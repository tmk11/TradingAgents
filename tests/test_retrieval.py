"""Tests for the optional RAG / semantic-memory layer.

All tests use the deterministic ``FakeEmbedder`` so they run fully
offline and don't require an OpenAI key. The Chroma store is created
in ephemeral mode (``:memory:``) where possible to keep the working
tree clean.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.retrieval import (
    FakeEmbedder,
    MemoryVectorStore,
    SemanticMemoryRetriever,
    create_embedder,
)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class TestFakeEmbedder:
    def test_returns_unit_vectors(self):
        emb = FakeEmbedder(dim=64)
        vec = emb.embed_query("Buy GLD on falling real yields")
        assert len(vec) == 64
        norm = sum(v * v for v in vec) ** 0.5
        # L2-normalised → norm ≈ 1.0 (allow float jitter)
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic(self):
        emb = FakeEmbedder(dim=32)
        a = emb.embed_query("DXY softening, gold catching a bid")
        b = emb.embed_query("DXY softening, gold catching a bid")
        assert a == b

    def test_similar_text_higher_cosine_than_unrelated(self):
        emb = FakeEmbedder(dim=128)
        anchor = emb.embed_query(
            "Fed cuts rates real yields drop bullish for gold"
        )
        related = emb.embed_query(
            "Fed cuts rates and real yields drop, supportive for gold"
        )
        unrelated = emb.embed_query(
            "Apple earnings beat services growth strong"
        )

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_related = cosine(anchor, related)
        sim_unrelated = cosine(anchor, unrelated)
        assert sim_related > sim_unrelated

    def test_factory_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            create_embedder(provider="not-a-real-provider")

    def test_factory_returns_fake(self):
        emb = create_embedder(provider="fake")
        assert isinstance(emb, FakeEmbedder)


# ---------------------------------------------------------------------------
# MemoryVectorStore + SemanticMemoryRetriever
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Persistent Chroma store on tmp_path (gets torn down with the path)."""
    return MemoryVectorStore(
        path=str(tmp_path / "chroma"),
        embedder=FakeEmbedder(),
        embedder_name="test-fake",
    )


@pytest.fixture()
def retriever(store):
    return SemanticMemoryRetriever(store)


class TestVectorStoreBasics:
    def test_empty_store_search_returns_empty(self, store):
        assert store.search("anything", n_results=5) == []
        assert store.count() == 0

    def test_upsert_and_count(self, store):
        store.upsert("NVDA:2026-01-10", "Buy NVDA AI capex", {"ticker": "NVDA"})
        assert store.count() == 1

    def test_upsert_idempotent(self, store):
        store.upsert("X:1", "A", {"ticker": "X"})
        store.upsert("X:1", "A revised", {"ticker": "X"})
        assert store.count() == 1

    def test_metadata_none_dropped(self, store):
        # None values must not crash the chroma metadata validator.
        store.upsert("X:1", "doc", {"ticker": "X", "alpha": None, "rating": "Buy"})
        hits = store.search("doc", n_results=1)
        assert hits and hits[0]["metadata"].get("ticker") == "X"
        assert "alpha" not in hits[0]["metadata"]


class TestRetrieverSearch:
    def _seed(self, retriever, *entries):
        for e in entries:
            retriever.index_entry(e)

    def test_index_entry_skipped_when_missing_keys(self, retriever):
        # Missing ticker / date is a parser bug — skip silently rather
        # than raise from inside the trading run.
        retriever.index_entry({"decision": "no ticker"})
        same, cross = retriever.search("anything", ticker="GLD", n_same=5, n_cross=3)
        assert same == [] and cross == []

    def test_same_ticker_query(self, retriever):
        self._seed(
            retriever,
            {
                "ticker": "GLD", "date": "2026-01-10", "rating": "Buy",
                "decision": "Buy GLD on Fed pivot dovish, real yields falling",
                "reflection": "Correct call, +3% in 5d", "pending": False,
            },
            {
                "ticker": "GLD", "date": "2026-02-10", "rating": "Sell",
                "decision": "Sell GLD on hawkish surprise, dollar bid",
                "reflection": "Right side, -2% reversal", "pending": False,
            },
            {
                "ticker": "AAPL", "date": "2026-01-05", "rating": "Buy",
                "decision": "Buy AAPL on services growth",
                "reflection": "Earnings confirmed", "pending": False,
            },
        )
        same, cross = retriever.search(
            query="Fed pivot dovish real yields drop gold supportive",
            ticker="GLD",
            n_same=5,
            n_cross=3,
        )
        # Same-ticker filter must not bleed AAPL into the same-list.
        assert all(h["metadata"]["ticker"] == "GLD" for h in same)
        # Cross-ticker filter must not return GLD into the cross-list.
        assert all(h["metadata"]["ticker"] != "GLD" for h in cross)
        # The most relevant same-ticker doc should be the dovish one,
        # not the hawkish one — fake embedder ranks shared tokens.
        assert same and "dovish" in same[0]["document"]

    def test_pending_excluded_from_search(self, retriever):
        # A still-pending entry must never be returned as past_context.
        self._seed(
            retriever,
            {
                "ticker": "GLD", "date": "2026-03-01", "rating": "Buy",
                "decision": "Buy GLD pending outcome",
                "reflection": "", "pending": True,
            },
        )
        same, cross = retriever.search("Buy GLD", ticker="GLD")
        assert same == []
        assert cross == []

    def test_cross_ticker_requires_reflection(self, retriever):
        self._seed(
            retriever,
            # Resolved but no reflection — cross-ticker search must skip.
            {
                "ticker": "AAPL", "date": "2026-01-05", "rating": "Buy",
                "decision": "Buy AAPL", "reflection": "", "pending": False,
            },
            # Resolved with reflection — eligible.
            {
                "ticker": "MSFT", "date": "2026-01-06", "rating": "Hold",
                "decision": "Hold MSFT", "reflection": "Lesson: timing matters",
                "pending": False,
            },
        )
        _, cross = retriever.search(
            "any", ticker="GLD", n_same=0, n_cross=5,
        )
        tickers = [h["metadata"]["ticker"] for h in cross]
        assert "MSFT" in tickers
        assert "AAPL" not in tickers

    def test_format_context_renders_headers(self, retriever):
        self._seed(
            retriever,
            {
                "ticker": "GLD", "date": "2026-01-10", "rating": "Buy",
                "decision": "Buy GLD on dovish Fed",
                "reflection": "Correct.", "pending": False,
            },
        )
        same, cross = retriever.search("dovish Fed", ticker="GLD")
        out = retriever.format_context("GLD", same, cross)
        assert "Past analyses of GLD" in out
        assert "Buy GLD on dovish Fed" in out

    def test_format_context_empty_returns_empty_string(self, retriever):
        out = retriever.format_context("GLD", [], [])
        assert out == ""


# ---------------------------------------------------------------------------
# TradingMemoryLog: RAG-enabled paths
# ---------------------------------------------------------------------------


def _rag_config(tmp_path, **overrides):
    """Build a memory-log config that wires the fake embedder + tmp store."""
    cfg = {
        "memory_log_path": str(tmp_path / "trading_memory.md"),
        "rag_enabled": True,
        "rag_embedding_provider": "fake",
        "rag_embedding_model": "test-fake",
        "rag_vector_store_path": str(tmp_path / "chroma"),
        "rag_n_same_ticker": 5,
        "rag_n_cross_ticker": 3,
    }
    cfg.update(overrides)
    return cfg


class TestTradingMemoryLogRagEnabled:
    def test_disabled_by_default_means_no_retriever_init(self, tmp_path):
        log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
        # Internal state — defensive: confirms the lazy guard truly
        # short-circuits when ``rag_enabled`` is missing/False.
        assert log.rag_enabled is False
        assert log._get_retriever() is None

    def test_index_on_store_decision(self, tmp_path):
        log = TradingMemoryLog(_rag_config(tmp_path))
        log.store_decision("GLD", "2026-01-10", "Rating: Buy\nBuy GLD.")
        retriever = log._get_retriever()
        assert retriever is not None
        # Pending entries are indexed but excluded from same-ticker search.
        same, cross = retriever.search("Buy", ticker="GLD")
        assert same == []
        # Index size should reflect the inserted (pending) entry.
        assert retriever._store.count() == 1

    def test_index_on_batch_update(self, tmp_path):
        log = TradingMemoryLog(_rag_config(tmp_path))
        log.store_decision("GLD", "2026-01-10", "Rating: Buy\nBuy GLD on dovish Fed.")
        log.batch_update_with_outcomes([{
            "ticker": "GLD",
            "trade_date": "2026-01-10",
            "raw_return": 0.03,
            "alpha_return": 0.015,
            "holding_days": 5,
            "reflection": "Correct call, dovish Fed confirmed.",
        }])

        retriever = log._get_retriever()
        same, _ = retriever.search("dovish Fed", ticker="GLD")
        assert same, "expected at least one same-ticker hit after resolution"
        # The resolved doc should now carry the reflection text.
        assert any("dovish Fed confirmed" in h["document"] for h in same)

    def test_get_past_context_semantic_returns_relevant_hit(self, tmp_path):
        log = TradingMemoryLog(_rag_config(tmp_path))
        # Two GLD analyses; one matches the query, the other doesn't.
        for date, decision, reflection in [
            ("2026-01-10", "Rating: Buy\nBuy GLD on falling real yields, dovish Fed",
             "Correct: real yields fell, GLD rallied"),
            ("2026-02-10", "Rating: Sell\nSell GLD on hawkish surprise, DXY bid",
             "Wrong call: gold held up despite DXY"),
        ]:
            log.store_decision("GLD", date, decision)
            log.batch_update_with_outcomes([{
                "ticker": "GLD", "trade_date": date,
                "raw_return": 0.03, "alpha_return": 0.01, "holding_days": 5,
                "reflection": reflection,
            }])

        ctx = log.get_past_context_semantic(
            query="Fed cuts rates real yields drop bullish gold",
            ticker="GLD",
            n_same=2,
            n_cross=0,
        )
        assert ctx, "semantic context should not be empty"
        # Top hit (first entry rendered after the section header) must
        # be the dovish-Fed entry, not the hawkish one. The dovish text
        # should appear before the hawkish text in the rendered string.
        dovish_pos = ctx.find("dovish Fed")
        hawkish_pos = ctx.find("hawkish surprise")
        assert dovish_pos != -1, f"dovish entry not in context: {ctx!r}"
        if hawkish_pos != -1:
            assert dovish_pos < hawkish_pos, (
                "dovish-Fed entry must rank above hawkish entry for a "
                "'Fed cuts rates' query"
            )

    def test_falls_back_to_recency_on_init_failure(self, tmp_path, monkeypatch):
        # Force the retriever import to fail. The log must silently
        # degrade to recency-based context — a trading run never breaks
        # because RAG infrastructure is misconfigured.
        import tradingagents.retrieval as retrieval_mod

        def _broken(*a, **kw):
            raise RuntimeError("simulated import failure")

        monkeypatch.setattr(retrieval_mod, "create_embedder", _broken)

        # Pre-seed a resolved entry through a plain (non-RAG) log
        # writing to the *same* markdown path the RAG log will read.
        cfg = _rag_config(tmp_path)
        plain = TradingMemoryLog({"memory_log_path": cfg["memory_log_path"]})
        plain.store_decision("GLD", "2026-01-10", "Rating: Buy\nBuy GLD")
        plain.batch_update_with_outcomes([{
            "ticker": "GLD", "trade_date": "2026-01-10",
            "raw_return": 0.02, "alpha_return": 0.01, "holding_days": 5,
            "reflection": "Recency lesson",
        }])

        rag_log = TradingMemoryLog(cfg)
        out = rag_log.get_past_context_semantic(
            query="anything", ticker="GLD", n_same=5, n_cross=0,
        )
        # Recency path includes the legacy "(most recent first)" header.
        assert "Past analyses of GLD" in out

    def test_bootstrap_indexes_existing_markdown(self, tmp_path):
        # 1. Write a resolved entry without RAG.
        cfg = _rag_config(tmp_path)
        plain = TradingMemoryLog({"memory_log_path": cfg["memory_log_path"]})
        plain.store_decision("GLD", "2026-01-10", "Rating: Buy\nBuy GLD on dovish Fed.")
        plain.batch_update_with_outcomes([{
            "ticker": "GLD", "trade_date": "2026-01-10",
            "raw_return": 0.03, "alpha_return": 0.015, "holding_days": 5,
            "reflection": "Correct call.",
        }])
        # 2. Now open the same log with RAG enabled. Bootstrap should
        #    index the pre-existing markdown entry on first retriever use.
        rag = TradingMemoryLog(cfg)
        retriever = rag._get_retriever()
        assert retriever is not None
        assert retriever._store.count() >= 1
