"""Tests for the news + macro archive RAG layer.

All tests run fully offline:

- ``FakeEmbedder`` (deterministic hash-based) provides embeddings.
- ``yfinance`` and the RSS fetchers are mocked at their module-level
  entry points so the dataflow integration tests don't need network.

The archive itself is built on the same Chroma adapter used in the
decision-log RAG, so the storage layer is exercised end-to-end (just
on a tmp_path).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import tradingagents.default_config as default_config
import tradingagents.dataflows._archive_indexer as indexer_mod
from tradingagents.dataflows.config import set_config
from tradingagents.retrieval import (
    ArchiveArticle,
    FakeEmbedder,
    MemoryVectorStore,
    NewsArchive,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def archive(tmp_path):
    """Real Chroma-backed NewsArchive on tmp_path with the fake embedder."""
    store = MemoryVectorStore(
        path=str(tmp_path / "news_archive"),
        embedder=FakeEmbedder(dim=128),
        collection_name="news_archive_test",
        embedder_name="test-fake",
    )
    return NewsArchive(store)


@pytest.fixture()
def archive_enabled_config(tmp_path, monkeypatch):
    """Set up dataflows config so the indexer facade routes to a tmp archive.

    Yields a tuple ``(config, archive_path)`` so individual tests can
    introspect both the config and the on-disk path. ``reset_cache``
    runs in teardown so the indexer does not hold a stale Chroma
    handle between tests (Chroma keeps an HNSW index in-process).
    """
    archive_path = tmp_path / "news_archive_chroma"
    cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
    cfg["news_archive_enabled"] = True
    cfg["news_archive_path"] = str(archive_path)
    cfg["rag_embedding_provider"] = "fake"
    cfg["rag_embedding_model"] = "test-fake"
    set_config(cfg)
    indexer_mod.reset_cache()
    yield cfg, archive_path
    indexer_mod.reset_cache()


# ---------------------------------------------------------------------------
# 1. ArchiveArticle dataclass + dict adapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArchiveArticle:
    def test_from_dict_handles_pub_date_alias(self):
        # yfinance hands us ``pub_date`` (datetime); the gold_news RSS
        # fetcher also uses ``pub_date``. The adapter must accept both.
        d = datetime(2026, 5, 10, tzinfo=timezone.utc)
        art = ArchiveArticle.from_dict(
            {"title": "x", "summary": "y", "publisher": "z",
             "link": "https://e/1", "pub_date": d},
            source="yfinance:ticker",
            ticker="GLD",
        )
        assert art.published_at == d
        assert art.ticker == "GLD"

    def test_from_dict_handles_iso_string(self):
        # API responses sometimes give ISO strings instead of datetimes.
        art = ArchiveArticle.from_dict(
            {"title": "t", "published_at": "2026-05-10T12:00:00Z"},
            source="rss:example.com",
        )
        assert art.published_at is not None
        assert art.published_at.year == 2026

    def test_from_dict_uppercases_ticker(self):
        art = ArchiveArticle.from_dict({"title": "t"}, source="x", ticker="gld")
        assert art.ticker == "GLD"

    def test_from_dict_tolerates_missing_fields(self):
        # An item with only a title must still index — the dataflow
        # parsers occasionally produce malformed rows and we don't
        # want to drop the whole batch over one of them.
        art = ArchiveArticle.from_dict({"title": "Hello"}, source="rss:x")
        assert art.title == "Hello"
        assert art.published_at is None
        assert art.summary == ""


# ---------------------------------------------------------------------------
# 2. NewsArchive — indexing + search semantics
# ---------------------------------------------------------------------------


def _article(
    *,
    title: str,
    summary: str = "",
    published_at: datetime | None = None,
    source: str = "rss:test.com",
    ticker: str | None = None,
    link: str = "",
) -> ArchiveArticle:
    return ArchiveArticle(
        title=title,
        summary=summary,
        publisher="TestPublisher",
        link=link or f"https://test/{title}",
        source=source,
        ticker=ticker.upper() if ticker else None,
        published_at=published_at,
    )


@pytest.mark.unit
class TestNewsArchive:
    def test_index_articles_dedup_by_link(self, archive):
        art = _article(title="Fed pivots dovish", link="https://x/1")
        n1 = archive.index_articles([art])
        n2 = archive.index_articles([art])
        # Both upserts succeed (idempotent), but the count stays at 1.
        assert n1 == 1
        assert n2 == 1
        assert archive.count() == 1

    def test_index_articles_skips_titleless(self, archive):
        # Without a title there's nothing to embed; skipping is the
        # correct behaviour rather than letting the rest of the batch
        # fail because of one malformed row.
        good = _article(title="Real")
        bad = _article(title="")
        written = archive.index_articles([good, bad])
        assert written == 1
        assert archive.count() == 1

    def test_search_articles_filters_by_ticker(self, archive):
        archive.index_articles([
            _article(title="GLD ETF inflows surge", ticker="GLD"),
            _article(title="NVDA AI capex", ticker="NVDA"),
            _article(title="Fed dovish pivot"),  # no ticker
        ])
        hits = archive.search_articles("anything", ticker="GLD")
        assert len(hits) == 1
        assert hits[0]["metadata"]["ticker"] == "GLD"

    def test_search_articles_excludes_macro_snapshots(self, archive):
        archive.index_articles([_article(title="GLD ETF inflows")])
        archive.index_macro_snapshot(
            "macro block", curr_date="2026-05-10", lookback_days=90
        )
        # Article search must not surface the macro snapshot even
        # though both records share the embedding space.
        hits = archive.search_articles("GLD")
        assert all(h["metadata"]["kind"] == "article" for h in hits)

    def test_search_articles_days_back_filters_old(self, archive):
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        new = datetime(2026, 5, 10, tzinfo=timezone.utc)
        archive.index_articles([
            _article(title="Old story", published_at=old, link="o"),
            _article(title="Recent story", published_at=new, link="n"),
        ])
        as_of = datetime(2026, 5, 11, tzinfo=timezone.utc)
        hits = archive.search_articles(
            "story", days_back=30, as_of=as_of
        )
        titles = [h["metadata"]["title"] for h in hits]
        assert "Recent story" in titles
        assert "Old story" not in titles

    def test_search_articles_as_of_blocks_lookahead(self, archive):
        # Backtesting: never retrieve articles dated in the analyst's future.
        future = datetime(2026, 6, 1, tzinfo=timezone.utc)
        past = datetime(2026, 5, 1, tzinfo=timezone.utc)
        archive.index_articles([
            _article(title="Past dovish call", published_at=past, link="p"),
            _article(title="Future Fed minutes", published_at=future, link="f"),
        ])
        as_of = datetime(2026, 5, 15, tzinfo=timezone.utc)
        hits = archive.search_articles(
            "Fed", days_back=None, as_of=as_of
        )
        titles = [h["metadata"]["title"] for h in hits]
        assert "Past dovish call" in titles
        assert "Future Fed minutes" not in titles

    def test_index_macro_snapshot_idempotent(self, archive):
        first = archive.index_macro_snapshot(
            "block A", curr_date="2026-05-10", lookback_days=90
        )
        second = archive.index_macro_snapshot(
            "block A revised", curr_date="2026-05-10", lookback_days=90
        )
        assert first is True and second is True
        # Same key, so upsert keeps a single row.
        assert archive.count() == 1

    def test_search_macro_snapshots_returns_only_snapshots(self, archive):
        archive.index_articles([_article(title="Some article")])
        archive.index_macro_snapshot(
            "DXY soft, real yields falling",
            curr_date="2026-05-10",
            lookback_days=90,
        )
        hits = archive.search_macro_snapshots("real yields falling")
        assert len(hits) == 1
        assert hits[0]["metadata"]["kind"] == "macro_snapshot"

    def test_format_articles_renders_markdown_block(self, archive):
        archive.index_articles([
            _article(
                title="Gold breaks $3000",
                summary="Spot gold pierced $3000/oz on dovish Fed expectations.",
                published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
                link="https://example/gold",
            )
        ])
        hits = archive.search_articles("gold $3000")
        rendered = archive.format_articles(hits, header="Test header")
        assert "## Test header" in rendered
        assert "Gold breaks $3000" in rendered
        assert "Spot gold pierced" in rendered
        assert "Link: https://example/gold" in rendered
        # Date tag rendered.
        assert "2026-05-10" in rendered

    def test_format_macro_snapshots(self, archive):
        archive.index_macro_snapshot(
            "DXY soft, real yields falling",
            curr_date="2026-05-10",
            lookback_days=90,
        )
        hits = archive.search_macro_snapshots("real yields")
        rendered = archive.format_macro_snapshots(hits)
        assert "Snapshot 2026-05-10" in rendered
        assert "90d window" in rendered
        assert "DXY soft" in rendered

    def test_format_articles_empty_hits_returns_empty(self, archive):
        assert archive.format_articles([]) == ""
        assert archive.format_macro_snapshots([]) == ""


# ---------------------------------------------------------------------------
# 3. _archive_indexer facade — config gating + caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArchiveIndexerFacade:
    def test_disabled_means_no_archive(self, monkeypatch):
        # Default config has news_archive_enabled=False.
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        indexer_mod.reset_cache()
        # Both record functions must be silent no-ops when disabled —
        # no exception, no Chroma init, no _archive_cache entry.
        indexer_mod.record_news_articles(
            [{"title": "x"}], source="yfinance:ticker", ticker="X"
        )
        indexer_mod.record_macro_snapshot(
            "macro", curr_date="2026-05-10", lookback_days=90
        )
        assert indexer_mod._archive_cache == {}

    def test_enabled_indexes_articles(self, archive_enabled_config):
        cfg, _ = archive_enabled_config
        indexer_mod.record_news_articles(
            [{"title": "Fed pivots", "summary": "Real yields fall."}],
            source="yfinance:global",
        )
        # Reach into the cache to verify the archive saw the row.
        archive = indexer_mod._get_archive()
        assert archive is not None
        assert archive.count() == 1

    def test_enabled_indexes_macro_snapshot(self, archive_enabled_config):
        indexer_mod.record_macro_snapshot(
            "DXY soft", curr_date="2026-05-10", lookback_days=90
        )
        archive = indexer_mod._get_archive()
        assert archive.count() == 1
        # Round-trip search must surface it.
        hits = archive.search_macro_snapshots("DXY soft")
        assert len(hits) == 1

    def test_record_news_handles_empty_list(self, archive_enabled_config):
        # Empty iterables are common (a fetch returns no items); they
        # must not pull a Chroma client into the cache.
        indexer_mod.record_news_articles(
            [], source="yfinance:global"
        )
        # Either no archive built or no rows written — but never raise.
        archive = indexer_mod._get_archive()
        if archive is not None:
            assert archive.count() == 0

    def test_init_failure_degrades_to_none(self, monkeypatch, tmp_path):
        # Force the embedder factory to raise. The facade must log and
        # return None instead of bubbling the failure up to the
        # dataflow caller (which is the fetch path of every analyst).
        import tradingagents.retrieval as retrieval_mod

        cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
        cfg["news_archive_enabled"] = True
        cfg["news_archive_path"] = str(tmp_path / "broken")
        set_config(cfg)
        indexer_mod.reset_cache()

        def _broken(*a, **kw):
            raise RuntimeError("simulated import failure")

        monkeypatch.setattr(retrieval_mod, "create_embedder", _broken)
        # Calling the public function must not raise.
        indexer_mod.record_news_articles(
            [{"title": "x"}], source="x"
        )
        # Cached entry resolved to None for this path so subsequent
        # calls short-circuit.
        assert indexer_mod._archive_cache.get(str(tmp_path / "broken")) is None


# ---------------------------------------------------------------------------
# 4. Dataflow integration — yfinance_news + gold_news + macro_data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataflowIndexingIntegration:
    """Integration: a fetch with archive enabled populates the archive.

    These tests mock the upstream fetchers (yfinance / urllib RSS) at
    the module boundary so the dataflow code under test runs end-to-end
    without network. The assertion is on archive contents — proving
    that the side-effect hook fires for every fetch path we care about.
    """

    def test_get_news_yfinance_indexes_articles(self, archive_enabled_config):
        from tradingagents.dataflows import yfinance_news

        fake_news = [
            {
                "content": {
                    "title": "GLD breaks $3000 on dovish Fed",
                    "summary": "Spot gold pierced $3000/oz amid pivot expectations.",
                    "provider": {"displayName": "Bloomberg"},
                    "canonicalUrl": {"url": "https://news/gld-3000"},
                    "pubDate": "2026-05-10T12:00:00Z",
                }
            }
        ]
        with patch("yfinance.Ticker") as mock_ticker:
            instance = MagicMock()
            instance.get_news.return_value = fake_news
            mock_ticker.return_value = instance
            yfinance_news.get_news_yfinance("GLD", "2026-05-09", "2026-05-11")

        archive = indexer_mod._get_archive()
        assert archive is not None
        assert archive.count() == 1
        hits = archive.search_articles("dovish Fed", ticker="GLD")
        assert any("GLD" == h["metadata"]["ticker"] for h in hits)

    def test_get_global_news_yfinance_indexes_articles(self, archive_enabled_config):
        from tradingagents.dataflows import yfinance_news

        fake_articles = [
            {
                "content": {
                    "title": "Fed cuts rates 25bp",
                    "summary": "Real yields fell sharply.",
                    "provider": {"displayName": "Reuters"},
                    "canonicalUrl": {"url": "https://news/fed-cut"},
                    "pubDate": "2026-05-10T08:00:00Z",
                }
            }
        ]
        fake_search = MagicMock()
        fake_search.news = fake_articles

        with patch("yfinance.Search", return_value=fake_search):
            yfinance_news.get_global_news_yfinance(
                "2026-05-11", look_back_days=7, limit=5
            )

        archive = indexer_mod._get_archive()
        assert archive is not None
        # Archive should contain the article keyed without a ticker.
        hits = archive.search_articles("Fed cuts rates")
        assert hits
        # No ticker on global articles.
        assert all("ticker" not in h["metadata"] for h in hits)

    def test_gold_news_fetch_feed_indexes_articles(self, archive_enabled_config):
        from tradingagents.dataflows import gold_news

        # Mock the underlying RSS fetcher rather than the network so
        # the parsed-item shape stays under test (item dict format).
        fake_items = [
            {
                "title": "Mining.com: GDX rallies on capex cuts",
                "link": "https://mining.com/x",
                "description": "Sector trims spend amid bullion strength.",
                "pub_date": datetime(2026, 5, 10, tzinfo=timezone.utc),
            }
        ]
        with patch.object(gold_news, "_fetch_rss_items", return_value=fake_items):
            feed = gold_news.GoldNewsFeed(
                label="Mining.com (test)",
                url="https://mining.com/feed/",
            )
            gold_news.fetch_feed(feed, look_back_days=7, limit=5, curr_date="2026-05-11")

        archive = indexer_mod._get_archive()
        assert archive is not None
        hits = archive.search_articles("GDX rallies")
        assert hits
        # Source tag must be derived from the RSS host (rss:mining.com).
        assert any(h["metadata"]["source"] == "rss:mining.com" for h in hits)

    def test_macro_data_indexes_snapshot(self, archive_enabled_config):
        from tradingagents.dataflows import macro_data

        # Skip every series fetch — the test only cares that the
        # rendered block triggers record_macro_snapshot.
        with patch.object(
            macro_data, "fetch_one_series", return_value="(stub series)\n"
        ):
            text = macro_data.fetch_gold_macro_data(
                "2026-05-10", lookback_days=30,
                series=[macro_data.MACRO_SERIES_GOLD[0]],  # 1 series is enough
            )
        assert "Gold-driver macro data" in text
        archive = indexer_mod._get_archive()
        assert archive is not None
        hits = archive.search_macro_snapshots("Gold-driver macro")
        assert len(hits) == 1
        assert hits[0]["metadata"]["lookback_days"] == 30
        assert hits[0]["metadata"]["curr_date"] == "2026-05-10"

    def test_disabled_archive_is_never_populated(self, tmp_path):
        # No fixture: the archive is intentionally disabled.
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        indexer_mod.reset_cache()

        from tradingagents.dataflows import macro_data

        with patch.object(
            macro_data, "fetch_one_series", return_value="(stub)\n"
        ):
            macro_data.fetch_gold_macro_data(
                "2026-05-10", lookback_days=30,
                series=[macro_data.MACRO_SERIES_GOLD[0]],
            )

        # The archive should never have been built.
        assert indexer_mod._archive_cache == {}


# ---------------------------------------------------------------------------
# 5. Tool wrappers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArchiveSearchTools:
    def test_search_news_archive_disabled_returns_placeholder(self):
        from tradingagents.agents.utils.archive_search_tools import (
            search_news_archive,
        )
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        indexer_mod.reset_cache()

        out = search_news_archive.func(query="anything")
        assert "[news archive disabled" in out

    def test_search_news_archive_empty_returns_placeholder(self, archive_enabled_config):
        from tradingagents.agents.utils.archive_search_tools import (
            search_news_archive,
        )
        out = search_news_archive.func(query="anything")
        assert "[news archive empty" in out

    def test_search_news_archive_returns_formatted_results(self, archive_enabled_config):
        from tradingagents.agents.utils.archive_search_tools import (
            search_news_archive,
        )
        # Seed via the indexer so the same archive instance is used.
        indexer_mod.record_news_articles(
            [
                {
                    "title": "GLD breaks $3000 on dovish Fed",
                    "summary": "Spot gold pierced $3000/oz.",
                    "publisher": "Bloomberg",
                    "link": "https://news/gld-3000",
                    "pub_date": datetime(2026, 5, 10, tzinfo=timezone.utc),
                }
            ],
            source="yfinance:ticker",
            ticker="GLD",
        )

        out = search_news_archive.func(
            query="dovish Fed", ticker="GLD", days_back=None,
        )
        assert "GLD breaks $3000" in out
        assert "ticker=GLD" in out  # header includes the filter

    def test_search_news_archive_no_match_message(self, archive_enabled_config):
        from tradingagents.agents.utils.archive_search_tools import (
            search_news_archive,
        )
        indexer_mod.record_news_articles(
            [{"title": "Generic article"}], source="rss:test.com"
        )
        # Filter to a ticker we never indexed — must produce a clear
        # "no matches" message rather than empty string or crash.
        out = search_news_archive.func(query="anything", ticker="NVDA")
        assert "no archived articles" in out or "[news archive empty" in out

    def test_search_macro_archive_returns_snapshot(self, archive_enabled_config):
        from tradingagents.agents.utils.archive_search_tools import (
            search_macro_archive,
        )
        indexer_mod.record_macro_snapshot(
            "DXY soft, real yields falling, gold supportive",
            curr_date="2026-05-10",
            lookback_days=90,
        )
        out = search_macro_archive.func(query="real yields falling")
        assert "Snapshot 2026-05-10" in out
        assert "DXY soft" in out


# ---------------------------------------------------------------------------
# 6. Analyst tool binding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalystToolBinding:
    """Confirm news_analyst and market_analyst pick up the search tools.

    We don't run the full agent loop; we just instantiate the node
    closure with a stub LLM that records the tool list passed to
    bind_tools. The closure must compose the same list every run, so
    one invocation is enough to lock in the wiring.
    """

    @staticmethod
    def _capturing_llm():
        captured = {"tools": None}

        class _Stub:
            def bind_tools(self, tools):
                captured["tools"] = list(tools)
                # bind_tools must be a fluent setter; return something
                # that accepts ``invoke`` so the chain composes.
                def _runner(_msgs):
                    return MagicMock(content="", tool_calls=[])
                fake = MagicMock()
                fake.invoke = _runner
                return fake

        return _Stub(), captured

    def _state(self, asset_type="stock"):
        return {
            "trade_date": "2026-05-22",
            "asset_type": asset_type,
            "company_of_interest": "GLD",
            "messages": [],
        }

    def test_news_analyst_includes_search_tool_when_enabled(
        self, archive_enabled_config
    ):
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        stub, captured = self._capturing_llm()
        node = create_news_analyst(stub)
        # Run once with archive enabled.
        node(self._state(asset_type="stock"))
        names = [t.name for t in captured["tools"]]
        assert "search_news_archive" in names

    def test_news_analyst_excludes_search_tool_when_disabled(self):
        from tradingagents.agents.analysts.news_analyst import create_news_analyst

        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        indexer_mod.reset_cache()

        stub, captured = self._capturing_llm()
        node = create_news_analyst(stub)
        node(self._state(asset_type="stock"))
        names = [t.name for t in captured["tools"]]
        assert "search_news_archive" not in names

    def test_market_analyst_commodity_includes_search_tool(
        self, archive_enabled_config
    ):
        from tradingagents.agents.analysts.market_analyst import (
            create_market_analyst,
        )

        stub, captured = self._capturing_llm()
        node = create_market_analyst(stub)
        node(self._state(asset_type="commodity"))
        names = [t.name for t in captured["tools"]]
        assert "search_macro_archive" in names
        assert "get_macro_data" in names  # original tool still present

    def test_market_analyst_stock_excludes_macro_tools(
        self, archive_enabled_config
    ):
        from tradingagents.agents.analysts.market_analyst import (
            create_market_analyst,
        )

        stub, captured = self._capturing_llm()
        node = create_market_analyst(stub)
        node(self._state(asset_type="stock"))
        names = [t.name for t in captured["tools"]]
        # Equity runs see the original tool set unchanged regardless
        # of whether the news/macro archive is enabled.
        assert "search_macro_archive" not in names
        assert "get_macro_data" not in names
