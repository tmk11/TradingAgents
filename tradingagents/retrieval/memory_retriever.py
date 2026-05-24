"""Semantic memory retriever — bridges the markdown log and the vector store.

:class:`SemanticMemoryRetriever` is a thin orchestrator:

- :meth:`index_entry` writes one parsed log entry into the vector store
  (idempotent via Chroma's upsert).
- :meth:`search` runs two scoped queries (same-ticker, cross-ticker)
  and returns ranked hits.
- :meth:`format_context` renders the hits in the same shape the legacy
  ``TradingMemoryLog.get_past_context`` produces, so the Portfolio
  Manager prompt does not need to change.

Keeping the formatting here (rather than in the log) means the log can
stay 100 % markdown-only and tests for the recency-based path don't
have to mock chromadb.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def _entry_id(ticker: str, date: str) -> str:
    """Stable id used by both the indexer and the markdown parser."""
    return f"{ticker}:{date}"


def _build_index_text(entry: Dict[str, Any]) -> str:
    """Render a parsed entry into the text that gets embedded.

    Keep this deterministic: a re-index of the same entry must produce
    byte-identical text so chroma's content-hash dedup works.
    """
    parts: List[str] = [
        f"Ticker: {entry.get('ticker', '')}",
        f"Date: {entry.get('date', '')}",
        f"Rating: {entry.get('rating', '')}",
    ]
    raw = entry.get("raw")
    alpha = entry.get("alpha")
    if raw and raw != "pending":
        parts.append(f"Raw return: {raw}")
    if alpha:
        parts.append(f"Alpha: {alpha}")

    decision = (entry.get("decision") or "").strip()
    if decision:
        # 2 KB of decision text is plenty for embedding signal; longer
        # passages dilute the vector and inflate API cost.
        parts.append("Decision:\n" + decision[:2000])

    reflection = (entry.get("reflection") or "").strip()
    if reflection:
        parts.append("Reflection:\n" + reflection)
    return "\n\n".join(parts)


def _build_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flat metadata for chroma ``where`` filters."""
    return {
        "ticker": entry.get("ticker", ""),
        "date": entry.get("date", ""),
        "rating": entry.get("rating", ""),
        "pending": bool(entry.get("pending", False)),
        "has_reflection": bool((entry.get("reflection") or "").strip()),
    }


class SemanticMemoryRetriever:
    """Indexer + searcher on top of a :class:`MemoryVectorStore`."""

    def __init__(self, vector_store) -> None:  # type: ignore[no-untyped-def]
        self._store = vector_store

    # ---- Indexing ------------------------------------------------------

    def index_entry(self, entry: Dict[str, Any]) -> None:
        """Index a single parsed log entry.

        Errors are logged and swallowed: a transient embedding failure
        must never break the trading run, only degrade RAG quality.
        """
        ticker = entry.get("ticker")
        date = entry.get("date")
        if not ticker or not date:
            return
        try:
            self._store.upsert(
                entry_id=_entry_id(ticker, date),
                document=_build_index_text(entry),
                metadata=_build_metadata(entry),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RAG index failed for %s:%s — %s", ticker, date, exc)

    def reindex(self, entries: Sequence[Dict[str, Any]]) -> int:
        """Bulk index. Returns the number of entries successfully written."""
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for entry in entries:
            ticker = entry.get("ticker")
            date = entry.get("date")
            if not ticker or not date:
                continue
            ids.append(_entry_id(ticker, date))
            docs.append(_build_index_text(entry))
            metas.append(_build_metadata(entry))
        if not ids:
            return 0
        try:
            self._store.upsert_many(ids=ids, documents=docs, metadatas=metas)
            return len(ids)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RAG bulk index failed (%d entries) — %s", len(ids), exc)
            return 0

    # ---- Search --------------------------------------------------------

    def search(
        self,
        query: str,
        ticker: str,
        n_same: int = 5,
        n_cross: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Two scoped semantic queries: same-ticker and cross-ticker.

        Same-ticker hits include all resolved entries (decision + any
        reflection). Cross-ticker hits require a non-empty reflection
        — reflections are the cross-ticker signal worth carrying;
        unresolved decisions for other tickers add noise.
        """
        same: List[Dict[str, Any]] = []
        cross: List[Dict[str, Any]] = []

        if n_same > 0:
            try:
                same = self._store.search(
                    query=query,
                    n_results=n_same,
                    where={
                        "$and": [
                            {"ticker": {"$eq": ticker}},
                            {"pending": {"$eq": False}},
                        ]
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("RAG same-ticker search failed: %s", exc)

        if n_cross > 0:
            try:
                cross = self._store.search(
                    query=query,
                    n_results=n_cross,
                    where={
                        "$and": [
                            {"ticker": {"$ne": ticker}},
                            {"has_reflection": {"$eq": True}},
                            {"pending": {"$eq": False}},
                        ]
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("RAG cross-ticker search failed: %s", exc)

        return same, cross

    # ---- Formatting ----------------------------------------------------

    def format_context(
        self,
        ticker: str,
        same_hits: Sequence[Dict[str, Any]],
        cross_hits: Sequence[Dict[str, Any]],
    ) -> str:
        """Render hits into the prompt-ready string.

        Header wording mirrors :meth:`TradingMemoryLog.get_past_context`
        but explicitly notes "most semantically relevant" so prompts
        downstream can reason about retrieval provenance.
        """
        if not same_hits and not cross_hits:
            return ""

        parts: List[str] = []
        if same_hits:
            parts.append(f"Past analyses of {ticker} (most semantically relevant):")
            for hit in same_hits:
                parts.append(hit.get("document", ""))
        if cross_hits:
            parts.append("Relevant cross-ticker lessons:")
            for hit in cross_hits:
                parts.append(hit.get("document", ""))
        return "\n\n".join(p for p in parts if p)
