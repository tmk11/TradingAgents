"""Lazy facade between dataflow modules and the news/macro archive.

The dataflow modules (``yfinance_news``, ``gold_news``, ``macro_data``)
must stay independent of the optional retrieval layer:

- They are imported during test collection on machines that don't
  have ``chromadb`` or an embedding API key.
- They are also called by the CLI on every run regardless of whether
  RAG is enabled.

This module provides three side-effect-free, never-raising functions
for those callers to use. Each one:

1. Reads the active dataflows config to decide whether the archive
   is enabled. ``False`` → no-op.
2. Lazily builds a singleton :class:`NewsArchive` on first use, with
   any failure (missing chroma, bad embedder, IO error) silently
   degrading to "archive disabled for this process".
3. Catches every exception inside the indexing call. A flaky archive
   must never fail a news fetch; it must only fail to record.

The functions are intentionally **fire-and-forget**. There is no
return value because callers should not branch on archive state —
the data they need is the same shape with or without indexing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


# Process-wide singletons keyed by the resolved archive path. Reusing
# one Chroma client per path keeps CPython happy (chroma maintains
# an HNSW index in-process; opening multiple writers to the same
# directory produces lock contention warnings).
_archive_cache: Dict[str, Optional[Any]] = {}


def _build_archive(config: Dict[str, Any]):
    """Construct a :class:`NewsArchive` from config. ``None`` on failure."""
    try:
        from tradingagents.retrieval import (
            MemoryVectorStore,
            NewsArchive,
            create_embedder,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("News archive imports failed (%s) — disabled.", exc)
        return None

    provider = config.get("rag_embedding_provider", "openai")
    model = config.get("rag_embedding_model", "text-embedding-3-small")
    path = config.get("news_archive_path") or ":memory:"

    try:
        embedder = create_embedder(provider=provider, model=model)
        store = MemoryVectorStore(
            path=path,
            embedder=embedder,
            collection_name="news_archive",
            embedder_name=f"{provider}:{model}",
        )
        return NewsArchive(store)
    except Exception as exc:
        logger.warning(
            "News archive init failed (path=%s, provider=%s): %s — disabled.",
            path, provider, exc,
        )
        return None


def _get_archive():
    """Return the cached archive for the active config, or ``None``."""
    # Late import — get_config lives in the same dataflows package and
    # importing at module top would create a circular dependency on
    # config.py at collection time.
    from tradingagents.dataflows.config import get_config

    config = get_config()
    if not config.get("news_archive_enabled"):
        return None

    path = str(config.get("news_archive_path") or ":memory:")
    if path not in _archive_cache:
        _archive_cache[path] = _build_archive(config)
    return _archive_cache[path]


def reset_cache() -> None:
    """Drop cached archive instances. Tests use this between cases."""
    _archive_cache.clear()


# ---------------------------------------------------------------------------
# Public side-effect API
# ---------------------------------------------------------------------------


def record_news_articles(
    articles: Iterable[Any],
    *,
    source: str,
    ticker: Optional[str] = None,
) -> None:
    """Index a batch of news article dicts.

    Accepts the loose dict shape the dataflow modules already produce
    (``title``, ``summary``/``description``, ``publisher``, ``link``,
    ``pub_date``/``published_at``).  Falls through to the
    :class:`ArchiveArticle.from_dict` adapter so callers don't have
    to import dataclasses.

    Args:
        articles: iterable of dicts — anything truthy with at least a
            ``title``.  Empty iterables are a fast no-op.
        source: tag identifying the fetcher (e.g.
            ``"yfinance:global"``, ``"rss:mining.com"``). Used both
            for the dedup key and as a metadata column.
        ticker: optional ticker symbol when the source is ticker-news.
    """
    archive = _get_archive()
    if archive is None:
        return
    try:
        # Late import keeps the no-op path free of dataclass machinery.
        from tradingagents.retrieval.news_archive import ArchiveArticle

        records = [
            ArchiveArticle.from_dict(a, source=source, ticker=ticker)
            for a in articles or []
            if isinstance(a, dict)
        ]
        if not records:
            return
        archive.index_articles(records)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("News archive record failed (source=%s): %s", source, exc)


def record_macro_snapshot(
    document: str,
    *,
    curr_date: str,
    lookback_days: int,
) -> None:
    """Index one macro-data snapshot rendered by ``fetch_gold_macro_data``."""
    archive = _get_archive()
    if archive is None:
        return
    try:
        archive.index_macro_snapshot(
            document, curr_date=curr_date, lookback_days=lookback_days
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Macro snapshot record failed (date=%s): %s", curr_date, exc
        )
