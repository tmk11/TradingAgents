"""LangChain ``@tool`` wrappers for the news + macro archive.

Two tools sit on top of :class:`tradingagents.retrieval.NewsArchive`:

- ``search_news_archive`` — semantic search over historical news
  articles, optionally scoped to a ticker and/or a recency window.
- ``search_macro_archive`` — semantic search over rendered macro
  snapshots (one per ``get_macro_data`` invocation).

Both tools are **safe to bind unconditionally**: when the archive is
disabled (``news_archive_enabled = False``) or empty (first run after
turning it on) they return a clearly-labelled placeholder string the
LLM can recognise as "nothing to retrieve here yet" rather than
crashing the analyst node.

The tools live in their own module so the import surface stays narrow:
``agent_utils`` already pulls in every fetch tool, and we don't want
the chromadb / langchain-openai chain dragged in for analysts that
never call these search tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper — resolves the archive once per invocation
# ---------------------------------------------------------------------------


def _get_archive():
    """Reach into the dataflow indexer's cache without re-implementing it.

    Both the indexer (write-side) and these tools (read-side) need the
    same singleton — opening two Chroma writers at the same path
    produces lock-contention warnings and risks corrupting the HNSW
    index. The indexer caches per-path; we just call its private
    accessor through a module-level alias to keep the surface narrow.
    """
    try:
        from tradingagents.dataflows._archive_indexer import _get_archive as _impl
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Archive indexer unavailable: %s", exc)
        return None
    try:
        return _impl()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Archive resolution failed: %s", exc)
        return None


def _parse_as_of(curr_date: Optional[str]) -> Optional[datetime]:
    """Normalise an upper-bound date string for back-tested look-aheads."""
    if not curr_date:
        return None
    try:
        return datetime.strptime(curr_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_news_archive(
    query: Annotated[
        str,
        "Free-text query, e.g. 'Fed dovish pivot real yields' or 'gold ETF outflows'",
    ],
    ticker: Annotated[
        Optional[str],
        "Optional ticker filter (case-insensitive). Pass None for cross-ticker macro search.",
    ] = None,
    days_back: Annotated[
        Optional[int],
        "Optional recency window — only return articles published within N days. "
        "Pass None to search the whole archive.",
    ] = 90,
    curr_date: Annotated[
        Optional[str],
        "Optional upper-bound date in yyyy-mm-dd. Required for back-tested runs "
        "to avoid retrieving future-dated articles.",
    ] = None,
    limit: Annotated[
        int,
        "Maximum number of articles to return (default 5).",
    ] = 5,
) -> str:
    """
    Semantic search over the persistent news archive.

    Use this when reasoning about a thesis whose drivers played out in the past
    (e.g. "how did gold react last time real yields fell this fast?"). The
    archive is populated automatically every time get_news, get_global_news,
    or get_gold_news runs, so analyses gradually accumulate a
    cross-run corpus the agent can mine.

    Returns a markdown block in the same shape as get_news so the LLM can mix
    archive results with fresh fetches without special-casing the formatting.
    Returns a clear placeholder string when the archive is disabled or empty.
    """
    archive = _get_archive()
    if archive is None:
        return (
            "[news archive disabled — set news_archive_enabled=true in config "
            "or TRADINGAGENTS_NEWS_ARCHIVE_ENABLED=true to enable]"
        )

    if archive.count() == 0:
        return (
            "[news archive empty — no articles indexed yet; this is normal on "
            "the first run after enabling the archive]"
        )

    hits = archive.search_articles(
        query=query,
        ticker=ticker,
        days_back=days_back,
        n_results=max(1, int(limit)),
        as_of=_parse_as_of(curr_date),
    )
    if not hits:
        return (
            f"[no archived articles match query={query!r}"
            + (f", ticker={ticker}" if ticker else "")
            + (f", days_back={days_back}" if days_back else "")
            + "]"
        )

    header = "Archived news (semantic match)"
    if ticker:
        header += f" — ticker={ticker.upper()}"
    if days_back:
        header += f", last {days_back}d"
    return archive.format_articles(hits, header=header)


@tool
def search_macro_archive(
    query: Annotated[
        str,
        "Free-text query, e.g. 'real yields falling DXY soft' or 'Fed balance sheet QT'",
    ],
    limit: Annotated[
        int,
        "Maximum number of historical snapshots to return (default 3).",
    ] = 3,
) -> str:
    """
    Semantic search over the persistent macro-snapshot archive.

    Each get_macro_data call indexes its rendered block as one snapshot keyed by
    date and lookback window. Use this to pull historical context — "what did
    DXY/real-yield/Fed-balance-sheet look like during the previous regime that
    matched today's setup?" — instead of paying a fresh fetch every analysis.

    Returns a markdown block with one snapshot per match. Returns a placeholder
    string when the archive is disabled or empty.
    """
    archive = _get_archive()
    if archive is None:
        return (
            "[macro archive disabled — set news_archive_enabled=true in config "
            "to enable]"
        )

    hits = archive.search_macro_snapshots(query=query, n_results=max(1, int(limit)))
    if not hits:
        return (
            f"[no archived macro snapshots match query={query!r} — this is "
            "normal until at least one get_macro_data call has run with the "
            "archive enabled]"
        )
    return archive.format_macro_snapshots(hits)
