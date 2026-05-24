"""Persistent news + macro archive built on top of :class:`MemoryVectorStore`.

The decision-log RAG (``SemanticMemoryRetriever``) is scoped to the
agent's own past decisions — short, structured, low-volume. The
**news/macro archive** is a different beast: high-volume free-text
documents pulled in by the dataflow tools every run.  Indexing them
in a separate collection keeps the two stores from contending on
embeddings, ``where`` filters, or rotation policy.

Two record kinds live here:

- **Articles** — a row per news item. Each carries a ``source`` tag
  (``yfinance:ticker``, ``yfinance:global``, ``rss:mining.com``…), an
  optional ``ticker`` (for ticker-news), and a ``published_at_ts``
  epoch timestamp so callers can do "last N days" `where` filters.
- **Macro snapshots** — one row per ``get_macro_data`` invocation.
  The full rendered block is the document, the trade date is the
  primary metadata, and ``kind="macro_snapshot"`` lets searches stay
  scoped without polluting the news ranking.

All operations are best-effort: any indexing or query failure logs at
WARNING and degrades gracefully.  A trading run never breaks because
the archive is misconfigured.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveArticle:
    """One news article ready to be indexed.

    The dataflow side speaks ``dict`` (yfinance returns dicts, RSS
    parsing returns dicts), so we provide ``from_dict`` for cheap
    interop without forcing every caller to import this dataclass.
    """

    title: str
    summary: str
    publisher: str
    link: str
    source: str                       # e.g. "yfinance:global", "rss:mining.com"
    ticker: Optional[str] = None      # set for ticker-news; None for global/macro
    published_at: Optional[datetime] = None

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        source: str,
        ticker: Optional[str] = None,
    ) -> "ArchiveArticle":
        """Build from the loose dicts the dataflow modules already produce."""
        published = data.get("published_at") or data.get("pub_date")
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                )
            except ValueError:
                published = None
        return cls(
            title=str(data.get("title") or "").strip(),
            summary=str(
                data.get("summary") or data.get("description") or ""
            ).strip(),
            publisher=str(
                data.get("publisher") or data.get("source") or ""
            ).strip(),
            link=str(data.get("link") or data.get("url") or "").strip(),
            source=source,
            ticker=ticker.upper() if ticker else None,
            published_at=published,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch(dt: Optional[datetime]) -> Optional[int]:
    """Convert a datetime to a UTC epoch integer Chroma can range-filter."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def _article_id(article: ArchiveArticle) -> str:
    """Deterministic id so re-fetches of the same article upsert in place.

    The link is the strongest dedup key but isn't always present (some
    feeds drop it or rotate it). Falling back to ``source + title``
    guarantees we still merge same-titled articles from the same feed
    rather than duplicating them on every run.
    """
    if article.link:
        key = f"{article.source}|{article.link}"
    else:
        key = f"{article.source}|{article.title}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"news:{digest}"


def _macro_id(curr_date: str, lookback_days: int) -> str:
    """One snapshot per (date, lookback) so re-runs upsert idempotently."""
    return f"macro:{curr_date}:{lookback_days}d"


def _build_article_text(article: ArchiveArticle) -> str:
    """Embedding text for an article.

    Title carries the strongest signal, summary fills in context. We
    deliberately omit publisher/link from the embedded text — they
    don't help similarity and would dilute the vector.
    """
    parts = [article.title]
    if article.summary:
        # Cap at ~2 KB; longer text dilutes the embedding and inflates
        # API cost without improving recall.
        parts.append(article.summary[:2000])
    return "\n\n".join(p for p in parts if p)


def _build_article_metadata(article: ArchiveArticle) -> Dict[str, Any]:
    """Flat metadata so chroma ``where`` filters work."""
    meta: Dict[str, Any] = {
        "kind": "article",
        "source": article.source,
        "title": article.title,
        "publisher": article.publisher,
        "link": article.link,
    }
    if article.ticker:
        meta["ticker"] = article.ticker
    epoch = _epoch(article.published_at)
    if epoch is not None:
        meta["published_at_ts"] = epoch
        meta["published_date"] = article.published_at.strftime("%Y-%m-%d")
    return meta


# ---------------------------------------------------------------------------
# NewsArchive
# ---------------------------------------------------------------------------


class NewsArchive:
    """Vector-indexed corpus of news articles + macro snapshots.

    The store is intentionally a thin wrapper: the heavy lifting
    (embedding, persistence, ``where`` filtering) lives in
    :class:`MemoryVectorStore`. We add three things on top:

    1. Stable IDs for upsert idempotency.
    2. Two record kinds (article / macro_snapshot) with kind-scoped
       search helpers.
    3. Markdown rendering that mirrors the existing news block shape,
       so analyst prompts treat archived results the same as fresh
       fetches.
    """

    def __init__(self, vector_store) -> None:  # type: ignore[no-untyped-def]
        self._store = vector_store

    # ---- Indexing -----------------------------------------------------

    def index_articles(self, articles: Sequence[ArchiveArticle]) -> int:
        """Upsert a batch of articles. Returns the count actually written."""
        rows: List[Dict[str, Any]] = []
        for art in articles:
            if not art.title:
                # Without a title there's nothing useful to embed;
                # skip silently rather than letting empty docs into
                # the index.
                continue
            rows.append({
                "id": _article_id(art),
                "document": _build_article_text(art),
                "metadata": _build_article_metadata(art),
            })
        if not rows:
            return 0
        try:
            self._store.upsert_many(
                ids=[r["id"] for r in rows],
                documents=[r["document"] for r in rows],
                metadatas=[r["metadata"] for r in rows],
            )
            return len(rows)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("News archive bulk index failed: %s", exc)
            return 0

    def index_macro_snapshot(
        self,
        document: str,
        *,
        curr_date: str,
        lookback_days: int,
    ) -> bool:
        """Upsert one macro-data snapshot keyed by (date, lookback).

        Returns ``True`` on success, ``False`` on any failure. Tests
        rely on the boolean to assert behaviour without poking at
        internals.
        """
        if not document.strip():
            return False
        try:
            self._store.upsert(
                entry_id=_macro_id(curr_date, lookback_days),
                document=document,
                metadata={
                    "kind": "macro_snapshot",
                    "curr_date": curr_date,
                    "lookback_days": int(lookback_days),
                    "indexed_at_ts": int(time.time()),
                },
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "News archive macro index failed for %s/%dd: %s",
                curr_date,
                lookback_days,
                exc,
            )
            return False

    # ---- Search -------------------------------------------------------

    def search_articles(
        self,
        query: str,
        *,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        n_results: int = 5,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search scoped to news articles.

        ``ticker`` filters to ticker-news for that symbol (case-insensitive).
        ``days_back`` filters by ``published_at_ts``; missing dates fall
        through (no filter applied to articles without timestamps).
        ``as_of`` sets the upper bound — useful for back-tested runs that
        must not retrieve future news.

        Returns ``[]`` on any error so analysts never crash because the
        archive is misconfigured.
        """
        where = self._build_article_where(
            ticker=ticker, days_back=days_back, as_of=as_of
        )
        try:
            return self._store.search(
                query=query, n_results=n_results, where=where
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("News archive article search failed: %s", exc)
            return []

    def search_macro_snapshots(
        self,
        query: str,
        *,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Semantic search scoped to macro snapshots."""
        try:
            return self._store.search(
                query=query,
                n_results=n_results,
                where={"kind": {"$eq": "macro_snapshot"}},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("News archive macro search failed: %s", exc)
            return []

    @staticmethod
    def _build_article_where(
        ticker: Optional[str],
        days_back: Optional[int],
        as_of: Optional[datetime],
    ) -> Dict[str, Any]:
        """Compose Chroma ``where`` AND-clause for article search.

        Chroma rejects a single-clause ``$and``, so we collapse
        gracefully when only one filter is active. ``kind == article``
        is always present so the macro snapshots never bleed into
        article ranking.
        """
        clauses: List[Dict[str, Any]] = [{"kind": {"$eq": "article"}}]

        if ticker:
            clauses.append({"ticker": {"$eq": ticker.upper()}})

        if days_back is not None and days_back > 0:
            anchor = (as_of or datetime.now(timezone.utc))
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=timezone.utc)
            cutoff = int(
                anchor.timestamp() - days_back * 86400
            )
            clauses.append({"published_at_ts": {"$gte": cutoff}})

        if as_of is not None:
            anchor = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            # Allow a small grace day so articles dated on as_of itself
            # are included even when timestamps are end-of-day UTC.
            upper = int(anchor.timestamp() + 86400)
            clauses.append({"published_at_ts": {"$lte": upper}})

        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # ---- Formatting ---------------------------------------------------

    @staticmethod
    def format_articles(
        hits: Sequence[Dict[str, Any]], *, header: str = "Archived news (semantic match)"
    ) -> str:
        """Render search hits in the existing news-block shape.

        Mirrors :func:`tradingagents.dataflows.yfinance_news.get_news_yfinance`
        output (``### title (source: publisher)`` + summary + link)
        so analyst prompts can stitch archive results next to fresh
        fetches without special-casing the markdown.
        """
        if not hits:
            return ""
        lines = [f"## {header}", ""]
        for hit in hits:
            meta = hit.get("metadata") or {}
            title = meta.get("title") or "(untitled)"
            publisher = meta.get("publisher") or meta.get("source") or "archive"
            date = meta.get("published_date") or ""
            tag = f" _( {date} )_" if date else ""
            lines.append(f"### {title} (source: {publisher}){tag}")
            doc = (hit.get("document") or "").strip()
            # Drop the title line we already rendered above to avoid
            # the duplicated heading the embedded text starts with.
            doc_lines = [ln for ln in doc.splitlines() if ln.strip() != title]
            body = "\n".join(doc_lines).strip()
            if body:
                lines.append(body)
            link = meta.get("link") or ""
            if link:
                lines.append(f"Link: {link}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def format_macro_snapshots(hits: Sequence[Dict[str, Any]]) -> str:
        """Render macro snapshots one block per hit."""
        if not hits:
            return ""
        sections = ["## Archived macro snapshots (semantic match)", ""]
        for hit in hits:
            meta = hit.get("metadata") or {}
            curr_date = meta.get("curr_date") or "(undated)"
            lookback = meta.get("lookback_days")
            label = f"### Snapshot {curr_date}"
            if lookback:
                label += f" ({lookback}d window)"
            sections.append(label)
            sections.append((hit.get("document") or "").strip())
            sections.append("")
        return "\n".join(sections).rstrip() + "\n"

    # ---- Maintenance --------------------------------------------------

    def count(self) -> int:
        return self._store.count()
