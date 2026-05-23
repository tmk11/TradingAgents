"""Gold-Edition news fetchers (RSS-based, no auth required).

A small registry of gold- and macro-relevant RSS feeds is wrapped here.
Each feed sits alongside (not replacing) the existing yfinance / Alpha
Vantage news pipeline; the wrapper tool ``get_gold_news`` is bound to
the News Analyst only when ``asset_type == "commodity"``, so equity and
crypto runs see no change.

Default feeds:

  - **Mining.com** — industry trade press; relevant for gold-miner ETFs
    (GDX/GDXJ) and supply-side context (production, M&A, jurisdiction).
  - **Investing.com Commodities & Futures** — direct commodity/gold
    coverage with frequent geopolitical-driver headlines.
  - **Investing.com Economy** — Fed / central-bank / inflation news;
    these are the macro inputs that move bullion most.

Adding more sources is intentionally cheap — append a ``GoldNewsFeed``
entry to ``GOLD_FEEDS``. The fetcher tolerates network failures and a
couple of common RSS pubDate dialects (RFC-2822 with weekday + offset,
plus the bare ``YYYY-MM-DD HH:MM:SS`` style Investing.com emits).

RSS is parsed with stdlib (``xml.etree.ElementTree`` + ``email.utils``)
so the framework picks up no new dependencies. Each fetcher degrades
gracefully — on any error it returns a placeholder string rather than
raising, matching the contract used by Reddit/StockTwits.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# A real browser-style UA: some publishers (Investing.com in particular)
# return a stub HTML page to obvious bot UAs. The stdlib default
# ``Python-urllib/x`` triggers that path; this UA does not.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "tradingagents-gold/0.3"
)

# Tag-stripping regex for RSS ``<description>`` payloads, which often
# include HTML.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|quot|lt|gt|#\d+|#x[0-9a-fA-F]+);")


@dataclass(frozen=True)
class GoldNewsFeed:
    """A single RSS feed entry in the gold-news registry."""

    label: str  # human-readable label, also used as section header
    url: str    # full RSS URL


# Registry of gold-relevant feeds. Append to extend.
#
# Why these three:
#   * Mining.com covers the supply side and miner-specific headlines
#     that yfinance/Alpha Vantage tend to skip.
#   * Investing.com Commodities & Futures surfaces gold-spot and
#     geopolitical drivers (Iran/Middle East, sanctions) directly.
#   * Investing.com Economy delivers the Fed/CPI/yields macro picture
#     that determines real rates — the dominant gold driver.
#
# Kitco's RSS endpoint went 404 after their 2025 site redesign and
# wasn't replaced; if they re-publish, add a ``GoldNewsFeed`` here.
GOLD_FEEDS: List[GoldNewsFeed] = [
    GoldNewsFeed(
        label="Mining.com (industry trade press — miners, supply, M&A)",
        url="https://www.mining.com/feed/",
    ),
    GoldNewsFeed(
        label="Investing.com Commodities & Futures (gold spot drivers, geopolitics)",
        url="https://www.investing.com/rss/news_11.rss",
    ),
    GoldNewsFeed(
        label="Investing.com Economy (Fed, CPI, real yields, central banks)",
        url="https://www.investing.com/rss/news_14.rss",
    ),
    GoldNewsFeed(
        label="Bloomberg Markets (institutional macro & cross-asset)",
        url="https://feeds.bloomberg.com/markets/news.rss",
    ),
]


def _strip_html(raw: str, max_len: int = 280) -> str:
    """Convert HTML-flavoured RSS description text to clean plaintext."""
    if not raw:
        return ""
    text = _HTML_TAG_RE.sub(" ", raw)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    text = _HTML_ENTITY_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


# Date formats Investing.com (and a few smaller publishers) emit instead
# of strict RFC-2822. They're treated as UTC because the field carries
# no timezone — the imprecision is acceptable for a 7-day filter window.
_FALLBACK_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
)


def _parse_pub_date(raw: str) -> Optional[datetime]:
    """Parse RSS pubDate to a UTC datetime. Returns ``None`` on failure.

    Tries strict RFC-2822 first (the spec), then a small set of
    fallback formats commonly seen in the wild. Failure simply means
    "include the article anyway" — the date filter only kicks in when
    a valid datetime is parsed.
    """
    if not raw:
        return None
    raw = raw.strip()

    # 1) RFC-2822 — what well-behaved feeds emit.
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    # 2) Bare ISO-ish formats (Investing.com, etc.)
    for fmt in _FALLBACK_DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _fetch_rss_items(url: str, *, timeout: float = 10.0) -> List[dict]:
    """Fetch + parse an RSS feed into a list of item dicts.

    Returns ``[]`` (not an exception) on any network or parse error so
    callers can format a placeholder message instead of crashing the
    News Analyst node.
    """
    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "application/rss+xml, application/xml; q=0.9, */*; q=0.5",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("RSS fetch failed for %s: %s", url, exc)
        return []

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed for %s: %s", url, exc)
        return []

    items: List[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = _strip_html(item.findtext("description") or "")
        pub_date = _parse_pub_date(item.findtext("pubDate") or "")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "description": description,
                "pub_date": pub_date,
            }
        )
    return items


def _filter_and_trim(
    items: List[dict],
    *,
    look_back_days: int,
    limit: int,
    curr_date: Optional[str],
) -> List[dict]:
    """Apply the date window and item limit shared by every feed."""
    if curr_date:
        try:
            anchor = datetime.strptime(curr_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            anchor = datetime.now(timezone.utc)
    else:
        anchor = datetime.now(timezone.utc)

    cutoff = anchor - timedelta(days=look_back_days)
    # Look-ahead guard: never let an article dated *after* the analysis
    # date leak into the report (matters for back-tested runs).
    upper = anchor + timedelta(days=1)

    filtered = []
    for it in items:
        pub = it.get("pub_date")
        if pub is not None:
            if pub < cutoff or pub > upper:
                continue
        filtered.append(it)

    filtered.sort(
        key=lambda it: it["pub_date"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return filtered[:limit]


def _format_block(
    source_label: str,
    items: List[dict],
    *,
    placeholder_when_empty: str,
) -> str:
    """Render a list of RSS items into the standard prompt block."""
    if not items:
        return f"## {source_label}\n\n{placeholder_when_empty}\n"

    lines = [f"## {source_label} (most recent first):", ""]
    for it in items:
        date_str = (
            it["pub_date"].strftime("%Y-%m-%d") if it["pub_date"] else "(undated)"
        )
        lines.append(f"### {it['title']}  _( {date_str} )_")
        if it["description"]:
            lines.append(it["description"])
        if it["link"]:
            lines.append(f"Link: {it['link']}")
        lines.append("")
    return "\n".join(lines)


def fetch_feed(
    feed: GoldNewsFeed,
    *,
    look_back_days: int = 7,
    limit: int = 10,
    curr_date: Optional[str] = None,
) -> str:
    """Fetch and format a single RSS feed.

    Useful for manual exploration and tests; the production path goes
    through :func:`fetch_gold_macro_news` to combine the registry.
    """
    raw_items = _fetch_rss_items(feed.url)
    items = _filter_and_trim(
        raw_items, look_back_days=look_back_days, limit=limit, curr_date=curr_date
    )
    return _format_block(
        feed.label,
        items,
        placeholder_when_empty=(
            f"<no articles available — {feed.url} empty or unreachable>"
        ),
    )


def fetch_gold_macro_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
    feeds: Optional[Sequence[GoldNewsFeed]] = None,
) -> str:
    """Combine every registered gold feed into a single prompt block.

    This is the entry point bound to the News Analyst (via the
    ``get_gold_news`` tool wrapper). Each feed renders as its own
    section so the LLM can attribute claims correctly.

    Args:
        curr_date: Current trading date in ``YYYY-MM-DD`` format. Used
            as the upper bound for article dates so back-tested runs
            do not leak future news.
        look_back_days: How many days back to include.
        limit: Max items per feed (combined output may reach
            ``len(feeds) * limit``).
        feeds: Optional override for the feed list. When omitted, uses
            the module-level :data:`GOLD_FEEDS` registry.
    """
    feeds = list(feeds if feeds is not None else GOLD_FEEDS)
    blocks = []
    for i, feed in enumerate(feeds):
        if i > 0:
            # Mild rate-limiting between distinct hosts; keeps us
            # friendly even under heavy debate-round invocation.
            time.sleep(0.2)
        blocks.append(
            fetch_feed(
                feed,
                look_back_days=look_back_days,
                limit=limit,
                curr_date=curr_date,
            )
        )

    header = (
        f"# Gold-complex news, last {look_back_days} day(s) ending {curr_date}\n\n"
        f"Combining {len(feeds)} gold-relevant RSS feed(s):"
        " industry trade press, commodity/futures coverage, and macro"
        " (Fed / inflation / central banks). Each section keeps its"
        " source label so you can attribute claims correctly.\n\n"
    )
    return header + "\n".join(blocks)


# ---------------------------------------------------------------------------
# Backwards-compat shims (kept so existing imports / smoke tests don't break)
# ---------------------------------------------------------------------------

def fetch_kitco_news(
    *,
    look_back_days: int = 7,
    limit: int = 10,
    curr_date: Optional[str] = None,
) -> str:
    """Compatibility shim — Kitco's public RSS endpoint went 404 in 2025.

    Kept so any caller still importing this name gets a clear placeholder
    instead of a crash. New code should call :func:`fetch_gold_macro_news`,
    which combines Mining.com + Investing.com Commodities + Economy.
    """
    return _format_block(
        "Kitco News (gold-focused)",
        [],
        placeholder_when_empty=(
            "<Kitco no longer publishes a public RSS feed; "
            "use fetch_gold_macro_news for current gold-news coverage>"
        ),
    )


def fetch_mining_news(
    *,
    look_back_days: int = 7,
    limit: int = 10,
    curr_date: Optional[str] = None,
) -> str:
    """Convenience wrapper for the Mining.com feed alone."""
    feed = next(
        (f for f in GOLD_FEEDS if "mining.com" in f.url),
        GoldNewsFeed(
            label="Mining.com (industry trade press)",
            url="https://www.mining.com/feed/",
        ),
    )
    return fetch_feed(
        feed,
        look_back_days=look_back_days,
        limit=limit,
        curr_date=curr_date,
    )
