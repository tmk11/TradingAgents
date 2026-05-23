"""Gold-Edition macro-data fetcher (real yields, DXY, VIX, etc.).

Gold's price is driven primarily by macro inputs — real yields, the
US dollar, inflation expectations, central-bank balance sheets — and
the framework's existing tools surface only price/news data for the
ticker itself. This module fills that gap by fetching the canonical
gold-driver time series and rendering each one as a compact summary
block (latest value, 1d/1w/1m change, window min/max) that the LLM can
reference directly without having to reason about raw OHLCV data.

Two providers are wired in, in priority order:

  1. **yfinance** — always available because the framework already
     uses it. Covers the most-watched market tickers:
     ``^TNX`` (10Y nominal Treasury yield), ``DX-Y.NYB`` (DXY US
     dollar index), ``^VIX`` (volatility / "fear gauge"), and ``TIP``
     (TIPS ETF — a price-based proxy for the 10Y real yield).

  2. **FRED CSV** — the St. Louis Fed's free public CSV download
     endpoint (``https://fred.stlouisfed.org/graph/fredgraph.csv``).
     No API key required. Adds the canonical series that yfinance
     does not expose directly: ``DFII10`` (10Y real yield), ``T10YIE``
     (10Y breakeven inflation), ``WALCL`` (Fed total balance sheet),
     ``DTWEXBGS`` (broad trade-weighted USD).

FRED is best-effort: a timeout, network failure, or 4xx response
degrades to a labelled placeholder rather than raising, so a partial
outage still yields useful output. The yfinance path is the load-
bearing one — if FRED is reachable, it adds real-yield precision and
Fed-balance-sheet context on top.

Adding a new series is one line — append a :class:`MacroSeries` entry
to :data:`MACRO_SERIES_GOLD`. Provider routing is automatic.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Browser-style UA: FRED rate-limits clearly-identified bots more
# aggressively, and yfinance is already wrapped elsewhere in the code.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "tradingagents-gold/0.3"
)

# FRED public CSV endpoint. ``id=<series>&cosd=<start>`` returns a
# two-column CSV (DATE,<series>). No auth.
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Network timeout for FRED. Kept on the short side so a flaky FRED
# does not stall the whole pipeline; yfinance carries the floor.
_FRED_TIMEOUT_SEC = 12.0


# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSeries:
    """One macro time series in the gold-driver registry."""

    provider: str          # "yfinance" or "fred"
    series_id: str         # yfinance ticker, or FRED series ID
    label: str             # human-readable label for the prompt block
    description: str       # one-liner: why this matters for gold


# Registry of gold-relevant macro series. Append to extend.
#
# Why these specifically:
#   * 10Y nominal Treasury yield (^TNX) and TIPS ETF (TIP) on yfinance
#     are the most-watched real-vs-nominal yield proxies and always
#     load.
#   * DXY (DX-Y.NYB) is gold's most direct currency headwind /
#     tailwind. Strong USD => weak gold, near-mechanically.
#   * VIX captures risk-off flows that historically chase bullion.
#   * GC=F gives the gold price itself for cross-reference (the LLM
#     can sanity-check claims against the spot tape).
#   * FRED DFII10 is the *exact* 10Y real yield series central banks
#     and macro desks quote — superior to TIP for that purpose when
#     reachable.
#   * T10YIE = 10Y breakeven inflation (DGS10 - DFII10) — the
#     inflation-expectation read.
#   * WALCL = Fed total assets (weekly). QE/QT regime indicator.
#   * DTWEXBGS = broad trade-weighted USD; complements DXY which is
#     EUR-heavy.
MACRO_SERIES_GOLD: List[MacroSeries] = [
    MacroSeries(
        provider="yfinance",
        series_id="^TNX",
        label="10Y Treasury Yield (^TNX, %)",
        description=(
            "Nominal 10-year US Treasury yield. Higher nominal yields "
            "raise the opportunity cost of holding non-yielding gold."
        ),
    ),
    MacroSeries(
        provider="yfinance",
        series_id="DX-Y.NYB",
        label="US Dollar Index (DXY)",
        description=(
            "Trade-weighted basket vs EUR/JPY/GBP/CAD/SEK/CHF. Gold is "
            "USD-denominated so a stronger DXY mechanically caps gold."
        ),
    ),
    MacroSeries(
        provider="yfinance",
        series_id="^VIX",
        label="VIX Volatility Index",
        description=(
            "S&P 500 30-day implied vol. Spikes accompany risk-off "
            "regimes that historically support safe-haven gold flows."
        ),
    ),
    MacroSeries(
        provider="yfinance",
        series_id="TIP",
        label="TIPS ETF (TIP)",
        description=(
            "iShares TIPS ETF price. Inverse proxy for the 10Y real "
            "yield — when TIP rallies, real yields are falling, which "
            "is structurally bullish for gold."
        ),
    ),
    MacroSeries(
        provider="yfinance",
        series_id="GC=F",
        label="Gold Futures (GC=F, USD/oz)",
        description=(
            "Front-month COMEX gold futures, included so other series "
            "can be cross-referenced against the actual price tape."
        ),
    ),
    # FRED series (best-effort; yfinance carries the floor).
    MacroSeries(
        provider="fred",
        series_id="DFII10",
        label="10Y Real Yield (FRED DFII10, %)",
        description=(
            "Treasury Inflation-Protected 10Y real yield. THE single "
            "most important macro driver of gold — the inverse-real-"
            "yield trade is structural."
        ),
    ),
    MacroSeries(
        provider="fred",
        series_id="T10YIE",
        label="10Y Breakeven Inflation (FRED T10YIE, %)",
        description=(
            "Implied 10-year inflation expectations (DGS10 minus "
            "DFII10). Rising breakevens with falling real yields is "
            "the textbook gold-bull regime."
        ),
    ),
    MacroSeries(
        provider="fred",
        series_id="WALCL",
        label="Fed Total Assets (FRED WALCL, $M, weekly)",
        description=(
            "Federal Reserve total assets. Expansion (QE) is gold-"
            "supportive; contraction (QT) historically caps gold."
        ),
    ),
    MacroSeries(
        provider="fred",
        series_id="DTWEXBGS",
        label="Broad Trade-Weighted USD (FRED DTWEXBGS)",
        description=(
            "Fed's broad trade-weighted dollar index. Complements DXY "
            "which is EUR-heavy; better captures EM-currency moves."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _parse_date(curr_date: Optional[str]) -> datetime:
    """Anchor ``curr_date`` to a UTC datetime (today on parse failure)."""
    if curr_date:
        try:
            return datetime.strptime(curr_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _summarise(observations: List[Tuple[datetime, float]]) -> Optional[dict]:
    """Compute the latest-value / 1d / 1w / 1m / window summary stats.

    ``observations`` is a list of ``(date, value)`` tuples sorted in
    chronological order. Returns ``None`` when there isn't enough data
    to report a current value.
    """
    if not observations:
        return None

    dates = [o[0] for o in observations]
    values = [o[1] for o in observations]
    last_dt, last = observations[-1]

    def _pct_change_back(target_dt: datetime) -> Optional[float]:
        """Find the closest observation at or before ``target_dt`` and
        return the percentage change from that value to ``last``.
        """
        candidate = None
        for dt, val in observations:
            if dt <= target_dt:
                candidate = val
            else:
                break
        if candidate is None or candidate == 0:
            return None
        return (last - candidate) / candidate * 100.0

    return {
        "last": last,
        "last_date": last_dt.strftime("%Y-%m-%d"),
        "change_1d_pct": _pct_change_back(last_dt - timedelta(days=1)),
        "change_1w_pct": _pct_change_back(last_dt - timedelta(days=7)),
        "change_1m_pct": _pct_change_back(last_dt - timedelta(days=30)),
        "window_min": min(values),
        "window_max": max(values),
        "n_obs": len(observations),
    }


def fetch_yfinance_series(
    ticker: str,
    *,
    lookback_days: int,
    curr_date: Optional[str],
) -> Optional[List[Tuple[datetime, float]]]:
    """Fetch close-price history for a yfinance ticker.

    Returns a chronological list of ``(datetime_utc, close)`` tuples,
    or ``None`` on failure / empty result. Limits the request window
    using ``lookback_days + buffer`` so the API call is small.
    """
    try:
        import yfinance as yf  # local import: keeps non-commodity runs cheap
    except ImportError as exc:
        logger.warning("yfinance unavailable: %s", exc)
        return None

    anchor = _parse_date(curr_date)
    # Add a small buffer for weekends / holidays so we get a full
    # ``lookback_days`` of trading observations.
    start = (anchor - timedelta(days=lookback_days + 7)).strftime("%Y-%m-%d")
    end = (anchor + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        history = yf.Ticker(ticker).history(start=start, end=end)
    except Exception as exc:  # network, parse, etc.
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None

    if history is None or history.empty or "Close" not in history.columns:
        return None

    out: List[Tuple[datetime, float]] = []
    for ts, close in history["Close"].items():
        try:
            f = float(close)
        except (TypeError, ValueError):
            continue
        # ``ts`` may be a pandas Timestamp; convert to plain UTC datetime.
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        out.append((dt, f))
    return out or None


def fetch_fred_series(
    series_id: str,
    *,
    lookback_days: int,
    curr_date: Optional[str],
    timeout: float = _FRED_TIMEOUT_SEC,
) -> Optional[List[Tuple[datetime, float]]]:
    """Fetch a FRED series via the public CSV endpoint. No API key needed.

    Returns ``None`` on any error (network, parse, empty body) so the
    caller can render a labelled placeholder rather than crashing the
    pipeline. FRED occasionally rate-limits or has slow handshakes, so
    keep the timeout aggressive — yfinance carries the floor.
    """
    anchor = _parse_date(curr_date)
    start = (anchor - timedelta(days=lookback_days + 7)).strftime("%Y-%m-%d")
    url = f"{_FRED_CSV_URL}?id={series_id}&cosd={start}"

    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "text/csv,text/plain,*/*",
            "Referer": "https://fred.stlouisfed.org/",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None
    except Exception as exc:  # belt-and-braces against ssl errors etc.
        logger.warning("FRED unexpected error for %s: %s", series_id, exc)
        return None

    if not payload.strip():
        return None

    out: List[Tuple[datetime, float]] = []
    reader = csv.reader(io.StringIO(payload))
    header = next(reader, None)
    if not header or len(header) < 2:
        return None
    upper = anchor + timedelta(days=1)
    cutoff = anchor - timedelta(days=lookback_days)

    for row in reader:
        if len(row) < 2:
            continue
        date_str, value_str = row[0].strip(), row[1].strip()
        # FRED uses "." for missing observations.
        if not date_str or value_str in ("", "."):
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            value = float(value_str)
        except ValueError:
            continue
        if dt < cutoff or dt > upper:
            continue
        out.append((dt, value))

    out.sort(key=lambda x: x[0])
    return out or None


# ---------------------------------------------------------------------------
# Block formatting + top-level fetcher
# ---------------------------------------------------------------------------


def _format_change(pct: Optional[float]) -> str:
    """Render a percentage change with a sign, or ``--`` when missing."""
    if pct is None:
        return "  -- "
    return f"{pct:+.2f}%"


def format_series_block(series: MacroSeries, summary: Optional[dict]) -> str:
    """Render a single series' summary as a markdown block."""
    if summary is None:
        return (
            f"### {series.label}\n"
            f"_{series.description}_\n\n"
            f"_(no data — provider {series.provider} unavailable for "
            f"{series.series_id} during this run)_\n"
        )

    return (
        f"### {series.label}\n"
        f"_{series.description}_\n\n"
        f"- Latest: **{summary['last']:.4g}**  ({summary['last_date']})\n"
        f"- 1d / 1w / 1m change: "
        f"{_format_change(summary['change_1d_pct'])} / "
        f"{_format_change(summary['change_1w_pct'])} / "
        f"{_format_change(summary['change_1m_pct'])}\n"
        f"- Window min / max: "
        f"{summary['window_min']:.4g} / {summary['window_max']:.4g}  "
        f"(n={summary['n_obs']} obs)\n"
    )


def fetch_one_series(
    series: MacroSeries,
    *,
    lookback_days: int,
    curr_date: Optional[str],
) -> str:
    """Fetch and format a single registered series.

    Provider routing is dispatched on ``series.provider``. Public
    surface used by tests and ad-hoc exploration; the production path
    goes through :func:`fetch_gold_macro_data`.
    """
    if series.provider == "yfinance":
        observations = fetch_yfinance_series(
            series.series_id, lookback_days=lookback_days, curr_date=curr_date
        )
    elif series.provider == "fred":
        observations = fetch_fred_series(
            series.series_id, lookback_days=lookback_days, curr_date=curr_date
        )
    else:
        logger.warning("Unknown provider %r for %s", series.provider, series.series_id)
        observations = None

    summary = _summarise(observations) if observations else None
    return format_series_block(series, summary)


def fetch_gold_macro_data(
    curr_date: str,
    lookback_days: int = 90,
    series: Optional[Sequence[MacroSeries]] = None,
) -> str:
    """Combine every registered macro series into one prompt block.

    This is the entry point bound to the Market Analyst (via the
    ``get_macro_data`` tool wrapper). Each series renders as its own
    section with summary stats so the LLM can cite specific numbers
    rather than reasoning about raw OHLCV.

    Args:
        curr_date: Current trading date in ``YYYY-MM-DD`` format. Used
            as the upper bound for observations so back-tested runs
            do not leak future data.
        lookback_days: How many days of history to fetch. The min/max
            window stats use the full lookback; 1d/1w/1m changes are
            computed against the closest prior observation.
        series: Optional override for the registry. When omitted, uses
            :data:`MACRO_SERIES_GOLD`.
    """
    series_list = list(series if series is not None else MACRO_SERIES_GOLD)

    sections = [
        f"# Gold-driver macro data, {lookback_days}-day window ending {curr_date}\n",
        (
            "Each series shows the latest value, short-term momentum "
            "(1d / 1w / 1m % change), and the min/max in the lookback "
            "window. Use these to ground the price-driver narrative — "
            "real yields, USD strength, breakeven inflation, and Fed "
            "balance sheet are the dominant gold inputs."
            "\n\n"
            "Provider notes: yfinance series always load; FRED series "
            "(DFII10, T10YIE, WALCL, DTWEXBGS) are best-effort and may "
            "show 'no data' if the upstream is unreachable.\n"
        ),
    ]

    by_provider: dict[str, List[str]] = {"yfinance": [], "fred": []}
    for s in series_list:
        block = fetch_one_series(s, lookback_days=lookback_days, curr_date=curr_date)
        by_provider.setdefault(s.provider, []).append(block)

    if by_provider.get("yfinance"):
        sections.append("## Market data (yfinance)\n")
        sections.extend(by_provider["yfinance"])
    if by_provider.get("fred"):
        sections.append("## Macro time series (FRED)\n")
        sections.extend(by_provider["fred"])

    return "\n".join(sections)
