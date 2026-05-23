"""Forward-return scoring for completed analyses.

Every completed analysis carries a ``final_decision`` on the canonical
5-tier scale (Buy / Overweight / Hold / Underweight / Sell), reasoned
over a specific ``ticker`` + ``analysis_date``. This module pulls the
actual price action that followed and grades whether the call was
right at multiple horizons.

The point isn't to claim accuracy — it's to **measure** it. Without
a feedback loop you can't tell whether changing the prompts, the
debate-round count, or the LLM provider actually improved anything.
That's exactly the gap this module fills.

Honest framing for callers:
    Even top-tier quant funds hit ~52-58% directional accuracy on
    short horizons for liquid commodities. A track-record table
    hovering around 50-55% is *good*, not broken — and a hit rate
    far above that on a small sample is almost certainly noise.

Caching: outcomes are stored back on the record under ``outcome``
so we don't hit yfinance more than once per (analysis, horizon).
A cached outcome with any unresolved horizon (whose target date has
since elapsed) triggers a recompute on the next read.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Forward windows in calendar days. Calendar (not trading) days keep
# the math simple — yfinance returns the closest available trading
# day's close for any calendar target via the helper below.
HORIZONS: Tuple[Tuple[str, int], ...] = (
    ("1d", 1),
    ("5d", 7),    # ~1 trading week
    ("21d", 30),  # ~1 trading month
    ("63d", 90),  # ~1 trading quarter
)


# Daily-noise floor for gold. Moves smaller than this are "flat",
# which is the ground-truth direction Hold calls aim at. 50 bps is
# roughly half the typical 1-day range on GLD over 2020-2025; tight
# enough to give Buy/Sell calls credit on real moves, loose enough
# that Hold isn't always wrong.
FLAT_THRESHOLD = 0.005


# 5-tier rating → expected forward direction. Buy and Overweight
# both expect up moves but with different conviction; we don't try
# to weigh them differently here because the only ground truth we
# have is realised return, not implied conviction.
_DIRECTION_BY_RATING = {
    "buy": "up",
    "overweight": "up",
    "hold": "flat",
    "underweight": "down",
    "sell": "down",
}


@dataclass
class HorizonOutcome:
    """One row of the per-analysis outcome table."""

    horizon: str          # "1d" / "5d" / "21d" / "63d"
    days: int             # calendar-day target offset
    target_date: str      # YYYY-MM-DD
    end_close: Optional[float]
    forward_return: Optional[float]   # decimal, e.g. 0.0123 = +1.23%
    actual_direction: str             # "up" / "down" / "flat" / "unknown"
    correct: Optional[bool]           # None until target date elapses


@dataclass
class AnalysisOutcome:
    decision: str                  # one of RATINGS_5_TIER (Title) or "Unknown"
    expected_direction: Optional[str]   # "up" / "down" / "flat" / None
    start_close: Optional[float]
    horizons: List[HorizonOutcome]
    computed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "expected_direction": self.expected_direction,
            "start_close": self.start_close,
            "computed_at": self.computed_at,
            "horizons": [asdict(h) for h in self.horizons],
        }


def _normalize_decision(raw: Optional[str]) -> str:
    """Map any free-form decision string back onto the 5-tier scale.

    The store usually carries the canonical rating already, but a
    failed run or a manually-edited record may have something like
    "BUY" or "Strong sell" — be lenient.
    """
    if not raw:
        return "Unknown"
    s = raw.strip().lower()
    for rating in ("overweight", "underweight", "buy", "hold", "sell"):
        # Order matters: "overweight" / "underweight" must be tested
        # before "buy" / "sell" so we don't accidentally classify
        # "Underweight" as containing "weight" of "Sell". (It doesn't
        # — but we also don't want "Buy with caution" → Underweight.)
        if rating in s:
            return rating.capitalize()
    return "Unknown"


def _classify_direction(forward_return: float) -> str:
    if forward_return > FLAT_THRESHOLD:
        return "up"
    if forward_return < -FLAT_THRESHOLD:
        return "down"
    return "flat"


def _expected_direction(decision: str) -> Optional[str]:
    return _DIRECTION_BY_RATING.get(decision.lower())


def _fetch_close_series(
    ticker: str, start: date, end: date
) -> Dict[date, float]:
    """Pull adjusted closes via yfinance. Empty dict on any failure.

    Lazily imported because the server itself doesn't require yfinance
    to start — only the runner thread and this module do, and we don't
    want a missing yfinance install to break unrelated endpoints.
    """
    try:
        import yfinance as yf  # local import keeps the import-graph cheap
    except ImportError:
        logger.warning("yfinance not installed; skipping outcome scoring")
        return {}

    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            # ``end`` in yfinance is exclusive — pad by 1 day so the
            # last calendar day we care about is actually fetched.
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:  # noqa: BLE001 — third-party may raise anything
        logger.exception("yfinance download failed for %s", ticker)
        return {}

    if df is None or df.empty:
        return {}

    closes: Dict[date, float] = {}
    # ``df["Close"]`` is a Series with a DatetimeIndex; iter via items()
    # for compatibility with both pandas 1.x and 2.x.
    try:
        col = df["Close"]
    except KeyError:
        return {}
    for ts, val in col.dropna().items():
        # yfinance >= 0.2.x sometimes returns a 2-D MultiIndex frame
        # when the ticker happens to match multiple instruments. Fall
        # back to scalar conversion when ``val`` is a Series.
        try:
            scalar = float(val)
        except (TypeError, ValueError):
            try:
                scalar = float(val.iloc[0])  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                continue
        if hasattr(ts, "date"):
            closes[ts.date()] = scalar
        else:
            try:
                closes[date.fromisoformat(str(ts)[:10])] = scalar
            except ValueError:
                continue
    return closes


def _close_on_or_after(
    closes: Dict[date, float], target: date
) -> Optional[Tuple[date, float]]:
    """Return (date, close) on the first trading day >= target.

    Markets are closed on weekends and holidays; the target may land
    on a non-trading day, in which case we want the *next* available
    close. ``None`` when the series doesn't reach far enough yet.
    """
    if not closes:
        return None
    for d in sorted(closes.keys()):
        if d >= target:
            return d, closes[d]
    return None


def score_analysis(
    record: Dict[str, Any], *, today: Optional[date] = None
) -> AnalysisOutcome:
    """Compute the forward-return outcome for one analysis record.

    Horizons whose target date hasn't elapsed yet come back with
    ``correct=None`` — they'll be filled in on a later refresh.
    Failures to fetch prices also produce ``correct=None`` so they
    don't poison the aggregate hit rate.
    """
    today = today or datetime.now(timezone.utc).date()
    decision = _normalize_decision(record.get("final_decision"))
    expected = _expected_direction(decision)

    try:
        start_date = datetime.strptime(
            record["analysis_date"], "%Y-%m-%d"
        ).date()
    except (KeyError, ValueError, TypeError):
        return AnalysisOutcome(
            decision=decision,
            expected_direction=expected,
            start_close=None,
            horizons=[],
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    ticker = (record.get("ticker") or "").strip()
    if not ticker:
        return AnalysisOutcome(
            decision=decision,
            expected_direction=expected,
            start_close=None,
            horizons=[],
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    # Pull a window covering the longest horizon plus a small buffer
    # for non-trading days at either end. One yfinance call per
    # analysis instead of one per horizon.
    max_days = max(d for _, d in HORIZONS)
    fetch_start = start_date - timedelta(days=7)
    fetch_end = min(today, start_date + timedelta(days=max_days + 7))
    closes = _fetch_close_series(ticker, fetch_start, fetch_end)

    start_pair = _close_on_or_after(closes, start_date)
    start_close = start_pair[1] if start_pair else None

    horizons: List[HorizonOutcome] = []
    for label, days in HORIZONS:
        target = start_date + timedelta(days=days)

        # Not enough time has passed — record placeholder and move on.
        if target > today:
            horizons.append(
                HorizonOutcome(
                    horizon=label,
                    days=days,
                    target_date=target.isoformat(),
                    end_close=None,
                    forward_return=None,
                    actual_direction="unknown",
                    correct=None,
                )
            )
            continue

        end_pair = _close_on_or_after(closes, target)
        if not end_pair or start_close is None:
            horizons.append(
                HorizonOutcome(
                    horizon=label,
                    days=days,
                    target_date=target.isoformat(),
                    end_close=end_pair[1] if end_pair else None,
                    forward_return=None,
                    actual_direction="unknown",
                    correct=None,
                )
            )
            continue

        end_close = end_pair[1]
        fr = (end_close - start_close) / start_close
        direction = _classify_direction(fr)
        correct = (expected == direction) if expected else None

        horizons.append(
            HorizonOutcome(
                horizon=label,
                days=days,
                target_date=target.isoformat(),
                end_close=end_close,
                forward_return=fr,
                actual_direction=direction,
                correct=correct,
            )
        )

    return AnalysisOutcome(
        decision=decision,
        expected_direction=expected,
        start_close=start_close,
        horizons=horizons,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def needs_refresh(
    cached: Dict[str, Any], today: Optional[date] = None
) -> bool:
    """Decide whether a cached outcome should be recomputed.

    Recompute when at least one horizon is still ``correct=None`` but
    its target date has already elapsed — that's the case where the
    price data exists now but didn't when we last computed.
    """
    today = today or datetime.now(timezone.utc).date()
    for h in cached.get("horizons", []) or []:
        if h.get("correct") is not None:
            continue
        try:
            target = datetime.strptime(h["target_date"], "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            continue
        if target <= today:
            return True
    return False


def get_or_compute_outcome(
    record: Dict[str, Any],
    store: Any,
    *,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Lazy outcome lookup. Computes + caches on first use; refreshes
    when previously-pending horizons can now be resolved.

    ``store`` is anything with an ``update(id, **kw)`` method — i.e.
    :class:`server.storage.AnalysisStore`. We accept ``Any`` to keep
    this module test-friendly without a circular import.
    """
    today = today or datetime.now(timezone.utc).date()
    cached = record.get("outcome")
    if cached and not needs_refresh(cached, today=today):
        return cached

    fresh = score_analysis(record, today=today).to_dict()
    try:
        store.update(record["id"], outcome=fresh)
    except Exception:  # noqa: BLE001
        # Persistence failure shouldn't block the caller from getting
        # the result — the worst case is we recompute next time.
        logger.exception("Failed to cache outcome for %s", record.get("id"))
    return fresh


def aggregate_track_record(
    records: List[Dict[str, Any]],
    store: Any,
    *,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Roll up per-analysis outcomes into hit-rate summaries.

    Returns a dict shaped for direct JSON serialisation:

        {
          "total_completed": int,
          "total_with_outcomes": int,
          "horizons": {
            "1d": { "total": int, "correct": int, "hit_rate": float|None,
                    "by_decision": { "Buy": {...}, "Hold": {...}, ... } },
            "5d": { ... },
            ...
          },
          "computed_at": str,
        }

    A horizon's ``hit_rate`` is ``None`` (not 0.0) when nothing has
    been scored at that horizon yet — distinguishing "we don't know
    yet" from "we tried and got 0%".
    """
    today = today or datetime.now(timezone.utc).date()

    horizons_stats: Dict[str, Dict[str, Any]] = {}
    for label, _days in HORIZONS:
        horizons_stats[label] = {
            "total": 0,
            "correct": 0,
            "hit_rate": None,
            "by_decision": {},
        }

    total_completed = 0
    total_scored = 0

    for r in records:
        if r.get("status") != "completed":
            continue
        total_completed += 1

        outcome = get_or_compute_outcome(r, store, today=today)
        decision = outcome.get("decision") or "Unknown"

        scored_any = False
        for h in outcome.get("horizons", []) or []:
            if h.get("correct") is None:
                continue
            label = h["horizon"]
            stats = horizons_stats.setdefault(
                label,
                {"total": 0, "correct": 0, "hit_rate": None, "by_decision": {}},
            )
            stats["total"] += 1
            if h["correct"]:
                stats["correct"] += 1

            by_dec = stats["by_decision"].setdefault(
                decision, {"total": 0, "correct": 0, "hit_rate": None}
            )
            by_dec["total"] += 1
            if h["correct"]:
                by_dec["correct"] += 1
            scored_any = True

        if scored_any:
            total_scored += 1

    # Fill in hit_rate ratios now that totals are stable.
    for label, stats in horizons_stats.items():
        if stats["total"]:
            stats["hit_rate"] = stats["correct"] / stats["total"]
        for _dec, sub in stats["by_decision"].items():
            if sub["total"]:
                sub["hit_rate"] = sub["correct"] / sub["total"]

    return {
        "total_completed": total_completed,
        "total_with_outcomes": total_scored,
        "horizons": horizons_stats,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
