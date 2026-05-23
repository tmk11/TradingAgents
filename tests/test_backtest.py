"""Tests for the forward-return scoring module.

Network access is mocked end-to-end so these run offline and finish
in milliseconds. The fixture replaces ``_fetch_close_series`` with a
hand-built dict of trading-day closes — exactly what the real
function returns once yfinance has been called.
"""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from typing import Dict
from unittest.mock import patch

from server import backtest


def _make_closes(start: date, daily_returns: list) -> Dict[date, float]:
    """Build a price series from a list of consecutive daily returns.

    Skips weekends so the dates look like real trading sessions.
    Starts at 100.0; each entry in ``daily_returns`` is the next day's
    decimal return (0.01 = +1%).
    """
    closes: Dict[date, float] = {}
    px = 100.0
    cur = start
    closes[cur] = px
    for r in daily_returns:
        # Advance one trading day (skip Sat/Sun).
        cur += timedelta(days=1)
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        px = px * (1.0 + r)
        closes[cur] = px
    return closes


class NormalizeDecisionTests(unittest.TestCase):
    def test_canonical_ratings_round_trip(self):
        for rating in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            self.assertEqual(backtest._normalize_decision(rating), rating)

    def test_lowercase_and_uppercase_inputs(self):
        self.assertEqual(backtest._normalize_decision("buy"), "Buy")
        self.assertEqual(backtest._normalize_decision("HOLD"), "Hold")

    def test_overweight_takes_precedence_over_buy(self):
        # Substring matching order matters: "Overweight" mustn't get
        # mis-mapped to "Buy" just because it contains alpha letters.
        self.assertEqual(
            backtest._normalize_decision("Overweight"), "Overweight"
        )
        self.assertEqual(
            backtest._normalize_decision("Underweight"), "Underweight"
        )

    def test_unknown_inputs_return_unknown(self):
        self.assertEqual(backtest._normalize_decision(""), "Unknown")
        self.assertEqual(backtest._normalize_decision(None), "Unknown")
        self.assertEqual(backtest._normalize_decision("???"), "Unknown")


class ClassifyDirectionTests(unittest.TestCase):
    def test_above_threshold_is_up(self):
        self.assertEqual(backtest._classify_direction(0.01), "up")

    def test_below_threshold_is_down(self):
        self.assertEqual(backtest._classify_direction(-0.01), "down")

    def test_inside_threshold_band_is_flat(self):
        self.assertEqual(backtest._classify_direction(0.001), "flat")
        self.assertEqual(backtest._classify_direction(-0.001), "flat")

    def test_exactly_threshold_is_flat(self):
        # Boundary is inclusive on the flat side — a +0.5% move
        # shouldn't flip a Hold call to "wrong".
        self.assertEqual(
            backtest._classify_direction(backtest.FLAT_THRESHOLD), "flat"
        )


class ExpectedDirectionTests(unittest.TestCase):
    def test_buy_and_overweight_expect_up(self):
        self.assertEqual(backtest._expected_direction("Buy"), "up")
        self.assertEqual(backtest._expected_direction("Overweight"), "up")

    def test_sell_and_underweight_expect_down(self):
        self.assertEqual(backtest._expected_direction("Sell"), "down")
        self.assertEqual(backtest._expected_direction("Underweight"), "down")

    def test_hold_expects_flat(self):
        self.assertEqual(backtest._expected_direction("Hold"), "flat")


class ScoreAnalysisTests(unittest.TestCase):
    """End-to-end scoring with the price-fetch boundary stubbed out."""

    ANALYSIS_DATE = "2026-01-05"  # Monday
    START = date(2026, 1, 5)

    def _record(self, decision: str = "Buy") -> Dict:
        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "ticker": "GLD",
            "asset_type": "commodity",
            "analysis_date": self.ANALYSIS_DATE,
            "language": "English",
            "status": "completed",
            "progress": {},
            "reports": {},
            "final_decision": decision,
            "error": None,
            "created_at": "2026-01-05T00:00:00+00:00",
            "completed_at": "2026-01-05T00:30:00+00:00",
        }

    def _patch_closes(self, closes: Dict[date, float]):
        return patch.object(
            backtest, "_fetch_close_series", return_value=closes
        )

    def test_buy_call_correct_when_price_rises(self):
        # +1% per trading day — comfortably above the 0.5% flat
        # threshold at every horizon, including 1d.
        closes = _make_closes(self.START, [0.01] * 80)
        with self._patch_closes(closes):
            outcome = backtest.score_analysis(
                self._record("Buy"), today=date(2026, 5, 1)
            )

        self.assertEqual(outcome.decision, "Buy")
        self.assertEqual(outcome.expected_direction, "up")
        self.assertEqual(outcome.start_close, 100.0)
        # All four horizons resolved.
        self.assertEqual(len(outcome.horizons), 4)
        for h in outcome.horizons:
            self.assertEqual(h.actual_direction, "up", msg=h.horizon)
            self.assertTrue(h.correct, msg=h.horizon)

    def test_sell_call_wrong_when_price_rises(self):
        closes = _make_closes(self.START, [0.01] * 80)
        with self._patch_closes(closes):
            outcome = backtest.score_analysis(
                self._record("Sell"), today=date(2026, 5, 1)
            )
        for h in outcome.horizons:
            self.assertEqual(h.actual_direction, "up")
            self.assertFalse(h.correct, msg=h.horizon)

    def test_hold_call_correct_when_price_stays_flat(self):
        # Alternating tiny up/down moves so the cumulative drift stays
        # inside the 0.5% flat band even at the 63d horizon.
        flat_returns = [0.0005, -0.0005] * 50
        closes = _make_closes(self.START, flat_returns)
        with self._patch_closes(closes):
            outcome = backtest.score_analysis(
                self._record("Hold"), today=date(2026, 5, 1)
            )
        for h in outcome.horizons:
            self.assertEqual(h.actual_direction, "flat", msg=h.horizon)
            self.assertTrue(h.correct, msg=h.horizon)

    def test_horizons_in_the_future_are_unresolved(self):
        # Simulate "today is the day after the analysis": only the 1d
        # horizon can resolve; longer ones should come back as None.
        closes = _make_closes(self.START, [0.01] * 5)
        # Shift today to just past the 1d horizon.
        today = self.START + timedelta(days=2)

        with self._patch_closes(closes):
            outcome = backtest.score_analysis(
                self._record("Buy"), today=today
            )

        by_horizon = {h.horizon: h for h in outcome.horizons}
        self.assertTrue(by_horizon["1d"].correct)
        for label in ("5d", "21d", "63d"):
            self.assertIsNone(by_horizon[label].correct, msg=label)
            self.assertEqual(by_horizon[label].actual_direction, "unknown")

    def test_missing_price_data_yields_unknown_outcomes(self):
        with self._patch_closes({}):
            outcome = backtest.score_analysis(
                self._record("Buy"), today=date(2026, 5, 1)
            )
        self.assertIsNone(outcome.start_close)
        for h in outcome.horizons:
            self.assertIsNone(h.correct)
            self.assertEqual(h.actual_direction, "unknown")

    def test_unknown_decision_produces_no_correct_judgement(self):
        # No expected direction → can't grade right/wrong, but we
        # should still record the actual direction for posterity.
        closes = _make_closes(self.START, [0.01] * 80)
        with self._patch_closes(closes):
            outcome = backtest.score_analysis(
                self._record(""), today=date(2026, 5, 1)
            )
        self.assertEqual(outcome.decision, "Unknown")
        for h in outcome.horizons:
            self.assertIsNone(h.correct)
            self.assertEqual(h.actual_direction, "up")


class NeedsRefreshTests(unittest.TestCase):
    def test_no_unresolved_horizons_means_no_refresh(self):
        cached = {
            "horizons": [
                {"horizon": "1d", "target_date": "2026-01-06", "correct": True},
            ]
        }
        self.assertFalse(backtest.needs_refresh(cached, today=date(2026, 5, 1)))

    def test_unresolved_but_target_in_future_does_not_refresh(self):
        cached = {
            "horizons": [
                {"horizon": "63d", "target_date": "2026-12-01", "correct": None},
            ]
        }
        self.assertFalse(backtest.needs_refresh(cached, today=date(2026, 6, 1)))

    def test_unresolved_with_elapsed_target_triggers_refresh(self):
        cached = {
            "horizons": [
                {"horizon": "1d", "target_date": "2026-01-06", "correct": True},
                {"horizon": "5d", "target_date": "2026-01-12", "correct": None},
            ]
        }
        self.assertTrue(backtest.needs_refresh(cached, today=date(2026, 5, 1)))


class GetOrComputeOutcomeTests(unittest.TestCase):
    """Confirm the cache-aware wrapper writes back through the store."""

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from server.storage import AnalysisStore

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = AnalysisStore(base_dir=Path(self._tmpdir.name))

    def _seed_record(self) -> Dict:
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            analysis_date="2026-01-05",
        )
        return self.store.update(
            rec["id"],
            status="completed",
            final_decision="Buy",
        )

    def test_first_call_computes_and_persists(self):
        rec = self._seed_record()
        closes = _make_closes(date(2026, 1, 5), [0.01] * 80)
        with patch.object(
            backtest, "_fetch_close_series", return_value=closes
        ) as fetch:
            outcome = backtest.get_or_compute_outcome(
                rec, self.store, today=date(2026, 5, 1)
            )
        self.assertEqual(fetch.call_count, 1)

        # Persisted on-record.
        persisted = self.store.get(rec["id"])
        self.assertEqual(persisted["outcome"], outcome)

    def test_second_call_uses_cache(self):
        rec = self._seed_record()
        closes = _make_closes(date(2026, 1, 5), [0.01] * 80)
        with patch.object(
            backtest, "_fetch_close_series", return_value=closes
        ) as fetch:
            backtest.get_or_compute_outcome(
                rec, self.store, today=date(2026, 5, 1)
            )
            # Re-fetch the freshly-cached record and read again.
            cached_rec = self.store.get(rec["id"])
            backtest.get_or_compute_outcome(
                cached_rec, self.store, today=date(2026, 5, 1)
            )
        # Only the first call should have hit the price fetcher.
        self.assertEqual(fetch.call_count, 1)


class AggregateTrackRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from server.storage import AnalysisStore

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = AnalysisStore(base_dir=Path(self._tmpdir.name))

    def _seed(self, decision: str, closes: Dict[date, float]) -> str:
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            analysis_date="2026-01-05",
        )
        self.store.update(rec["id"], status="completed", final_decision=decision)
        # Pre-cache the outcome to avoid re-patching during aggregate.
        with patch.object(
            backtest, "_fetch_close_series", return_value=closes
        ):
            backtest.get_or_compute_outcome(
                self.store.get(rec["id"]), self.store, today=date(2026, 5, 1)
            )
        return rec["id"]

    def test_aggregate_hit_rate_across_two_correct_one_wrong(self):
        up_closes = _make_closes(date(2026, 1, 5), [0.01] * 80)
        down_closes = _make_closes(date(2026, 1, 5), [-0.01] * 80)

        self._seed("Buy", up_closes)     # 4 horizons correct
        self._seed("Buy", up_closes)     # 4 horizons correct
        self._seed("Buy", down_closes)   # 4 horizons wrong

        records = self.store.list(summary_only=True)
        agg = backtest.aggregate_track_record(
            records, self.store, today=date(2026, 5, 1)
        )

        self.assertEqual(agg["total_completed"], 3)
        self.assertEqual(agg["total_with_outcomes"], 3)
        # Each horizon: 3 calls scored, 2 correct, 1 wrong.
        for label, _days in backtest.HORIZONS:
            stats = agg["horizons"][label]
            self.assertEqual(stats["total"], 3, msg=label)
            self.assertEqual(stats["correct"], 2, msg=label)
            self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=4)

    def test_pending_analyses_excluded_from_aggregate(self):
        # A pending analysis must not change the totals.
        self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-01-05"
        )
        agg = backtest.aggregate_track_record(
            self.store.list(summary_only=True),
            self.store,
            today=date(2026, 5, 1),
        )
        self.assertEqual(agg["total_completed"], 0)
        for stats in agg["horizons"].values():
            self.assertIsNone(stats["hit_rate"])

    def test_per_decision_breakdown_keeps_buy_and_sell_separate(self):
        up_closes = _make_closes(date(2026, 1, 5), [0.01] * 80)

        self._seed("Buy", up_closes)   # Buy x up = correct
        self._seed("Sell", up_closes)  # Sell x up = wrong

        agg = backtest.aggregate_track_record(
            self.store.list(summary_only=True),
            self.store,
            today=date(2026, 5, 1),
        )
        for _label, stats in agg["horizons"].items():
            buy = stats["by_decision"]["Buy"]
            sell = stats["by_decision"]["Sell"]
            self.assertEqual(buy["correct"], 1)
            self.assertEqual(buy["total"], 1)
            self.assertEqual(sell["correct"], 0)
            self.assertEqual(sell["total"], 1)


if __name__ == "__main__":
    unittest.main()
