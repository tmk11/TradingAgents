"""Tests for ``ScheduleStore`` and the pure scheduler decision logic.

Network and the runner thread are mocked end-to-end; ``BackgroundScheduler``
itself is exercised through ``tick(now_utc)`` so tests stay deterministic.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from server import scheduler as sched_mod
from server.schedules import (
    DEFAULT_DAILY_PARAMS,
    DEFAULT_VOLATILITY_PARAMS,
    SCHEDULE_KINDS,
    ScheduleStore,
)


# ---------------------------------------------------------------------------
# ScheduleStore CRUD
# ---------------------------------------------------------------------------


class ScheduleStoreCreateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ScheduleStore(base_dir=Path(self._tmp.name))

    def test_create_daily_uses_default_params(self):
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            kind="daily_after_close",
        )
        self.assertEqual(rec["ticker"], "GLD")
        self.assertEqual(rec["kind"], "daily_after_close")
        self.assertEqual(
            rec["params"]["fire_hour_utc"], DEFAULT_DAILY_PARAMS["fire_hour_utc"]
        )
        self.assertTrue(rec["enabled"])
        self.assertIsNone(rec["last_run_at"])
        # Auto-generated name should mention the ticker.
        self.assertIn("GLD", rec["name"])

    def test_create_volatility_clamps_params(self):
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            kind="volatility_trigger",
            # threshold above the cap should clamp to 10
            params={"threshold_pct": 99},
        )
        self.assertLessEqual(rec["params"]["threshold_pct"], 10.0)
        # check_interval has a 5-minute floor
        rec2 = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            kind="volatility_trigger",
            params={"check_interval_minutes": 1},
        )
        self.assertGreaterEqual(rec2["params"]["check_interval_minutes"], 5)

    def test_create_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            self.store.create(
                ticker="GLD", asset_type="commodity", kind="bogus_kind"
            )

    def test_lowercase_ticker_normalised(self):
        rec = self.store.create(
            ticker="gld",
            asset_type="commodity",
            kind="daily_after_close",
        )
        self.assertEqual(rec["ticker"], "GLD")


class ScheduleStoreLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ScheduleStore(base_dir=Path(self._tmp.name))

    def _seed(self, **kw: Any) -> Dict[str, Any]:
        defaults = {
            "ticker": "GLD",
            "asset_type": "commodity",
            "kind": "daily_after_close",
        }
        defaults.update(kw)
        return self.store.create(**defaults)

    def test_list_returns_all_in_creation_order(self):
        a = self._seed()
        b = self._seed(ticker="IAU")
        ids = [r["id"] for r in self.store.list()]
        self.assertEqual(ids, [a["id"], b["id"]])

    def test_get_returns_none_for_missing(self):
        self.assertIsNone(self.store.get("00000000-0000-0000-0000-000000000000"))

    def test_update_partial(self):
        rec = self._seed()
        updated = self.store.update(rec["id"], enabled=False)
        self.assertFalse(updated["enabled"])
        # Untouched fields preserved.
        self.assertEqual(updated["ticker"], "GLD")

    def test_update_revalidates_params_on_kind_change(self):
        rec = self._seed()
        # Switch to volatility with an out-of-band threshold.
        updated = self.store.update(
            rec["id"],
            kind="volatility_trigger",
            params={"threshold_pct": 50},
        )
        self.assertEqual(updated["kind"], "volatility_trigger")
        self.assertLessEqual(updated["params"]["threshold_pct"], 10.0)

    def test_delete(self):
        rec = self._seed()
        self.assertTrue(self.store.delete(rec["id"]))
        self.assertIsNone(self.store.get(rec["id"]))

    def test_mark_fired_records_analysis_id(self):
        rec = self._seed()
        updated = self.store.mark_fired(
            rec["id"], analysis_id="11111111-1111-1111-1111-111111111111"
        )
        self.assertEqual(
            updated["last_run_analysis_id"],
            "11111111-1111-1111-1111-111111111111",
        )
        self.assertIsNotNone(updated["last_run_at"])

    def test_invalid_id_path_raises(self):
        # Defence against path traversal — uuid4 never produces these
        # but the API accepts arbitrary strings.
        with self.assertRaises(ValueError):
            self.store._path("../escape")


# ---------------------------------------------------------------------------
# Pure decision-logic tests
# ---------------------------------------------------------------------------


# Anchor: a Tuesday so the weekday checks have weekday()==1.
TUESDAY = datetime(2026, 6, 2, 21, 30, tzinfo=timezone.utc)


class ShouldFireDailyTests(unittest.TestCase):
    def _call(self, **kw: Any) -> bool:
        defaults = dict(
            now_utc=TUESDAY,
            last_run_at=None,
            fire_hour_utc=21,
            fire_minute_utc=30,
            weekdays_only=True,
        )
        defaults.update(kw)
        return sched_mod.should_fire_daily(**defaults)

    def test_fires_at_target_with_no_prior_run(self):
        self.assertTrue(self._call())

    def test_does_not_fire_before_target(self):
        early = TUESDAY.replace(hour=21, minute=29)
        self.assertFalse(self._call(now_utc=early))

    def test_does_not_fire_on_weekend_with_weekdays_only(self):
        sunday = datetime(2026, 6, 7, 22, 0, tzinfo=timezone.utc)
        self.assertFalse(self._call(now_utc=sunday))

    def test_fires_on_weekend_when_weekdays_only_disabled(self):
        sunday = datetime(2026, 6, 7, 22, 0, tzinfo=timezone.utc)
        self.assertTrue(self._call(now_utc=sunday, weekdays_only=False))

    def test_does_not_fire_twice_same_day(self):
        # Yesterday's run + today's evaluation → fires
        yesterday_run = TUESDAY - timedelta(days=1, hours=2)
        self.assertTrue(
            self._call(last_run_at=yesterday_run.isoformat())
        )
        # A run that already happened today → skip
        today_run = TUESDAY.replace(minute=31)  # one minute past target
        self.assertFalse(
            self._call(
                now_utc=TUESDAY.replace(minute=45),
                last_run_at=today_run.isoformat(),
            )
        )

    def test_fires_after_missed_target(self):
        # Server was asleep; we wake up at 22:30 with no run today.
        late = TUESDAY.replace(hour=22, minute=30)
        self.assertTrue(self._call(now_utc=late))


class ShouldFireVolatilityTests(unittest.TestCase):
    def _call(self, **kw: Any) -> bool:
        defaults = dict(
            now_utc=TUESDAY,
            last_run_at=None,
            last_check_at=None,
            threshold_pct=1.5,
            throttle_hours=6,
            check_interval_minutes=15,
            intraday_return_pct=2.0,
        )
        defaults.update(kw)
        return sched_mod.should_fire_volatility(**defaults)

    def test_fires_when_return_exceeds_threshold(self):
        self.assertTrue(self._call(intraday_return_pct=2.0))
        self.assertTrue(self._call(intraday_return_pct=-2.0))

    def test_does_not_fire_inside_threshold(self):
        self.assertFalse(self._call(intraday_return_pct=1.0))
        self.assertFalse(self._call(intraday_return_pct=-1.4))

    def test_does_not_fire_when_check_interval_recent(self):
        recent = TUESDAY - timedelta(minutes=5)
        self.assertFalse(self._call(last_check_at=recent.isoformat()))

    def test_does_not_fire_inside_throttle_window(self):
        recent_fire = TUESDAY - timedelta(hours=2)
        self.assertFalse(self._call(last_run_at=recent_fire.isoformat()))

    def test_fires_after_throttle_elapses(self):
        old_fire = TUESDAY - timedelta(hours=8)
        self.assertTrue(self._call(last_run_at=old_fire.isoformat()))

    def test_unknown_return_does_not_fire(self):
        self.assertFalse(self._call(intraday_return_pct=None))


# ---------------------------------------------------------------------------
# BackgroundScheduler.tick — integration with stub stores + runner
# ---------------------------------------------------------------------------


class _StubRunner:
    """In-memory drop-in for ``AnalysisRunner`` so tests don't spin
    a worker thread or the LLM stack."""

    def __init__(self) -> None:
        self.submitted: List[str] = []

    def submit(self, analysis_id: str) -> None:
        self.submitted.append(analysis_id)

    def start(self) -> None:  # pragma: no cover — interface compat
        pass

    def stop(self) -> None:  # pragma: no cover — interface compat
        pass


class SchedulerTickTests(unittest.TestCase):
    def setUp(self) -> None:
        from server.storage import AnalysisStore

        self._tmp_a = tempfile.TemporaryDirectory()
        self._tmp_s = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_a.cleanup)
        self.addCleanup(self._tmp_s.cleanup)
        self.analyses = AnalysisStore(base_dir=Path(self._tmp_a.name))
        self.schedules = ScheduleStore(base_dir=Path(self._tmp_s.name))
        self.runner = _StubRunner()
        self.scheduler = sched_mod.BackgroundScheduler(
            analysis_store=self.analyses,
            schedule_store=self.schedules,
            runner=self.runner,
        )

    def test_disabled_schedule_is_skipped(self):
        self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="daily_after_close",
            enabled=False,
        )
        self.scheduler.tick(TUESDAY)
        self.assertEqual(self.runner.submitted, [])

    def test_daily_schedule_fires_and_marks_run(self):
        rec = self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="daily_after_close",
        )
        self.scheduler.tick(TUESDAY)
        self.assertEqual(len(self.runner.submitted), 1)
        analysis_id = self.runner.submitted[0]

        # Schedule got mark_fired'd with the new analysis id.
        updated = self.schedules.get(rec["id"])
        self.assertEqual(updated["last_run_analysis_id"], analysis_id)

        # Analysis was created on the right ticker / date.
        analysis = self.analyses.get(analysis_id)
        self.assertEqual(analysis["ticker"], "GLD")
        self.assertEqual(
            analysis["analysis_date"], TUESDAY.date().isoformat()
        )
        self.assertEqual(analysis["max_debate_rounds"], 3)

    def test_daily_schedule_does_not_fire_twice(self):
        self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="daily_after_close",
        )
        self.scheduler.tick(TUESDAY)
        self.scheduler.tick(TUESDAY.replace(minute=45))
        self.assertEqual(len(self.runner.submitted), 1)

    def test_volatility_schedule_fires_when_yfinance_returns_big_move(self):
        self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="volatility_trigger",
        )
        with patch.object(
            sched_mod, "fetch_intraday_return_pct", return_value=2.5
        ) as m:
            self.scheduler.tick(TUESDAY)
        self.assertEqual(m.call_count, 1)
        self.assertEqual(len(self.runner.submitted), 1)

    def test_volatility_schedule_polls_but_does_not_fire_on_calm_market(self):
        rec = self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="volatility_trigger",
        )
        with patch.object(
            sched_mod, "fetch_intraday_return_pct", return_value=0.4
        ):
            self.scheduler.tick(TUESDAY)
        self.assertEqual(self.runner.submitted, [])
        # last_check_at must still bump so the polling cadence works.
        updated = self.schedules.get(rec["id"])
        self.assertIsNotNone(updated["last_check_at"])
        self.assertIsNone(updated["last_run_at"])

    def test_volatility_throttle_skips_yfinance_call(self):
        rec = self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="volatility_trigger",
        )
        # Simulate "we fired 30 minutes ago" — well inside the
        # default 6-hour throttle.
        recent_fire = (TUESDAY - timedelta(minutes=30)).isoformat()
        self.schedules.update(rec["id"], last_run_at=recent_fire)
        with patch.object(
            sched_mod, "fetch_intraday_return_pct"
        ) as m:
            self.scheduler.tick(TUESDAY)
        # Throttle pre-check must avoid the network call entirely.
        m.assert_not_called()
        self.assertEqual(self.runner.submitted, [])

    def test_unknown_kind_logged_but_does_not_crash(self):
        # Persist a record with a hand-rolled bogus kind by bypassing
        # ``create`` (which validates) and using ``update``.
        rec = self.schedules.create(
            ticker="GLD",
            asset_type="commodity",
            kind="daily_after_close",
        )
        # Force an unknown kind via the file itself — the validator
        # only kicks in when ``params`` or ``kind`` are passed via
        # update; raw .update() lets through anything else.
        self.schedules.update(rec["id"], some_other_field=True)
        # Manually patch the stored kind — easier than fighting
        # validation for this corner case.
        path = self.schedules._path(rec["id"])
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace('"daily_after_close"', '"bogus_kind"'),
            encoding="utf-8",
        )
        # Should not raise.
        self.scheduler.tick(TUESDAY)
        self.assertEqual(self.runner.submitted, [])


# ---------------------------------------------------------------------------
# Recommended-defaults seeding
# ---------------------------------------------------------------------------


class SeedRecommendedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ScheduleStore(base_dir=Path(self._tmp.name))

    def test_creates_two_schedules_when_empty(self):
        out = sched_mod.seed_recommended_schedules(self.store)
        self.assertIn("daily", out)
        self.assertIn("volatility", out)
        listed = self.store.list()
        kinds = sorted(r["kind"] for r in listed)
        self.assertEqual(kinds, ["daily_after_close", "volatility_trigger"])
        # Both should be Medium depth (3) per the recommended workflow.
        for r in listed:
            self.assertEqual(r["max_debate_rounds"], 3)
            self.assertEqual(r["max_risk_discuss_rounds"], 3)

    def test_idempotent_when_already_seeded(self):
        sched_mod.seed_recommended_schedules(self.store)
        sched_mod.seed_recommended_schedules(self.store)
        self.assertEqual(len(self.store.list()), 2)

    def test_does_not_touch_other_tickers(self):
        # User had a non-default schedule on IAU; seeding GLD must
        # not delete it or duplicate it under the wrong ticker.
        existing = self.store.create(
            ticker="IAU",
            asset_type="commodity",
            kind="daily_after_close",
        )
        sched_mod.seed_recommended_schedules(self.store, ticker="GLD")
        ids = {r["id"] for r in self.store.list()}
        self.assertIn(existing["id"], ids)
        self.assertEqual(len(self.store.list()), 3)


if __name__ == "__main__":
    unittest.main()
