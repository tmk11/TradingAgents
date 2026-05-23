"""End-to-end tests for the ``/api/schedules*`` surface.

The runner and scheduler thread are stubbed so requests never hit
yfinance or LangGraph; we're verifying request validation, response
shape, and that the store persists the right thing.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

from server.api import create_app
from server.scheduler import BackgroundScheduler
from server.schedules import ScheduleStore
from server.storage import AnalysisStore


class _StubRunner:
    def __init__(self) -> None:
        self.submitted: List[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit(self, analysis_id: str) -> None:
        with self._lock:
            self.submitted.append(analysis_id)


class _StubScheduler:
    """We test scheduler internals separately. Here we only need the
    lifespan hooks to be no-ops so the TestClient can spin up."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _build_client(tmp: Path):
    store = AnalysisStore(base_dir=tmp / "analyses")
    schedule_store = ScheduleStore(base_dir=tmp / "schedules")
    runner = _StubRunner()
    scheduler = _StubScheduler()
    app = create_app(
        store=store,
        runner=runner,
        schedule_store=schedule_store,
        scheduler=scheduler,  # type: ignore[arg-type]
        static_dir=False,
        enable_cors=False,
    )
    return TestClient(app), store, schedule_store, runner


class ScheduleCrudTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client, self.store, self.schedules, self.runner = _build_client(
            Path(self._tmp.name)
        )

    # ---- list / empty -------------------------------------------------

    def test_list_empty_initially(self):
        r = self.client.get("/api/schedules")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    # ---- create -------------------------------------------------------

    def test_create_daily_schedule(self):
        r = self.client.post(
            "/api/schedules",
            json={"ticker": "GLD", "kind": "daily_after_close"},
        )
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["ticker"], "GLD")
        self.assertEqual(body["asset_type"], "commodity")
        self.assertEqual(body["kind"], "daily_after_close")
        self.assertTrue(body["enabled"])
        # Defaults: Medium depth.
        self.assertEqual(body["max_debate_rounds"], 3)

    def test_create_volatility_schedule_with_custom_threshold(self):
        r = self.client.post(
            "/api/schedules",
            json={
                "ticker": "GLD",
                "kind": "volatility_trigger",
                "params": {"threshold_pct": 2.0, "throttle_hours": 4},
            },
        )
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["params"]["threshold_pct"], 2.0)
        self.assertEqual(body["params"]["throttle_hours"], 4)

    def test_create_rejects_unknown_kind(self):
        r = self.client.post(
            "/api/schedules",
            json={"ticker": "GLD", "kind": "totally_made_up"},
        )
        self.assertEqual(r.status_code, 422)

    def test_create_rejects_invalid_ticker(self):
        r = self.client.post(
            "/api/schedules",
            json={"ticker": "NOT A TICKER!!", "kind": "daily_after_close"},
        )
        self.assertEqual(r.status_code, 422)

    # ---- patch --------------------------------------------------------

    def test_patch_enables_and_disables(self):
        rec = self.client.post(
            "/api/schedules",
            json={"ticker": "GLD", "kind": "daily_after_close"},
        ).json()
        # Disable
        r = self.client.patch(
            f"/api/schedules/{rec['id']}", json={"enabled": False}
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["enabled"])
        # Re-enable
        r = self.client.patch(
            f"/api/schedules/{rec['id']}", json={"enabled": True}
        )
        self.assertTrue(r.json()["enabled"])

    def test_patch_404_for_missing(self):
        r = self.client.patch(
            "/api/schedules/00000000-0000-0000-0000-000000000000",
            json={"enabled": False},
        )
        self.assertEqual(r.status_code, 404)

    # ---- delete -------------------------------------------------------

    def test_delete(self):
        rec = self.client.post(
            "/api/schedules",
            json={"ticker": "GLD", "kind": "daily_after_close"},
        ).json()
        r = self.client.delete(f"/api/schedules/{rec['id']}")
        self.assertEqual(r.status_code, 204)
        self.assertEqual(self.client.get("/api/schedules").json(), [])

    def test_delete_404(self):
        r = self.client.delete(
            "/api/schedules/00000000-0000-0000-0000-000000000000"
        )
        self.assertEqual(r.status_code, 404)

    # ---- run-now ------------------------------------------------------

    def test_run_now_creates_analysis_and_submits_to_runner(self):
        rec = self.client.post(
            "/api/schedules",
            json={"ticker": "GLD", "kind": "daily_after_close"},
        ).json()
        r = self.client.post(f"/api/schedules/{rec['id']}/run-now")
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["ticker"], "GLD")
        self.assertEqual(body["status"], "pending")
        # Runner queue saw it.
        self.assertEqual(self.runner.submitted, [body["id"]])
        # Schedule timestamp was updated.
        sched = self.client.get("/api/schedules").json()[0]
        self.assertEqual(sched["last_run_analysis_id"], body["id"])
        self.assertIsNotNone(sched["last_run_at"])

    def test_run_now_404_for_missing_schedule(self):
        r = self.client.post(
            "/api/schedules/00000000-0000-0000-0000-000000000000/run-now"
        )
        self.assertEqual(r.status_code, 404)

    # ---- seed-recommended --------------------------------------------

    def test_seed_recommended_creates_two_schedules(self):
        r = self.client.post("/api/schedules/seed-recommended")
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(len(body), 2)
        kinds = sorted(s["kind"] for s in body)
        self.assertEqual(kinds, ["daily_after_close", "volatility_trigger"])
        # Both should target GLD by default.
        for s in body:
            self.assertEqual(s["ticker"], "GLD")
            self.assertEqual(s["max_debate_rounds"], 3)

    def test_seed_recommended_is_idempotent(self):
        self.client.post("/api/schedules/seed-recommended")
        self.client.post("/api/schedules/seed-recommended")
        self.assertEqual(len(self.client.get("/api/schedules").json()), 2)

    def test_seed_recommended_with_custom_ticker(self):
        r = self.client.post(
            "/api/schedules/seed-recommended?ticker=IAU&language=Vietnamese"
        )
        self.assertEqual(r.status_code, 201, r.text)
        for sched in r.json():
            self.assertEqual(sched["ticker"], "IAU")
            self.assertEqual(sched["language"], "Vietnamese")


if __name__ == "__main__":
    unittest.main()
