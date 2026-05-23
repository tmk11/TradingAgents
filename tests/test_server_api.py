"""Tests for the FastAPI surface — analyses CRUD + progress.

The real ``AnalysisRunner`` would fire up TradingAgentsGraph and an
LLM client, so these tests inject a stub runner that just records
which jobs were submitted. Storage is scoped to a tmp_path so tests
are hermetic.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import List

from fastapi.testclient import TestClient

from server.api import create_app
from server.runner import AnalysisRunner
from server.storage import AnalysisStore


class _StubRunner:
    """Minimal AnalysisRunner stand-in for tests.

    Records the IDs submitted to ``submit`` so tests can assert that
    ``POST /api/analyses`` actually queues the job. ``start``/``stop``
    are no-ops.
    """

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


def _build_client(tmp_path: Path):
    store = AnalysisStore(base_dir=tmp_path)
    runner = _StubRunner()
    # static_dir=False disables the SPA fallback so 404s on unknown
    # routes stay clean for assertion.
    app = create_app(
        store=store, runner=runner, static_dir=False, enable_cors=False
    )
    client = TestClient(app)
    return client, store, runner


class HealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.client, _, _ = _build_client(Path(self._tmpdir.name))

    def test_health_returns_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})


class CreateAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.client, self.store, self.runner = _build_client(
            Path(self._tmpdir.name)
        )

    def test_create_returns_201_with_pending_status(self):
        r = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2026-05-22"},
        )
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertEqual(body["ticker"], "GLD")
        self.assertEqual(body["asset_type"], "commodity")
        self.assertEqual(body["status"], "pending")
        # List response shape: no ``reports`` field on summary.
        self.assertNotIn("reports", body)

    def test_create_routes_known_gold_tickers_to_commodity(self):
        for ticker, expected in [
            ("GLD", "commodity"),
            ("GC=F", "commodity"),
            ("XAUUSD=X", "commodity"),
            ("AAPL", "stock"),
            ("BTC-USD", "crypto"),
        ]:
            r = self.client.post(
                "/api/analyses",
                json={"ticker": ticker, "analysis_date": "2026-05-22"},
            )
            self.assertEqual(r.status_code, 201, f"failed for {ticker}: {r.text}")
            self.assertEqual(r.json()["asset_type"], expected, f"wrong route for {ticker}")

    def test_create_submits_to_runner(self):
        r = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2026-05-22"},
        )
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.runner.submitted, [r.json()["id"]])

    def test_create_rejects_future_date(self):
        r = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2099-01-01"},
        )
        self.assertEqual(r.status_code, 422)
        # FastAPI/pydantic v2 puts the error message under .detail[*].msg
        self.assertIn("future", r.text)

    def test_create_rejects_invalid_ticker_chars(self):
        r = self.client.post(
            "/api/analyses",
            json={"ticker": "AAPL; rm -rf /", "analysis_date": "2026-05-22"},
        )
        self.assertEqual(r.status_code, 422)

    def test_create_rejects_malformed_date(self):
        r = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "tomorrow"},
        )
        self.assertEqual(r.status_code, 422)

    def test_create_with_custom_language_persists_choice(self):
        r = self.client.post(
            "/api/analyses",
            json={
                "ticker": "GLD",
                "analysis_date": "2026-05-22",
                "language": "Vietnamese",
            },
        )
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["language"], "Vietnamese")


class ListAndGetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.client, self.store, _ = _build_client(Path(self._tmpdir.name))

    def test_list_empty_returns_array(self):
        r = self.client.get("/api/analyses")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_list_returns_newest_first(self):
        a = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2026-05-22"},
        ).json()
        b = self.client.post(
            "/api/analyses",
            json={"ticker": "IAU", "analysis_date": "2026-05-22"},
        ).json()
        listed = self.client.get("/api/analyses").json()
        # Latest created first (b after a).
        self.assertEqual([x["id"] for x in listed], [b["id"], a["id"]])

    def test_get_returns_full_detail_with_reports_field(self):
        rec = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2026-05-22"},
        ).json()
        # Simulate completion with stored reports.
        self.store.update(
            rec["id"],
            status="completed",
            reports={"market_report": "## Market\nTrend up."},
            final_decision="Buy",
        )
        r = self.client.get(f"/api/analyses/{rec['id']}")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "completed")
        self.assertIn("market_report", body["reports"])
        self.assertEqual(body["final_decision"], "Buy")

    def test_get_missing_id_returns_404(self):
        r = self.client.get("/api/analyses/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_get_invalid_id_returns_400(self):
        r = self.client.get("/api/analyses/..%2Fetc%2Fpasswd")
        # FastAPI may return 400 (our explicit raise) or 404 depending
        # on URL decoding; either is fine — we just need it to not
        # leak filesystem contents.
        self.assertIn(r.status_code, (400, 404))


class DeleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.client, self.store, _ = _build_client(Path(self._tmpdir.name))

    def test_delete_existing_returns_204_and_removes_record(self):
        rec = self.client.post(
            "/api/analyses",
            json={"ticker": "GLD", "analysis_date": "2026-05-22"},
        ).json()
        r = self.client.delete(f"/api/analyses/{rec['id']}")
        self.assertEqual(r.status_code, 204)
        # Round-trip: get now 404s.
        self.assertEqual(self.client.get(f"/api/analyses/{rec['id']}").status_code, 404)

    def test_delete_missing_returns_404(self):
        r = self.client.delete("/api/analyses/does-not-exist")
        self.assertEqual(r.status_code, 404)


class AssetTypeDetectionTests(unittest.TestCase):
    """Sanity-check the routing helper inside the API module."""

    def test_routing_table(self):
        from server.api import detect_asset_type

        self.assertEqual(detect_asset_type("GLD"), "commodity")
        self.assertEqual(detect_asset_type("gc=f"), "commodity")
        self.assertEqual(detect_asset_type("XAUUSD=X"), "commodity")
        self.assertEqual(detect_asset_type("AAPL"), "stock")
        self.assertEqual(detect_asset_type("CNC.TO"), "stock")
        self.assertEqual(detect_asset_type("BTC-USD"), "crypto")
        self.assertEqual(detect_asset_type("eth-usd"), "crypto")


if __name__ == "__main__":
    unittest.main()
