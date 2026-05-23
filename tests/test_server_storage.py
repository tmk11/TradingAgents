"""Tests for the web backend's filesystem-based analysis store."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from server.storage import PIPELINE_STEPS, AnalysisStore


class AnalysisStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # tmp_path equivalent without pytest fixtures so this file
        # also runs cleanly under plain ``python -m unittest``.
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = AnalysisStore(base_dir=Path(self._tmpdir.name))

    # ---- create --------------------------------------------------------

    def test_create_returns_pending_record_with_uuid_id(self):
        rec = self.store.create(
            ticker="gld",
            asset_type="commodity",
            analysis_date="2026-05-22",
            language="Vietnamese",
        )
        # Ticker is normalised to upper-case on create.
        self.assertEqual(rec["ticker"], "GLD")
        self.assertEqual(rec["status"], "pending")
        self.assertEqual(rec["asset_type"], "commodity")
        self.assertEqual(rec["language"], "Vietnamese")
        self.assertIsNone(rec["final_decision"])
        self.assertIsNone(rec["completed_at"])
        # UUID4 has 4 hyphens in canonical form.
        self.assertEqual(rec["id"].count("-"), 4)

    def test_create_initialises_every_pipeline_step_to_pending(self):
        rec = self.store.create(
            ticker="GC=F",
            asset_type="commodity",
            analysis_date="2026-05-22",
        )
        for step in PIPELINE_STEPS:
            self.assertEqual(
                rec["progress"][step],
                "pending",
                msg=f"step {step} not initialised",
            )

    def test_create_persists_to_disk_atomically(self):
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            analysis_date="2026-05-22",
        )
        path = Path(self._tmpdir.name) / f"{rec['id']}.json"
        self.assertTrue(path.exists())
        # And no leftover .tmp file from the atomic-rename dance.
        leftovers = list(Path(self._tmpdir.name).glob("*.tmp"))
        self.assertEqual(leftovers, [])

    # ---- get / list ----------------------------------------------------

    def test_get_round_trips_create(self):
        rec = self.store.create(
            ticker="GLD",
            asset_type="commodity",
            analysis_date="2026-05-22",
        )
        roundtripped = self.store.get(rec["id"])
        self.assertEqual(roundtripped, rec)

    def test_get_returns_none_for_missing_id(self):
        self.assertIsNone(self.store.get("does-not-exist-id-1234"))

    def test_get_rejects_path_traversal_ids(self):
        with self.assertRaises(ValueError):
            self.store.get("../etc/passwd")

    def test_list_returns_newest_first(self):
        # Create three with distinguishable created_at by patching the
        # values directly — uuid4 collisions are vanishingly unlikely.
        a = self.store.create(
            ticker="A", asset_type="stock", analysis_date="2026-05-22"
        )
        b = self.store.create(
            ticker="B", asset_type="stock", analysis_date="2026-05-22"
        )
        c = self.store.create(
            ticker="C", asset_type="stock", analysis_date="2026-05-22"
        )
        # Force an out-of-order created_at on disk to confirm sort.
        self.store.update(a["id"], created_at="2026-05-22T00:00:00+00:00")
        self.store.update(b["id"], created_at="2026-05-22T01:00:00+00:00")
        self.store.update(c["id"], created_at="2026-05-22T02:00:00+00:00")

        listed = self.store.list()
        self.assertEqual([r["ticker"] for r in listed], ["C", "B", "A"])

    def test_list_summary_strips_reports(self):
        rec = self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-05-22"
        )
        self.store.update(rec["id"], reports={"market_report": "x" * 1000})

        summary_only = self.store.list(summary_only=True)
        self.assertNotIn("reports", summary_only[0])

        full = self.store.list(summary_only=False)
        self.assertIn("reports", full[0])

    def test_list_skips_corrupt_files(self):
        # Drop a malformed JSON file in the dir; ``list`` should
        # log+ignore and return only the valid records.
        rec = self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-05-22"
        )
        bad = Path(self._tmpdir.name) / "00000000-bad-bad-bad-000000000000.json"
        bad.write_text("{not-json", encoding="utf-8")

        listed = self.store.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], rec["id"])

    # ---- update --------------------------------------------------------

    def test_update_applies_partial_changes(self):
        rec = self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-05-22"
        )
        updated = self.store.update(
            rec["id"],
            status="completed",
            final_decision="Buy",
        )
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["final_decision"], "Buy")
        # Untouched fields preserved.
        self.assertEqual(updated["ticker"], "GLD")
        self.assertEqual(updated["analysis_date"], "2026-05-22")

    def test_update_returns_none_for_missing_id(self):
        self.assertIsNone(self.store.update("missing-id", status="completed"))

    def test_update_progress_only_touches_that_step(self):
        rec = self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-05-22"
        )
        upd = self.store.update_progress(rec["id"], "market_analyst", "completed")
        self.assertEqual(upd["progress"]["market_analyst"], "completed")
        # Other steps still pending.
        self.assertEqual(upd["progress"]["news_analyst"], "pending")

    # ---- delete --------------------------------------------------------

    def test_delete_removes_file_and_returns_true(self):
        rec = self.store.create(
            ticker="GLD", asset_type="commodity", analysis_date="2026-05-22"
        )
        self.assertTrue(self.store.delete(rec["id"]))
        self.assertIsNone(self.store.get(rec["id"]))

    def test_delete_returns_false_for_missing_id(self):
        self.assertFalse(self.store.delete("does-not-exist"))


if __name__ == "__main__":
    unittest.main()
