"""Filesystem-based storage for web-driven analysis runs.

One JSON file per analysis, atomically written via tempfile + rename
so a crash mid-write never leaves a half-baked file. A reentrant lock
serialises mutations within the process; cross-process safety relies
on the rename being atomic at the filesystem level (which it is on
POSIX and on NTFS via ``os.replace``).

The schema is intentionally flat and forgiving — extra fields are
preserved on read/update so future migrations don't need a v-bump.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default directory for stored analyses. Lives under the same
# ``~/.tradingagents/`` tree as memory log + checkpoints, so users
# who already manage that dir's lifecycle don't pick up a new path.
DEFAULT_BASE_DIR = Path.home() / ".tradingagents" / "web" / "analyses"


# Canonical pipeline steps reported through ``progress``. The runner
# updates these as each LangGraph node finishes; the frontend renders
# a checkmark per completed step.
PIPELINE_STEPS = (
    "market_analyst",
    "sentiment_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "bull_researcher",
    "bear_researcher",
    "research_manager",
    "trader",
    "risk_aggressive",
    "risk_conservative",
    "risk_neutral",
    "portfolio_manager",
)


@dataclass
class AnalysisRecord:
    """In-memory view of a stored analysis. Convenience type."""

    id: str
    ticker: str
    asset_type: str
    analysis_date: str
    language: str
    status: str
    progress: Dict[str, str]
    reports: Dict[str, Any]
    final_decision: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        return cls(
            id=data["id"],
            ticker=data["ticker"],
            asset_type=data.get("asset_type", "stock"),
            analysis_date=data["analysis_date"],
            language=data.get("language", "English"),
            status=data.get("status", "pending"),
            progress=data.get("progress", {}),
            reports=data.get("reports", {}),
            final_decision=data.get("final_decision"),
            error=data.get("error"),
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
        )


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp. Matches what the rest of the framework emits."""
    return datetime.now(timezone.utc).isoformat()


class AnalysisStore:
    """JSON-on-disk store for analysis records."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- CRUD ----------------------------------------------------------

    def create(
        self,
        *,
        ticker: str,
        asset_type: str,
        analysis_date: str,
        language: str = "English",
    ) -> Dict[str, Any]:
        """Create a new ``pending`` analysis and persist it.

        Returns the stored record dict (not the dataclass) so the API
        layer can pass it straight into ``jsonable_encoder``.
        """
        analysis_id = str(uuid.uuid4())
        record = {
            "id": analysis_id,
            "ticker": ticker.strip().upper(),
            "asset_type": asset_type,
            "analysis_date": analysis_date,
            "language": language,
            "status": "pending",
            "progress": {step: "pending" for step in PIPELINE_STEPS},
            "reports": {},
            "final_decision": None,
            "error": None,
            "created_at": _utcnow_iso(),
            "completed_at": None,
        }
        with self._lock:
            self._write(record)
        return record

    def get(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(analysis_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt analysis file %s: %s", path, exc)
            return None

    def list(self, *, summary_only: bool = True) -> List[Dict[str, Any]]:
        """Return all stored analyses, newest first.

        ``summary_only`` strips the heavy ``reports`` field — the list
        endpoint does not need full markdown blobs and dropping them
        keeps payloads small even with hundreds of past runs.
        """
        with self._lock:
            results: List[Dict[str, Any]] = []
            for f in self.base_dir.glob("*.json"):
                try:
                    record = json.loads(f.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if summary_only:
                    record = {k: v for k, v in record.items() if k != "reports"}
                results.append(record)
            results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            return results

    def update(self, analysis_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
        """Apply a partial update and rewrite the file atomically."""
        with self._lock:
            record = self.get(analysis_id)
            if record is None:
                return None
            record.update(changes)
            self._write(record)
            return record

    def update_progress(
        self, analysis_id: str, step: str, status: str
    ) -> Optional[Dict[str, Any]]:
        """Convenience: flip a single pipeline step's status."""
        with self._lock:
            record = self.get(analysis_id)
            if record is None:
                return None
            progress = dict(record.get("progress", {}))
            progress[step] = status
            record["progress"] = progress
            self._write(record)
            return record

    def delete(self, analysis_id: str) -> bool:
        with self._lock:
            path = self._path(analysis_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    # ---- internals -----------------------------------------------------

    def _path(self, analysis_id: str) -> Path:
        # Reject ids that would escape base_dir. uuid4() never produces
        # path separators, but the public ``get/delete`` accept arbitrary
        # strings from the API layer, and FastAPI path params are strings.
        if "/" in analysis_id or "\\" in analysis_id or analysis_id.startswith("."):
            raise ValueError(f"invalid analysis id: {analysis_id!r}")
        return self.base_dir / f"{analysis_id}.json"

    def _write(self, record: Dict[str, Any]) -> None:
        """Atomic JSON write. Tempfile is in the same dir so the rename
        is on the same filesystem (cross-fs renames aren't atomic)."""
        path = self._path(record["id"])
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(record, indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
