"""Storage for recurring-analysis schedules.

A schedule is a small JSON record describing *when* to auto-create an
analysis (daily after US close, on a >1.5% intraday gold move, etc.)
plus the parameters to pass to ``AnalysisStore.create``. The actual
firing logic lives in ``server.scheduler``; this module is just the
persistence layer, mirroring the shape of ``server.storage`` so the
codebase has one consistent storage idiom.

We deliberately keep the surface small:

  * Two schedule "kinds" today — ``daily_after_close`` and
    ``volatility_trigger`` — chosen because they cover the realistic
    decision cadence for gold (daily routine + reactive on big moves)
    without dragging in a full cron parser.
  * Every schedule keeps its own ``last_run_at`` / ``last_run_analysis_id``
    so the scheduler thread can decide whether to fire without needing
    to scan the analysis store.
  * Schemas are forgiving: extra fields on disk are preserved through
    update/read, so future kinds can land without migrating old files.
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


DEFAULT_BASE_DIR = Path.home() / ".tradingagents" / "web" / "schedules"


# Recognised schedule kinds. Adding a new kind is a 3-step process:
#   1. Add the literal here (also update the frontend ScheduleKind type).
#   2. Implement should_fire_* + fire-side logic in server.scheduler.
#   3. Whitelist the params in _validate_params below.
SCHEDULE_KINDS = ("daily_after_close", "volatility_trigger")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Defaults match the recommended workflow:
#
#   * Daily run lands at 21:30 UTC, which is 16:30 EST in winter and
#     17:30 EDT in summer — comfortably after the 16:00 ET equity
#     close in both regimes, and during the 18:00 ET globex pause for
#     gold futures.
#   * Volatility threshold of 1.5% matches gold's typical daily noise
#     ceiling; below that, "trigger every move" produces too many runs.
DEFAULT_DAILY_PARAMS: Dict[str, Any] = {
    "fire_hour_utc": 21,
    "fire_minute_utc": 30,
    "weekdays_only": True,
}
DEFAULT_VOLATILITY_PARAMS: Dict[str, Any] = {
    "threshold_pct": 1.5,            # absolute %, e.g. 1.5 means ±1.5%
    "throttle_hours": 6,             # min gap between two fires
    "check_interval_minutes": 15,    # how often the scheduler polls price
}


def _validate_params(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalised params dict for the given schedule kind.

    Falls back to the documented defaults for missing keys; coerces
    types where reasonable so the frontend can pass strings from form
    inputs without us forcing a parse on the JS side.
    """
    if kind == "daily_after_close":
        merged = {**DEFAULT_DAILY_PARAMS, **(params or {})}
        merged["fire_hour_utc"] = int(merged["fire_hour_utc"]) % 24
        merged["fire_minute_utc"] = int(merged["fire_minute_utc"]) % 60
        merged["weekdays_only"] = bool(merged["weekdays_only"])
        return merged
    if kind == "volatility_trigger":
        merged = {**DEFAULT_VOLATILITY_PARAMS, **(params or {})}
        # Clamp the threshold to a sane band — 0.1%..10%. Below 0.1%
        # we'd fire on noise; above 10% the trigger never fires and
        # the user may as well delete the schedule.
        threshold = float(merged["threshold_pct"])
        merged["threshold_pct"] = max(0.1, min(10.0, threshold))
        merged["throttle_hours"] = max(1, int(merged["throttle_hours"]))
        merged["check_interval_minutes"] = max(
            5, int(merged["check_interval_minutes"])
        )
        return merged
    raise ValueError(f"unknown schedule kind: {kind!r}")


@dataclass
class ScheduleRecord:
    """Convenience view of a stored schedule. Callers usually work
    with the dict form directly — this just documents the shape."""

    id: str
    name: str
    ticker: str
    asset_type: str
    kind: str
    params: Dict[str, Any]
    language: str
    max_debate_rounds: int
    max_risk_discuss_rounds: int
    enabled: bool
    last_run_at: Optional[str]
    last_run_analysis_id: Optional[str]
    last_check_at: Optional[str]
    created_at: str


class ScheduleStore:
    """JSON-on-disk store. Same atomic-write idiom as AnalysisStore."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = (
            Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- CRUD ----------------------------------------------------------

    def create(
        self,
        *,
        ticker: str,
        asset_type: str,
        kind: str,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        language: str = "English",
        max_debate_rounds: int = 3,
        max_risk_discuss_rounds: int = 3,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        if kind not in SCHEDULE_KINDS:
            raise ValueError(f"unknown schedule kind: {kind!r}")

        validated = _validate_params(kind, params or {})

        # Auto-name when the caller didn't provide one. Easier on the
        # frontend — the form can stay focused on ticker/kind only.
        normalised_ticker = ticker.strip().upper()
        if not name:
            if kind == "daily_after_close":
                name = f"{normalised_ticker} · daily after US close"
            elif kind == "volatility_trigger":
                threshold = validated["threshold_pct"]
                name = f"{normalised_ticker} · volatility ≥ {threshold:.1f}%"
            else:
                name = f"{normalised_ticker} · {kind}"

        record = {
            "id": str(uuid.uuid4()),
            "name": name,
            "ticker": normalised_ticker,
            "asset_type": asset_type,
            "kind": kind,
            "params": validated,
            "language": language,
            "max_debate_rounds": int(max_debate_rounds),
            "max_risk_discuss_rounds": int(max_risk_discuss_rounds),
            "enabled": bool(enabled),
            "last_run_at": None,
            "last_run_analysis_id": None,
            "last_check_at": None,
            "created_at": _utcnow_iso(),
        }
        with self._lock:
            self._write(record)
        return record

    def get(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(schedule_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt schedule file %s: %s", path, exc)
            return None

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            results: List[Dict[str, Any]] = []
            for f in self.base_dir.glob("*.json"):
                try:
                    record = json.loads(f.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                results.append(record)
            results.sort(key=lambda r: r.get("created_at", ""))
            return results

    def update(
        self, schedule_id: str, **changes: Any
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self.get(schedule_id)
            if record is None:
                return None
            # Re-validate params if either kind or params is being
            # changed — keeps disk records well-formed even if a
            # caller passes a partial dict.
            if "params" in changes or "kind" in changes:
                kind = changes.get("kind", record["kind"])
                params = changes.get("params", record.get("params", {}))
                changes["params"] = _validate_params(kind, params or {})
                changes["kind"] = kind
            record.update(changes)
            self._write(record)
            return record

    def delete(self, schedule_id: str) -> bool:
        with self._lock:
            path = self._path(schedule_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    def mark_fired(
        self,
        schedule_id: str,
        *,
        analysis_id: str,
        fired_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convenience helper used by the scheduler after a fire."""
        return self.update(
            schedule_id,
            last_run_at=fired_at or _utcnow_iso(),
            last_run_analysis_id=analysis_id,
        )

    def mark_checked(
        self, schedule_id: str, *, checked_at: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Used by volatility schedules to record a no-op poll."""
        return self.update(
            schedule_id,
            last_check_at=checked_at or _utcnow_iso(),
        )

    # ---- internals -----------------------------------------------------

    def _path(self, schedule_id: str) -> Path:
        if (
            "/" in schedule_id
            or "\\" in schedule_id
            or schedule_id.startswith(".")
        ):
            raise ValueError(f"invalid schedule id: {schedule_id!r}")
        return self.base_dir / f"{schedule_id}.json"

    def _write(self, record: Dict[str, Any]) -> None:
        path = self._path(record["id"])
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(record, indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
