"""Background analysis runner.

A single worker thread drains a queue and runs ``TradingAgentsGraph``
to completion for each job, updating the store as it goes so the
frontend can poll progress live.

We deliberately keep this single-threaded:

  * LangGraph + the LLM clients hold their own thread state and are
    not designed for concurrent re-entry from a single graph object.
  * Most operators run this on a workstation against paid LLM APIs;
    fan-out parallelism mostly increases bill, not throughput.

If you need concurrency, run multiple processes (one
``AnalysisRunner`` each) sharing the same ``base_dir`` — the store's
atomic writes make that safe.
"""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from .storage import AnalysisStore

logger = logging.getLogger(__name__)


# Mapping from LangGraph chunk keys to the canonical pipeline step
# names exposed in ``progress``. Keys without a chunk equivalent are
# updated explicitly in ``_dispatch_progress``.
REPORT_TO_STEP = {
    "market_report": "market_analyst",
    "sentiment_report": "sentiment_analyst",
    "news_report": "news_analyst",
    "fundamentals_report": "fundamentals_analyst",
    "investment_plan": "research_manager",
    "trader_investment_plan": "trader",
}


# Stock runs include the Fundamentals Analyst; commodity / crypto
# runs auto-skip it in the graph, so its progress entry should be
# pre-marked "skipped" instead of left "pending" forever.
def _initial_progress_for_asset(asset_type: str) -> Dict[str, str]:
    from .storage import PIPELINE_STEPS

    progress = {step: "pending" for step in PIPELINE_STEPS}
    if asset_type in ("commodity", "crypto"):
        progress["fundamentals_analyst"] = "skipped"
    return progress


class AnalysisRunner:
    """Single-worker background runner. Thread-safe ``submit``/``stop``."""

    def __init__(self, store: AnalysisStore) -> None:
        self.store = store
        self._queue: "Queue[Optional[str]]" = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_lock = threading.Lock()

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        with self._started_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="analysis-runner", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        # Wake the worker if it's idle on ``queue.get``.
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, analysis_id: str) -> None:
        self._queue.put(analysis_id)

    # ---- worker --------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                analysis_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if analysis_id is None:  # sentinel from ``stop``
                continue
            try:
                self._run_one(analysis_id)
            except Exception:  # noqa: BLE001
                # Keep the worker alive across job failures.
                tb = traceback.format_exc()
                logger.exception("Analysis %s crashed", analysis_id)
                self.store.update(
                    analysis_id,
                    status="failed",
                    error=tb,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

    def _run_one(self, analysis_id: str) -> None:
        # Late imports keep the server importable on machines that
        # don't have the LLM stack installed (e.g. a frontend-only
        # development setup that talks to a remote backend).
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        record = self.store.get(analysis_id)
        if record is None:
            logger.warning("Analysis %s vanished before runner picked it up", analysis_id)
            return

        config = DEFAULT_CONFIG.copy()
        if record.get("language"):
            config["output_language"] = record["language"]

        # Equity runs include fundamentals; everything else (crypto,
        # commodity) drops it because there's no real fundamentals
        # data and the prompts mislead the LLM.
        if record["asset_type"] == "stock":
            analysts: List[str] = ["market", "social", "news", "fundamentals"]
        else:
            analysts = ["market", "social", "news"]

        # Mark started before constructing the graph: graph init takes
        # a few seconds (LLM client setup) and the user wants to see
        # the run leave the queue immediately.
        self.store.update(
            analysis_id,
            status="running",
            progress=_initial_progress_for_asset(record["asset_type"]),
        )

        graph = TradingAgentsGraph(
            selected_analysts=analysts, debug=False, config=config
        )

        # Drive the graph manually so we can hook every chunk and
        # surface progress to the store.
        init_state = graph.propagator.create_initial_state(
            record["ticker"],
            record["analysis_date"],
            asset_type=record["asset_type"],
        )
        args = graph.propagator.get_graph_args()

        trace = []
        for chunk in graph.graph.stream(init_state, **args):
            trace.append(chunk)
            self._dispatch_progress(analysis_id, chunk)

        # Streamed chunks are per-node deltas — merge into a final
        # state matching the non-debug invoke() return shape.
        final_state: Dict[str, Any] = {}
        for chunk in trace:
            final_state.update(chunk)

        decision = graph.process_signal(final_state["final_trade_decision"])

        # Persist a flat reports dict that the frontend can render
        # section-by-section. The debate sub-states travel as nested
        # dicts because the markdown renderer iterates them.
        self.store.update(
            analysis_id,
            status="completed",
            reports={
                "market_report": final_state.get("market_report", ""),
                "sentiment_report": final_state.get("sentiment_report", ""),
                "news_report": final_state.get("news_report", ""),
                "fundamentals_report": final_state.get("fundamentals_report", ""),
                "investment_plan": final_state.get("investment_plan", ""),
                "trader_investment_plan": final_state.get("trader_investment_plan", ""),
                "final_trade_decision": final_state.get("final_trade_decision", ""),
                "investment_debate_state": final_state.get(
                    "investment_debate_state", {}
                ),
                "risk_debate_state": final_state.get("risk_debate_state", {}),
            },
            final_decision=decision,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _dispatch_progress(self, analysis_id: str, chunk: Dict[str, Any]) -> None:
        """Translate a LangGraph chunk into progress updates."""
        # Direct report-key → step mappings.
        for chunk_key, step in REPORT_TO_STEP.items():
            if chunk.get(chunk_key):
                self.store.update_progress(analysis_id, step, "completed")

        # Investment debate state — bull / bear / research-manager.
        ids = chunk.get("investment_debate_state") or {}
        if ids.get("bull_history"):
            self.store.update_progress(analysis_id, "bull_researcher", "completed")
        if ids.get("bear_history"):
            self.store.update_progress(analysis_id, "bear_researcher", "completed")
        if ids.get("judge_decision"):
            self.store.update_progress(analysis_id, "research_manager", "completed")

        # Risk debate state — three risk analysts + portfolio manager.
        rds = chunk.get("risk_debate_state") or {}
        if rds.get("aggressive_history"):
            self.store.update_progress(analysis_id, "risk_aggressive", "completed")
        if rds.get("conservative_history"):
            self.store.update_progress(analysis_id, "risk_conservative", "completed")
        if rds.get("neutral_history"):
            self.store.update_progress(analysis_id, "risk_neutral", "completed")
        if rds.get("judge_decision"):
            self.store.update_progress(analysis_id, "portfolio_manager", "completed")
