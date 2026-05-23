"""FastAPI app for the Gold-Edition web frontend.

Three layers, all wired together by ``create_app``:

  - REST API under ``/api`` — list / create / get / delete analyses.
  - Static frontend mount at ``/`` — serves ``frontend/dist`` when
    that directory exists (i.e. after ``npm run build``). In dev,
    Vite runs separately on port 5173 and proxies ``/api`` to here.
  - Lifespan hook — starts the background runner thread on app
    startup and stops it on shutdown.

The factory pattern (``create_app(store=..., runner=...)``) is so
tests can inject a temp-dir store and skip the real LangGraph runner
entirely.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .runner import AnalysisRunner
from .storage import AnalysisStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset-type detection (mirrors cli.utils.detect_asset_type)
# ---------------------------------------------------------------------------

_GOLD_TICKERS = {
    "GC=F", "MGC=F",
    "XAUUSD=X", "XAU=X",
    "GLD", "IAU", "SGOL", "BAR", "AAAU", "GLDM", "OUNZ",
    "GDX", "GDXJ", "RING", "NUGT", "JNUG",
    "^XAU",
}

_CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")


def detect_asset_type(ticker: str) -> str:
    """Same routing as ``cli.utils.detect_asset_type``.

    Lifted here as a tiny duplicate so the server module doesn't pull
    in ``cli.*`` (which depends on questionary / typer / rich).
    """
    norm = ticker.strip().upper()
    if norm in _GOLD_TICKERS:
        return "commodity"
    if norm.endswith(_CRYPTO_SUFFIXES):
        return "crypto"
    return "stock"


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


_TICKER_RE = re.compile(r"^[A-Za-z0-9._\-^=]{1,32}$")


class CreateAnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32)
    analysis_date: str = Field(
        ...,
        description="YYYY-MM-DD analysis date (must not be in the future).",
    )
    language: str = Field(default="English", min_length=2, max_length=32)
    max_debate_rounds: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Number of Bull/Bear back-and-forth rounds. The graph "
            "stops the investment debate when the speaker count "
            "reaches ``2 * max_debate_rounds``."
        ),
    )
    max_risk_discuss_rounds: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Number of Aggressive/Conservative/Neutral risk-debate "
            "rounds. The graph stops the risk debate when the "
            "speaker count reaches ``3 * max_risk_discuss_rounds``."
        ),
    )

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        if not _TICKER_RE.match(v):
            raise ValueError(
                "ticker must be 1-32 chars of letters/digits/._-^="
            )
        return v.strip().upper()

    @field_validator("analysis_date")
    @classmethod
    def _validate_date(cls, v: str) -> str:
        try:
            d = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("analysis_date must be YYYY-MM-DD") from exc
        if d > datetime.utcnow().date():
            raise ValueError("analysis_date cannot be in the future")
        return v


class AnalysisSummary(BaseModel):
    """List-view shape — no ``reports`` to keep payloads small."""

    id: str
    ticker: str
    asset_type: str
    analysis_date: str
    language: str
    status: str
    progress: Dict[str, str]
    final_decision: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    # Optional so analyses created before this field existed (older
    # JSON files on disk) still validate. New records always have
    # both fields populated by ``AnalysisStore.create``.
    max_debate_rounds: Optional[int] = None
    max_risk_discuss_rounds: Optional[int] = None


class AnalysisDetail(AnalysisSummary):
    reports: Dict[str, Any]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    store: Optional[AnalysisStore] = None,
    runner: Optional[AnalysisRunner] = None,
    static_dir: Optional[Path] = None,
    enable_cors: bool = True,
) -> FastAPI:
    """Build and return a configured FastAPI app.

    Args:
        store: Override the default file-backed store. Used by tests
            to scope state to a tmp_path.
        runner: Override the default runner. Pass a no-op stub in
            tests to skip the LLM stack.
        static_dir: Where to find the built React frontend
            (``frontend/dist``). Defaults to the sibling ``frontend/
            dist`` of the repo root. Pass ``False`` to disable static
            mounting entirely.
        enable_cors: Allow same-origin browser requests from the Vite
            dev server (port 5173). Off by default in tests.
    """
    store = store if store is not None else AnalysisStore()
    runner = runner if runner is not None else AnalysisRunner(store)

    if static_dir is None:
        static_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="TradingAgents Gold Edition",
        version="0.3.0",
        description=(
            "Background-runner API + React UI for the Gold-Edition "
            "TradingAgents framework. Runs analyses asynchronously, "
            "persists them to disk, and exposes them for review or "
            "deletion through a small REST surface."
        ),
        lifespan=lifespan,
    )

    if enable_cors:
        # Vite dev server defaults to 5173. We don't bother with auth
        # because this is a local-developer tool — the assumption is
        # the API only listens on localhost.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ---- API routes ----------------------------------------------------

    def get_store() -> AnalysisStore:
        return store

    def get_runner() -> AnalysisRunner:
        return runner

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/analyses", response_model=List[AnalysisSummary])
    def list_analyses(s: AnalysisStore = Depends(get_store)) -> List[Dict[str, Any]]:
        return s.list(summary_only=True)

    @app.post(
        "/api/analyses",
        response_model=AnalysisSummary,
        status_code=status.HTTP_201_CREATED,
    )
    def create_analysis(
        req: CreateAnalysisRequest,
        s: AnalysisStore = Depends(get_store),
        r: AnalysisRunner = Depends(get_runner),
    ) -> Dict[str, Any]:
        record = s.create(
            ticker=req.ticker,
            asset_type=detect_asset_type(req.ticker),
            analysis_date=req.analysis_date,
            language=req.language,
            max_debate_rounds=req.max_debate_rounds,
            max_risk_discuss_rounds=req.max_risk_discuss_rounds,
        )
        r.submit(record["id"])
        # Drop ``reports`` to match AnalysisSummary shape.
        return {k: v for k, v in record.items() if k != "reports"}

    @app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail)
    def get_analysis(
        analysis_id: str, s: AnalysisStore = Depends(get_store)
    ) -> Dict[str, Any]:
        try:
            record = s.get(analysis_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid analysis id",
            )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="analysis not found",
            )
        # Always include ``reports`` (may be empty for pending/running).
        record.setdefault("reports", {})
        return record

    @app.delete(
        "/api/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT
    )
    def delete_analysis(
        analysis_id: str, s: AnalysisStore = Depends(get_store)
    ) -> None:
        try:
            ok = s.delete(analysis_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid analysis id",
            )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="analysis not found",
            )

    # ---- Frontend (production build) ----------------------------------

    if static_dir is not False:
        static_path = Path(static_dir)
        if static_path.exists():
            assets_dir = static_path / "assets"
            if assets_dir.exists():
                app.mount(
                    "/assets",
                    StaticFiles(directory=assets_dir),
                    name="assets",
                )

            @app.get("/", include_in_schema=False)
            @app.get("/{full_path:path}", include_in_schema=False)
            def spa_fallback(full_path: str = "") -> FileResponse:
                """Serve the SPA for any non-API GET.

                FastAPI matches more-specific routes (``/api/...``,
                ``/assets/...``) first, so anything that lands here is
                expected to be a frontend route handled by React Router.
                """
                index = static_path / "index.html"
                if index.exists():
                    return FileResponse(index)
                # Soft-fail with a hint when the user starts the
                # backend before building the frontend.
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": (
                            "frontend not built — run "
                            "`npm install && npm run build` in ./frontend"
                        )
                    },
                )

    return app
