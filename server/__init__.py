"""TradingAgents Gold Edition — web backend.

A small FastAPI service that runs analyses in the background and
persists them to disk so a browser-based frontend can list, view, and
delete them later. Storage is plain JSON files (one per analysis)
under ``~/.tradingagents/web/analyses/`` — no extra database
dependency.

Layered intentionally:

  - ``storage.AnalysisStore`` — atomic CRUD on JSON files. No
    knowledge of FastAPI or LangGraph.
  - ``runner.AnalysisRunner`` — single worker thread that pulls jobs
    from a queue and drives ``TradingAgentsGraph.propagate``. Updates
    the store as each agent in the LangGraph completes so the
    frontend can poll progress without re-reading the whole graph.
  - ``api.create_app`` — FastAPI factory that wires the two together
    and serves the React frontend (``/`` falls through to
    ``frontend/dist/index.html``).
"""

from .storage import AnalysisRecord, AnalysisStore  # noqa: F401
from .runner import AnalysisRunner  # noqa: F401
from .api import create_app  # noqa: F401
