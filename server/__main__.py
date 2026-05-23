"""``python -m server`` — convenience launcher for the web backend.

Usage:
    python -m server                # listen on 127.0.0.1:8765
    python -m server --port 8000    # custom port
    python -m server --host 0.0.0.0 # bind to all interfaces

Picked port 8765 as default because it's an unprivileged port that's
unlikely to clash with anything else a developer has running.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .api import create_app


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m server",
        description="TradingAgents Gold Edition web backend",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    p.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="uvicorn log level",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "uvicorn is required to run the server. Install with:\n"
            "    pip install uvicorn fastapi\n"
        )
        return 2

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
