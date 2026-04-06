"""
server/app.py — OpenEnv-compliant entry point for the Adaptive Cyber Defense environment.

Re-exports the fully-featured FastAPI `app` from the project root `app.py` and provides
the `main()` function required by [project.scripts] server = "server.app:main" in
pyproject.toml so that `openenv serve`, `uv run`, and `python -m` deployment modes work.

Endpoints (all implemented in ../app.py):
    POST /reset         — start a new episode
    POST /step          — apply one action
    GET  /state         — current observation
    GET  /analytics     — SOC performance metrics
    GET  /threat-intel  — live threat-intelligence feed
    GET  /              — health-check / root
"""

import os
import sys

# Ensure the project root is on sys.path so `import app` resolves correctly
# regardless of how this module is invoked (uv run, python -m, uvicorn, etc.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import app  # re-export the canonical FastAPI application


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """
    Entry point for:
        uv run server            (via [project.scripts] server = "server.app:main")
        openenv serve .
        python -m server.app

    Port resolution order:
        1. `port` argument (if provided)
        2. PORT environment variable
        3. Default 7860 (matches openenv.yaml docker.port)
    """
    import uvicorn

    resolved_port = port if port is not None else int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=resolved_port, reload=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Cyber Defense environment server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: PORT env or 7860)")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
