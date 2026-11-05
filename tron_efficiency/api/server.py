"""Run the Compute Efficiency Engine's API + dashboard with uvicorn.

Port convention mirrors ``queue_server.py``'s ``TRON_PORT`` env var, but on
its own variable and default port so the two servers never collide:
``TRON_EFFICIENCY_PORT`` (default 9100).
"""

from __future__ import annotations

import os


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    from tron_efficiency.api.app import app

    host = host or os.environ.get("TRON_EFFICIENCY_HOST", "0.0.0.0")
    port = port or int(os.environ.get("TRON_EFFICIENCY_PORT", 9100))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
