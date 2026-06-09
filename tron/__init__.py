"""
TRON - Make Distributed Computing Feel Like Native Python

Key exports:
- @remote: Turn any function into distributed code (NEW - recommended)
- @task: Legacy decorator support
- config(): Configure server URL
- Tron: SDK client (legacy)
- serialize/deserialize: For custom serialization
"""

# New magic layer
from .remote import remote
from .config import (
    set_config_url,
    get_config,
    add_server,
    get_server,
    list_servers,
    use_server,
    start_local_server,
    ensure_server,
    stop_local_server,
    start_local_worker,
    stop_local_worker,
    start_local_environment,
)
from .magic_future import MagicFuture

# Backward compatibility - old APIs still work
from .decorators import task
from .sdk import Tron
from .serializer import serialize, deserialize
from .client import submit, status

__all__ = [
    # New API (recommended)
    "remote",
    "MagicFuture",
    
    # Configuration
    "config",
    "add_server",
    "get_server",
    "list_servers",
    "use_server",
    "start_local_server",
    "ensure_server",
    "stop_local_server",
    "start_local_worker",
    "stop_local_worker",
    "start_local_environment",

    # Legacy API (still works)
    "task",
    "Tron",
    "serialize",
    "deserialize",
    "submit",
    "status",
]


def config(url: str = None, name: str = None) -> None:
    """Configure TRON server URL or select a registered named server."""
    if url and name:
        raise ValueError("Specify either url or name, not both")

    if url:
        set_config_url(url)
    elif name:
        use_server(name)
    else:
        print(f"TRON Server: {get_config().url}")

