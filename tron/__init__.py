"""
TRON - Unified Distributed Computing, Orchestration & GPU Infrastructure

Key exports:
- @remote: Turn any function into distributed code (NEW - recommended)
- @task: Legacy decorator support
- TrainingOrchestrator: ML orchestration with adapters (TRON-II integration)
- VirtualGPUCluster: Aggregated GPU cluster management (vGPU integration)
- config(): Configure server URL
- Tron: SDK client (legacy)
- serialize/deserialize: For custom serialization

Integrated Layers:
- tron._runtime: DAGs, pipelines, scheduling, execution
- tron.orchestrator: TRON-II orchestration & ML adapters
- tron.gpu: vGPU cluster management & virtual GPU runtime
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

# Integrated layers - TRON-II Orchestrator
try:
    from .orchestrator import TrainingOrchestrator, TrainingConfig
except ImportError:
    pass

# Integrated layers - Carrier automation (official Verizon / AT&T APIs)
try:
    from .carriers import CarrierGateway, CarrierOp, OpKind
except ImportError:
    pass

# Integrated layers - vGPU Cluster
try:
    from .gpu import VirtualGPUCluster, VirtualGPUNode, VirtualGPURuntime
except ImportError:
    pass

# Integrated layers - Runtime (formerly tkron)
# NOTE: Runtime modules auto-connect and may block import.
# Import explicitly: from tron._runtime import DAG, Pipeline
# (not loaded here to avoid blocking main module import)

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
    
    # Integrated orchestration
    "TrainingOrchestrator",
    "TrainingConfig",
    
    # Integrated carrier automation
    "CarrierGateway",
    "CarrierOp",
    "OpKind",

    # Integrated GPU cluster
    "VirtualGPUCluster",
    "VirtualGPUNode",
    "VirtualGPURuntime",
    
    # Runtime (import from tron._runtime explicitly)
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

