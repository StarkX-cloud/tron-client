"""Measured fleet capacity for Autopilot — the "without limits" substrate.

Reads live worker registrations (memory, GPU, efficiency runtime) and
topology signals. Never fabricates capacity. ``dedicated`` still means
"human/other system must provision" — this module does not call a cloud
marketplace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tron_router.decision import PoolStatus, query_pool_status


@dataclass
class WorkerCapacity:
    name: str
    memory_gb: float
    free_memory_gb: float | None
    gpu: bool
    efficiency_runtime: bool
    latency_ms: float | None = None
    bandwidth_mbps: float | None = None
    sku: str | None = None
    hourly_rate_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FleetCapacity:
    """Snapshot of what Autopilot can actually schedule onto right now."""

    master_url: str
    reachable: bool
    workers: list[WorkerCapacity] = field(default_factory=list)
    error: str | None = None

    @property
    def alive_count(self) -> int:
        return len(self.workers)

    @property
    def efficiency_worker_count(self) -> int:
        return sum(1 for w in self.workers if w.efficiency_runtime)

    @property
    def total_memory_gb(self) -> float:
        return sum(w.memory_gb for w in self.workers)

    @property
    def total_free_memory_gb(self) -> float | None:
        vals = [w.free_memory_gb for w in self.workers if w.free_memory_gb is not None]
        if not vals:
            return None
        return sum(vals)

    @property
    def gpu_worker_count(self) -> int:
        return sum(1 for w in self.workers if w.gpu)

    def can_run_efficiency_pooled(self) -> bool:
        return self.reachable and self.efficiency_worker_count > 0

    def to_dict(self) -> dict:
        return {
            "master_url": self.master_url,
            "reachable": self.reachable,
            "error": self.error,
            "alive_count": self.alive_count,
            "efficiency_worker_count": self.efficiency_worker_count,
            "gpu_worker_count": self.gpu_worker_count,
            "total_memory_gb": self.total_memory_gb,
            "total_free_memory_gb": self.total_free_memory_gb,
            "workers": [w.to_dict() for w in self.workers],
            "min_hourly_rate_usd": self.min_hourly_rate_usd,
            "limits_honestly": [
                "capacity is only what registered workers report right now",
                "dedicated provisioner is opt-in via TRON_DEDICATED_PROVISIONER",
                "efficiency_runtime must be true on a worker to run real-model jobs there",
            ],
        }

    @property
    def min_hourly_rate_usd(self) -> float | None:
        rates = [
            w.hourly_rate_usd
            for w in self.workers
            if isinstance(w.hourly_rate_usd, (int, float))
        ]
        return min(rates) if rates else None


def detect_efficiency_runtime() -> bool:
    """True when this process can run tron_efficiency workloads."""
    try:
        import accelerate  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def system_memory_gb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / 1_000_000_000.0
    except Exception:  # noqa: BLE001
        return None


def query_fleet_capacity(
    master_url: str | None = None,
    pool_status: PoolStatus | None = None,
    workers_payload: dict | None = None,
) -> FleetCapacity:
    """Build FleetCapacity from a live master (or injected worker dict for tests)."""
    import requests

    from tron_router.decision import DEFAULT_MASTER_URL

    url = (master_url or DEFAULT_MASTER_URL).rstrip("/")
    status = pool_status or query_pool_status(url)
    if not status.reachable and workers_payload is None:
        return FleetCapacity(
            master_url=url, reachable=False, error=status.error, workers=[]
        )

    workers = workers_payload
    if workers is None:
        try:
            resp = requests.get(f"{url}/workers", timeout=5.0)
            resp.raise_for_status()
            workers = resp.json()
        except Exception as exc:  # noqa: BLE001
            return FleetCapacity(
                master_url=url, reachable=False, error=str(exc), workers=[]
            )

    import time

    now = time.time()
    out: list[WorkerCapacity] = []
    for name, w in (workers or {}).items():
        if not isinstance(w, dict):
            continue
        if now - w.get("last_heartbeat", 0) > 60.0 and workers_payload is None:
            continue
        link = (status.measured_links or {}).get(name) if status.measured_links else None
        caps = w.get("capabilities") if isinstance(w.get("capabilities"), dict) else {}
        efficiency = bool(
            w.get("efficiency_runtime", caps.get("efficiency_runtime", False))
        )
        free = w.get("free_memory_gb")
        if free is None and link and isinstance(link, dict):
            free = link.get("memory_gb")  # topology sometimes mirrors free mem
        sku = w.get("sku") or caps.get("sku")
        rate = w.get("hourly_rate_usd", caps.get("hourly_rate_usd"))
        try:
            rate_f = float(rate) if rate is not None else None
        except (TypeError, ValueError):
            rate_f = None
        out.append(
            WorkerCapacity(
                name=name,
                memory_gb=float(w.get("memory_gb") or caps.get("memory_gb") or 0),
                free_memory_gb=float(free) if isinstance(free, (int, float)) else None,
                gpu=bool(w.get("gpu") or caps.get("gpu", False)),
                efficiency_runtime=efficiency,
                latency_ms=(link or {}).get("latency_ms") if isinstance(link, dict) else None,
                bandwidth_mbps=(link or {}).get("bandwidth_mbps") if isinstance(link, dict) else None,
                sku=str(sku) if sku else None,
                hourly_rate_usd=rate_f,
            )
        )
    return FleetCapacity(master_url=url, reachable=True, workers=out, error=None)
