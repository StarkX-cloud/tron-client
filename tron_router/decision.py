"""Decide where a job should run: this machine, the pooled fleet, or real
dedicated infrastructure.

This is the first thing in this repo that connects two previously
separate systems:

- ``tron_efficiency.advisor`` -- can *this* machine run the job, and how
  (device/dtype/policy), using real measured memory thresholds.
- ``tron``'s spine/matcher (``queue_server.py``, ``tron/spine/topology.py``)
  -- is there a live pool of other machines registered and reachable right
  now, using real worker heartbeats, not assumed capacity.

What this module honestly is: a recommendation, in the same spirit as
``tron_efficiency.advisor.recommend`` -- it picks a target and explains
why, with a confidence level calibrated to how much was actually measured.
It does **not** provision anything. "dedicated" means "this job needs real
guaranteed infrastructure, which this project doesn't provision" -- it is
a flag for a human (or another system) to act on, not an API call this
code makes to AWS/GCP/a GPU marketplace on your behalf. "pooled" means
"the spine/matcher's existing distributed training/job path is the real
mechanism this would route into" -- this module decides *that* a job
should go there, it does not reimplement the matcher's own placement
logic (``tron/spine/matcher.py`` already does real Hungarian-algorithm
placement across registered workers once a job reaches that path).

Also honestly: every "worker" this project has ever measured against was
a local or same-machine process, not a real external fleet (see
ROADMAP.md's distributed-training section, "no real second physical
host"). A "pooled" recommendation here reflects that a live pool was
found and looks usable by the signals available (worker count, reported
memory, heartbeat freshness, and now real measured latency/bandwidth
from ``queue_server.py``'s ``/topology`` endpoint when available) -- it
is not evidence that pooling real external, less-trusted machines works
at any meaningful scale. That gap is inherited here, not closed by this
module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware

DEFAULT_MASTER_URL = "http://localhost:9000"
# A worker that hasn't heartbeat within this window is treated as dead,
# not just slow -- queue_server.py itself doesn't publish a fixed
# staleness threshold via its API, so this is a judgment call stated
# explicitly rather than hidden inside a magic number. Deliberately
# generous (heartbeat intervals in this project's worker bootstraps are
# on the order of seconds, not minutes) so a merely-slow worker isn't
# wrongly written off.
DEFAULT_WORKER_STALENESS_SECONDS = 60.0


@dataclass
class PoolStatus:
    """What was actually learned by querying a live master -- or why
    nothing was learned."""

    reachable: bool
    master_url: str
    alive_worker_count: int = 0
    total_reported_memory_gb: float = 0.0
    checked_worker_count: int = 0
    error: str | None = None
    # Real latency_ms/bandwidth_mbps per alive worker, from queue_server.py's
    # /topology endpoint -- the same measured numbers tron/spine/matcher.py
    # already uses internally for placement (tron/spine/topology.py). None
    # per-field means "never measured yet, or aged out" -- never fabricated.
    # The whole dict is None (not {}) when /topology itself wasn't reachable
    # or didn't exist, distinct from "reachable but nothing measured yet".
    measured_links: dict | None = None

    def to_dict(self) -> dict:
        return {
            "reachable": self.reachable,
            "master_url": self.master_url,
            "alive_worker_count": self.alive_worker_count,
            "total_reported_memory_gb": self.total_reported_memory_gb,
            "checked_worker_count": self.checked_worker_count,
            "error": self.error,
            "measured_links": self.measured_links,
        }


def query_pool_status(
    master_url: str | None = None,
    timeout_seconds: float = 2.0,
    staleness_seconds: float = DEFAULT_WORKER_STALENESS_SECONDS,
) -> PoolStatus:
    """Real HTTP query against a live ``queue_server.py`` instance's
    ``/workers`` endpoint. Never raises -- an unreachable master (nothing
    running, wrong URL, network error) is a real, expected case, reported
    as ``reachable=False`` with the actual error, not a crash.
    """
    import time

    import requests

    url = master_url or DEFAULT_MASTER_URL
    try:
        response = requests.get(f"{url.rstrip('/')}/workers", timeout=timeout_seconds)
        response.raise_for_status()
        workers = response.json()
    except Exception as exc:  # noqa: BLE001 - any network/parse failure means "not reachable"
        return PoolStatus(reachable=False, master_url=url, error=f"{type(exc).__name__}: {exc}")

    now = time.time()
    alive_names = [
        name for name, w in workers.items()
        if isinstance(w, dict) and (now - w.get("last_heartbeat", 0)) <= staleness_seconds
    ]
    alive = [workers[name] for name in alive_names]
    total_memory_gb = sum(w.get("memory_gb", 0) or 0 for w in alive)

    # Best-effort: /topology is a real endpoint on this same master, but
    # an older queue_server.py (or any other HTTP server answering the
    # same /workers shape in a test) might not have it. Its absence
    # doesn't make the pool unreachable -- /workers already answered --
    # so a failure here leaves measured_links as None, not a crash.
    measured_links = None
    try:
        topo_response = requests.get(f"{url}/topology", timeout=timeout_seconds)
        topo_response.raise_for_status()
        all_links = topo_response.json()
        if isinstance(all_links, dict):
            measured_links = {name: all_links.get(name) for name in alive_names if name in all_links}
    except Exception:  # noqa: BLE001 - genuinely optional; /workers succeeding is what matters
        measured_links = None

    return PoolStatus(
        reachable=True,
        master_url=url,
        alive_worker_count=len(alive),
        total_reported_memory_gb=total_memory_gb,
        checked_worker_count=len(workers),
        measured_links=measured_links,
    )


@dataclass
class JobRequest:
    """What's actually known about the job before deciding where to run
    it. ``model_name`` feeds the same real size-estimation
    ``tron_efficiency.advisor`` uses (config.json only, no download) --
    this is not a separate, invented sizing mechanism.
    """

    model_name: str
    tolerant: bool = True  # can this job survive a pooled worker dropping out / retrying?
    requires_multi_node: bool = False  # must it be split across more than one machine regardless of local fit?
    gpu_budget_bytes_override: int | None = None


@dataclass
class PlacementDecision:
    target: str  # "local", "pooled", or "dedicated"
    confidence: str  # "high", "medium", "low"
    rationale: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    local_recommendation: dict | None = None  # the tron_efficiency Recommendation, if consulted
    pool_status: dict | None = None  # the PoolStatus, if consulted

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "local_recommendation": self.local_recommendation,
            "pool_status": self.pool_status,
        }


def decide_placement(
    job: JobRequest,
    hardware: HardwareInfo | None = None,
    pool_status: PoolStatus | None = None,
    master_url: str | None = None,
) -> PlacementDecision:
    """Pick local / pooled / dedicated for ``job``, consulting real
    signals at each step rather than assuming. ``hardware`` and
    ``pool_status`` are injectable (tests use this; production callers
    normally omit them and let this function detect/query for real)."""
    from tron_efficiency.advisor.recommend import recommend

    hardware = hardware or detect_hardware()
    rationale: list = []
    warnings: list = []

    if not job.requires_multi_node:
        local_rec = recommend(
            job.model_name, hardware=hardware,
            gpu_budget_bytes_override=job.gpu_budget_bytes_override,
        )
        fits_without_offload = local_rec.policy == "baseline"
        if fits_without_offload and local_rec.confidence in ("high", "medium"):
            rationale.append(
                f"fits on this machine without offload: {local_rec.device}/{local_rec.dtype}, "
                f"advisor confidence {local_rec.confidence}"
            )
            return PlacementDecision(
                target="local",
                confidence=local_rec.confidence,
                rationale=rationale,
                warnings=list(local_rec.warnings),
                local_recommendation=local_rec.to_dict(),
            )
        rationale.append(
            f"does not comfortably fit locally without offload (advisor: "
            f"policy={local_rec.policy}, confidence={local_rec.confidence}) -- checking the pool"
        )
    else:
        local_rec = None
        rationale.append("job explicitly requires multiple nodes -- checking the pool directly")

    status = pool_status or query_pool_status(master_url)
    status_dict = status.to_dict()

    if not status.reachable:
        warnings.append(
            f"no live pool reachable at {status.master_url} ({status.error}) -- "
            "a 'pooled' recommendation would be unfounded; this job needs real "
            "dedicated infrastructure or a pool to actually be started"
        )
        return PlacementDecision(
            target="dedicated",
            confidence="low",
            rationale=rationale + ["no reachable pool -- dedicated infrastructure is the honest answer"],
            warnings=warnings,
            local_recommendation=local_rec.to_dict() if local_rec else None,
            pool_status=status_dict,
        )

    if status.alive_worker_count == 0:
        warnings.append(
            f"pool at {status.master_url} is reachable but has 0 alive workers -- "
            "nothing registered or everything is stale"
        )
        return PlacementDecision(
            target="dedicated",
            confidence="low",
            rationale=rationale + ["pool reachable but empty -- dedicated infrastructure is the honest answer"],
            warnings=warnings,
            local_recommendation=local_rec.to_dict() if local_rec else None,
            pool_status=status_dict,
        )

    rationale.append(
        f"pool at {status.master_url} has {status.alive_worker_count} alive worker(s) "
        f"reporting {status.total_reported_memory_gb:.1f}GB total"
    )

    if status.measured_links is None:
        warnings.append(
            "could not read real measured latency/bandwidth from the master's "
            "/topology endpoint -- this pool-health check only has worker count "
            "and self-reported memory to go on, not real network cost"
        )
    else:
        measured = [v for v in status.measured_links.values() if v and v.get("latency_ms") is not None]
        if not measured:
            warnings.append(
                "no alive worker has a real measured latency/bandwidth sample yet "
                "(none have completed a heartbeat probe) -- the matcher will place "
                "the first job to each of them blind to real network cost"
            )
        else:
            avg_latency = sum(v["latency_ms"] for v in measured) / len(measured)
            bw_values = [v["bandwidth_mbps"] for v in measured if v.get("bandwidth_mbps") is not None]
            if bw_values:
                avg_bw = sum(bw_values) / len(bw_values)
                rationale.append(
                    f"{len(measured)}/{status.alive_worker_count} alive worker(s) have real "
                    f"measured network data: avg latency {avg_latency:.1f}ms, avg bandwidth {avg_bw:.1f}Mbps"
                )
            else:
                rationale.append(
                    f"{len(measured)}/{status.alive_worker_count} alive worker(s) have a real "
                    f"measured latency sample (avg {avg_latency:.1f}ms) but no bandwidth sample yet"
                )

    if not job.tolerant:
        warnings.append(
            "job is marked intolerant of failure, but this project has no real "
            "reliability guarantee for pooled workers (see ROADMAP.md -- no defense "
            "against an unreliable or malicious participant) -- recommending "
            "dedicated infrastructure instead of the pool for a job that can't "
            "tolerate a worker dropping out"
        )
        return PlacementDecision(
            target="dedicated",
            confidence="medium",
            rationale=rationale,
            warnings=warnings,
            local_recommendation=local_rec.to_dict() if local_rec else None,
            pool_status=status_dict,
        )

    confidence = "medium" if status.alive_worker_count >= 2 else "low"
    if status.alive_worker_count < 2:
        warnings.append(
            "only 1 alive worker -- the matcher's placement logic "
            "(tron/spine/matcher.py) has little to actually optimize across"
        )

    return PlacementDecision(
        target="pooled",
        confidence=confidence,
        rationale=rationale,
        warnings=warnings,
        local_recommendation=local_rec.to_dict() if local_rec else None,
        pool_status=status_dict,
    )
