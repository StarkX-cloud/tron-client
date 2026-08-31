"""Outcome tracking and feedback for TRON-II decisions."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime


@dataclass
class TrainingOutcome:
    """Record of a training run outcome vs expectations."""
    artifact_id: str
    adapter_name: str
    module_id: str
    expected_capability_gain: float
    actual_capability_gain: float
    expected_cost: float
    actual_cost: float
    success: bool
    timestamp: str
    # Which placement nodes (spine worker names) took part in this run, if
    # any — e.g. the shard-N workers in a distributed local-SGD / LoRA
    # session. Empty for outcomes that aren't tied to a placed node (the
    # in-process hybrid-adapter path). Optional and defaulted so every
    # existing caller and every persisted outcomes.json predating this
    # field keeps loading unchanged. Feeds OutcomeLog.node_quality, which
    # matcher.py can use to let real training results — not just measured
    # latency/bandwidth — bias future placement.
    node_ids: List[str] = field(default_factory=list)

    def accuracy(self) -> float:
        """How close was the estimate to reality? 0.0 to 1.0."""
        if self.expected_capability_gain <= 0.0:
            return 0.0
        error = abs(self.expected_capability_gain - self.actual_capability_gain)
        return max(0.0, 1.0 - error / max(self.expected_capability_gain, 1.0))

    def to_dict(self) -> Dict:
        return {
            "artifact_id": self.artifact_id,
            "adapter_name": self.adapter_name,
            "module_id": self.module_id,
            "expected_capability_gain": self.expected_capability_gain,
            "actual_capability_gain": self.actual_capability_gain,
            "expected_cost": self.expected_cost,
            "actual_cost": self.actual_cost,
            "success": self.success,
            "timestamp": self.timestamp,
            "accuracy": self.accuracy(),
            "node_ids": list(self.node_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingOutcome":
        return cls(
            artifact_id=data["artifact_id"],
            adapter_name=data["adapter_name"],
            module_id=data["module_id"],
            expected_capability_gain=float(data["expected_capability_gain"]),
            actual_capability_gain=float(data["actual_capability_gain"]),
            expected_cost=float(data["expected_cost"]),
            actual_cost=float(data["actual_cost"]),
            success=bool(data["success"]),
            timestamp=data["timestamp"],
            node_ids=list(data.get("node_ids", [])),
        )


class OutcomeLog:
    """Track training outcomes for feedback and learning."""
    def __init__(self, storage_path: Optional[str] = None):
        self.outcomes: List[TrainingOutcome] = []
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            try:
                self.load(self.storage_path)
            except Exception:
                self.outcomes = []

    def record(self, outcome: TrainingOutcome) -> None:
        """Record a training outcome."""
        self.outcomes.append(outcome)

    def adapter_accuracy(self, adapter_name: str) -> Optional[float]:
        """Average accuracy for an adapter across all outcomes."""
        adapter_outcomes = [o for o in self.outcomes if o.adapter_name == adapter_name]
        if not adapter_outcomes:
            return None
        return sum(o.accuracy() for o in adapter_outcomes) / len(adapter_outcomes)

    def adapter_success_rate(self, adapter_name: str) -> Optional[float]:
        """Success rate for an adapter."""
        adapter_outcomes = [o for o in self.outcomes if o.adapter_name == adapter_name]
        if not adapter_outcomes:
            return None
        successes = sum(1 for o in adapter_outcomes if o.success)
        return successes / len(adapter_outcomes)

    def module_outcomes(self, module_id: str) -> List[TrainingOutcome]:
        """Get all outcomes for a specific module."""
        return [o for o in self.outcomes if o.module_id == module_id]

    def _pair_outcomes(self, adapter_name: str, module_id: str) -> List[TrainingOutcome]:
        return [
            o for o in self.outcomes
            if o.adapter_name == adapter_name and o.module_id == module_id
        ]

    def pair_accuracy(self, adapter_name: str, module_id: str) -> Optional[float]:
        """Average accuracy for this exact (adapter, module) pair — a
        sharper signal than adapter_accuracy when an adapter behaves very
        differently across task types (e.g. transformers doing well on NLP
        modules but poorly on vision ones)."""
        pair_outcomes = self._pair_outcomes(adapter_name, module_id)
        if not pair_outcomes:
            return None
        return sum(o.accuracy() for o in pair_outcomes) / len(pair_outcomes)

    def pair_success_rate(self, adapter_name: str, module_id: str) -> Optional[float]:
        """Success rate for this exact (adapter, module) pair."""
        pair_outcomes = self._pair_outcomes(adapter_name, module_id)
        if not pair_outcomes:
            return None
        successes = sum(1 for o in pair_outcomes if o.success)
        return successes / len(pair_outcomes)

    def estimate_capability_gain(
        self, adapter_name: str, module_id: Optional[str] = None
    ) -> Optional[float]:
        """What capability gain has this adapter actually delivered, on
        average, in the past? Prefers the (adapter, module) pair's own
        history — a module-specific estimate is sharper than an
        adapter-wide one — and falls back to every outcome for this
        adapter, regardless of module, when the pair has no history yet.
        Returns None with no history at all, so callers can fall back to
        their own prior (a hardcoded guess) without this ever forcing a
        0.0 onto a never-seen adapter."""
        if module_id:
            pair_outcomes = self._pair_outcomes(adapter_name, module_id)
            if pair_outcomes:
                return sum(o.actual_capability_gain for o in pair_outcomes) / len(pair_outcomes)
        adapter_outcomes = [o for o in self.outcomes if o.adapter_name == adapter_name]
        if not adapter_outcomes:
            return None
        return sum(o.actual_capability_gain for o in adapter_outcomes) / len(adapter_outcomes)

    def estimate_cost(
        self, adapter_name: str, module_id: Optional[str] = None
    ) -> Optional[float]:
        """What has this adapter actually cost, on average, in the past?
        Same (adapter, module) -> adapter-wide -> None fallback chain as
        estimate_capability_gain."""
        if module_id:
            pair_outcomes = self._pair_outcomes(adapter_name, module_id)
            if pair_outcomes:
                return sum(o.actual_cost for o in pair_outcomes) / len(pair_outcomes)
        adapter_outcomes = [o for o in self.outcomes if o.adapter_name == adapter_name]
        if not adapter_outcomes:
            return None
        return sum(o.actual_cost for o in adapter_outcomes) / len(adapter_outcomes)

    def node_quality(self, node_id: str) -> Optional[float]:
        """Success rate across every recorded outcome this node took part
        in (0.0-1.0), or None if the node has never appeared in a
        completed run's node_ids. This is the placement-feedback signal:
        matcher.py can nudge a job toward a node that has a real track
        record of successful training outcomes, on top of (not instead
        of) the latency/bandwidth signals topology.py measures directly.
        A node is only ever penalized or rewarded by runs it actually
        participated in — a node with no outcomes yet is neither trusted
        nor distrusted."""
        touched = [o for o in self.outcomes if node_id in o.node_ids]
        if not touched:
            return None
        successes = sum(1 for o in touched if o.success)
        return successes / len(touched)

    def save(self, path: Optional[Path] = None) -> None:
        """Persist outcomes to JSON."""
        p = Path(path) if path else self.storage_path
        if p is None:
            raise ValueError("No storage path configured for OutcomeLog")
        data = [o.to_dict() for o in self.outcomes]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    def load(self, path: Optional[Path] = None) -> None:
        """Load outcomes from JSON."""
        p = Path(path) if path else self.storage_path
        if p is None or not p.exists():
            return
        raw = json.loads(p.read_text())
        self.outcomes = [TrainingOutcome.from_dict(d) for d in raw]
