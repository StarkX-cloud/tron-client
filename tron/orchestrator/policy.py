from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .outcomes import OutcomeLog


class TrainingPolicy:
    def __init__(self, outcome_log: Optional["OutcomeLog"] = None):
        self.outcome_log = outcome_log

    def decide_reuse(self, existing_score: float, retrain_score: float) -> bool:
        return existing_score >= retrain_score

    def decide_stop(self, marginal_utility: float, threshold: float) -> bool:
        return marginal_utility < threshold

    def decide_substrate(self, module_hint: str, substrates: Dict[str, float]) -> Optional[str]:
        if not substrates:
            return None
        return max(substrates, key=substrates.get)

    def decide_adapter(
        self,
        estimates: Dict[str, Dict[str, float]],
        preferred_substrate: Optional[str],
        available: List[str],
        requested: Optional[str] = None,
        module_id: Optional[str] = None,
    ) -> Optional[str]:
        if requested and requested in available:
            return requested

        best_name: Optional[str] = None
        best_score = float("-inf")
        for name in available:
            estimate = estimates.get(name)
            if not isinstance(estimate, dict):
                continue
            score = self.score_estimate(estimate, preferred_substrate, name, module_id=module_id)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def score_estimate(
        self,
        estimate: Dict[str, float],
        preferred_substrate: Optional[str],
        adapter_name: str,
        module_id: Optional[str] = None,
    ) -> float:
        capability = float(estimate.get("capability_gain", 0.0))
        cost = float(estimate.get("cost", 1.0))
        score = capability / max(cost, 1e-6)

        # Adjust score based on historical accuracy if available. Prefer
        # the (adapter, module) pair's own track record — an adapter can
        # be reliable for one kind of module and not another — and fall
        # back to the adapter-wide numbers when this exact pair has no
        # history yet.
        if self.outcome_log:
            accuracy = None
            success_rate = None
            if module_id:
                accuracy = self.outcome_log.pair_accuracy(adapter_name, module_id)
                success_rate = self.outcome_log.pair_success_rate(adapter_name, module_id)
            if accuracy is None:
                accuracy = self.outcome_log.adapter_accuracy(adapter_name)
            if success_rate is None:
                success_rate = self.outcome_log.adapter_success_rate(adapter_name)
            if accuracy is not None:
                score *= (0.7 + 0.3 * accuracy)  # Weight historical accuracy at 30%
            if success_rate is not None:
                score *= (0.7 + 0.3 * success_rate)  # Weight success rate at 30%

        if preferred_substrate == "gpu" and adapter_name == "ray":
            score *= 1.1
        if preferred_substrate == "cpu" and adapter_name == "sb3":
            score *= 1.05
        return score

