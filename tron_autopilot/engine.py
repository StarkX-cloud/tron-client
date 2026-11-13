"""Unified Economic Autopilot decision engine.

Connects existing TRON components through their public APIs:

1. Resources required → ``tron_efficiency.advisor.recommend``
2. Available machines → ``tron_router.decision.query_pool_status``
3. Memory / performance fit → advisor + ``estimate_required_memory_gb``
4. Min expected cost among feasible options → this module
5. Where to execute → chosen candidate (local / pooled / dedicated flag)
6. Recovery contract → ``tron_autopilot.safety.safety_summary`` (existing watchdog)
7. Feedback → ``execute`` + ``RuntimeHistory`` (predicted vs actual)

Does not rewrite matcher placement, memory policies, or the watchdog.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable

from tron_efficiency.advisor.recommend import Recommendation, recommend
from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware
from tron_router.decision import (
    JobRequest,
    PlacementDecision,
    decide_placement,
    estimate_required_memory_gb,
    query_pool_status,
)

from tron_autopilot.safety import reliability_allows_pool, safety_summary
from tron_autopilot.store import (
    DecisionLog,
    RuntimeHistory,
    cost_from_runtime,
    new_plan_id,
    now,
)
from tron_autopilot.types import CandidateOption, WorkloadPlan, WorkloadSpec

DEFAULT_LOG_PATH = os.environ.get(
    "TRON_AUTOPILOT_LOG", "tron_autopilot_decisions.jsonl"
)
DEFAULT_HISTORY_PATH = os.environ.get(
    "TRON_AUTOPILOT_HISTORY", "tron_autopilot_history.json"
)


class EconomicAutopilot:
    def __init__(
        self,
        decision_log: DecisionLog | None = None,
        history: RuntimeHistory | None = None,
        hardware: HardwareInfo | None = None,
        log_path: str | None = None,
        history_path: str | None = None,
        recommend_fn: Callable | None = None,
        decide_placement_fn: Callable | None = None,
        query_pool_status_fn: Callable | None = None,
    ):
        self.hardware = hardware
        self.decision_log = decision_log or DecisionLog(log_path or DEFAULT_LOG_PATH)
        self.history = history or RuntimeHistory(history_path or DEFAULT_HISTORY_PATH)
        # Injectable for tests — production leaves these as None and uses
        # the real tron_efficiency / tron_router entrypoints.
        self._recommend = recommend_fn or recommend
        self._decide_placement = decide_placement_fn or decide_placement
        self._query_pool_status = query_pool_status_fn or query_pool_status

    def plan(
        self,
        spec: WorkloadSpec,
        pool_status=None,
        strategy: str = "autopilot",
    ) -> WorkloadPlan:
        """Build a full plan: candidates, feasibility, chosen option."""
        hardware = self.hardware or detect_hardware()
        tolerant = reliability_allows_pool(spec.reliability)
        job = JobRequest(
            model_name=spec.model_name,
            tolerant=tolerant,
            requires_multi_node=spec.requires_multi_node,
            gpu_budget_bytes_override=spec.gpu_budget_bytes_override,
        )

        # Keep router placement as an explicit signal/explanation layer —
        # Autopilot may still pick a different economic option among the
        # feasible set (e.g. cheaper pool vs comfortable local).
        try:
            placement = self._decide_placement(
                job, hardware=hardware, pool_status=pool_status, master_url=spec.master_url
            )
        except Exception as exc:  # noqa: BLE001 - network/config failures must not crash planning
            placement = PlacementDecision(
                target="dedicated",
                confidence="low",
                rationale=[f"decide_placement failed: {type(exc).__name__}: {exc}"],
                warnings=["router placement unavailable — Autopilot continues with measured pool query"],
            )

        status = pool_status
        if status is None:
            status = self._query_pool_status(spec.master_url)

        try:
            local_rec = self._recommend(
                spec.model_name,
                hardware=hardware,
                gpu_budget_bytes_override=spec.gpu_budget_bytes_override,
            )
        except Exception as exc:  # noqa: BLE001
            local_rec = Recommendation(
                model_name=spec.model_name,
                estimated_param_count=0,
                param_count_is_estimate=True,
                device="cpu",
                dtype="fp32",
                policy="baseline",
                memory_budget_bytes=None,
                confidence="low",
                rationale=[f"advisor failed ({type(exc).__name__}: {exc}); safest CPU baseline fallback"],
                warnings=["advisor/recommend raised — using CPU baseline fallback, not a fabricated fit"],
            )

        candidates = self._build_candidates(spec, local_rec, status, placement)
        chosen = self._select(candidates, strategy=strategy, spec=spec)

        rationale = list(placement.rationale)
        warnings = list(placement.warnings) + list(local_rec.warnings)
        if chosen is None:
            rationale.append(
                "no feasible candidate satisfied budget/deadline/reliability/"
                "memory constraints — refusing rather than guessing"
            )
        else:
            rationale.append(
                f"strategy={strategy!r} chose {chosen.option_id} "
                f"(target={chosen.target}, predicted_cost_usd={chosen.predicted_cost_usd})"
            )
            rationale.extend(chosen.rationale)

        plan = WorkloadPlan(
            plan_id=new_plan_id(),
            spec=spec,
            chosen=chosen,
            candidates=candidates,
            placement_decision=placement.to_dict(),
            local_recommendation=local_rec.to_dict(),
            pool_status=status.to_dict() if hasattr(status, "to_dict") else status,
            safety=safety_summary(),
            rationale=rationale,
            warnings=warnings,
            timestamp=now(),
        )
        self.decision_log.append_plan(plan)
        return plan

    def _predicted_runtime(
        self,
        spec: WorkloadSpec,
        target: str,
        policy: str | None,
        device: str | None,
        dtype: str | None,
    ) -> tuple[float | None, str]:
        key = RuntimeHistory.key(spec.model_name, target, policy, device, dtype)
        hist = self.history.mean_runtime(key)
        if hist is not None:
            return hist, "measured_history_mean"
        if spec.expected_runtime_seconds is not None:
            return float(spec.expected_runtime_seconds), "user_expected_runtime"
        return None, "unknown_no_history_or_user_hint"

    def _build_candidates(self, spec, local_rec, status, placement) -> list[CandidateOption]:
        candidates: list[CandidateOption] = []
        dtype = local_rec.dtype
        try:
            required_gb = estimate_required_memory_gb(spec.model_name, dtype)
        except Exception:  # noqa: BLE001 - config fetch failures → unknown, not a guess
            required_gb = None
        if required_gb is None and local_rec.estimated_param_count:
            # Derive from the advisor's already-obtained param count when the
            # separate estimate_model_size call can't reach the hub.
            from tron_efficiency.advisor.model_info import BYTES_PER_PARAM

            bpp = BYTES_PER_PARAM.get(dtype or "fp16")
            if bpp is not None:
                required_gb = (local_rec.estimated_param_count * bpp) / 1_000_000_000.0


        # --- local with advisor policy ---
        pred_rt, pred_src = self._predicted_runtime(
            spec, "local", local_rec.policy, local_rec.device, local_rec.dtype
        )
        local_rate = spec.hourly_rate_usd
        local_cost = (
            cost_from_runtime(local_rate, pred_rt) if pred_rt is not None else None
        )
        local_reasons = []
        local_feasible = True
        if spec.requires_multi_node:
            local_feasible = False
            local_reasons.append("requires_multi_node — local single-machine path excluded")
        if local_rec.confidence == "low" and local_rec.policy == "offload":
            # Still allowed, but warn; low confidence is not an automatic reject.
            pass
        if spec.budget_usd is not None and local_cost is not None and local_cost > spec.budget_usd:
            local_feasible = False
            local_reasons.append(
                f"predicted cost ${local_cost:.6f} exceeds budget ${spec.budget_usd:.6f}"
            )
        if (
            spec.deadline_seconds is not None
            and pred_rt is not None
            and pred_rt > spec.deadline_seconds
        ):
            local_feasible = False
            local_reasons.append(
                f"predicted runtime {pred_rt:.3f}s exceeds deadline {spec.deadline_seconds:.3f}s"
            )
        # If cost/runtime unknown, still allow local when advisor has a plan —
        # refusing unknown would block every first run. Mark as warning.
        local_rationale = [
            f"advisor: {local_rec.device}/{local_rec.dtype}/{local_rec.policy} "
            f"(confidence={local_rec.confidence})",
            f"runtime prediction source: {pred_src}",
        ]
        if pred_rt is None:
            local_rationale.append(
                "predicted cost unknown until history or expected_runtime_seconds is set"
            )
        candidates.append(
            CandidateOption(
                option_id=f"local-{local_rec.policy}",
                target="local",
                policy=local_rec.policy,
                device=local_rec.device,
                dtype=local_rec.dtype,
                memory_budget_bytes=local_rec.memory_budget_bytes,
                required_memory_gb=required_gb,
                hourly_rate_usd=local_rate,
                predicted_runtime_seconds=pred_rt,
                predicted_cost_usd=local_cost,
                predicted_peak_memory_bytes=(
                    int(required_gb * 1_000_000_000) if required_gb is not None else None
                ),
                feasible=local_feasible,
                rejection_reasons=local_reasons,
                rationale=local_rationale,
                source_signals={
                    "advisor": local_rec.to_dict(),
                    "runtime_prediction_source": pred_src,
                },
            )
        )

        # Also expose a fixed baseline-local option when advisor chose offload,
        # so the "fixed-resource" strategy and comparisons have a real baseline.
        if local_rec.policy != "baseline":
            b_rt, b_src = self._predicted_runtime(
                spec, "local", "baseline", local_rec.device, local_rec.dtype
            )
            b_cost = cost_from_runtime(local_rate, b_rt) if b_rt is not None else None
            b_reasons = []
            b_feasible = not spec.requires_multi_node
            # Baseline may not actually fit — advisor said offload. Mark
            # infeasible for autopilot selection, but keep for fixed strategy.
            b_reasons.append(
                "advisor selected offload — baseline may OOM / exceed memory; "
                "kept as fixed-resource baseline option only"
            )
            candidates.append(
                CandidateOption(
                    option_id="local-baseline-forced",
                    target="local",
                    policy="baseline",
                    device=local_rec.device,
                    dtype=local_rec.dtype,
                    memory_budget_bytes=None,
                    required_memory_gb=required_gb,
                    hourly_rate_usd=local_rate,
                    predicted_runtime_seconds=b_rt,
                    predicted_cost_usd=b_cost,
                    predicted_peak_memory_bytes=(
                        int(required_gb * 1_000_000_000) if required_gb is not None else None
                    ),
                    feasible=False,  # autopilot must not pick this when advisor says offload
                    rejection_reasons=b_reasons,
                    rationale=[f"runtime prediction source: {b_src}"],
                    source_signals={"runtime_prediction_source": b_src, "forced_baseline": True},
                )
            )

        # --- pooled ---
        pool_rate = (
            spec.pooled_hourly_rate_usd
            if spec.pooled_hourly_rate_usd is not None
            else spec.hourly_rate_usd
        )
        p_rt, p_src = self._predicted_runtime(
            spec, "pooled", local_rec.policy, local_rec.device, local_rec.dtype
        )
        p_cost = cost_from_runtime(pool_rate, p_rt) if p_rt is not None else None
        p_reasons = []
        p_feasible = True
        p_rationale = [f"runtime prediction source: {p_src}"]

        if not reliability_allows_pool(spec.reliability):
            p_feasible = False
            p_reasons.append("reliability=strict — pool excluded (no malicious-worker defense)")
        if not status.reachable:
            p_feasible = False
            p_reasons.append(f"pool unreachable: {status.error}")
        elif status.alive_worker_count <= 0:
            p_feasible = False
            p_reasons.append("pool reachable but has 0 alive workers")
        else:
            p_rationale.append(
                f"pool has {status.alive_worker_count} alive worker(s), "
                f"{status.total_reported_memory_gb:.1f}GB reported"
            )
            if required_gb is not None and status.total_reported_memory_gb < required_gb:
                # Soft signal: total fleet memory < job need. Not a hard reject
                # if a single large worker might still fit — but we don't get
                # per-worker free memory from PoolStatus totals alone. Warn.
                p_rationale.append(
                    "fleet total reported memory is below estimated requirement — "
                    "matcher.memory_fits will still hard-filter per worker at assign time"
                )
            if status.measured_links:
                measured = [
                    v for v in status.measured_links.values()
                    if v and v.get("latency_ms") is not None
                ]
                if measured:
                    avg_lat = sum(v["latency_ms"] for v in measured) / len(measured)
                    p_rationale.append(
                        f"measured avg pool latency {avg_lat:.1f}ms "
                        f"({len(measured)} workers with samples)"
                    )

        if spec.budget_usd is not None and p_cost is not None and p_cost > spec.budget_usd:
            p_feasible = False
            p_reasons.append(
                f"predicted cost ${p_cost:.6f} exceeds budget ${spec.budget_usd:.6f}"
            )
        if (
            spec.deadline_seconds is not None
            and p_rt is not None
            and p_rt > spec.deadline_seconds
        ):
            p_feasible = False
            p_reasons.append(
                f"predicted runtime {p_rt:.3f}s exceeds deadline {spec.deadline_seconds:.3f}s"
            )

        candidates.append(
            CandidateOption(
                option_id="pooled-matcher",
                target="pooled",
                policy=local_rec.policy,
                device=local_rec.device,
                dtype=local_rec.dtype,
                memory_budget_bytes=local_rec.memory_budget_bytes,
                required_memory_gb=required_gb if required_gb is not None else placement.required_memory_gb,
                hourly_rate_usd=pool_rate,
                predicted_runtime_seconds=p_rt,
                predicted_cost_usd=p_cost,
                predicted_peak_memory_bytes=(
                    int(required_gb * 1_000_000_000) if required_gb is not None else None
                ),
                feasible=p_feasible,
                rejection_reasons=p_reasons,
                rationale=p_rationale,
                source_signals={
                    "pool_status": status.to_dict(),
                    "runtime_prediction_source": p_src,
                    "router_target": placement.target,
                },
            )
        )

        # --- dedicated flag (never auto-executed) ---
        candidates.append(
            CandidateOption(
                option_id="dedicated-flag",
                target="dedicated",
                policy=None,
                device=None,
                dtype=None,
                memory_budget_bytes=None,
                required_memory_gb=required_gb,
                hourly_rate_usd=pool_rate,
                predicted_runtime_seconds=None,
                predicted_cost_usd=None,
                predicted_peak_memory_bytes=None,
                feasible=False,
                rejection_reasons=[
                    "dedicated is a flag for real provisioned infrastructure — "
                    "Autopilot does not call a cloud API or GPU marketplace"
                ],
                rationale=["surfaced for honesty when local/pool cannot serve the job"],
                source_signals={},
            )
        )
        return candidates

    def _select(
        self,
        candidates: list[CandidateOption],
        strategy: str,
        spec: WorkloadSpec,
    ) -> CandidateOption | None:
        if strategy == "fixed":
            # Fixed-resource baseline: always force local baseline, even if
            # advisor prefers offload (may fail — that's the baseline's point).
            for c in candidates:
                if c.option_id in ("local-baseline-forced", "local-baseline") or (
                    c.target == "local" and c.policy == "baseline"
                ):
                    return replace(
                        c,
                        feasible=True,
                        rationale=list(c.rationale)
                        + ["fixed strategy: ignore advisor/pool; run local baseline"],
                    )
            return None

        if strategy == "cheapest":
            # Cheapest-available: pick lowest expected $/run among options
            # that are *reachable*, ignoring advisor memory-fit hard blocks.
            # That is intentional — the benchmark contrasts this naive cost
            # minimization against Autopilot's constraint-aware choice.
            available: list[CandidateOption] = []
            for c in candidates:
                if c.target == "dedicated":
                    continue
                reasons = " ".join(c.rejection_reasons)
                if c.target == "pooled" and (
                    "unreachable" in reasons
                    or "0 alive workers" in reasons
                    or "reliability=strict" in reasons
                ):
                    continue
                if c.target in ("local", "pooled"):
                    available.append(c)

            if not available:
                return None

            def cheap_key(c: CandidateOption):
                # Prefer known predicted cost; else fall back to hourly rate
                # alone (documented in rationale — not a fabricated runtime).
                if c.predicted_cost_usd is not None:
                    return (0, c.predicted_cost_usd, c.hourly_rate_usd)
                return (1, c.hourly_rate_usd, c.hourly_rate_usd)

            chosen = min(available, key=cheap_key)
            return replace(
                chosen,
                feasible=True,
                rationale=list(chosen.rationale)
                + [
                    "cheapest strategy: minimize expected $/run using hourly "
                    "rate (and predicted runtime when known); memory-fit "
                    "warnings are not treated as hard blocks"
                ],
            )

        # autopilot (default): only feasible candidates; minimize predicted cost;
        # if costs unknown, prefer router placement target among feasible.
        feasible = [c for c in candidates if c.feasible and c.target in ("local", "pooled")]
        if not feasible:
            return None

        known_cost = [c for c in feasible if c.predicted_cost_usd is not None]
        if known_cost:
            return min(known_cost, key=lambda c: c.predicted_cost_usd)  # type: ignore[arg-type, return-value]

        # No measured/user runtime → cannot compare $; follow router target
        # if that candidate is feasible, else first feasible.
        router_target = None
        # placement is not passed here; use candidate source_signals
        for c in feasible:
            rt = c.source_signals.get("router_target")
            if rt:
                router_target = rt
                break
        if router_target:
            for c in feasible:
                if c.target == router_target:
                    c.rationale = list(c.rationale) + [
                        "predicted costs unknown — deferred to tron_router.decide_placement target"
                    ]
                    return c
        feasible[0].rationale = list(feasible[0].rationale) + [
            "predicted costs unknown — selected first feasible candidate"
        ]
        return feasible[0]

    def record_feedback(self, plan: WorkloadPlan, outcome) -> None:
        """Persist outcome and update measured runtime history on success."""
        self.decision_log.append_outcome(outcome)
        if not outcome.success or plan.chosen is None:
            return
        key = RuntimeHistory.key(
            plan.spec.model_name,
            plan.chosen.target,
            plan.chosen.policy,
            plan.chosen.device,
            plan.chosen.dtype,
        )
        self.history.record(key, outcome.actual_runtime_seconds)
