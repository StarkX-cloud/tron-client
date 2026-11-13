"""Stable Python SDK for Autopilot plan/run/fleet + pool submit.

Sends ``X-TRON-AUTH`` for both Autopilot and submit tokens when set.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

from tron_pilot.provider import ProviderRegistry
from tron_pilot.provider_stats import ProviderOutcome, ProviderStatsStore
from tron_pilot.security import auth_headers, current_trust_model
from tron_pilot.tenancy import TenantContext
from tron_pilot.usage import UsageEvent, UsageLedger


class TronClient:
    """Enterprise integration client for the TRON control plane."""

    def __init__(
        self,
        *,
        autopilot_url: str | None = None,
        master_url: str | None = None,
        customer_id: str = "default",
        tenant_id: str | None = None,
        project_id: str | None = None,
        timeout: float = 60.0,
        usage_ledger: UsageLedger | None = None,
        provider_stats: ProviderStatsStore | None = None,
        provider_registry: ProviderRegistry | None = None,
    ):
        self.autopilot_url = (
            autopilot_url
            or os.environ.get("TRON_AUTOPILOT_URL")
            or "http://127.0.0.1:9100"
        ).rstrip("/")
        self.master_url = (
            master_url
            or os.environ.get("TRON_MASTER_URL")
            or "http://127.0.0.1:9000"
        ).rstrip("/")
        self.tenant = TenantContext(
            customer_id=customer_id, tenant_id=tenant_id, project_id=project_id
        )
        self.timeout = timeout
        self.usage = usage_ledger or UsageLedger()
        self.provider_stats = provider_stats or ProviderStatsStore()
        self.providers = provider_registry or ProviderRegistry()

    def trust_model(self) -> dict:
        return current_trust_model().__dict__

    def health(self) -> dict:
        ap = requests.get(f"{self.autopilot_url}/health", timeout=self.timeout)
        ap.raise_for_status()
        master = requests.get(f"{self.master_url}/health", timeout=self.timeout)
        master.raise_for_status()
        return {
            "autopilot": ap.json(),
            "master": master.json(),
            "trust": self.trust_model(),
        }

    def fleet(self) -> dict:
        r = requests.get(
            f"{self.autopilot_url}/fleet",
            params={"master_url": self.master_url},
            headers=auth_headers(autopilot=True),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def plan(self, **workload) -> dict:
        body = self._workload_body(workload)
        r = requests.post(
            f"{self.autopilot_url}/plan",
            json=body,
            headers=auth_headers(autopilot=True),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def run(self, **workload) -> dict:
        body = self._workload_body(workload)
        body.setdefault("efficiency", workload.get("efficiency", False))
        body.setdefault(
            "wait_timeout_seconds", workload.get("wait_timeout_seconds", 600.0)
        )
        # Prefer costtrace for customer accounting when not set.
        body.setdefault(
            "costtrace_task",
            workload.get("costtrace_task")
            or f"customer:{self.tenant.customer_id}",
        )
        r = requests.post(
            f"{self.autopilot_url}/run",
            json=body,
            headers=auth_headers(autopilot=True),
            timeout=max(self.timeout, float(body.get("wait_timeout_seconds") or 60) + 30),
        )
        r.raise_for_status()
        data = r.json()
        self._record_from_run(data)
        return data

    def submit_callable(
        self,
        func: Callable[..., Any],
        *args,
        wait: bool = True,
        wait_timeout_seconds: float = 120.0,
        poll: float = 0.2,
        required_memory_gb: float | None = 0.25,
        **kwargs,
    ) -> Any:
        """Submit an arbitrary callable via the pool (cloudpickle)."""
        from tron_router.decision import JobRequest, PlacementDecision
        from tron_router.handoff import submit_to_pool

        job = JobRequest(model_name="pilot/callable", tolerant=True)
        decision = PlacementDecision(
            target="pooled",
            confidence="medium",
            rationale=["tron_pilot.submit_callable"],
            required_memory_gb=required_memory_gb,
        )
        future = submit_to_pool(
            job,
            decision,
            func,
            func_args=args,
            func_kwargs=kwargs or {},
            master_url=self.master_url,
            required_memory_gb=required_memory_gb,
            min_memory_gb=required_memory_gb or 1.0,
            customer_id=self.tenant.customer_id,
            tenant_id=self.tenant.tenant_id,
            project_id=self.tenant.project_id,
        )
        if not wait:
            return future
        result = future.get(timeout=wait_timeout_seconds, poll=poll)
        status = future.status() if hasattr(future, "status") else {}
        self.usage.record(
            UsageEvent(
                customer_id=self.tenant.customer_id,
                event_type="workload_complete",
                plan_id=getattr(future, "job_id", None),
                model_name="pilot/callable",
                target="pooled",
                success=True,
                runtime_seconds=float((status or {}).get("runtime") or 0.0),
                cost_usd=None,
                tenant_id=self.tenant.tenant_id,
                project_id=self.tenant.project_id,
                extra={"via": "submit_callable"},
            )
        )
        return result

    def register_provider(self, **kwargs):
        kwargs.setdefault("master_url", self.master_url)
        return self.providers.register(**kwargs)

    def usage_report(self, **kwargs) -> dict:
        from tron_pilot.usage import usage_report as _report

        kwargs.setdefault("customer_id", self.tenant.customer_id)
        return _report(self.usage, **kwargs)

    def _workload_body(self, workload: dict) -> dict:
        body = {
            "model_name": workload.get("model_name", "pilot/workload"),
            "hourly_rate_usd": float(workload.get("hourly_rate_usd", 0.5)),
            "reliability": workload.get("reliability", "tolerant"),
            "requires_multi_node": bool(workload.get("requires_multi_node", False)),
            "master_url": workload.get("master_url") or self.master_url,
            "prompt": workload.get(
                "prompt", "The quick brown fox jumps over the lazy dog, and"
            ),
            "max_new_tokens": int(workload.get("max_new_tokens", 32)),
            "requires_efficiency_runtime": bool(
                workload.get("requires_efficiency_runtime", False)
            ),
            "strategy": workload.get("strategy", "autopilot"),
            "parallelizable": bool(workload.get("parallelizable", False)),
            "map_op": workload.get("map_op", "sum_squares"),
            "min_shard_size": int(workload.get("min_shard_size", 1)),
            "bytes_per_item": float(workload.get("bytes_per_item", 64.0)),
            "requires_efficiency_runtime": bool(
                workload.get(
                    "requires_efficiency_runtime",
                    workload.get("efficiency", False),
                )
            ),
            "preferred_location_strict": bool(
                workload.get("preferred_location_strict", False)
            ),
        }
        optional = (
            "budget_usd",
            "deadline_seconds",
            "pooled_hourly_rate_usd",
            "expected_runtime_seconds",
            "costtrace_task",
            "partition_items",
            "max_parallel_workers",
            "preferred_region",
        )
        for k in optional:
            if k in workload and workload.get(k) is not None:
                body[k] = workload[k]
        return body

    def _record_from_run(self, data: dict) -> None:
        plan = data.get("plan") or {}
        outcome = data.get("outcome") or {}
        spec = plan.get("spec") or {}
        success = bool(outcome.get("success"))
        self.usage.record(
            UsageEvent(
                customer_id=self.tenant.customer_id,
                event_type="workload_complete" if success else "workload_failed",
                plan_id=plan.get("plan_id") or outcome.get("plan_id"),
                model_name=spec.get("model_name"),
                target=outcome.get("target"),
                success=success,
                runtime_seconds=float(outcome.get("actual_runtime_seconds") or 0.0),
                cost_usd=outcome.get("actual_cost_usd"),
                strategy=outcome.get("strategy"),
                tenant_id=self.tenant.tenant_id,
                project_id=self.tenant.project_id,
                extra={
                    "feedback": outcome.get("feedback"),
                    "recovery": outcome.get("recovery"),
                    "recorded_at": time.time(),
                },
            )
        )
        # Provider stats: when pool/distributed, attach worker if known.
        worker = None
        result = outcome.get("result")
        if isinstance(result, dict):
            worker = result.get("worker") or result.get("worker_name")
            shards = result.get("shards") or []
            for s in shards:
                if isinstance(s, dict) and s.get("job_id"):
                    self.provider_stats.record(
                        ProviderOutcome(
                            worker_name=str(s.get("worker_name") or worker or "unknown"),
                            provider_id=None,
                            success=bool(s.get("success", success)),
                            runtime_seconds=float(outcome.get("actual_runtime_seconds") or 0.0),
                            cost_usd=outcome.get("actual_cost_usd"),
                            requeue_count=float(s.get("requeue_count") or 0),
                            target=outcome.get("target"),
                            customer_id=self.tenant.customer_id,
                            error=s.get("error"),
                        )
                    )
        if worker:
            self.provider_stats.record(
                ProviderOutcome(
                    worker_name=str(worker),
                    provider_id=None,
                    success=success,
                    runtime_seconds=float(outcome.get("actual_runtime_seconds") or 0.0),
                    cost_usd=outcome.get("actual_cost_usd"),
                    target=outcome.get("target"),
                    customer_id=self.tenant.customer_id,
                    error=outcome.get("error"),
                )
            )
