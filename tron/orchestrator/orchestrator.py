"""Training orchestrator that coordinates hybrid adapters and TRON-II decision-making."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .mission import MissionPlanner, MissionPlan
from .registry import ArtifactRegistry, ArtifactMetadata
from .substrate import SubstrateManager, Substrate
from .utility import UtilityEngine
from .manager import TrainingManager
from .module import CapabilityModule
from .policy import TrainingPolicy
from .depin import DePINClient
from .hybrid import HybridIntegrator
from .outcomes import OutcomeLog, TrainingOutcome
from pathlib import Path
import time


@dataclass
class TrainingConfig:
    """Configuration for training run."""

    task_name: str
    total_timesteps: int = 10_000
    adapter_type: str = "auto"  # auto, sb3, transformers, ray, etc.
    objective: str = "improve capability"
    budget: float = 1.0
    modules: List[CapabilityModule] = field(default_factory=list)
    enable_reuse: bool = True
    adapter_estimates: Optional[Dict[str, Dict[str, float]]] = None
    depin_master_url: Optional[str] = None
    depin_priority: int = 1
    depin_requires_gpu: bool = False


class TrainingOrchestrator:
    """High-level orchestrator for coordinating training via hybrid adapters.

    This class shows how TRON-II's decision-making (via TrainingManager)
    can work with pluggable adapters for different ML frameworks.
    """

    def __init__(self, policy: Optional[TrainingPolicy] = None, artifact_registry_path: Optional[str] = None, outcome_log_path: Optional[str] = None):
        # Initialize TRON-II components
        planner = MissionPlanner()
        registry = ArtifactRegistry(storage_path=artifact_registry_path)
        substrate_manager = SubstrateManager()
        substrate_manager.register(Substrate(name="cpu", type="hardware", cost_per_cycle=0.1, capabilities={"compute": 1.0}))
        substrate_manager.register(Substrate(name="gpu", type="hardware", cost_per_cycle=0.5, capabilities={"compute": 2.0}))
        utility_engine = UtilityEngine()

        # Initialize outcome log before policy so policy can use it
        self.outcome_log = OutcomeLog(storage_path=outcome_log_path)

        self.manager = TrainingManager(
            planner=planner,
            registry=registry,
            substrate_manager=substrate_manager,
            utility_engine=utility_engine,
            policy=policy or TrainingPolicy(outcome_log=self.outcome_log),
        )
        self.integrator = HybridIntegrator.default()
        self.adapters = {adapter.name: adapter for adapter in self.integrator.adapters}
        self.training_log: List[Dict[str, Any]] = []
        self.last_plan: Optional[MissionPlan] = None
        self.registry = registry
        self.substrate_manager = substrate_manager

    def _plan_mission(self, config: TrainingConfig) -> Optional[MissionPlan]:
        if not config.modules:
            return None

        self.manager.create_mission(
            name=config.task_name,
            objective=config.objective,
            budget=config.budget,
            modules=config.modules,
        )
        plan = self.manager.plan_mission(config.task_name)
        self.last_plan = plan
        return plan

    def _estimate_adapter_performance(
        self,
        plan: Optional[MissionPlan],
        module_id: Optional[str] = None,
        overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if overrides is not None:
            return overrides

        estimates: Dict[str, Dict[str, float]] = {}
        substrate_name = plan.phases[0].substrate if plan and plan.phases else None

        for adapter_name in self.integrator.available_names():
            estimates[adapter_name] = {
                "capability_gain": self._estimate_capability_gain(adapter_name, plan, module_id),
                "cost": self._estimate_adapter_cost(adapter_name, substrate_name, module_id),
            }

        return estimates

    def _estimate_capability_gain(
        self, adapter_name: str, plan: Optional[MissionPlan], module_id: Optional[str] = None
    ) -> float:
        # Closing the outcomes feedback loop: what this adapter has
        # actually delivered in past runs is a better estimate than a
        # fixed guess, once there's enough history to ask. Falls through
        # to the hardcoded priors below for an adapter/module pair that
        # has never run yet.
        if self.outcome_log:
            observed = self.outcome_log.estimate_capability_gain(adapter_name, module_id)
            if observed is not None:
                return observed

        if adapter_name == "sb3":
            return 1.0
        if adapter_name == "ray":
            return 0.9
        if adapter_name == "transformers":
            return 0.8
        if adapter_name == "scikit-learn":
            return 0.4
        return 0.5

    def _estimate_adapter_cost(
        self, adapter_name: str, substrate_name: Optional[str], module_id: Optional[str] = None
    ) -> float:
        if self.outcome_log:
            observed = self.outcome_log.estimate_cost(adapter_name, module_id)
            if observed is not None:
                return observed

        if adapter_name == "ray":
            return 0.8 if substrate_name == "gpu" else 1.2
        if adapter_name == "sb3":
            return 1.0 if substrate_name == "cpu" else 1.5
        if adapter_name == "transformers":
            return 2.0
        if adapter_name == "scikit-learn":
            return 0.3
        return 1.0

    def _build_depin_payload(self, config: TrainingConfig, model: Any, env: Any) -> Dict[str, Any]:
        return {
            "task_name": config.task_name,
            "adapter_type": config.adapter_type,
            "total_timesteps": config.total_timesteps,
            "budget": config.budget,
            "modules": [
                {
                    "module_id": module.module_id,
                    "preferred_substrate": module.preferred_substrate,
                    "complexity": module.complexity,
                    "task_type": module.task_type,
                }
                for module in config.modules
            ],
            "model_description": str(type(model).__name__) if model is not None else None,
            "env_description": str(type(env).__name__) if env is not None else None,
        }

    def run_training(self, config: TrainingConfig, model: Any, env: Any) -> bool:
        """Execute a training session using the specified adapter.

        Args:
            config: Training configuration.
            model: Model to train (framework-specific).
            env: Environment (framework-specific).

        Returns:
            True if training completed successfully.
        """
        # Record training intent in TRON-II (mission planning)
        try:
            print(f"[TRON-II] Planning training mission: {config.task_name}")
        except Exception:
            pass  # Graceful fallback if planning fails

        try:
            if config.depin_master_url:
                client = DePINClient(config.depin_master_url)
                payload = self._build_depin_payload(config, model, env)
                response = client.submit_job(
                    task_type="tron_ii_training",
                    payload=payload,
                    runtime_seconds=None,
                    priority=config.depin_priority,
                    requires_gpu=config.depin_requires_gpu,
                    metadata={"task_name": config.task_name, "adapter_type": config.adapter_type},
                )
                self.training_log.append({
                    "task": config.task_name,
                    "adapter": "depin",
                    "status": "submitted",
                    "job_id": response.get("job_id"),
                })
                print(f"[Orchestrator] Submitted job {response.get('job_id')} to DePIN master")
                return True

            if config.adapter_type not in ("auto", "", None) and config.adapter_type not in self.integrator.available_names():
                raise ValueError(f"Unknown adapter: {config.adapter_type}")

            dataset = {"model": model, "env": env}
            mission_plan = self._plan_mission(config)
            if mission_plan and mission_plan.phases:
                dataset["selected_substrate"] = mission_plan.phases[0].substrate
                print(f"[TRON-II] Mission phase substrate: {mission_plan.phases[0].substrate}")

            if config.enable_reuse and config.modules:
                reuse_artifacts = []
                for module in config.modules:
                    artifact = self.manager.best_reuse_artifact(module)
                    if artifact and self.manager.should_reuse(module, retrain_cost=1.0):
                        reuse_artifacts.append(artifact.artifact_id)

                if reuse_artifacts:
                    log_entry = {
                        "task": config.task_name,
                        "adapter": "reuse",
                        "timesteps": 0,
                        "status": "reused",
                        "artifact_ids": reuse_artifacts,
                    }
                    self.training_log.append(log_entry)
                    print(f"[Orchestrator] Reusing existing artifact(s): {reuse_artifacts}")
                    return True

            # Adapter selection happens once per run, not once per module,
            # so the primary (first) module is what estimate-refinement and
            # pair-specific outcome scoring key off of.
            primary_module_id = config.modules[0].module_id if config.modules else None

            dataset["adapter_estimates"] = self._estimate_adapter_performance(
                mission_plan, module_id=primary_module_id, overrides=config.adapter_estimates
            )

            adapter_name = (
                None if config.adapter_type in ("auto", "", None) else config.adapter_type
            )
            selected_adapter_name = self.manager.policy.decide_adapter(
                estimates=dataset["adapter_estimates"],
                preferred_substrate=dataset.get("selected_substrate"),
                available=self.integrator.available_names(),
                requested=adapter_name,
                module_id=primary_module_id,
            )
            if selected_adapter_name is None:
                raise RuntimeError("No adapter was selected by policy")

            adapter = self.integrator.select(selected_adapter_name)
            print(f"[Orchestrator] Using adapter: {adapter.name}")

            result = adapter.train(
                dataset=dataset,
                config={
                    "total_timesteps": config.total_timesteps,
                    "save_path": f"{config.task_name}_model",
                    "capability_gain": 1.0,
                    "cost": 1.0,
                },
            )

            self._register_training_artifacts(config, adapter.name, result, dataset.get("selected_substrate"))
            self._record_training_outcome(
                config=config,
                adapter_name=adapter.name,
                result=result,
                estimates=dataset.get("adapter_estimates", {}),
            )

            log_entry = {
                "task": config.task_name,
                "adapter": adapter.name,
                "timesteps": config.total_timesteps,
                "status": "success",
                "result": result,
            }
            self.training_log.append(log_entry)

            print(f"[Orchestrator] Training completed for {config.task_name}")
            return True

        except Exception as e:
            log_entry = {
                "task": config.task_name,
                "adapter": config.adapter_type,
                "status": "failed",
                "error": str(e),
            }
            self.training_log.append(log_entry)
            print(f"[Orchestrator] Training failed: {e}")
            return False

    def _register_training_artifacts(
        self,
        config: TrainingConfig,
        adapter_name: str,
        result: Dict[str, Any],
        substrate_name: Optional[str],
    ) -> None:
        if not config.modules:
            return

        metrics = {
            key: float(value)
            for key, value in result.items()
            if isinstance(value, (int, float))
        }
        # Prefer reported capability_gain but fall back to module baseline when adapter reports no gain
        reported = metrics.get("capability_gain")
        if reported is None or reported <= 0.0:
            metrics["capability_gain"] = config.modules[0].metrics.get("capability_gain", 0.0)

        training_config = {
            "total_timesteps": config.total_timesteps,
            "adapter_type": config.adapter_type,
            "budget": config.budget,
        }

        for module in config.modules:
            artifact_id = f"{config.task_name}-{module.module_id}-{adapter_name}"
            artifact = ArtifactMetadata(
                artifact_id=artifact_id,
                module=module.module_id,
                version="v1",
                adapter=adapter_name,
                task_name=config.task_name,
                metrics=metrics,
                size_bytes=int(result.get("size_bytes", 1)),
                substrates=[substrate_name] if substrate_name else [],
                training_config=training_config,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.manager.registry.register(artifact)

        # best-effort persist registry to disk
        try:
            save_path = self.manager.registry.storage_path or Path("tron_artifacts.json")
            self.manager.registry.save(save_path)
        except Exception:
            pass

    def _record_training_outcome(
        self,
        config: TrainingConfig,
        adapter_name: str,
        result: Dict[str, Any],
        estimates: Dict[str, Dict[str, float]],
    ) -> None:
        """Record actual training outcome for feedback learning."""
        if not config.modules:
            return

        adapter_estimate = estimates.get(adapter_name, {})
        expected_gain = adapter_estimate.get("capability_gain", 0.5)
        expected_cost = adapter_estimate.get("cost", 1.0)

        actual_gain = float(result.get("capability_gain", 0.0))
        if actual_gain <= 0.0:
            actual_gain = config.modules[0].metrics.get("capability_gain", 0.0)

        actual_cost = float(result.get("cost", 1.0))

        for module in config.modules:
            artifact_id = f"{config.task_name}-{module.module_id}-{adapter_name}"
            outcome = TrainingOutcome(
                artifact_id=artifact_id,
                adapter_name=adapter_name,
                module_id=module.module_id,
                expected_capability_gain=expected_gain,
                actual_capability_gain=actual_gain,
                expected_cost=expected_cost,
                actual_cost=actual_cost,
                success=actual_gain > 0.0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.outcome_log.record(outcome)

        # best-effort persist outcome log
        try:
            outcome_path = self.outcome_log.storage_path or Path("tron_outcomes.json")
            self.outcome_log.save(outcome_path)
        except Exception:
            pass


    def get_training_summary(self) -> Dict[str, Any]:
        """Return a summary of all training runs."""
        return {
            "total_runs": len(self.training_log),
            "successful": sum(1 for log in self.training_log if log["status"] == "success"),
            "failed": sum(1 for log in self.training_log if log["status"] == "failed"),
            "log": self.training_log,
        }

    def get_last_plan(self) -> Optional[MissionPlan]:
        """Return the most recently planned mission."""
        return self.last_plan
