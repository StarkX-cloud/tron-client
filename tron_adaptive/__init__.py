"""TRON Adaptive Compute Intelligence.

Closes the predicted → actual → updated-decision loop on top of
``tron_efficiency`` and ``tron_autopilot``. Does **not** invent unified
VRAM across independent GPUs.
"""

from tron_adaptive.loop import AdaptiveIntelligence, close_feedback_loop
from tron_adaptive.optimizer import OptimizationProfile, AdaptiveOptimizer
from tron_adaptive.performance_store import PerformanceSample, PerformanceStore
from tron_adaptive.metrics import aggregate_run_metrics, sample_gpu_utilization

__all__ = [
    "AdaptiveIntelligence",
    "AdaptiveOptimizer",
    "OptimizationProfile",
    "PerformanceSample",
    "PerformanceStore",
    "aggregate_run_metrics",
    "close_feedback_loop",
    "sample_gpu_utilization",
]
