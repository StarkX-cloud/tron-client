"""TRON Cost Trace — request-level cost/outcome recording for live LLM calls.

Separate, standalone package (zero coupling to ``tron/``, ``tron_efficiency/``,
``tron_router/``). Those packages measure TRON's own local inference/training
workloads; this one measures a caller's real production LLM API traffic, so
that "cost per successful task" — not cost per token or per call — can be
reported from real usage before any routing or model-switching is built.

See README.md's "Cost Trace" section for usage. Reuses the reporting
discipline from ``tron_efficiency.results.report`` (every number traceable to
a real recorded event, missing/unknown data stated explicitly, failures shown
not hidden) rather than that module's code, since the workload here — live
API calls, not local model loading — is different.
"""

from tron_costtrace.tracer import trace
from tron_costtrace.schema import CallRecord

__all__ = ["trace", "CallRecord"]
