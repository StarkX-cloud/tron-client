"""CLI for the Economic Autopilot.

    python3 -m tron_autopilot.cli plan --model EleutherAI/pythia-70m --hourly-rate 0.5
    python3 -m tron_autopilot.cli run --model EleutherAI/pythia-70m --hourly-rate 0.5 --sleep 0.1
    python3 -m tron_autopilot.cli bench --trials 3 --live-pool
"""

from __future__ import annotations

import argparse
import json
import time

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.report import format_outcome, format_plan
from tron_autopilot.types import WorkloadSpec


def cmd_plan(args) -> int:
    spec = WorkloadSpec(
        model_name=args.model,
        hourly_rate_usd=args.hourly_rate,
        pooled_hourly_rate_usd=args.pooled_hourly_rate,
        budget_usd=args.budget,
        deadline_seconds=args.deadline,
        reliability="strict" if args.strict else "tolerant",
        expected_runtime_seconds=args.expected_runtime,
        requires_multi_node=args.multi_node,
        master_url=args.master_url,
    )
    ap = EconomicAutopilot()
    plan = ap.plan(spec, strategy=args.strategy)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, default=str))
    else:
        print(format_plan(plan))
    return 0 if plan.chosen is not None else 2


def cmd_run(args) -> int:
    spec = WorkloadSpec(
        model_name=args.model,
        hourly_rate_usd=args.hourly_rate,
        pooled_hourly_rate_usd=args.pooled_hourly_rate,
        budget_usd=args.budget,
        deadline_seconds=args.deadline,
        reliability="strict" if args.strict else "tolerant",
        expected_runtime_seconds=args.expected_runtime or (
            None if args.real_model else args.sleep
        ),
        master_url=args.master_url,
        max_new_tokens=args.max_new_tokens,
        requires_efficiency_runtime=bool(args.real_model),
    )
    ap = EconomicAutopilot()
    plan = ap.plan(spec, strategy=args.strategy)
    print(format_plan(plan))

    if args.real_model:
        outcome = execute_plan(plan, efficiency=True, strategy=args.strategy)
    else:

        def _work():
            time.sleep(args.sleep)
            return {"slept": args.sleep, "model": args.model}

        outcome = execute_plan(plan, func=_work, strategy=args.strategy)
    ap.record_feedback(plan, outcome)
    print(format_outcome(outcome))
    if args.json:
        print(json.dumps({"plan": plan.to_dict(), "outcome": outcome.to_dict()}, indent=2, default=str))
    return 0 if outcome.success else 1


def cmd_bench(args) -> int:
    from benchmarks.economic_autopilot import run_benchmark

    out = args.out
    if args.real_model and out == "artifacts/economic_autopilot_benchmark.json":
        out = "artifacts/economic_autopilot_real_model_benchmark.json"
    run_benchmark(
        trials=args.trials,
        hourly_rate=args.hourly_rate,
        pooled_hourly_rate=args.pooled_hourly_rate or 0.20,
        live_pool=args.live_pool,
        master_url=args.master_url,
        out_path=out,
        duration_seconds=args.sleep,
        real_model=args.real_model,
        model_name=args.model if args.real_model else "EleutherAI/pythia-70m",
        max_new_tokens=args.max_new_tokens,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tron_autopilot")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--hourly-rate", type=float, required=True)
        sp.add_argument("--pooled-hourly-rate", type=float, default=None)
        sp.add_argument("--budget", type=float, default=None)
        sp.add_argument("--deadline", type=float, default=None)
        sp.add_argument("--expected-runtime", type=float, default=None)
        sp.add_argument("--strict", action="store_true")
        sp.add_argument("--multi-node", action="store_true")
        sp.add_argument("--master-url", default=None)
        sp.add_argument(
            "--strategy",
            choices=["autopilot", "fixed", "cheapest"],
            default="autopilot",
        )
        sp.add_argument("--json", action="store_true")

    plan_p = sub.add_parser("plan", help="decide without executing")
    add_common(plan_p)
    plan_p.set_defaults(func=cmd_plan)

    run_p = sub.add_parser("run", help="plan + execute a callable stand-in workload")
    add_common(run_p)
    run_p.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="synthetic workload duration seconds (not a model inference)",
    )
    run_p.add_argument(
        "--real-model",
        action="store_true",
        help="execute via tron_efficiency Baseline/Offload (downloads weights)",
    )
    run_p.add_argument("--max-new-tokens", type=int, default=16)
    run_p.set_defaults(func=cmd_run)

    bench_p = sub.add_parser("bench", help="fixed vs cheapest vs autopilot benchmark")
    bench_p.add_argument("--trials", type=int, default=3)
    bench_p.add_argument("--hourly-rate", type=float, default=0.50)
    bench_p.add_argument("--pooled-hourly-rate", type=float, default=0.20)
    bench_p.add_argument("--live-pool", action="store_true")
    bench_p.add_argument("--master-url", default=None)
    bench_p.add_argument("--sleep", type=float, default=0.05)
    bench_p.add_argument("--real-model", action="store_true")
    bench_p.add_argument("--model", default="EleutherAI/pythia-70m")
    bench_p.add_argument("--max-new-tokens", type=int, default=16)
    bench_p.add_argument("--out", default="artifacts/economic_autopilot_benchmark.json")
    bench_p.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
