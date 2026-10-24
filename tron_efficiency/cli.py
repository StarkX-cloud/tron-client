"""CLI for the TRON Compute Efficiency Engine.

    python -m tron_efficiency.cli hardware
    python -m tron_efficiency.cli run --policy baseline
    python -m tron_efficiency.cli run --policy offload --memory-budget-mb 128
    python -m tron_efficiency.cli compare --memory-budget-mb 128
    python -m tron_efficiency.cli report <run_id> [<run_id> ...]
    python -m tron_efficiency.cli serve
"""

from __future__ import annotations

import argparse
import json
import sys

from tron_efficiency.hardware.detect import detect_hardware
from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy
from tron_efficiency.results.report import generate_report
from tron_efficiency.results.writer import list_runs, save_run, write_csv
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import run_workload


def _print_outcome(result) -> None:
    o = result.outcome
    if not o.success:
        print(f"[{o.policy_name}] FAILED: {o.error}")
        return
    print(
        f"[{o.policy_name}] load={o.load_time_seconds:.3f}s "
        f"ttft={o.time_to_first_token_seconds:.3f}s "
        f"e2e={o.end_to_end_latency_seconds:.3f}s "
        f"tok/s={o.tokens_per_second:.2f} "
        f"peak_cpu_mem={o.peak_cpu_memory_bytes} "
        f"run_id={result.run_id}"
    )
    for w in result.warnings:
        print(f"  warning: {w}")


def _build_config(args) -> WorkloadConfig:
    return WorkloadConfig(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )


def _build_policy(args):
    if args.policy == "baseline":
        return BaselinePolicy()
    budget_bytes = args.memory_budget_mb * 1024 * 1024
    return OffloadPolicy(memory_budget_bytes=budget_bytes)


def cmd_hardware(args) -> int:
    hw = detect_hardware()
    print(json.dumps(hw.to_dict(), indent=2))
    return 0


def cmd_run(args) -> int:
    from tron_efficiency.results.schema import BenchmarkResult

    hardware = detect_hardware()
    config = _build_config(args)
    policy = _build_policy(args)
    outcome = run_workload(config, policy)
    result = BenchmarkResult.build(hardware, config, policy.describe(), outcome)
    save_run(result)
    _print_outcome(result)
    return 0 if outcome.success else 1


def cmd_compare(args) -> int:
    from tron_efficiency.benchmarks.engine import run_suite

    config = _build_config(args)
    budget_bytes = args.memory_budget_mb * 1024 * 1024
    policies = [BaselinePolicy(), OffloadPolicy(memory_budget_bytes=budget_bytes)]
    results = run_suite(config, policies)
    for r in results:
        save_run(r)
        _print_outcome(r)
    print()
    print(generate_report(results, hourly_rate_usd=args.hourly_rate))
    return 0


def cmd_report(args) -> int:
    all_runs = {r.run_id: r for r in list_runs()}
    if args.run_ids:
        selected = [all_runs[rid] for rid in args.run_ids if rid in all_runs]
        missing = [rid for rid in args.run_ids if rid not in all_runs]
        for rid in missing:
            print(f"warning: run_id {rid} not found, skipping", file=sys.stderr)
    else:
        selected = list(all_runs.values())
    print(generate_report(selected, hourly_rate_usd=args.hourly_rate))
    if args.csv:
        write_csv(selected, args.csv)
        print(f"wrote {args.csv}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    from tron_efficiency.api.server import serve

    serve(port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tron_efficiency")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="report detected hardware").set_defaults(
        func=cmd_hardware
    )

    def add_workload_args(p):
        p.add_argument("--model", default="EleutherAI/pythia-70m")
        p.add_argument("--prompt", default=WorkloadConfig().prompt)
        p.add_argument("--max-new-tokens", type=int, default=32)
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    run_p = sub.add_parser("run", help="run one workload under one policy")
    add_workload_args(run_p)
    run_p.add_argument("--policy", choices=["baseline", "offload"], default="baseline")
    run_p.add_argument("--memory-budget-mb", type=int, default=100)
    run_p.set_defaults(func=cmd_run)

    compare_p = sub.add_parser(
        "compare", help="run baseline and offload back-to-back and report"
    )
    add_workload_args(compare_p)
    compare_p.add_argument("--memory-budget-mb", type=int, default=100)
    compare_p.add_argument("--hourly-rate", type=float, default=None)
    compare_p.set_defaults(func=cmd_compare)

    report_p = sub.add_parser("report", help="render a report from saved runs")
    report_p.add_argument("run_ids", nargs="*")
    report_p.add_argument("--hourly-rate", type=float, default=None)
    report_p.add_argument("--csv", default=None)
    report_p.set_defaults(func=cmd_report)

    serve_p = sub.add_parser("serve", help="run the API + dashboard server")
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
