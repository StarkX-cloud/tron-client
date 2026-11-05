"""Toy agent demo for tron_costtrace — produces the first real report.

Two modes:

  --mock   Synthetic responses, no network calls, no API key needed. Runs
           today. Clearly NOT real traffic — it exists to exercise the real
           tron_costtrace code end to end and show what the report looks
           like, not to claim a real-world cost finding.

  (default) Real mode. Reads OPENAI_API_KEY or ANTHROPIC_API_KEY from your
           own environment (never pass one on the command line or paste one
           into chat) and makes real, real-money API calls. Requires
           `pip install openai` or `pip install anthropic`.

Simulates one small agent doing two tasks across several calls:
  - "classify_ticket": most calls use a cheap model; some (deliberately,
    to mirror the Gap-B hypothesis from the strategic audit) escalate to a
    more expensive model "just in case" even though the task is simple;
    a few fail and are retried once before succeeding.
  - "summarize_thread": a single bigger call, occasionally fails outright
    with no retry, to show failed-cost-as-overhead in the report.

Run (from the repo root, as a module — matches examples/distributed_training's
convention, needed so `tron_costtrace` resolves without installing the package):
    python -m examples.costtrace_demo.run_toy_agent --mock
    python -m examples.costtrace_demo.run_toy_agent --mock --path my_log.jsonl --seed 3
    python -m tron_costtrace.cli report tron_costtrace_demo_log.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from types import SimpleNamespace

from tron_costtrace import trace
from tron_costtrace.report import generate_task_report
from tron_costtrace.store import JsonlStore

DEFAULT_PATH = "tron_costtrace_demo_log.jsonl"


def _mock_response(model: str, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


def run_mock(store: JsonlStore, n_tickets: int, seed: int) -> None:
    rng = random.Random(seed)

    for _ in range(n_tickets):
        with trace("classify_ticket", store=store) as t:
            # Gap-B hypothesis: ~25% of calls escalate to a pricier model
            # for a task simple enough for the cheap one.
            if rng.random() < 0.25:
                resp = _mock_response("gpt-4o", input_tokens=120, output_tokens=20)
            else:
                resp = _mock_response("gpt-4o-mini", input_tokens=120, output_tokens=20)

            # ~15% of calls fail once and succeed on retry.
            if rng.random() < 0.15:
                t.retry()

            t.record(resp)
            t.mark_success(True)

    n_summaries = max(1, n_tickets // 4)
    for _ in range(n_summaries):
        with trace("summarize_thread", store=store) as t:
            resp = _mock_response("gpt-4o", input_tokens=800, output_tokens=150)
            t.record(resp)
            # ~20% of summaries fail outright, no retry attempted.
            t.mark_success(rng.random() >= 0.20)


def run_real_openai(store: JsonlStore, n_tickets: int) -> None:
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set in this environment. Set it yourself "
              "(never paste it into chat) and re-run.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI()
    prompts = [
        "Classify this support ticket as billing, bug, or feature-request: "
        "'My invoice shows double the price I agreed to.'",
    ]

    for i in range(n_tickets):
        with trace("classify_ticket", store=store) as t:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
                max_tokens=20,
            )
            t.record(resp)
            t.mark_success(True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use synthetic responses, no API key needed")
    parser.add_argument("--provider", choices=["openai"], default="openai", help="Real-mode provider")
    parser.add_argument("--path", default=DEFAULT_PATH, help="JSONL log path")
    parser.add_argument("--n", type=int, default=20, help="Number of classify_ticket calls to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Mock mode RNG seed, for reproducibility")
    args = parser.parse_args()

    store = JsonlStore(args.path)

    if args.mock:
        run_mock(store, n_tickets=args.n, seed=args.seed)
    elif args.provider == "openai":
        run_real_openai(store, n_tickets=args.n)

    records = store.read_all()
    print(generate_task_report(records))
    print(f"(log written to {args.path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
