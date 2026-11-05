"""CLI: python -m tron_costtrace.cli report [path]

Reads a JSONL log written by tron_costtrace.trace() and prints the
cost-per-successful-task markdown report to stdout.
"""

from __future__ import annotations

import argparse
import sys

from tron_costtrace.report import generate_task_report
from tron_costtrace.store import JsonlStore
from tron_costtrace.tracer import DEFAULT_LOG_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tron_costtrace")
    sub = parser.add_subparsers(dest="command", required=True)

    report_parser = sub.add_parser("report", help="Print the cost-per-successful-task report")
    report_parser.add_argument("path", nargs="?", default=DEFAULT_LOG_PATH)

    args = parser.parse_args(argv)

    if args.command == "report":
        store = JsonlStore(args.path)
        records = store.read_all()
        print(generate_task_report(records))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
