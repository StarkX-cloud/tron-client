import argparse
import sys
import time

from .config import (
    start_local_server,
    stop_local_server,
    start_local_worker,
    stop_local_worker,
    start_local_environment,
    get_config,
)


def _run_until_cancelled(cleanup_callback=None):
    try:
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        if cleanup_callback:
            cleanup_callback()
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tron",
        description="TRON CLI for local runtime and server management.",
    )

    subparsers = parser.add_subparsers(dest="command")

    server = subparsers.add_parser("server", help="Manage the local TRON server")
    server_sub = server.add_subparsers(dest="action")
    server_start = server_sub.add_parser("start", help="Start the local TRON server")
    server_start.add_argument("--host", default="127.0.0.1", help="Host to bind the TRON server")
    server_start.add_argument("--port", type=int, default=9000, help="Port for the TRON server")
    server_start.add_argument("--reload", action="store_true", help="Run server with reload enabled")

    worker = subparsers.add_parser("worker", help="Manage the local TRON worker")
    worker_sub = worker.add_subparsers(dest="action")
    worker_start = worker_sub.add_parser("start", help="Start a local TRON worker")
    worker_start.add_argument("--queue-url", default=None, help="TRON queue server URL")

    env = subparsers.add_parser("env", help="Start a local TRON development environment")
    env.add_argument("--host", default="127.0.0.1", help="Host to bind the TRON server")
    env.add_argument("--port", type=int, default=9000, help="Port for the TRON server")
    env.add_argument("--reload", action="store_true", help="Run server with reload enabled")

    status = subparsers.add_parser("status", help="Show TRON SDK/server status")

    args = parser.parse_args(argv)

    if args.command == "server":
        if args.action == "start":
            url = start_local_server(
                port=args.port,
                host=args.host,
                reload_flag=args.reload,
            )
            print(f"TRON server started at {url}")
            return _run_until_cancelled(cleanup_callback=stop_local_server)
        parser.print_help()
        return 1

    if args.command == "worker":
        if args.action == "start":
            queue_url = start_local_worker(queue_url=args.queue_url)
            print(f"TRON worker connected to {queue_url}")
            return _run_until_cancelled(cleanup_callback=stop_local_worker)
        parser.print_help()
        return 1

    if args.command == "env":
        url = start_local_environment(
            port=args.port,
            host=args.host,
            reload_flag=args.reload,
        )
        print(f"Local TRON environment started at {url}")
        return _run_until_cancelled(cleanup_callback=lambda: (stop_local_worker(), stop_local_server()))

    if args.command == "status":
        config = get_config()
        print(f"TRON server URL: {config.url}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
