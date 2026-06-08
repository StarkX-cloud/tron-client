import argparse
import tron
from tron_sdk import Tron


def main():

    parser = argparse.ArgumentParser(prog="tron")
    parser.add_argument("--url", default="http://127.0.0.1:9000")

    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a prompt job")
    submit_parser.add_argument("prompt", help="Prompt text to run")
    submit_parser.add_argument("--stream", action="store_true", help="Stream output live")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for completion")

    infer_parser = subparsers.add_parser("infer", help="Run inference on a prompt")
    infer_parser.add_argument("prompt", help="Prompt text to infer")
    infer_parser.add_argument("--stream", action="store_true", help="Stream output live")

    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID to query")

    stream_parser = subparsers.add_parser("stream", help="Stream job events")
    stream_parser.add_argument("job_id", help="Job ID to stream")

    server_parser = subparsers.add_parser("server", help="Manage TRON server connections")
    server_subparsers = server_parser.add_subparsers(dest="server_command", required=True)

    server_add = server_subparsers.add_parser("add", help="Register a named server")
    server_add.add_argument("name", help="Server name")
    server_add.add_argument("url", help="Server URL")

    server_use = server_subparsers.add_parser("use", help="Switch to a named server")
    server_use.add_argument("name", help="Server name")

    server_list = server_subparsers.add_parser("list", help="List registered servers")

    server_ensure = server_subparsers.add_parser("ensure", help="Ensure a TRON server is available")
    server_ensure.add_argument("--port", type=int, default=9000, help="Local server port")
    server_ensure.add_argument("--host", default="127.0.0.1", help="Local server host")
    server_ensure.add_argument("--no-wait", action="store_true", help="Do not wait for local server startup")

    server_start = server_subparsers.add_parser("start", help="Start a local TRON server")
    server_start.add_argument("--port", type=int, default=9000, help="Local server port")
    server_start.add_argument("--host", default="127.0.0.1", help="Local server host")
    server_start.add_argument("--no-wait", action="store_true", help="Do not wait for local server startup")

    server_stop = server_subparsers.add_parser("stop", help="Stop the local TRON server")

    args = parser.parse_args()

    tron_client = Tron(args.url)

    if args.command == "submit":
        print("⚡ TRON SUBMITTING:", args.prompt)
        job = tron_client.submit(args.prompt)
        print("JOB ID:", job.job_id)
        if args.stream:
            tron_client.stream(job.job_id)
        elif args.wait:
            print(job.wait())

    elif args.command == "infer":
        print("⚡ TRON INFER:", args.prompt)
        if args.stream:
            job = tron_client.run(args.prompt)
            tron_client.stream(job.job_id)
        else:
            result = tron_client.infer(args.prompt)
            print("\n✅ RESULT:\n", result)

    elif args.command == "status":
        status = tron_client.status(args.job_id)
        print(status)

        if status.get("status") == "queued":
            workers = tron_client.workers()
            queue = tron_client.queue()
            print("\nWorker pool:", len(workers) if isinstance(workers, dict) else workers)
            print("Queue size:", len(queue.get("queue", [])) if isinstance(queue, dict) else queue)

    elif args.command == "stream":
        tron_client.stream(args.job_id)

    elif args.command == "server":
        if args.server_command == "add":
            tron.add_server(args.name, args.url)
            print(f"Registered server '{args.name}' -> {args.url}")
        elif args.server_command == "use":
            url = tron.use_server(args.name)
            print(f"Using server '{args.name}' -> {url}")
        elif args.server_command == "list":
            servers = tron.list_servers()
            for name, url in servers.items():
                print(f"{name}: {url}")
        elif args.server_command == "ensure":
            url = tron.ensure_server(
                port=args.port,
                host=args.host,
                wait=not args.no_wait,
            )
            print(f"Connected to TRON server: {url}")
        elif args.server_command == "start":
            url = tron.start_local_server(
                port=args.port,
                host=args.host,
                wait=not args.no_wait,
            )
            print(f"Started local TRON server at {url}")
        elif args.server_command == "stop":
            stopped = tron.stop_local_server()
            print("Stopped local TRON server" if stopped else "No local server to stop")


if __name__ == "__main__":
    main()
