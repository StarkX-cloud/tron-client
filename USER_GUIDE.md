# TRON User Guide

This guide is for developers using a TRON server deployed by their team or organization.

## What is TRON?

TRON is a distributed computing platform. Your team runs a TRON server, and you use the SDK to run code on it.

TRON feels language-like because it gives Python functions distributed execution semantics, but it is used from Python rather than as a separate standalone language.

Instead of writing loops or managing threads, you write normal Python functions and decorate them with `@tron.remote`. They execute on your team's infrastructure, in parallel, with results streaming back.

## Install the SDK

```bash
# Local install while the package is not yet published to PyPI
python -m pip install dist/tron_client-0.1.0-py3-none-any.whl
```

If the wheel is published as a GitHub Release asset, developers can install directly from that release URL:

```bash
pip install https://github.com/<org>/<repo>/releases/download/v0.1.0/tron_client-0.1.0-py3-none-any.whl
```

Once `tron-client` is published, this can become:

```bash
python -m pip install tron-client
```

That's all you need.

## Developer client setup

This section is for application developers who want to use TRON from Python.

- Install the SDK.
- Get the TRON server URL from your team.
- Configure the SDK with `tron.config(...)` or `TRON_SERVER`.
- Write `@tron.remote` functions.
- Call `.get()` on the returned future.

You do not need to run or modify `queue_server.py` to use TRON as a developer.
That file is the backend server implementation, and it is managed by the infrastructure or operations team.

## Write a remote function

```python
import tron

@tron.remote
def add(a, b):
    return a + b
```

## Connect to your team's TRON server

Ask your team for the TRON server URL. It might look like:

- `https://tron.mycompany.com`
- `https://tron-prod.onrender.com`
- `http://10.0.1.5:9000` (internal IP)

Add it to your code:

```python
import tron

tron.config("https://your-team-tron-server")
```

You can also set it via environment variable (useful in CI/CD):

```bash
export TRON_SERVER=https://your-team-tron-server
```

Then in code:

```python
import os
import tron

tron.config(os.getenv("TRON_SERVER"))
```

### Named servers and local convenience

You can register named servers for easy selection, or connect to the local TRON server:

```python
import tron

tron.add_server("local", "http://127.0.0.1:9000")
tron.config(name="local")
```

To start a local TRON server and worker from the SDK:

```python
import tron

tron.start_local_environment()
```

This starts both the local queue server and a worker automatically so developers can work without explicit infrastructure commands.

To start a local TRON server from the SDK:

```python
import tron

tron.start_local_server()
```

To make TRON auto-provision a server when none is configured:

```python
import tron

tron.ensure_server()
```

If the server is already running locally, the SDK will connect to it automatically.

## Run the task

```python
result = add(2, 3).get()
print(result)
```

## Operator / server customization

TRON is self-hosted, which means your organization owns the backend server.
The server implementation lives in `queue_server.py` and is deployed by your infrastructure team.

- Operators and infrastructure engineers should follow `SELF_HOST.md` or use the deploy scripts.
- Developers should receive a shared TRON server URL and use the SDK.
- Local server helpers like `tron.start_local_server()` and `tron.ensure_server()` are convenience tools for local development only.

If you need to change the server deployment, scale it, or configure authentication, do that on the server side.
Developers do not need to edit or run the backend server code to write TRON-powered Python applications.

## Notes

- TRON is self-hosted by your team or organization
- You should never need to run `queue_server.py` yourself
- The server is deployed and managed by your team's infrastructure/devops team
- You just install the SDK and write `@tron.remote` functions
- The SDK automatically handles serialization, submission, and result fetching

## When you have issues

If the TRON server is down or unreachable:

1. Check the server URL with your team
2. Verify your network connection
3. Contact your team's infrastructure team

If you need to run tasks locally while the server is down, you can temporarily fall back to local execution:

```python
@tron.remote(local_only=True)
def my_task(x):
    return x * 2
```

But normally, always use the shared team server.

## Example

```python
import tron

tron.config("http://localhost:9000")

@tron.remote
def square(x):
    return x * x

print(square(5).get())
```
