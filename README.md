# TRON

**TRON is a self-hosted distributed computing platform.**

You run your own TRON server. Your developers point their code at it. Done.

- **SDK:** lightweight client (local wheel install today; future PyPI `tron-client`)
- **Server:** run it on Cloud Run, Fly.io, Docker, VPS, laptop, wherever
- **Zero cloud vendor lock-in** — you own the infrastructure

Make distributed computing feel local and seamless:

```python
import tron

# Ensure a TRON server is available automatically.
# This will connect to an existing server or start a local one.
tron.ensure_server()

@tron.remote
def expensive_task(x):
    return x * 2

result = expensive_task(10).get()
```

With the current SDK, developers can treat TRON as a client-first platform: the SDK auto-discovers or starts a local runtime as needed, so they rarely need to run `queue_server.py` manually.

For local development, the SDK also supports worker lifecycle helpers:

```python
tron.start_local_worker()
# ... run remote tasks ...
tron.stop_local_worker()
```

## Why TRON?

- **No infrastructure complexity:** One command deploys a full distributed system
- **Works anywhere:** Cloud, on-prem, laptop, VPS, Kubernetes
- **Own your data:** Your server, your rules, no vendor lock-in
- **Scales smoothly:** From local testing to multi-worker production
- **Feels like native Python:** `@tron.remote`, `.get()`, and you're done

## Install the SDK

```bash
# Local install while the package is not yet published to PyPI
python -m pip install dist/tron_client-0.1.0-py3-none-any.whl
```

If the wheel is published as a GitHub Release asset, developers can install directly from the release URL:

```bash
pip install https://github.com/<org>/<repo>/releases/download/v0.1.0/tron_client-0.1.0-py3-none-any.whl
```

Once `tron-client` is published, this can become:

```bash
python -m pip install tron-client
```

If you are developing the repo itself, use:

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Basic usage

```python
import tron

@tron.remote
def add_numbers(a, b):
    return a + b

result = add_numbers(1, 2).get()
print(result)
```

## Client-side flow

Users should only ever do this:

1. install the SDK locally for now: `python -m pip install dist/tron_client-0.1.0-py3-none-any.whl`
   - once published, this becomes: `python -m pip install tron-client`
2. write Python functions with `@tron.remote`
3. optionally configure the backend URL with `tron.config(...)`
4. call `.get()` on futures

Example:

```python
import tron

tron.config("http://localhost:9000")

@tron.remote
def add(a, b):
    return a + b

print(add(2, 3).get())
```

## Quick Start: Deploy your own TRON

Pick your platform and deploy in minutes:

### Cloud Run (free tier, easiest)
```bash
./deploy/cloud-run-quick.sh
```

### Fly.io (global, free tier)
```bash
./deploy/fly-quick.sh
```

### Render (simple web service)
```bash
./deploy/render-quick.sh
```

### Local Docker (dev/testing)
```bash
docker compose up
```

See [SELF_HOST.md](SELF_HOST.md) for detailed guides and customization.

This repo includes a `Dockerfile` and `docker-compose.yml` for an always-on backend.

```bash
docker compose up --build
```

If you want background mode:

```bash
docker compose up --build -d
```

The local API will be available at:

- `http://localhost:9000`

If Docker is unstable, use the direct Python run above instead.

## Developer workflow

Once your team deploys a TRON server, developers:

1. Install the SDK locally for now: `python -m pip install dist/tron_client-0.1.0-py3-none-any.whl`
   - once published, this becomes: `python -m pip install tron-client`
2. Get the server URL from your team
3. Add to code:
   ```python
   import tron
   tron.config("https://your-team-server")
   ```
4. Write normal Python with `@tron.remote`
5. Call `.get()` to fetch results

Developers do not need to run or change `queue_server.py`.
That file is the backend server implementation, and it is managed by your infrastructure or operations team.

See [USER_GUIDE.md](USER_GUIDE.md) for detailed examples.

## How TRON is structured

- `tron/` — client SDK, `@remote` decorator, `MagicFuture` for transparent `.get()`
- `queue_server.py` — FastAPI backend, job submission, status tracking
- `worker.py` — task execution, resource management
- `Dockerfile` — containerized runtime for any cloud
- `.github/workflows/deploy-cloud-run.yml` — auto-deploy to Cloud Run on push

## Architecture

```
Developer
    |
    | python -m pip install dist/tron_client-0.1.0-py3-none-any.whl
    |
    v
[ @tron.remote decorator ]
    |
    | tron.config("https://server")
    |
    v
[ TRON Backend (self-hosted) ]
    |
    +---> Queue (job storage)
    +---> Workers (parallel execution)
    +---> Results (stream back to SDK)
```

## For operators / infrastructure teams

Your job: deploy TRON once, keep it running.

Developers' job: use the SDK.

Deployment is simple:

1. Pick a platform (Cloud Run, Fly.io, Docker, etc.)
2. Run the deploy script or follow [SELF_HOST.md](SELF_HOST.md)
3. Share the server URL with developers
4. Done

No vendor lock-in. You own the infrastructure.

### Quick deploy

- **Cloud Run:** `bash deploy/cloud-run-quick.sh`
- **Fly.io:** `bash deploy/fly-quick.sh`
- **Local/VPS:** `docker compose up`

See [SELF_HOST.md](SELF_HOST.md) for detailed guides, scaling, and troubleshooting.

## Building the package

If you want to build the client package for distribution:

```bash
python -m pip install --upgrade build
python -m build --sdist --wheel
```

Test locally:

```bash
python -m pip install dist/tron_client-0.1.0-py3-none-any.whl
python -c "import tron; print(tron.__name__)"
```

## Documentation

- [SELF_HOST.md](SELF_HOST.md) - Complete self-hosting guide (Cloud Run, Fly.io, Render, Docker, etc.)
- [USER_GUIDE.md](USER_GUIDE.md) - For developers using TRON
- [QUICKSTART.md](QUICKSTART.md) - Quick usage tutorial
- [deploy/README.md](deploy/README.md) - Deployment reference
- [MAGIC_GUIDE.md](MAGIC_GUIDE.md) - Advanced TRON behavior and design
