# TRON Operators Guide

This guide is for infrastructure engineers and operators who deploy TRON for developers.

## What operators deliver

Your team should provide:

- a running TRON server endpoint, e.g. `https://tron-bank.internal`
- a packaged SDK wheel for clients, ideally as a release asset
- documentation for client install and server URL configuration
- operational controls for security, TLS, auth, and logging

## When to use GitHub Releases

For an early internal distribution, GitHub Releases is the simplest clean path.

1. Build the wheel:

```bash
python -m pip install --upgrade build
python -m build --sdist --wheel
```

2. Verify the artifacts exist:

```bash
ls dist/
# dist/tron_client-0.1.0-py3-none-any.whl
# dist/tron_client-0.1.0.tar.gz
```

3. Create a GitHub Release for the version tag, and attach the wheel.

4. Share the install link with developers:

```bash
pip install https://github.com/<org>/<repo>/releases/download/v0.1.0/tron_client-0.1.0-py3-none-any.whl
```

## Why this is better than repo checkout

If developers must clone the repo just to access `dist/`, they are tied to the source tree.
A release asset is a standalone distributed package, so devs can install the wheel directly without pulling the full codebase.

## Recommended operator checklist

- Run TRON server inside the customer environment.
- Use HTTPS for the TRON API endpoint.
- Add authentication: API keys, OAuth/OIDC, or mTLS.
- Use an internal artifact registry or release asset for SDK distribution.
- Host Docker images in the bank's internal registry.
- Restrict outbound network access from the server/processes.
- Enable logging and auditing for compliance.
- Document the exact install command for developers.

## Client install options

### From a GitHub Release

```bash
pip install https://github.com/<org>/<repo>/releases/download/v0.1.0/tron_client-0.1.0-py3-none-any.whl
```

### From a repo checkout

If developers do clone the repo, they can install from the wheel in `dist/`:

```bash
pip install dist/tron_client-0.1.0-py3-none-any.whl
```

### Internal package index

If your organization uses an internal artifact repo, publish the wheel there and use:

```bash
pip install --index-url https://your-internal-index/simple tron-client
```

## Server deployment notes

For on-prem or bank deployment, follow `SELF_HOST.md` and consider these additional steps:

- expose TRON only inside the bank network
- use a reverse proxy or load balancer with TLS termination
- implement request authentication and authorization
- keep `queue_server.py` and model routing logic under operator control

## Keep developer and operator roles separate

- Developers get the SDK and a server URL.
- Operators manage `queue_server.py`, deployment, security, and infrastructure.
- Avoid giving developers the backend server source as part of their normal workflow.
