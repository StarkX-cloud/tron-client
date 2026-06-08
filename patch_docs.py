from pathlib import Path
replacements = [
    (
        Path("README.md"),
        "- **SDK:** lightweight client (`pip install tron-client`)\n- **Server:** run it on Cloud Run, Fly.io, Docker, VPS, laptop, wherever\n- **Zero cloud vendor lock-in** — you own the infrastructure\n",
        "- **SDK:** lightweight client (local wheel install today; future PyPI `tron-client`)\n- **Server:** run it on Cloud Run, Fly.io, Docker, VPS, laptop, wherever\n- **Zero cloud vendor lock-in** — you own the infrastructure\n",
    ),
    (
        Path("README.md"),
        "## Install the SDK\n\n```bash\npip install tron-client\n```\n\nIf you are developing the repo itself, use:\n",
        "## Install the SDK\n\n```bash\n# Temporary local install while the package is not yet published\npip install dist/tron_client-0.1.0-py3-none-any.whl\n```\n\nIf you are developing the repo itself, use:\n",
    ),
    (
        Path("README.md"),
        "## Developer workflow\n\nOnce your team deploys a TRON server, developers:\n\n1. Install: `pip install tron-client`\n2. Get the server URL from your team\n3. Add to code:\n",
        "## Developer workflow\n\nOnce your team deploys a TRON server, developers:\n\n1. Install the SDK locally for now: `pip install dist/tron_client-0.1.0-py3-none-any.whl`\n   - Once published, this becomes: `pip install tron-client`\n2. Get the server URL from your team\n3. Add to code:\n",
    ),
    (
        Path("README.md"),
        "Developer\n    |\n    | pip install tron-client\n    |\n    v\n[ @tron.remote decorator ]\n",
        "Developer\n    |\n    | pip install dist/tron_client-0.1.0-py3-none-any.whl\n    |\n    v\n[ @tron.remote decorator ]\n",
    ),
    (
        Path("USER_GUIDE.md"),
        "## Install the SDK\n\n```bash\npip install tron-client\n```\n\nThat's all you need. The server is managed by your team.\n",
        "## Install the SDK\n\n```bash\n# Local install while the package is not published to PyPI\npip install dist/tron_client-0.1.0-py3-none-any.whl\n```\n\nOnce `tron-client` is published, this will become:\n\n```bash\npip install tron-client\n```\n\nThat's all you need. The server is managed by your team.\n",
    ),
]
for path, old, new in replacements:
    text = path.read_text()
    if old not in text:
        raise ValueError(f"Expected text not found in {path}")
    path.write_text(text.replace(old, new, 1))
print('patched docs')
