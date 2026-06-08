import requests
from .config import get_config


def submit(payload):
    """Submit a job to TRON queue."""
    config = get_config()
    r = requests.post(
        f"{config.url}/submit",
        json={"function": payload},
        timeout=10
    )
    return r.json()["job_id"]


def status(job_id):
    """Get job status."""
    config = get_config()
    r = requests.get(
        f"{config.url}/status/{job_id}",
        timeout=5
    )
    return r.json()
