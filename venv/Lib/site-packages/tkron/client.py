import requests


QUEUE_URL = "http://127.0.0.1:9000"


def submit(payload):

    r = requests.post(

        f"{QUEUE_URL}/submit",

        json={
            "function": payload
        }
    )

    return r.json()["job_id"]


def status(job_id):

    r = requests.get(
        f"{QUEUE_URL}/status/{job_id}"
    )

    return r.json()