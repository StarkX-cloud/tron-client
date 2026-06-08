import requests


class Tron:

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    # =========================
    # SESSION (CREATE CONTEXT)
    # =========================
    def session(self):

        try:
            r = requests.post(
                f"{self.base_url}/create_session"
            )
            return r.json()

        except Exception as e:
            print("[SESSION ERROR]", e)
            return None

    # =========================
    # RUN JOB
    # =========================
    def run(self, prompt, **kwargs):

        payload = {
            "task_type": "inference",
            "prompt": prompt,
            "priority": kwargs.get("priority", 1),
            "gpu": kwargs.get("gpu", False),
            "memory_gb": kwargs.get("memory_gb", 2),
            "session_id": kwargs.get("session_id"),
            "graph_id": kwargs.get("graph_id"),
            "depends_on": kwargs.get("depends_on")
        }

        try:
            r = requests.post(
                f"{self.base_url}/submit_ai",
                json=payload,
                timeout=10
            )
            return r.json()

        except Exception as e:
            print("[TRON RUN ERROR]", e)
            return None

    # =========================
    # STATUS
    # =========================
    def status(self, job_id):

        try:
            r = requests.get(
                f"{self.base_url}/status/{job_id}",
                timeout=5
            )
            return r.json()

        except Exception as e:
            print("[STATUS ERROR]", e)
            return None

    # =========================
    # STREAM
    # =========================
    def stream(self, job_id):

        try:
            r = requests.get(
                f"{self.base_url}/stream/{job_id}",
                stream=True,
                timeout=10
            )

            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))

        except Exception as e:
            print("[STREAM ERROR]", e)

    # =========================
    # GRAPH
    # =========================
    def graph(self, graph_id):

        try:
            r = requests.get(
                f"{self.base_url}/graph/{graph_id}")
            return r.json()

        except Exception as e:
            print("[GRAPH ERROR]", e)
            return None