import json
import time
from typing import Optional
from urllib.parse import urlparse

import requests

from job import Job

def _get_ensure_server():
    from tron.config import ensure_server
    return ensure_server


class Tron:

    # =========================
    # INIT
    # =========================
    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            base_url = _get_ensure_server()()
        else:
            base_url = base_url.rstrip("/")
            parsed = urlparse(base_url)
            if parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost"):
                try:
                    requests.get(base_url, timeout=1)
                except Exception:
                    base_url = _get_ensure_server()(
                        host=parsed.hostname,
                        port=parsed.port or 9000,
                    )

        self.base_url = base_url.rstrip("/")
        self._dag_nodes: list = []

    # =========================
    # INTERNAL GET
    # =========================
    def _get(self, path, timeout=10):

        try:
            r = requests.get(
                f"{self.base_url}{path}",
                timeout=timeout
            )
            return r.json()

        except Exception as e:
            print("[GET ERROR]", path, e)
            return None

    # =========================
    # INTERNAL POST
    # =========================
    def _post(self, path, payload=None, timeout=10):

        try:
            r = requests.post(
                f"{self.base_url}{path}",
                json=payload or {},
                timeout=timeout
            )
            return r.json()

        except Exception as e:
            print("[POST ERROR]", path, e)
            return None

    # =========================
    # HEALTH
    # =========================
    def health(self):
        return self._get("/")

    # =========================
    # SESSION
    # =========================
    def session(self):
        return self._post("/create_session")

    # =========================
    # GRAPH
    # =========================
    def create_graph(self):
        return self._post("/create_graph")

    # =========================
    # RUN JOB
    # =========================
    def run(self, *args, **kwargs):
        if len(args) == 0 and not kwargs:
            return self._run_dag()
        return self._run_remote(*args, **kwargs)

    def _run_remote(self, prompt, **kwargs):
        payload = {
            "prompt": prompt,
            **kwargs
        }

        result = self._post("/submit_ai", payload)

        if result and isinstance(result, dict) and result.get("job_id"):
            return Job(self, result["job_id"])

        return result

    def submit(self, prompt, **kwargs):
        return self.run(prompt, **kwargs)

    def _run_dag(self, tasks=None):
        from tron.decorators import TaskWrapper

        if tasks is None:
            tasks = list(self._dag_nodes)

        normalized = []
        for task in tasks:
            if isinstance(task, TaskWrapper):
                normalized.append(task)
            elif callable(task):
                normalized.append(task)
            else:
                raise ValueError("Tron.run() expects TaskWrapper objects or tasks created with tron.node()")

        ordered = self._topological_sort(normalized)
        futures = []

        for task in ordered:
            for dep in getattr(task, "dependencies", []):
                dep.wait()
            task.submit()
            futures.append(task)

        return futures

    def _topological_sort(self, tasks):
        visited = {}
        order = []

        def visit(node):
            if visited.get(node) == "temp":
                raise RuntimeError("Cycle detected in DAG")
            if visited.get(node) == "perm":
                return

            visited[node] = "temp"
            for dep in getattr(node, "dependencies", []):
                visit(dep)
            visited[node] = "perm"
            order.append(node)

        for task in tasks:
            visit(task)

        return order

    def node(self, task_or_fn):
        from tron.decorators import TaskWrapper, task

        if isinstance(task_or_fn, TaskWrapper):
            if task_or_fn not in self._dag_nodes:
                self._dag_nodes.append(task_or_fn)
            return task_or_fn

        if callable(task_or_fn):
            wrapped = task(task_or_fn)
            if wrapped not in self._dag_nodes:
                self._dag_nodes.append(wrapped)
            return wrapped

        raise ValueError("Tron.node() expects a decorated task or callable")

    def connect(self, parent, child):
        if not hasattr(child, "after"):
            raise ValueError("Tron.connect() requires a task wrapper with an after() method")
        child.after(parent)
        if child not in self._dag_nodes:
            self._dag_nodes.append(child)
        if parent not in self._dag_nodes:
            self._dag_nodes.append(parent)
        return child

    # =========================
    # STATUS
    # =========================
    def status(self, job_id):
        return self._get(f"/status/{job_id}")

    # =========================
    # RESULT
    # =========================
    def result(self, job_id):
        return self._get(f"/result/{job_id}")

    # =========================
    # METRICS
    # =========================
    def metrics(self):
        return self._get("/metrics")

    def workers(self):
        return self._get("/workers")

    def queue(self):
        return self._get("/queue")

    def history(self):
        return self._get("/history")

    # =========================
    # WAIT (POLLING SAFE)
    # =========================
    def wait(self, job_id, poll_interval=1):

        while True:

            result = self.result(job_id)

            if result and result.get("status") == "completed":
                return result

            time.sleep(poll_interval)

    # =========================
    # COMPUTE (STREAM-POWERED EXECUTION)
    # =========================
    def compute(self, prompt, **kwargs):

        job = self.run(prompt, **kwargs)

        if not job:
            raise Exception("TRON submission failed")

        job_id = job.job_id if hasattr(job, "job_id") else job.get("job_id")

        result_data = None

        try:
            r = requests.get(
                f"{self.base_url}/stream/{job_id}",
                stream=True,
                timeout=60
            )

            for line in r.iter_lines():

                if not line:
                    continue

                decoded = line.decode("utf-8")

                # ignore heartbeat noise
                if "heartbeat" in decoded:
                    continue

                print(decoded)

                # detect completion event
                if '"type": "completed"' in decoded:

                    payload = json.loads(
                        decoded.replace("data: ", "")
                    )

                    result_data = payload["data"]["result"]
                    break

        except Exception as e:
            print("[STREAM COMPUTE ERROR]", e)

        if result_data is None:
            raise Exception("TRON execution failed or no completion event received")

        return result_data

    # =========================
    # INFER (MAIN API)
    # =========================
    def infer(self, prompt, **kwargs):
        return self.compute(prompt, **kwargs)

    def infer_local(self, prompt, **kwargs):
        return self.infer(prompt, **kwargs)

    # =========================
    # STREAM (DEBUG MODE)
    # =========================
    def stream(self, job_id, timeout=60):

        print("[STREAM START]")

        waiting_message_shown = False

        try:
            r = requests.get(
                f"{self.base_url}/stream/{job_id}",
                stream=True,
                timeout=timeout
            )

            if r.status_code != 200:
                body = r.text.strip()
                print(f"[STREAM ERROR] server returned status {r.status_code}")
                if body:
                    print(body)
                return

            buffer = ""

            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk is None:
                    continue

                buffer += chunk

                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        decoded = line[len("data: "):]
                    else:
                        decoded = line

                    if "heartbeat" in decoded:
                        if not waiting_message_shown:
                            print("[STREAM] waiting for execution...")
                            waiting_message_shown = True
                        continue

                    try:
                        payload = json.loads(decoded)
                        event_type = payload.get("type")
                        data = payload.get("data", {})

                        if event_type == "queued":
                            print("[STREAM] queued")
                        elif event_type == "started":
                            worker = data.get("worker")
                            task_type = data.get("task_type")
                            print(f"[STREAM] started on {worker} ({task_type})")
                        elif event_type == "completed":
                            print("[STREAM] completed")
                            print(data.get("result"))
                            return
                        elif event_type == "error":
                            print("[STREAM] error", data)
                            return
                        else:
                            print(f"[STREAM] {event_type}: {data}")

                    except Exception:
                        print(decoded)

        except Exception as e:
            print("[STREAM ERROR]", e)
            print("Hint: make sure a TRON worker is running and the server supports /stream")

    # =========================
    # GRAPH FETCH
    # =========================
    def graph(self, graph_id):
        return self._get(f"/graph/{graph_id}")