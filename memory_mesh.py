import time
import threading


class MemoryMesh:

    def __init__(self):

        self.global_store = {}
        self.lock = threading.Lock()

    # =========================
    # WRITE TO MESH
    # =========================

    def write(self, key, value, source=None):

        with self.lock:

            self.global_store[key] = {
                "value": value,
                "source": source,
                "timestamp": time.time()
            }

    # =========================
    # READ FROM MESH
    # =========================

    def read(self, key):

        with self.lock:

            return self.global_store.get(key, {}).get("value")

    # =========================
    # CONTEXT INJECTION
    # =========================

    def inject_context(self, job):

        context = {}

        # pull system-wide memory signals
        context["global_prompt_bias"] = self.read("prompt_pattern")
        context["system_load_hint"] = self.read("load_pattern")

        job["memory_context"] = context

        return job