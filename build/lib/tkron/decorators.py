from tron.client import submit, status
from tron.serializer import serialize
import time


class TaskWrapper:

    def __init__(self, fn):

        self.fn = fn
        self.job_id = None
        self._cache = {}
        self._last_poll = 0

    # =========================
    # SUBMIT JOB
    # =========================

    def submit(self, *args, **kwargs):

        payload = serialize((self.fn, args, kwargs))

        self.job_id = submit(payload)

        print(f"[TASK SUBMITTED] {self.job_id}")

        return self

    # =========================
    # INTERNAL STATUS FETCH (cached)
    # =========================

    def _get_status(self, force=False):

        if not self.job_id:
            return None

        now = time.time()

        # simple throttle (prevents API spam)
        if not force and now - self._last_poll < 0.3:
            return self._cache.get("status", {})

        try:
            s = status(self.job_id)

        except Exception as e:
            print("[STATUS ERROR]", e)
            return self._cache.get("status", {})

        if not isinstance(s, dict):
            return self._cache.get("status", {})

        self._cache["status"] = s
        self._last_poll = now

        return s

    # =========================
    # DONE CHECK
    # =========================

    def done(self):

        s = self._get_status()

        if not s:
            return False

        if "status" not in s:
            return False

        return s["status"] == "completed"

    # =========================
    # RESULT
    # =========================

    def result(self):

        s = self._get_status(force=True)

        if not s:
            return None

        return s.get("result")

    # =========================
    # LOGS (NEW POWER FEATURE)
    # =========================

    def logs(self):

        s = self._get_status()

        if not s:
            return []

        return s.get("logs", [])

    # =========================
    # RUNTIME (NEW POWER FEATURE)
    # =========================

    def runtime(self):

        s = self._get_status(force=True)

        if not s:
            return None

        return s.get("runtime")

    # =========================
    # RAW STATUS (DEBUG MODE)
    # =========================

    def status(self):

        return self._get_status(force=True)


# =========================
# DECORATOR
# =========================

def task(fn):

    return TaskWrapper(fn)