import time
from .client import submit, status
from .serializer import serialize


class TaskWrapper:

    def __init__(self, fn, resources=None):
        self.fn = fn
        self.resources = resources or {}
        self.job_id = None
        self._cache = {}
        self._last_poll = 0
        self.dependencies = []

    def submit(self, *args, **kwargs):
        """Submit task to queue."""
        payload = serialize((self.fn, args, kwargs))
        self.job_id = submit(payload)
        print(f"[TASK SUBMITTED] {self.job_id}")
        return self

    def _get_status(self, force=False):
        if not self.job_id:
            return None

        now = time.time()

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

    def done(self):
        s = self._get_status()

        if not s:
            return False

        return s.get("status") == "completed"

    def result(self):
        s = self._get_status(force=True)

        if not s:
            return None

        return s.get("result") or s.get("output")

    def logs(self):
        s = self._get_status()

        if not s:
            return []

        return s.get("logs", [])

    def runtime(self):
        s = self._get_status(force=True)

        if not s:
            return None

        return s.get("runtime")

    def status(self):
        return self._get_status(force=True)

    def after(self, *dependencies):
        """Set task dependencies for DAG execution."""
        self.dependencies = dependencies
        return self
    
    def wait(self):
        """Wait for task completion and return result."""
        return self.result()


def task(fn=None, gpu=False, memory=None, memory_gb=None, priority=1, **kwargs):
    """
    @task decorator - supports both @task and @task(gpu=True, memory=8)
    
    Backward compatible with old code that uses resource hints.
    """
    # Memory parameter can be passed as either 'memory' or 'memory_gb'
    if memory is not None and memory_gb is None:
        memory_gb = memory
    if memory_gb is None:
        memory_gb = 1
    
    # Store resource hints for later use
    resources = {
        "gpu": gpu,
        "memory_gb": memory_gb,
        "priority": priority,
        **kwargs
    }
    
    def decorator(func):
        wrapper = TaskWrapper(func, resources)
        return wrapper
    
    # Handle both @task and @task(...) syntax
    if fn is not None:
        # Called as @task (no parameters)
        return decorator(fn)
    else:
        # Called as @task(...) (with parameters)
        return decorator
