import time
import json
from collections import defaultdict


class StreamEngine:

    def __init__(self):

        self.streams = defaultdict(list)

    # =========================
    # PUSH EVENT
    # =========================

    def emit(self, job_id, event_type, data=None):

        self.streams[job_id].append({
            "type": event_type,
            "data": data or {},
            "time": time.time()
        })

    # =========================
    # GET EVENTS
    # =========================

    def get(self, job_id):

        return self.streams.get(job_id, [])

    # =========================
    # HEARTBEAT EVENT
    # =========================

    def heartbeat(self, job_id):

        self.emit(job_id, "heartbeat", {
            "status": "alive"
        })