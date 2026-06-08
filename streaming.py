from collections import defaultdict
import time

# ====================================
# GLOBAL EVENT BUS
# ====================================

event_bus = defaultdict(list)

# ====================================
# EMIT EVENT
# ====================================

def emit(job_id, event_type, data):

    event = {
        "job_id": job_id,
        "type": event_type,
        "data": data,
        "timestamp": time.time()
    }

    event_bus[job_id].append(event)

# ====================================
# GET STREAM
# ====================================

def get_stream(job_id):

    return event_bus.get(job_id, [])