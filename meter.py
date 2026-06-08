usage_events = []

def log_usage(user_id, job_id, cost, job_type):

    event = {
        "user_id": user_id,
        "job_id": job_id,
        "cost": cost,
        "job_type": job_type,
        "time": __import__("time").time()
    }

    usage_events.append(event)

    return event