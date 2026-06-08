# =========================
# TRON PREDICTOR V1
# =========================

import statistics

# =========================
# RUNTIME HISTORY
# =========================

runtime_history = {}

# =========================
# RECORD RUNTIME
# =========================

def record_runtime(task_type, runtime):

    if task_type not in runtime_history:
        runtime_history[task_type] = []

    runtime_history[task_type].append(runtime)

    # keep only latest 100
    runtime_history[task_type] = \
        runtime_history[task_type][-100:]

# =========================
# PREDICT RUNTIME
# =========================

def predict_runtime(task_type):

    history = runtime_history.get(task_type, [])

    if len(history) == 0:
        return 3.0

    return round(statistics.mean(history), 2)

# =========================
# WORKER SPEED SCORE
# =========================

def worker_speed(worker):

    load = worker.get("load", 0)

    memory = worker.get("memory_gb", 1)

    base = memory * 10

    penalty = load * 5

    return max(base - penalty, 1)

# =========================
# CONGESTION RISK
# =========================

def congestion_risk(job_queue, running_jobs):

    q = len(job_queue)
    r = len(running_jobs)

    total = q + (r * 2)

    if total < 10:
        return "low"

    elif total < 25:
        return "medium"

    else:
        return "high"

# =========================
# EXECUTION FORECAST
# =========================

def execution_forecast(
    task_type,
    worker,
    job_queue,
    running_jobs
):

    predicted_runtime = predict_runtime(task_type)

    speed = worker_speed(worker)

    congestion = congestion_risk(
        job_queue,
        running_jobs
    )

    adjusted = predicted_runtime

    adjusted /= max(speed / 10, 1)

    if congestion == "medium":
        adjusted *= 1.2

    elif congestion == "high":
        adjusted *= 1.5

    return round(adjusted, 2)