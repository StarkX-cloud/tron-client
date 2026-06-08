# ====================================
# TRON ORCHESTRATOR V1
# ====================================

def can_run(job, worker):

    # GPU REQUIREMENT
    if job.get("gpu") and not worker.get("gpu"):
        return False

    # MEMORY CHECK
    if worker.get("memory_gb", 0) < job.get("memory_gb", 0):
        return False

    return True


def compute_score(job, worker):

    score = 0

    # PRIORITY BOOST
    score += job.get("priority", 1) * 20

    # MEMORY CAPABILITY
    score += worker.get("memory_gb", 0)

    # LOWER LOAD = BETTER
    score -= worker.get("load", 0) * 5

    # GPU BONUS
    if job.get("gpu") and worker.get("gpu"):
        score += 50

    # TASK BOOSTS
    task_type = job.get("task_type", "")

    if task_type == "training":
        score += 20

    elif task_type == "inference":
        score += 10

    elif task_type == "pipeline":
        score += 15

    return score


def orchestrate(worker_name, workers, job_queue, running_jobs):

    # WORKER EXIST?
    if worker_name not in workers:
        return None

    worker = workers[worker_name]

    best_job = None
    best_score = float("-inf")
    best_index = None

    # LOOP THROUGH QUEUE
    for i, job in enumerate(job_queue):

        # CAN THIS WORKER RUN IT?
        if not can_run(job, worker):
            continue

        score = compute_score(job, worker)

        if score > best_score:
            best_score = score
            best_job = job
            best_index = i

    if not best_job:
        return None

    return best_job, best_index