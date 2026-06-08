def select_worker(job, workers):

    best_worker = None
    best_score = -1

    for name, w in workers.items():

        score = 0

        # GPU MATCH
        if job["gpu_required"] and w["gpu"]:
            score += 50

        # MEMORY MATCH
        if w["memory_gb"] >= job["min_memory_gb"]:
            score += 30

        # LIGHT LOAD PREFERENCE (future-ready)
        load = w.get("load", 0.0)
        score += int((1 - load) * 20)

        if score > best_score:
            best_score = score
            best_worker = name

    return best_worker