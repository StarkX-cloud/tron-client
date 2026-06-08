from tron import task
from tron.sdk import Tron
import time

tron = Tron()

# =========================
# TASK
# =========================

@task()
def heavy_task(i):

    print(f"Running task {i}")

    time.sleep(5)

    return f"JOB {i} COMPLETE"


# =========================
# SUBMIT MANY TASKS
# =========================

futures = []

for i in range(10):

    f = heavy_task.submit(i)

    futures.append(f)

# =========================
# WAIT FOR RESULTS
# =========================

for f in futures:

    print(f.result())