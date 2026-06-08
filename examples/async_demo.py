from tron.remote import remote

import time

@remote
def gpu_task():

    print("GPU TASK...")
    time.sleep(5)

    return {
        "gpu": "done"
    }

@remote
def cpu_task():

    print("CPU TASK...")
    time.sleep(2)

    return {
        "cpu": "done"
    }

print("\n=== TRON ASYNC EXECUTION ===\n")

a = gpu_task()

b = cpu_task()

print("TASKS RUNNING IN BACKGROUND\n")

while True:

    gpu_ready = a.ready()
    cpu_ready = b.ready()

    print(
        f"GPU: {gpu_ready} | CPU: {cpu_ready}"
    )

    if gpu_ready and cpu_ready:
        break

    time.sleep(1)

print("\nRESULTS:\n")

print(a.get())
print(b.get())
