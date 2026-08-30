import time
import tron

@tron.remote(remote_only=True)
def expensive_add(x, y):
    return {"sum": x + y, "worker": "distributed"}

print("Submitting distributed job...")
future = expensive_add(5, 7)
print("Job submitted, waiting for result...")
result = future.get()
print("Result:", result)
print("Done")
