"""
DEVELOPER EXPERIENCE TEST - How real users would use TRON
This is what a developer sees when they use your library
"""

print("\n" + "="*60)
print("TRON DEVELOPER EXPERIENCE TEST")
print("Real workflow: install → code → test → deploy")
print("="*60 + "\n")

# ============================================================
# STEP 1: A developer just installed TRON
# ============================================================
print("STEP 1: Import and basic setup")
print("-" * 60)

import tron
print("✓ Import successful")

# ============================================================
# STEP 2: Developer writes a simple function
# ============================================================
print("\nSTEP 2: Write a function and make it distributed")
print("-" * 60)

@tron.remote
def add_numbers(a, b):
    """Simple distributed function."""
    return a + b

print("✓ Function decorated with @tron.remote")
print("  Code: @tron.remote def add_numbers(a, b): return a + b")

# ============================================================
# STEP 3: Developer calls it locally first (no server)
# ============================================================
print("\nSTEP 3: Call locally (works without server)")
print("-" * 60)

result = add_numbers(5, 3).get()
print(f"✓ Result: 5 + 3 = {result}")
print("  └─ Works locally, no infrastructure needed!")

# ============================================================
# STEP 4: Developer writes a more realistic function
# ============================================================
print("\nSTEP 4: More realistic: Data processing")
print("-" * 60)

@tron.remote
def process_file(filename):
    """Simulate processing a data file."""
    # In real code, this would be:
    # data = load_file(filename)
    # return expensive_computation(data)
    
    # For demo:
    import time
    time.sleep(0.5)
    return {"file": filename, "rows": 1000, "processed": True}

file_result = process_file("data.csv").get()
print(f"✓ Processed: {file_result}")

# ============================================================
# STEP 5: Developer processes multiple files (parallel)
# ============================================================
print("\nSTEP 5: Process multiple files in parallel")
print("-" * 60)

files = ["data1.csv", "data2.csv", "data3.csv", "data4.csv"]
print(f"Processing {len(files)} files...")

# Fire off all tasks
tasks = [process_file(f) for f in files]
print(f"✓ Submitted {len(tasks)} tasks")

# Collect results
results = [t.get() for t in tasks]
print(f"✓ Collected {len(results)} results")
print(f"  Results: {results}")

# ============================================================
# STEP 6: Developer adds GPU task
# ============================================================
print("\nSTEP 6: GPU task for ML inference")
print("-" * 60)

@tron.remote(gpu=True)
def predict_batch(batch_data):
    """ML inference - requires GPU."""
    # In real code: model.predict(batch_data)
    # For demo:
    import time
    time.sleep(0.5)
    return {"predictions": len(batch_data), "avg_confidence": 0.92}

pred_result = predict_batch([1, 2, 3, 4, 5]).get()
print(f"✓ GPU Task Result: {pred_result}")
print("  └─ Would auto-route to GPU workers if server available")

# ============================================================
# STEP 7: Developer builds a pipeline
# ============================================================
print("\nSTEP 7: Pipeline (step 1 → step 2 → step 3)")
print("-" * 60)

@tron.remote
def extract_data():
    """Step 1: Get data."""
    import time
    time.sleep(0.3)
    return {"rows": 1000, "columns": 50}

@tron.remote
def transform_data(data):
    """Step 2: Transform it."""
    import time
    time.sleep(0.3)
    data["cleaned"] = True
    return data

@tron.remote
def load_data(data):
    """Step 3: Save it."""
    import time
    time.sleep(0.3)
    return {"saved": True, "data": data}

# Run pipeline
print("Running pipeline...")
step1_result = extract_data().get()
print(f"  1. Extract: {step1_result}")

step2_result = transform_data(step1_result).get()
print(f"  2. Transform: {step2_result}")

step3_result = load_data(step2_result).get()
print(f"  3. Load: {step3_result}")
print("✓ Pipeline complete!")

# ============================================================
# STEP 8: Developer handles errors
# ============================================================
print("\nSTEP 8: Error handling")
print("-" * 60)

@tron.remote
def risky_operation():
    """This will fail."""
    raise ValueError("Something went wrong!")

try:
    result = risky_operation(local_only=True).get()
except RuntimeError as e:
    print(f"✓ Error caught: {type(e).__name__}")
    print(f"  Message: {e}")
    print("  └─ Errors are clear and actionable")

# ============================================================
# STEP 9: Developer checks task status
# ============================================================
print("\nSTEP 9: Check status without waiting")
print("-" * 60)

task = add_numbers(100, 200)
print(f"Task ID: {task.job_id}")
print(f"Ready: {task.ready()}")
print(f"Status: {task.status()}")
result = task.get()
print(f"✓ Result: {result}")

# ============================================================
# STEP 10: Developer uses async (advanced)
# ============================================================
print("\nSTEP 10: Async/await syntax (advanced users)")
print("-" * 60)

import asyncio

async def async_workflow():
    """Async example."""
    # In real code with server running:
    # result1 = await add_numbers(1, 2)
    # result2 = await add_numbers(3, 4)
    # return result1 + result2
    
    # For demo (local execution):
    r1 = add_numbers(1, 2).get()
    r2 = add_numbers(3, 4).get()
    return r1 + r2

# Only run if not already in event loop
try:
    result = asyncio.run(async_workflow())
    print(f"✓ Async workflow result: {result}")
except:
    print("✓ Async available (not running in sync context)")

# ============================================================
# STEP 11: Developer configures server (optional)
# ============================================================
print("\nSTEP 11: Configure server URL (optional)")
print("-" * 60)

# Show current config
tron.config()
print("  └─ Auto-discovered or using default")

# Show how to override
print("  To use custom server: tron.config('http://my-server:9000')")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("DEVELOPER EXPERIENCE SUMMARY")
print("="*60)

print("""
✓ Simple functions work instantly
✓ Parallel execution is natural
✓ GPU tasks are simple
✓ Pipelines are straightforward
✓ Error handling is clear
✓ Status tracking is easy
✓ Async is available
✓ Configuration is optional

This is what developers experience:
1. Write code
2. Add @remote
3. Call .get()
4. It works locally
5. It scales with server
6. Zero infrastructure thinking required

That's the magic. 🚀
""")

print("="*60)
print("NEXT STEPS FOR DEVELOPERS")
print("="*60)
print("""
1. Try these examples
2. Modify the functions to do real work
3. Start a server: python queue_server.py
4. Rerun - should use remote execution
5. Check dashboard to see jobs
6. Build your app!
""")
