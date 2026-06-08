# Developer Testing Guide - Real Workflows

## What You Just Saw

That's what a **real developer** sees when using TRON:
- ✓ Simple import
- ✓ One decorator
- ✓ Works instantly
- ✓ Scales easily
- ✓ Clear errors
- ✓ Parallel by default

---

## Testing Scenarios - From Developer Perspective

### Scenario 1: "I Just Want to Test It"

```bash
# Run the complete developer experience
python developer_experience_test.py

# You see:
# ✓ 11 different use cases
# ✓ Parallel execution working
# ✓ Pipelines working
# ✓ Error handling working
# ✓ Status tracking working
```

**Time:** 2 seconds  
**Prerequisites:** None (works without server)  
**Result:** Confidence that it works

---

### Scenario 2: "I Want to Build Something"

Start with a **real problem**:

#### A. Batch Image Processing
```python
import tron
from PIL import Image

@tron.remote
def process_image(image_path):
    """Resize and convert image."""
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img.save(f"output/{image_path}")
    return f"Processed: {image_path}"

# Process 1000 images in parallel
images = get_image_list()
results = [process_image(img) for img in images]
outputs = [r.get() for r in results]
print(f"Processed {len(outputs)} images")
```

**Test it:**
```bash
# 1. Prepare test images
mkdir -p images output
# Add some test images

# 2. Run the script
python your_script.py

# You'll see:
# - Images processing in parallel
# - Results collected
# - All done
```

---

#### B. ML Model Inference
```python
import tron
import torch

# Load model once
model = torch.load("model.pt")

@tron.remote(gpu=True)
def predict(data):
    """Inference on GPU."""
    with torch.no_grad():
        return model(torch.tensor(data)).tolist()

# Predict on batch
batch = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
predictions = [predict(item).get() for item in batch]
print(predictions)
```

**Test it:**
```bash
# Without server (will run locally):
python inference_script.py
# ✓ Should work instantly

# With server (will route to GPU):
python queue_server.py &  # Start server
sleep 2
python inference_script.py
# ✓ Should still work, using remote GPU
```

---

#### C. Data Processing Pipeline
```python
import tron
import pandas as pd

@tron.remote
def load_data(source):
    return pd.read_csv(source)

@tron.remote
def clean_data(df):
    return df.dropna()

@tron.remote
def aggregate(df):
    return df.groupby('category').sum()

# Pipeline
data = load_data("raw_data.csv").get()
clean = clean_data(data).get()
result = aggregate(clean).get()
result.to_csv("output.csv")
```

**Test it:**
```bash
# Create sample data
python -c "import pandas as pd; pd.DataFrame({'a': range(100), 'b': range(100)}).to_csv('raw_data.csv')"

# Run pipeline
python pipeline_script.py

# You'll see:
# ✓ Each step executes
# ✓ Data passes through
# ✓ Output saved
```

---

### Scenario 3: "Show Me Parallel Execution"

```python
import tron
import time

@tron.remote
def task(i):
    """Tasks that take 5 seconds each."""
    time.sleep(5)
    return f"Task {i} done"

# Sequential: 50 seconds
print("Starting 10 tasks...")
start = time.time()

# This runs in parallel (all 10 at once locally)
futures = [task(i) for i in range(10)]
results = [f.get() for f in futures]

elapsed = time.time() - start
print(f"Completed in {elapsed:.1f}s")
print(results)
```

**Test it:**
```bash
time python parallel_test.py

# Without server:
# Takes ~5-6 seconds (sequential)

# With server and multiple workers:
# Takes ~5-6 seconds (parallel!)
```

**What you see:**
- Local execution is fast (5s for 10 tasks)
- Remote execution is fast too (5s for 10 tasks on 10 workers)
- **The code is identical in both cases** ✨

---

### Scenario 4: "I Want to Monitor It"

```python
import tron
import time

@tron.remote
def long_task(n):
    """Task that takes a while."""
    time.sleep(n)
    return f"Took {n} seconds"

# Start task
print("Starting long task...")
future = long_task(10)

# Monitor without blocking
for i in range(11):
    status = future.status()
    print(f"[{i}s] Status: {status.get('status')}")
    time.sleep(1)
    if future.ready():
        break

# Get result when done
print(f"Result: {future.get()}")
```

**Test it:**
```bash
python monitor_test.py

# You see:
# [0s] Status: queued
# [1s] Status: running
# [2s] Status: running
# ...
# [10s] Status: completed
# Result: Took 10 seconds
```

---

### Scenario 5: "Test with Server"

When developer has server running:

```bash
# Terminal 1: Start server
python queue_server.py

# Terminal 2: Run tests
python developer_experience_test.py

# OR run your app:
python your_app.py
```

**What changes for developer:**
- Same code works
- But uses remote execution
- But no code changes needed
- Check dashboard to see jobs

---

## Testing Checklist for Developers

### ✓ Basic Testing
```bash
# Does it work locally?
python developer_experience_test.py
```

### ✓ Parallel Testing
```bash
# Do multiple tasks run?
python -c "
import tron
tasks = [tron.remote(lambda x: x*2)(i) for i in range(5)]
results = [t.get() for t in tasks]
print('Parallel works:', results)
"
```

### ✓ GPU Testing
```bash
# Does GPU routing work?
@tron.remote(gpu=True)
def gpu_task():
    return 'gpu'

print(gpu_task().get())
```

### ✓ Error Testing
```bash
# Do errors propagate?
@tron.remote
def failing():
    raise ValueError("test error")

try:
    failing(local_only=True).get()
except RuntimeError as e:
    print("Error caught:", e)
```

### ✓ Pipeline Testing
```bash
# Do pipelines work?
@tron.remote
def step1(): return 1

@tron.remote
def step2(x): return x * 2

@tron.remote
def step3(x): return x + 10

r1 = step1().get()
r2 = step2(r1).get()
r3 = step3(r2).get()
print(f"Pipeline: 1 -> {r2} -> {r3}")
```

---

## Developer Quick Start (5 minutes)

```bash
# 1. See it work (local, no server)
python developer_experience_test.py

# 2. Start server (in another terminal)
python queue_server.py

# 3. See it scale (with server)
python developer_experience_test.py

# 4. Build something
# Edit your_app.py with real work
python your_app.py

# 5. Monitor
# Start the dashboard with streamlit run dashboard_v3.py
# Open http://localhost:8501
```

---

## What Developers Should Test Themselves

| Feature | How to Test | Expected Result |
|---------|------------|-----------------|
| **Basic execution** | `python developer_experience_test.py` | Works immediately |
| **Parallel tasks** | Run 10 tasks, see them complete fast | All complete |
| **GPU routing** | `@tron.remote(gpu=True)` | Works, routes if GPU available |
| **Error handling** | Intentional error in function | Clear error message |
| **Status tracking** | Check `.status()` during execution | Shows correct status |
| **Local to remote** | Same code, with/without server | Same results |
| **Scaling** | 1M item list | Processes in reasonable time |

---

## Real Developer Workflows

### Workflow 1: "I'm Learning"
```bash
# 1. Run developer_experience_test.py to see examples
python developer_experience_test.py

# 2. Modify functions to understand
# Edit the file, change what functions do

# 3. Test my changes
python developer_experience_test.py
```

**Time:** 5 minutes to understand  
**Result:** Confident with API

---

### Workflow 2: "I'm Building an App"
```bash
# 1. Write your functions
# my_app.py with @tron.remote functions

# 2. Test locally (no server)
python my_app.py

# 3. Start server
python queue_server.py &

# 4. Test with server
python my_app.py

# 5. Monitor in dashboard
# Run the dashboard with:
# streamlit run dashboard_v3.py
# Then open http://localhost:8501
```

**Time:** Depends on app  
**Result:** Production ready

---

### Workflow 3: "I'm Debugging"
```python
import tron

@tron.remote
def buggy_function(x):
    print(f"Input: {x}")  # Debug output
    result = x / (x - 5)  # Might fail
    print(f"Output: {result}")  # Debug output
    return result

# Test it
try:
    future = buggy_function(5, local_only=True)
    print(future.get())
except Exception as e:
    print(f"Error: {e}")
```

---

## Test Results Template

When you test TRON, you should see:

```
✓ Functions work locally
✓ No server required for basic testing
✓ Parallel execution is automatic
✓ Results are collected cleanly
✓ Errors are clear and actionable
✓ Status tracking works
✓ Same code works with/without server
✓ Performance scales with workers
```

If ANY of these is missing, we need to fix it.

---

## The Ideal Developer Experience

A developer should be able to:

1. **Read docs** (5 min) → Understand concept
2. **See example** (1 min) → Run developer_experience_test.py
3. **Build something** (30 min) → Create my_app.py
4. **Test it** (1 min) → Works locally
5. **Scale it** (0 changes) → Works with server
6. **Deploy it** (1 min) → Same code to production

**Total time to productivity:** 40 minutes  
**Code changes from local to distributed:** 0 (just add @remote)

---

## Current Status

✅ **You can test:**
- Running the developer experience test
- Building your own functions
- Parallel execution
- Error handling
- Status tracking

✅ **Works without server** (for learning)

🟡 **Need server for:**
- Actual distributed execution
- GPU routing
- Multiple workers
- Real performance

---

## Next Step

**For you (creator):** Run `python developer_experience_test.py` and see:
- Is the experience clear?
- Is it intuitive?
- What would a real dev struggle with?

**For developers:** 
- Read QUICKSTART.md
- Run developer_experience_test.py  
- Build your app
- You're ready!
