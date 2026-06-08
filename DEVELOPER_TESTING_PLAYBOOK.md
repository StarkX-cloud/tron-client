# Developer Testing Playbook

**From a developer's perspective, here's how to test TRON:**

---

## 🎯 Level 1: First Time Using TRON (5 minutes)

### Goal
See that TRON works and understand the basic pattern

### Test Script
```bash
python developer_getting_started.py
```

### What You'll See
- Part 1: Simple `@remote` function
- Part 2: Parallel processing
- Part 3: GPU tasks
- Part 4: Pipelines
- Part 5: Template for your code

### Success Criteria
✅ Everything runs
✅ You understand the pattern
✅ You see parallel execution
✅ You can modify the examples

---

## 🎯 Level 2: Building Your First App (20 minutes)

### Goal
Build a real distributed function for your use case

### Steps

1. **Pick a function you want to distribute**
   - Image processing?
   - Data cleaning?
   - ML inference?
   - API calls?

2. **Create `my_first_app.py`**
   ```python
   import tron
   
   @tron.remote
   def your_function(input_data):
       # Your logic here
       return result
   
   # Test it
   result = your_function(test_data).get()
   print(result)
   ```

3. **Test it locally**
   ```bash
   python my_first_app.py
   # Should work immediately!
   ```

4. **Modify to process multiple items**
   ```python
   tasks = [your_function(item) for item in items]
   results = [t.get() for t in tasks]
   print(results)
   ```

5. **Verify it works**
   ```bash
   python my_first_app.py
   # Should process all items
   ```

### Success Criteria
✅ Function works locally
✅ Multiple calls work
✅ Results are correct
✅ No errors

---

## 🎯 Level 3: Parallel Execution (10 minutes)

### Goal
Verify that parallel execution actually works

### Test
```python
import tron
import time

@tron.remote
def slow_task(n):
    time.sleep(2)
    return n * 2

# Measure parallel execution
start = time.time()
tasks = [slow_task(i) for i in range(10)]
results = [t.get() for t in tasks]
elapsed = time.time() - start

print(f"Processed 10 tasks in {elapsed:.1f}s")
print(f"Expected: ~2s (parallel), Not 20s (serial)")
```

### What You'll See (Local)
```
Processed 10 tasks in 2.1s
✓ Parallel execution confirmed!
```

### What Happens With Server
```
Processed 10 tasks in 2.1s
✓ Same speed but distributed across workers!
```

### Success Criteria
✅ 10 tasks complete in ~2s (not 20s)
✅ Same result with or without server

---

## 🎯 Level 4: Real-World Scenarios

### Scenario A: Image Processing

```python
import tron

@tron.remote
def process_image(image_path):
    # from PIL import Image
    # img = Image.open(image_path)
    # img.resize((256, 256))
    # img.save(f"output/{image_path}")
    return f"Processed {image_path}"

# Get list of images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Process in parallel
results = [process_image(img).get() for img in images]
print(results)
```

**Test:**
```bash
# Create test images
python process_images.py
# ✓ Should process quickly
```

---

### Scenario B: Data Pipeline

```python
import tron

@tron.remote
def extract(source):
    return {"raw": 1000}

@tron.remote
def transform(data):
    data["clean"] = 950
    return data

@tron.remote
def load(data):
    return {"saved": True, "rows": data["clean"]}

# Pipeline
e = extract("source").get()
t = transform(e).get()
l = load(t).get()
print(l)
```

**Test:**
```bash
python pipeline.py
# ✓ Each step completes
# ✓ Data flows through
```

---

### Scenario C: Batch Prediction

```python
import tron

# Simulate model
def predict(x):
    return x * 2.5

@tron.remote(gpu=True)
def predict_batch(batch):
    return [predict(x) for x in batch]

# Predict on batches
batches = [[1,2,3], [4,5,6], [7,8,9]]
results = [predict_batch(b).get() for b in batches]
print(results)
```

**Test:**
```bash
python predictions.py
# ✓ All batches process
# ✓ Results correct
```

---

## 🎯 Level 5: Error Cases

### Test Error Handling

```python
import tron

@tron.remote
def failing_function():
    raise ValueError("Intentional error")

try:
    result = failing_function(local_only=True).get()
except RuntimeError as e:
    print(f"✓ Error caught: {e}")
    print("✓ Error is clear and actionable")
```

**Success Criteria**
✅ Error is caught
✅ Error message is clear
✅ You know what went wrong

---

## 🎯 Level 6: With Server (10 minutes)

### Goal
See TRON scale from local to distributed

### Steps

**Terminal 1: Start server**
```bash
cd /path/to/TRON
python queue_server.py
```

**Terminal 2: Run your app**
```bash
python developer_getting_started.py
# Now it uses the server!
```

### What Changes
- Same code
- Same results
- But executes on remote workers
- Check dashboard: http://localhost:8501 (run `streamlit run dashboard_v3.py`)

### What Stays the Same
- Code is identical
- Results are identical
- `.get()` works the same

### Success Criteria
✅ App works with server running
✅ Same results as local
✅ See jobs in dashboard
✅ Jobs complete successfully

---

## 📋 Testing Checklist

### Basic Functionality
- [ ] Import TRON succeeds
- [ ] `@remote` decorator works
- [ ] `.get()` returns result
- [ ] Function works locally

### Parallel Execution
- [ ] Multiple tasks can run
- [ ] Results collected correctly
- [ ] Parallel is faster than serial
- [ ] All results are correct

### Real-World Patterns
- [ ] Processing lists/files works
- [ ] Pipelines execute in order
- [ ] GPU tasks work
- [ ] Error handling works

### Integration
- [ ] Works without server (local)
- [ ] Works with server (distributed)
- [ ] Same code in both cases
- [ ] Results identical in both cases

### Performance
- [ ] 10 tasks complete quickly
- [ ] 100 tasks process smoothly
- [ ] Memory doesn't explode
- [ ] No hanging/deadlocks

---

## 🚀 Testing Scripts You Can Copy

### Script 1: Quick Test
```bash
python developer_getting_started.py
```

### Script 2: Comprehensive Test
```bash
python developer_experience_test.py
```

### Script 3: Your Own Test
```bash
python my_app.py
```

---

## 📊 Expected Results

### Local Execution (No Server)
```
✓ Functions execute instantly
✓ Parallel tasks run sequentially
✓ Results available immediately
✓ Perfect for testing/development
```

### Remote Execution (With Server)
```
✓ Functions execute on workers
✓ Parallel tasks run in parallel
✓ Results collected when ready
✓ Scales to thousands of workers
```

---

## ❓ Common Issues & Solutions

### "It's not using my GPU"
**Solution:** Ensure `gpu=True` in decorator and server has GPU

### "Parallel doesn't seem faster"
**Solution:** Local execution can't parallelize well. Use server with multiple workers.

### "Errors are unclear"
**Solution:** Add logging to your function for debugging

### "Status always shows 'queued'"
**Solution:** Check if server is running: `python queue_server.py`

---

## 🎓 Learning Path

1. **Understand the concept** (5 min)
   - Read QUICKSTART.md
   - Run developer_getting_started.py

2. **Try it yourself** (10 min)
   - Create your own function
   - Make it work locally

3. **Test parallel execution** (5 min)
   - Process multiple items
   - Verify speedup

4. **Build real app** (20 min)
   - Use real data
   - Real functions
   - Real results

5. **Scale with server** (5 min)
   - Start server
   - Rerun same code
   - See it scale

**Total:** 45 minutes to productivity

---

## ✅ You Know TRON Works When

- [ ] You can decorate any function with `@remote`
- [ ] You can call it and get a result with `.get()`
- [ ] You can process multiple items in parallel
- [ ] You can build pipelines (step1 -> step2 -> step3)
- [ ] Errors are clear and actionable
- [ ] Same code works local and distributed
- [ ] You can monitor job status
- [ ] You understand when to use GPU

---

## 🚀 You're Ready to Ship When

- [ ] All tests pass locally
- [ ] Works with server running
- [ ] Error handling is solid
- [ ] Performance is acceptable
- [ ] Documentation is clear
- [ ] Examples work end-to-end
- [ ] You've tested edge cases

---

## Next Steps

**After testing:**
1. ✅ Build your real app
2. ✅ Deploy with confidence
3. ✅ Monitor in production
4. ✅ Scale as needed
5. ✅ Share your success!

You're ready. Go build! 🚀
