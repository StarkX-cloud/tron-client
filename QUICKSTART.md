# 🚀 TRON Quick Start - The Magic Way

## Install & Setup

```bash
# Python is all you need
pip install -r requirements.txt
```

## 5-Minute Tutorial

### 1. Import TRON
```python
import tron
```

### 2. Define Work (One Decorator)
```python
@tron.remote
def expensive_task(x):
    """This runs distributed automatically."""
    return x ** 2

@tron.remote(gpu=True)  # GPU tasks get routed correctly
def train_model(data):
    return train(data)
```

### 3. Call It (It Just Works)
```python
# Single task
result = expensive_task(100).get()

# Parallel tasks
r1 = expensive_task(1).get()
r2 = expensive_task(2).get()
r3 = expensive_task(3).get()

# Chained tasks
data = load_data().get()
model = train_model(data).get()
eval = evaluate(model).get()
```

### 4. Done!
That's literally it. TRON handles:
- ✅ Finding your server (auto-discovery)
- ✅ Routing to the right workers
- ✅ Running locally when smart
- ✅ Running remote when needed
- ✅ Serialization & deserialization
- ✅ Error handling

---
## Local development made even easier
If you want TRON to start a local runtime for you, use:
```python
import tron

tron.start_local_environment()
```
This launches a local TRON server and worker process automatically so you can work without manually running `queue_server.py`.

You can also use the CLI after installation:
```bash
tron env start
```

---

## API Reference

### Basic Decorator
```python
@tron.remote
def my_func(x):
    return x * 2
```

### With Resource Hints
```python
@tron.remote(gpu=True, memory_gb=8)
def train():
    return model

@tron.remote(priority=10)  # Higher priority = faster
def urgent_task():
    pass
```

### Call Options
```python
result = task()              # Auto local+remote
result = task(local_only=True)  # Force local
result = task(remote_only=True) # Skip local, go remote
```

### Getting Results
```python
# Blocking (simple, recommended)
result = task().get()

# Non-blocking (advanced)
future = task()
if future.ready():
    print(future.result())

# Async (advanced)
result = await task()
```

### Configure Server (Optional)
```python
# Environment variable
import os
os.environ["TRON_URL"] = "http://my-cluster:9000"

# Or explicit
import tron
tron.config("http://my-cluster:9000")

# Or just let auto-discovery work 🎉
```

---

## Examples

### Simple Parallel Pipeline
```python
import tron
import time

@tron.remote
def fetch_data():
    time.sleep(1)
    return {"data": [1,2,3]}

@tron.remote(gpu=True)
def process(data):
    time.sleep(2)
    return {"processed": data}

@tron.remote
def save(result):
    time.sleep(1)
    print("Done!")
    return result

# Run it
data = fetch_data()
processed = process(data.get())
final = save(processed.get())

print(final.get())
```

### Batch Processing
```python
@tron.remote
def process_item(item):
    return item * 2

items = [1, 2, 3, 4, 5]

# Fire off all tasks
futures = [process_item(x) for x in items]

# Collect results
results = [f.get() for f in futures]

print(results)  # [2, 4, 6, 8, 10]
```

---

## FAQ

**Q: Do I need to run a server?**  
A: TRON auto-discovers it. If not found, tasks run locally anyway.

**Q: What if something fails?**  
A: It raises an exception with a clear error message. TRON auto-retries (coming soon).

**Q: Can I mix local and remote?**  
A: Yes! That's the whole point. TRON is smart about it.

**Q: Does my code need to change?**  
A: Just add `@tron.remote` and call `.get()` on the result. That's it.

**Q: What gets serialized?**  
A: The function and its arguments. Normal Python pickle rules apply.

---

## Under the Hood (Optional Reading)

```
User Code (@remote decorator)
         ↓
    Wrapper Function
         ↓
    Local Execution First? 
    Yes → Returns LocalFuture (instant)
    No  → Serialize & Submit to Queue
         ↓
    Server Receives Job
         ↓
    Orchestrator Matches to Worker
         ↓
    Worker Executes & Returns Result
         ↓
    .get() Polls for Result (transparent)
         ↓
    Result to User (feels instant)
```

---

## The Magic ✨

Compare old vs new:

```python
# OLD: Complex
@tron.remote
def task():
    return 1

result = task()
while not result.done():
    time.sleep(1)
output = result.result()

# NEW: Magic
@tron.remote
def task():
    return 1

output = task().get()
```

**That's the entire transformation.** The rest is invisible.

---

## Next Steps

1. Run `python test_magic_layer.py` to verify setup
2. Try `python magic_example.py` with your server
3. Read [MAGIC_GUIDE.md](MAGIC_GUIDE.md) for deep dive
4. Start building!
