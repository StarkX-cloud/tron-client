"""
TRON Developer Getting Started
From zero to working distributed app in 10 minutes
"""

# ============================================================
# PART 1: THE SIMPLEST POSSIBLE EXAMPLE
# ============================================================

print("\n" + "="*60)
print("PART 1: Hello TRON")
print("="*60 + "\n")

import tron

# Step 1: Write a function
def double(x):
    return x * 2

# Step 2: Add @tron.remote
@tron.remote
def double_distributed(x):
    return x * 2

# Step 3: Use it
result = double_distributed(21).get()
print(f"Result: {result}")
print("✓ That's all it takes!\n")

# ============================================================
# PART 2: REALISTIC EXAMPLE - DATA PROCESSING
# ============================================================

print("="*60)
print("PART 2: Process CSV Files")
print("="*60 + "\n")

@tron.remote
def process_csv(filename):
    """Process a CSV file."""
    # In real code:
    # import pandas as pd
    # df = pd.read_csv(filename)
    # df['doubled'] = df['value'] * 2
    # return df
    
    # For demo:
    return {"file": filename, "rows": 100, "processed": True}

# Process multiple files
files = ["file1.csv", "file2.csv", "file3.csv"]
print(f"Processing {len(files)} files...\n")

# Start all tasks
tasks = []
for file in files:
    task = process_csv(file)
    tasks.append(task)
    print(f"  Started: {file}")

print()

# Collect results
results = []
for file, task in zip(files, tasks):
    result = task.get()
    results.append(result)
    print(f"  Completed: {file} -> {result}")

print(f"\n✓ Processed {len(results)} files\n")

# ============================================================
# PART 3: REALISTIC EXAMPLE - ML INFERENCE
# ============================================================

print("="*60)
print("PART 3: ML Model Inference")
print("="*60 + "\n")

# Simulate a model
class FakeModel:
    def predict(self, x):
        return x * 2.5

model = FakeModel()

@tron.remote(gpu=True)
def predict_batch(batch):
    """Predict on a batch."""
    # In real code: return model.predict(batch)
    # For demo:
    return [model.predict(x) for x in batch]

# Process batches
batches = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

print("Running inference on batches...\n")

predictions = []
for batch in batches:
    result = predict_batch(batch).get()
    predictions.append(result)
    print(f"  Batch {batch} -> {result}")

print(f"\n✓ Inference complete\n")

# ============================================================
# PART 4: REALISTIC EXAMPLE - PIPELINE
# ============================================================

print("="*60)
print("PART 4: Multi-Step Pipeline")
print("="*60 + "\n")

@tron.remote
def extract():
    """Step 1: Extract data."""
    return {"raw_rows": 1000}

@tron.remote
def transform(data):
    """Step 2: Transform data."""
    data["cleaned_rows"] = data["raw_rows"] - 50  # remove nulls
    return data

@tron.remote
def load(data):
    """Step 3: Load data."""
    return {"status": "saved", "rows": data["cleaned_rows"]}

# Run pipeline
print("ETL Pipeline:\n")

print("1. Extract...", end=" ")
step1 = extract().get()
print(f"OK ({step1})")

print("2. Transform...", end=" ")
step2 = transform(step1).get()
print(f"OK ({step2})")

print("3. Load...", end=" ")
step3 = load(step2).get()
print(f"OK ({step3})")

print(f"\n✓ Pipeline complete\n")

# ============================================================
# PART 5: NOW YOU TRY
# ============================================================

print("="*60)
print("PART 5: Your Turn - Template")
print("="*60 + "\n")

print("""
Template for your own distributed function:

```python
import tron

@tron.remote
def your_function(input_data):
    # Write your logic here
    result = do_expensive_work(input_data)
    return result

# Use it
result = your_function(data).get()

# Or in parallel
tasks = [your_function(item) for item in items]
results = [t.get() for t in tasks]
```

What to build:
1. Image processing
2. Data validation
3. API calls
4. ML inference
5. File conversion
6. Report generation
7. Anything compute-heavy!
""")

# ============================================================
# SUMMARY
# ============================================================

print("="*60)
print("YOU LEARNED")
print("="*60)

print("""
✓ How to add @tron.remote to functions
✓ How to call distributed functions with .get()
✓ How to run multiple tasks in parallel
✓ How to build pipelines
✓ How to use GPU-accelerated tasks

✓ It works locally (no server needed)
✓ It scales to distributed (add server)
✓ No code changes between local and distributed

WHAT HAPPENS NEXT:

1. Start a server: python queue_server.py
2. Your code automatically uses it
3. Same results, but distributed
4. See jobs in dashboard
5. Scale to 1000s of workers

That's all. Build something amazing! 🚀
""")

print("="*60)
