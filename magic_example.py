"""
TRON MAGIC DEMO - This is what "dramatically local" feels like.

No complexity. No polling loops. No server config.
Just write Python, and it runs faster.
"""

import tron

# =========================
# CONFIGURE TRON (Optional - auto-discovers)
# =========================
# tron.config("http://your-server:9000")


# =========================
# DEFINE WORK
# =========================

@tron.remote
def preprocess_data(size: int):
    """Preprocesses data - runs distributed if needed."""
    import time
    time.sleep(1)
    return {"processed": size * 2}


@tron.remote(gpu=True)
def train_model(data: dict):
    """GPU task - auto-scheduled to GPU workers."""
    import time
    time.sleep(3)
    return {"model": "trained", "accuracy": 0.95}


@tron.remote
def evaluate_model(model: dict):
    """Final step - stays distributed."""
    import time
    time.sleep(1)
    return {"result": model, "status": "complete"}


# =========================
# RUN IT - No ceremony, no complexity
# =========================

if __name__ == "__main__":
    print("\n🚀 TRON - Distributed Magic\n")

    # Single call - auto handles everything
    data = preprocess_data(1000)
    model = train_model(data.get())  # .get() blocks cleanly until done
    result = evaluate_model(model.get())

    # One final get() to see the result
    print("✅ Pipeline complete!")
    print(result.get())

    # OR use async/await syntax
    print("\n📡 Async example:")

    async def async_pipeline():
        data = preprocess_data(2000)
        model = train_model(await data)
        result = evaluate_model(await model)
        return await result

    # Run async (requires async runner)
    import asyncio
    final = asyncio.run(async_pipeline())
    print(final)
