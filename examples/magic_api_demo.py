from tron.remote import remote

import time

# =========================
# REMOTE FUNCTIONS
# =========================

@remote
def embeddings():

    print("GENERATING EMBEDDINGS...")
    time.sleep(3)

    return {
        "vectors": 4096
    }

@remote
def train_model():

    print("TRAINING MODEL...")
    time.sleep(5)

    return {
        "model": "TRON-TITAN-X"
    }

# =========================
# EXECUTION
# =========================

print("\n=== TRON MAGIC API ===\n")

a = embeddings()

b = train_model()

print("\nTASKS SENT\n")

print(a.get())

print(b.get())
