from tron.remote import remote
import time

print("\n=== TRON MAGIC REMOTE EXECUTION ===\n")


@remote
def generate_embeddings():

    time.sleep(4)

    return {
        "vectors": 4096
    }


@remote
def train_model():

    time.sleep(6)

    return {
        "model": "TRON-TITAN"
    }


# =========================
# FIRE BOTH TASKS
# =========================

embeddings = generate_embeddings()

training = train_model()

print("TASKS SENT TO TRON\n")

# =========================
# WAIT
# =========================

while not embeddings.done():
    time.sleep(1)

while not training.done():
    time.sleep(1)

print("EMBEDDINGS COMPLETE")
print("MODEL COMPLETE\n")

print(embeddings.result())
print(training.result())