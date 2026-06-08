from tron.remote import remote
import time

print("\n=== TRON REMOTE FUNCTION DEMO ===\n")


@remote
def generate_embeddings():

    time.sleep(3)

    return {
        "embeddings": 2048
    }


@remote
def train_llm():

    time.sleep(6)

    return {
        "model": "TRON-GPT-X"
    }


# SUBMIT BOTH

emb = generate_embeddings.submit()

llm = train_llm.submit()


# WAIT

while not emb.done():
    time.sleep(1)

print("EMBEDDINGS READY")


while not llm.done():
    time.sleep(1)

print("MODEL READY")

print("\nRESULTS:\n")

print(emb.result())
print(llm.result())
