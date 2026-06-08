from tron import task
import time


# =========================
# STAGE 1
# =========================

@task
def preprocess():

    print("PREPROCESSING DATA...")
    time.sleep(3)

    return {
        "dataset": "clean_data"
    }


# =========================
# STAGE 2
# =========================

@task
def tokenize():

    print("TOKENIZING...")
    time.sleep(4)

    return {
        "tokens": 50000
    }


# =========================
# STAGE 3
# =========================

@task
def embeddings():

    print("GENERATING EMBEDDINGS...")
    time.sleep(5)

    return {
        "vectors": 2048
    }


# =========================
# STAGE 4
# =========================

@task
def train():

    print("TRAINING MODEL...")
    time.sleep(8)

    return {
        "model": "TRON-GPT"
    }


# =========================
# STAGE 5
# =========================

@task
def inference():

    print("RUNNING INFERENCE...")
    time.sleep(3)

    return {
        "output": "Distributed AI pipeline complete"
    }


# =========================
# EXECUTION GRAPH
# =========================

print("\n=== TRON DISTRIBUTED AI PIPELINE ===\n")


# PREPROCESS
p = preprocess.submit()

while not p.done():
    time.sleep(1)

print("PREPROCESS COMPLETE")


# PARALLEL EXECUTION
t = tokenize.submit()
e = embeddings.submit()

while not t.done() or not e.done():
    time.sleep(1)

print("TOKENIZATION COMPLETE")
print("EMBEDDINGS COMPLETE")


# TRAIN
tr = train.submit()

while not tr.done():
    time.sleep(1)

print("TRAINING COMPLETE")


# INFERENCE
i = inference.submit()

while not i.done():
    time.sleep(1)

print("INFERENCE COMPLETE")


print("\n=== PIPELINE FINISHED ===\n")

print(i.result())