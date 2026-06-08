from tron import task
from tron.sdk import Tron

import time

tron = Tron()

# =========================
# TASKS
# =========================

@task()
def preprocess():

    time.sleep(3)

    print("PREPROCESS")

    return "PREPROCESS"


@task()
def tokenize():

    time.sleep(5)

    print("TOKENIZE")

    return "TOKENIZE"


@task()
def embeddings():

    time.sleep(5)

    print("EMBEDDINGS")

    return "EMBEDDINGS"


@task()
def train():

    time.sleep(3)

    print("TRAIN")

    return "TRAIN"


# =========================
# BUILD DAG
# =========================

a = tron.node(preprocess)

b = tron.node(tokenize)

c = tron.node(embeddings)

d = tron.node(train)

# dependencies
tron.connect(a, b)
tron.connect(a, c)

tron.connect(b, d)
tron.connect(c, d)

# =========================
# RUN DAG
# =========================

futures = tron.run()

# =========================
# RESULTS
# =========================

for f in futures:

    print(f.result())