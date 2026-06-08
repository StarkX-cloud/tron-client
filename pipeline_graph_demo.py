import time

from tron.decorators import task
from tron.pipeline import Pipeline


# =========================
# TASKS
# =========================

@task
def preprocess():

    time.sleep(3)

    return "cleaned"


@task
def tokenize():

    time.sleep(4)

    return "tokens"


@task
def embeddings():

    time.sleep(5)

    return "vectors"


@task
def train():

    time.sleep(8)

    return "model"


@task
def inference():

    time.sleep(3)

    return "done"


# =========================
# PIPELINE
# =========================

pipe = Pipeline()

a = pipe.task(preprocess)

b = pipe.task(
    tokenize,
    depends=[a]
)

c = pipe.task(
    embeddings,
    depends=[a]
)

d = pipe.task(
    train,
    depends=[b, c]
)

e = pipe.task(
    inference,
    depends=[d]
)

pipe.run()