from tron import task


# =========================
# TASKS
# =========================

@task(
    gpu=False,
    memory=1
)
def preprocess():

    return "PREPROCESS COMPLETE"


@task(
    gpu=True,
    memory=8
)
def train():

    return "TRAINING COMPLETE"


@task(
    gpu=False,
    memory=1
)
def inference():

    return "INFERENCE COMPLETE"


# =========================
# DAG FLOW
# =========================

train.after(preprocess)

inference.after(train)


# =========================
# SUBMIT
# =========================

a = preprocess.submit()

b = train.submit()

c = inference.submit()


# =========================
# WAIT
# =========================

print(a.wait())

print(b.wait())

print(c.wait())