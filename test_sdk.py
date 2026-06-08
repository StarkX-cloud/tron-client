import os
from tron_sdk import Tron

tron = Tron(os.getenv("TRON_URL", "http://127.0.0.1:9000"))

session = tron.session()
graph = tron.create_graph()

print("[SESSION]", session)
print("[GRAPH]", graph)

job = tron.run(
    "train model on images",
    session_id=session["session_id"],
    graph_id=graph["graph_id"]
)

job_id = job.job_id

tron.stream(job_id)