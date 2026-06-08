import uuid
import time


class ExecutionGraph:

    def __init__(self):

        self.graphs = {}

    # =========================
    # CREATE GRAPH
    # =========================

    def create_graph(self):

        graph_id = str(uuid.uuid4())

        self.graphs[graph_id] = {
            "created_at": time.time(),
            "nodes": {},
            "edges": []
        }

        return graph_id

    # =========================
    # ADD NODE
    # =========================

    def add_node(
        self,
        graph_id,
        job
    ):

        node_id = job["id"]

        self.graphs[graph_id]["nodes"][node_id] = {
            "job": job,
            "status": "queued"
        }

        return node_id

    # =========================
    # ADD EDGE
    # =========================

    def add_edge(
        self,
        graph_id,
        parent,
        child
    ):

        self.graphs[graph_id]["edges"].append({
            "from": parent,
            "to": child
        })

    # =========================
    # UPDATE NODE
    # =========================

    def update_status(
        self,
        graph_id,
        node_id,
        status
    ):

        if node_id in self.graphs[graph_id]["nodes"]:

            self.graphs[graph_id]["nodes"][node_id]["status"] = status

    # =========================
    # GET GRAPH
    # =========================

    def get(self, graph_id):

        return self.graphs.get(graph_id)