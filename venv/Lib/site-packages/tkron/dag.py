# =========================
# DAG NODE
# =========================

class DAGNode:

    def __init__(self, future):

        self.future = future

        self.children = []

        self.parents = []

        self.job_id = None


# =========================
# DAG BUILDER
# =========================

class DAG:

    def __init__(self):

        self.nodes = []

    # =========================
    # ADD EDGE
    # =========================

    def connect(self, parent, child):

        parent.children.append(child)

        child.parents.append(parent)

    # =========================
    # BUILD EXECUTION STAGES
    # =========================

    def build_stages(self):

        stages = []

        remaining = set(self.nodes)

        while remaining:

            ready = []

            for node in remaining:

                deps_done = all(
                    p not in remaining
                    for p in node.parents
                )

                if deps_done:

                    ready.append(node)

            stages.append(ready)

            for r in ready:

                remaining.remove(r)

        return stages