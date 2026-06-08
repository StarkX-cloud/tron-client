import time


class VirtualMemory:

    def __init__(self):

        self.memory = {}

    # =========================
    # CREATE SPACE
    # =========================

    def create(self, session_id):

        self.memory[session_id] = {
            "created_at": time.time(),
            "context": {},
            "cache": {},
            "execution_graph": []
        }

    # =========================
    # STORE
    # =========================

    def store(
        self,
        session_id,
        key,
        value
    ):

        if session_id not in self.memory:
            self.create(session_id)

        self.memory[session_id]["context"][key] = value

    # =========================
    # LOAD
    # =========================

    def load(
        self,
        session_id,
        key
    ):

        if session_id not in self.memory:
            return None

        return self.memory[session_id]["context"].get(key)

    # =========================
    # CACHE
    # =========================

    def cache(
        self,
        session_id,
        key,
        value
    ):

        if session_id not in self.memory:
            self.create(session_id)

        self.memory[session_id]["cache"][key] = value

    # =========================
    # GRAPH
    # =========================

    def add_execution(
        self,
        session_id,
        execution
    ):

        if session_id not in self.memory:
            self.create(session_id)

        self.memory[session_id]["execution_graph"].append(
            execution
        )

    # =========================
    # FULL STATE
    # =========================

    def get(self, session_id):

        return self.memory.get(session_id)