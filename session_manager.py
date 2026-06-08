import uuid
import time


class SessionManager:

    def __init__(self):

        self.sessions = {}

    # =========================
    # CREATE SESSION
    # =========================

    def create_session(self):

        session_id = str(uuid.uuid4())

        self.sessions[session_id] = {
            "created_at": time.time(),
            "jobs": [],
            "state": {},
            "active": True
        }

        return session_id

    # =========================
    # ADD JOB
    # =========================

    def add_job(self, session_id, job):

        if session_id not in self.sessions:
            return False

        self.sessions[session_id]["jobs"].append(job)

        return True

    # =========================
    # GET SESSION
    # =========================

    def get(self, session_id):

        return self.sessions.get(session_id)

    # =========================
    # UPDATE STATE
    # =========================

    def update_state(
        self,
        session_id,
        key,
        value
    ):

        if session_id not in self.sessions:
            return False

        self.sessions[session_id]["state"][key] = value

        return True