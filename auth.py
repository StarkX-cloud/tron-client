import uuid

api_keys = {}
usage_log = {}

def create_api_key(user_id):

    key = "sk_" + str(uuid.uuid4())

    api_keys[key] = {
        "user_id": user_id,
        "created_at": __import__("time").time(),
        "active": True
    }

    return key


def validate_key(key):

    return key in api_keys and api_keys[key]["active"]