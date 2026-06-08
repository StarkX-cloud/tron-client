import uuid

OBJECT_STORE = {}

def put(obj):

    object_id = str(
        uuid.uuid4()
    )

    OBJECT_STORE[
        object_id
    ] = obj

    return object_id

def get(object_id):

    return OBJECT_STORE.get(
        object_id
    )