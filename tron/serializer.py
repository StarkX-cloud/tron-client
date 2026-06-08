import cloudpickle
import base64


def serialize(fn):
    return base64.b64encode(cloudpickle.dumps(fn)).decode()


def deserialize(blob):
    return cloudpickle.loads(base64.b64decode(blob))
