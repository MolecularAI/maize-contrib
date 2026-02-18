from typing import Any
import pickle, base64

def serialize(d: Any) -> str:
    """
    Serializes a Python object to a base64-encoded UTF-8 string.

    Parameters
    ----------
    d : Any
        The Python object to serialize. Must be compatible with `pickle`.

    Returns
    -------
    str
        The base64-encoded string representation of the object.
    """
    return base64.b64encode(pickle.dumps(d)).decode("utf-8")


def deserialize(s: str) -> Any:
    """
    Deserializes a base64-encoded UTF-8 string back to a Python object.

    Parameters
    ----------
    s : str
        The base64-encoded string to deserialize.

    Returns
    -------
    Any
        The deserialized Python object.
    """
    return pickle.loads(base64.b64decode(s.encode("utf-8")))