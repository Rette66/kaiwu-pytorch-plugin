"""Transfer NumPy arrays between the model and Kaiwu solver processes."""

import io

import numpy as np


def send_array(stream, array: np.ndarray) -> None:
    """Write one length-prefixed NumPy array."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    data = buffer.getvalue()
    stream.write(len(data).to_bytes(8, "big"))
    stream.write(data)
    stream.flush()


def receive_array(stream) -> np.ndarray:
    """Read one length-prefixed NumPy array."""
    header = stream.read(8)
    if len(header) != 8:
        raise EOFError("Solver process closed the stream")
    length = int.from_bytes(header, "big")
    data = stream.read(length)
    if len(data) != length:
        raise EOFError("Solver process returned an incomplete array")
    return np.load(io.BytesIO(data), allow_pickle=False)
