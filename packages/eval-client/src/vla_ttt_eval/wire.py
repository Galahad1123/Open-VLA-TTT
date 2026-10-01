"""Versioned binary transport; never deserialize pickle or object arrays."""

import math

import msgpack
import numpy as np

PROTOCOL = "vla-ttt/1"
MAX_BYTES = 32 * 1024 * 1024


def _dtype(value):
    dtype = np.dtype(value)
    if dtype.kind not in "buif" or dtype.itemsize > 8:
        raise ValueError("Only numeric/bool dtype arrays are allowed")
    return dtype


def _encode(value):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported wire value: {type(value).__name__}")
    dtype = _dtype(value.dtype)
    if value.nbytes > MAX_BYTES:
        raise ValueError("Array exceeds message size limit")
    return msgpack.ExtType(42, msgpack.packb([dtype.str, value.shape, value.tobytes()]))


def _decode(code, payload):
    if code != 42:
        raise ValueError("Unknown extension")
    dtype_name, shape, data = msgpack.unpackb(payload, raw=False)
    dtype = _dtype(dtype_name)
    if not isinstance(shape, list) or len(shape) > 8:
        raise ValueError("Invalid array rank")
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid array shape")
    if not isinstance(data, bytes) or math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError("Array byte length does not match shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def pack(value):
    data = msgpack.packb(value, default=_encode, use_bin_type=True)
    if len(data) > MAX_BYTES:
        raise ValueError("Message exceeds size limit")
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError("Expected bounded binary frame")
    return msgpack.unpackb(data, ext_hook=_decode, raw=False, strict_map_key=True)
