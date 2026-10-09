# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Client side of the policy server (numpy + msgpack only). Messages: 8-byte big-endian length + msgpack."""

from __future__ import annotations

import socket
import struct
import time
from typing import Any

import msgpack
import numpy as np

_HEADER = struct.Struct(">Q")


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {"__ndarray__": True, "dtype": array.dtype.str, "shape": list(array.shape), "data": array.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _hook(obj: dict) -> Any:
    if obj.get("__ndarray__") is True:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def send(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = msgpack.packb(message, default=_default, use_bin_type=True)
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def recv(sock: socket.socket) -> dict[str, Any]:
    def exact(size: int) -> bytes:
        chunks = []
        while size:
            chunk = sock.recv(min(size, 1 << 20))
            if not chunk:
                raise ConnectionError("connection closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    (size,) = _HEADER.unpack(exact(_HEADER.size))
    return msgpack.unpackb(exact(size), object_hook=_hook, raw=False, strict_map_key=False)


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 18000, wait_seconds: float = 1800.0) -> None:
        deadline = time.monotonic() + wait_seconds  # the server may still be loading the model
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=30)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(5)
        self.sock.settimeout(None)
        self.metadata = recv(self.sock)

    def __call__(self, request: dict[str, Any]) -> np.ndarray:
        send(self.sock, request)
        reply = recv(self.sock)
        if reply.get("type") == "error":
            raise RuntimeError(f"policy server error: {reply.get('message')}")
        return reply["actions"]

    def close(self) -> None:
        self.sock.close()
