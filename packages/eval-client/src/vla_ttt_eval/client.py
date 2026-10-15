"""Synchronous, fail-fast client. Reconnect requires a new episode reset."""

from websockets.sync.client import connect

from .wire import MAX_BYTES, PROTOCOL, pack, unpack


class PolicyClient:
    def __init__(self, uri, *, benchmark, timeout=120):
        self.timeout = timeout
        self.connection = connect(
            uri, compression=None, max_size=MAX_BYTES, open_timeout=timeout, proxy=None
        )
        try:
            self.metadata = unpack(self.connection.recv(timeout=timeout))
            if self.metadata.get("protocol") != PROTOCOL:
                raise ValueError("Incompatible protocol or busy server")
            if self.metadata.get("benchmark") != benchmark:
                raise ValueError("Policy/benchmark mismatch")
        except BaseException:
            self.connection.close()
            raise
        self.request_id = 0
        self.episode_id = None

    def _request(self, kind, **fields):
        request = {"type": kind, "request_id": self.request_id, **fields}
        try:
            self.connection.send(pack(request))
            response = unpack(self.connection.recv(timeout=self.timeout))
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Policy server error"))
            if response.get("request_id") != self.request_id:
                raise RuntimeError("Response request_id mismatch")
            self.request_id += 1
            return response["data"]
        except BaseException:
            # A timed out inference may have advanced the policy; never retry it.
            self.close()
            raise

    def reset(self, episode_id, seed):
        self._request("reset", episode_id=episode_id, seed=seed)
        self.episode_id = episode_id

    def consume_demo(self, observation):
        if self.episode_id is None:
            raise ValueError("reset is required before demo")
        return self._request("demo", episode_id=self.episode_id, payload=observation)

    def infer(self, observation):
        if self.episode_id is None:
            raise ValueError("reset is required before infer")
        return self._request("infer", episode_id=self.episode_id, payload=observation)["action"]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
