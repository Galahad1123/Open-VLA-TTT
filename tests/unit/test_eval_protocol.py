import importlib

import numpy as np
import pytest


def test_numeric_wire_preserves_noncontiguous_arrays_and_rejects_objects():
    wire = importlib.import_module("vla_ttt_eval.wire")
    value = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[::-1, ::-1]
    decoded = wire.unpack(wire.pack({"image": value, "task": "移动方块"}))
    np.testing.assert_array_equal(decoded["image"], value)
    assert decoded["image"].dtype == np.uint8
    assert decoded["task"] == "移动方块"
    with pytest.raises(ValueError, match="dtype"):
        wire.pack(np.array([object()], dtype=object))


def test_wire_rejects_malformed_array_and_text_frames():
    import msgpack

    wire = importlib.import_module("vla_ttt_eval.wire")
    malformed = msgpack.ExtType(42, msgpack.packb(["<f4", [10], b"1234"]))
    with pytest.raises(ValueError):
        wire.unpack(msgpack.packb(malformed))
    with pytest.raises(ValueError):
        wire.unpack("not a binary frame")


def test_session_enforces_reset_episode_sequence_and_action_shape():
    server = importlib.import_module("lerobot_policy_vla_ttt.evaluation.server")

    class Policy:
        metadata = {"benchmark": "libero", "action_dim": 7, "state_dim": 8}

        def reset(self, seed):
            self.state = seed

        def infer(self, observation):
            self.state += 1
            return np.full(7, self.state, dtype=np.float32)

    session = server.Session(Policy())
    with pytest.raises(ValueError, match="reset"):
        session.handle({"type": "infer", "request_id": 0, "episode_id": "a", "payload": {}})
    session.handle({"type": "reset", "request_id": 0, "episode_id": "a", "seed": 5})
    request = {"type": "infer", "request_id": 1, "episode_id": "a", "payload": {}}
    result = session.handle(request)
    np.testing.assert_array_equal(result["action"], np.full(7, 6))
    with pytest.raises(ValueError, match="request_id"):
        session.handle(request)
    with pytest.raises(ValueError, match="episode"):
        session.handle({**request, "request_id": 2, "episode_id": "b"})
    session.handle({"type": "reset", "request_id": 2, "episode_id": "b", "seed": 5})
    result = session.handle({**request, "request_id": 3, "episode_id": "b"})
    np.testing.assert_array_equal(result["action"], np.full(7, 6))


def test_real_socket_client_reset_infer_and_contract_mismatch():
    from threading import Thread

    client = importlib.import_module("vla_ttt_eval.client")
    server = importlib.import_module("lerobot_policy_vla_ttt.evaluation.server")

    class Policy:
        metadata = {"benchmark": "libero", "action_dim": 7, "state_dim": 8}

        def reset(self, seed):
            self.seed = seed

        def infer(self, observation):
            return observation["state"][:7] + self.seed

    with server.make_server(Policy(), "127.0.0.1", 0) as service:
        thread = Thread(target=service.serve_forever, daemon=True)
        thread.start()
        uri = f"ws://127.0.0.1:{service.socket.getsockname()[1]}"
        try:
            with client.PolicyClient(uri, benchmark="libero") as remote:
                remote.reset("task/0", seed=3)
                result = remote.infer({"state": np.arange(8, dtype=np.float32)})
                np.testing.assert_array_equal(result, [3, 4, 5, 6, 7, 8, 9])
            with pytest.raises(ValueError, match="benchmark"):
                client.PolicyClient(uri, benchmark="robomme")
        finally:
            service.shutdown()
            thread.join(timeout=5)
