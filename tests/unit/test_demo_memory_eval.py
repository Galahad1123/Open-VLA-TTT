import numpy as np
import pytest
import torch


def test_demo_writes_keep_execution_clock_and_slow_weights(monkeypatch):
    from observation_memory.assets import patch_assets
    from observation_memory.test_policy import policy_config

    from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig
    from lerobot_policy_vla_ttt.models.t2mem.modeling_t2mem import T2MemPolicy

    patch_assets(monkeypatch)
    p = T2MemPolicy(policy_config(T2MemConfig, chunk_size=6, n_action_steps=4)).eval()
    b = {
        "observation.state": torch.zeros(1, 4),
        "observation.images.image": torch.zeros(1, 3, 16, 16),
        "task": ["move"],
    }
    before = {k: v.clone() for k, v in p.state_dict().items()}
    rng = torch.get_rng_state().clone()
    p.consume_demo(b)
    p.consume_demo(b)
    assert torch.equal(rng, torch.get_rng_state())
    assert p.clock.last == -1 and p._chunk is None
    assert p.inference_state[0].update_count.tolist() == [2]
    p.select_action(b)
    assert p.inference_state[0].update_count.tolist() == [3]
    assert p._last_action_step == 0
    with pytest.raises(ValueError, match="before"):
        p.consume_demo(b)
    for key, value in p.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    p.reset()
    assert p.inference_state is None


@pytest.mark.parametrize("length, expected", [(1, []), (11, [0, 5]), (12, [0, 5, 10])])
def test_demo_prefix_excludes_execution_frame(length, expected):
    from vla_ttt_eval.robomme import RoboMMEEnvironment

    env = RoboMMEEnvironment.__new__(RoboMMEEnvironment)
    env.task = "move"
    env._reset_observations = {
        "front_rgb_list": [np.full((256, 256, 3), i, np.uint8) for i in range(length)],
        "wrist_rgb_list": [np.zeros((256, 256, 3), np.uint8)] * length,
        "joint_state_list": [np.ones(7) * i for i in range(length)],
        "gripper_state_list": [np.zeros(2)] * length,
    }
    frames = list(env.demo_observations(5))
    assert [x["images"]["image"][0, 0, 0] for x in frames] == expected
    assert env._reset_observations is None
    assert all(set(f) == {"images", "state", "task"} for f in frames)


def test_demo_rpc_requires_reset_and_rejects_after_inference():
    from lerobot_policy_vla_ttt.evaluation.server import Session

    class Policy:
        metadata = {"action_dim": 8, "demonstration_mode": "write_only_prefix"}

        def reset(self, seed):
            self.writes = 0

        def consume_demo(self, payload):
            self.writes += 1

        def infer(self, payload):
            return np.ones(8) * self.writes

    p = Policy()
    s = Session(p)
    with pytest.raises(ValueError, match="reset"):
        s.handle({"type": "demo", "request_id": 0, "episode_id": "a", "payload": {}})
    s.handle({"type": "reset", "request_id": 0, "episode_id": "a", "seed": 1})
    s.handle({"type": "demo", "request_id": 1, "episode_id": "a", "payload": {}})
    result = s.handle({"type": "infer", "request_id": 2, "episode_id": "a", "payload": {}})
    np.testing.assert_equal(result["action"], np.ones(8))
    with pytest.raises(ValueError, match="before"):
        s.handle({"type": "demo", "request_id": 3, "episode_id": "a", "payload": {}})


def test_runner_sends_demo_between_reset_and_first_action():
    from vla_ttt_eval.runner import run_episode

    calls = []

    class Env:
        def reset(self, episode, seed):
            return "execution"

        def demo_observations(self, stride):
            assert stride == 5
            yield "demo0"
            yield "demo5"

        def step(self, action):
            return "next", True, True

    class Client:
        metadata = {"demonstration_mode": "write_only_prefix", "demo_stride": 5}

        def reset(self, *args):
            calls.append("reset")

        def consume_demo(self, frame):
            calls.append(frame)

        def infer(self, frame):
            calls.append(frame)
            return np.zeros(8)

    result = run_episode(Env(), Client(), "RouteStick", 0, 1000, 1300)
    assert calls == ["reset", "demo0", "demo5", "execution"]
    assert result["demo_writes"] == 2 and result["success"]


def test_real_demo_rpc():
    from threading import Thread

    from vla_ttt_eval.client import PolicyClient

    from lerobot_policy_vla_ttt.evaluation.server import make_server

    class Policy:
        metadata = {
            "benchmark": "robomme",
            "action_dim": 8,
            "demonstration_mode": "write_only_prefix",
            "demo_stride": 5,
        }

        def reset(self, seed):
            self.count = 0

        def consume_demo(self, obs):
            self.count += 1

        def infer(self, obs):
            return np.full(8, self.count, dtype=np.float32)

    with make_server(Policy(), "127.0.0.1", 0) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with PolicyClient(
                f"ws://127.0.0.1:{server.socket.getsockname()[1]}", benchmark="robomme"
            ) as client:
                client.reset("a", 1)
                client.consume_demo({})
                client.consume_demo({})
                np.testing.assert_equal(client.infer({}), np.full(8, 2))
                client.reset("b", 2)
                np.testing.assert_equal(client.infer({}), np.zeros(8))
        finally:
            server.shutdown()
            thread.join(timeout=5)
