import json
from types import SimpleNamespace

import numpy as np
import pytest
from vla_ttt_eval.rmbench_diagnostics import EpisodeDiagnostics


class Env:
    take_action_cnt = 0
    step_lim = 10
    max_reward = 0
    eval_success = False
    stage_id = 0
    press_cnt = 0
    press_flag = False
    calls = 0

    def check_success(self):
        self.calls += 1
        self.press_cnt += 1
        self.stage_id = 1
        return False


def test_observer_preserves_call_results_and_restores_on_error(tmp_path):
    env = Env()
    original = env.check_success
    path = tmp_path / "episode.jsonl"
    with pytest.raises(RuntimeError, match="policy failed"):
        with EpisodeDiagnostics(env, path, task="other", episode=0, seed=100000) as log:
            for _ in range(2):
                assert env.check_success() is False
            log.action(np.zeros(14), np.zeros(14), frame=1)
            raise RuntimeError("policy failed")
    assert env.calls == 2
    assert env.check_success == original
    assert "check_success" not in vars(env)
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "transition", "transition", "action", "error"]
    assert rows[1]["before"]["press_cnt"] == 0
    assert rows[1]["after"]["press_cnt"] == 1
    assert rows[-1]["error"] == "RuntimeError('policy failed')"
    assert rows[-2]["video_frame"] == 1


def test_rearrange_snapshot_exposes_near_miss_without_calling_scorer(tmp_path):
    env = Env()

    def actor(p):
        return SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array(p)))

    env.block1 = actor([0.02, -0.1, 0.76])
    env.block2 = actor([0.13, -0.1, 0.76])
    env.final_occupy_mat = actor([0.02, -0.1, 0.7415])
    env.mid_mat = actor([0.26, -0.1, 0.7415])
    env.block_y_lim = -0.1
    env.robot = SimpleNamespace(right_gripper_val=0.8, left_gripper_val=1.0)
    env.get_current_button_value = lambda name: np.float32(-0.006)
    with EpisodeDiagnostics(
        env, tmp_path / "d.jsonl", task="rearrange_blocks", episode=2, seed=100002
    ) as log:
        state = log.snapshot()
        assert state["rearrange"]["conditions"]["right_gripper_open"] is False
        assert state["rearrange"]["conditions"]["button_below_threshold"] is True
        assert state["rearrange"]["conditions"]["block2_at_center"] is True
        assert env.calls == 0
    assert env.calls == 0


def test_runner_enabled_disabled_equivalence(tmp_path, monkeypatch):
    from contextlib import nullcontext

    from vla_ttt_eval import rmbench_runner as runner

    class Simulation(Env):
        def setup_demo(self, **kwargs):
            self.take_action_cnt = 0
            self.eval_success = False
            self.plan_success = True
            self.max_reward = 0
            self.stage_id = self.press_cnt = 0
            self.press_flag = False
            self.calls = 0

        def play_once(self):
            self.eval_success = True
            return {"info": {}}

        def close_env(self, **kwargs):
            pass

        def check_success(self):
            self.calls += 1
            self.stage_id = self.take_action_cnt
            return self.eval_success or self.take_action_cnt >= 2

        def set_instruction(self, **kwargs):
            pass

        def get_obs(self):
            return {
                "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
                "observation": {
                    c: {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}
                    for c in ("head_camera", "left_camera", "right_camera")
                },
            }

        def take_action(self, action, **kwargs):
            self.take_action_cnt += 1
            self.eval_success = self.check_success()

    client = SimpleNamespace(reset=lambda *a: None, infer=lambda obs: np.zeros(14))
    monkeypatch.setattr(
        runner, "H264Video", lambda *a: nullcontext(SimpleNamespace(write=lambda f: None))
    )
    results = []
    for enabled in (False, True):
        out = tmp_path / str(enabled)
        out.mkdir()
        env = Simulation()
        runner.evaluate(
            "swap_T",
            env,
            {"clear_cache_freq": 10},
            client,
            out,
            seed=0,
            episodes=1,
            max_candidates=1,
            describe=lambda *a: [{"seen": ["test instruction"]}],
            diagnostics=enabled,
        )
        results.append((env.calls, json.loads((out / "episodes.jsonl").read_text())))
    assert results[0][0] == results[1][0] == 2
    for key in ("success", "steps", "seed", "max_reward", "instruction"):
        assert results[0][1][key] == results[1][1][key]
    path = results[1][1]["diagnostics"]
    with open(path) as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-1]["event"] == "end"
    assert rows[-1]["state"]["eval_success"] is True
    assert [r["video_frame"] for r in rows if r["event"] == "action"] == [1, 2]
    assert not (tmp_path / "False" / "diagnostics").exists()


def actor_at(x, y, z):
    return SimpleNamespace(get_pose=lambda: SimpleNamespace(p=np.array([x, y, z])))


@pytest.mark.parametrize("task", ["put_back_block", "swap_blocks"])
def test_additional_tasks_snapshot_and_transient_invalid_flag(tmp_path, task):
    env = Env()
    env.robot = SimpleNamespace(right_gripper_val=0.9, left_gripper_val=0.1)
    if task == "put_back_block":
        env.block = actor_at(0.1, -0.1, 0.76)
        env.center_pose = [0.1, -0.1, 0.765, 1, 0, 0, 0]
        env.target_pose = [0.3, -0.1, 0.7415]
        env.get_current_button_value = lambda name: -0.006
    else:
        env.flag = True
        env.button = object()
        env.with_block_basket_idx_1, env.with_block_basket_idx_2 = 0, 1
        env.empty_basket_idx = 2
        env.baskets = [actor_at(x, 0, 0.74) for x in (-0.2, 0.2, 0)]
        env.block1, env.block2 = actor_at(0.2, 0, 0.76), actor_at(-0.2, 0, 0.76)
        env.get_current_button_value = lambda name, button: -0.006

        def invalidate():
            env.flag = False
            return False

        env.check_success = invalidate
    with EpisodeDiagnostics(env, tmp_path / "d.jsonl", task=task, episode=0, seed=0) as log:
        snap = log.snapshot()[task]
        if task == "put_back_block":
            assert snap["conditions"]["block_at_center"]
            assert not snap["conditions"]["block_at_original"]
        else:
            assert snap["conditions"]["blocks_at_swapped_targets"]
            assert snap["conditions"]["valid_history"]
            assert not snap["conditions"]["invalid_proximity_now"]
            env.check_success()
    if task == "swap_blocks":
        assert env.check_success is invalidate
        rows = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text().splitlines()]
        event = next(r for r in rows if r["event"] == "transition")
        assert event["before"]["flag"] is True
        assert event["after"]["flag"] is False
