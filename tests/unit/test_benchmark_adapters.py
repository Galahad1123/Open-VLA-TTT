import importlib

import numpy as np
import pytest


def test_libero_config_requires_local_assets_and_refuses_mismatched_existing_config(tmp_path):
    setup = importlib.import_module("vla_ttt_eval.libero_setup")
    assets = tmp_path / "assets"
    with pytest.raises(FileNotFoundError):
        setup.write_config(tmp_path / "config", tmp_path / "package", assets)
    for folder in (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
        "scenes",
    ):
        (assets / folder).mkdir(parents=True)
    path = setup.write_config(tmp_path / "config", tmp_path / "package", assets)
    assert path.is_file()
    assert setup.write_config(tmp_path / "config", tmp_path / "package", assets) == path
    with pytest.raises(ValueError):
        setup.write_config(tmp_path / "config", tmp_path / "another-package", assets)


def test_libero_images_rotated_once_and_xyzw_converted_to_axis_angle():
    adapters = importlib.import_module("vla_ttt_eval.observations")
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw = {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": [1, 2, 3],
        "robot0_eef_quat": [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        "robot0_gripper_qpos": [0.02, -0.02],
    }
    obs = adapters.libero_observation(raw, "pick")
    np.testing.assert_array_equal(obs["images"]["image"][0, 0], [15, 16, 17])
    np.testing.assert_allclose(obs["state"], [1, 2, 3, 0, 0, np.pi / 2, 0.02, -0.02])
    assert obs["task"] == "pick"


def test_robomme_latest_frame_joint_state_and_no_privileged_info():
    adapters = importlib.import_module("vla_ttt_eval.observations")
    raw = {
        "front_rgb_list": [np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)],
        "wrist_rgb_list": [np.zeros((2, 2, 3), np.uint8)],
        "joint_state_list": [np.arange(7)],
        "gripper_state_list": [np.array([0.03, 0.04])],
        "grounded_subgoal_online": "secret location",
    }
    obs = adapters.robomme_observation(raw, "remember")
    assert set(obs) == {"images", "state", "task"}
    np.testing.assert_array_equal(obs["images"]["image"], np.ones((2, 2, 3)))
    np.testing.assert_allclose(obs["state"], [0, 1, 2, 3, 4, 5, 6, 0.03])


def test_episode_rollout_stops_on_success_and_reports_timeout():
    runner = importlib.import_module("vla_ttt_eval.runner")

    class Environment:
        def __init__(self, succeeds):
            self.succeeds = succeeds

        def reset(self, episode, seed):
            self.n = 0
            return {"state": np.zeros(8)}

        def step(self, action):
            assert action.shape == (7,)
            self.n += 1
            return {"state": np.zeros(8)}, self.succeeds and self.n == 2, False

    class Remote:
        def reset(self, episode_id, seed):
            pass

        def infer(self, obs):
            return np.zeros(7, np.float32)

    result = runner.run_episode(Environment(True), Remote(), "task", 0, 5, 10)
    assert result["success"] is True and result["steps"] == 2
    result = runner.run_episode(Environment(False), Remote(), "task", 0, 5, 3)
    assert result["success"] is False and result["steps"] == 3
    assert result["termination"] == "timeout"
    with pytest.raises(ValueError):
        runner.run_episode(Environment(False), Remote(), "task", 0, 5, 0)
