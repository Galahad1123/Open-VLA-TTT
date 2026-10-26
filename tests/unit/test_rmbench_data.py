import h5py
import numpy as np
import pytest

from lerobot_policy_vla_ttt.baselines.rmbench_data import episode_frames


def test_legacy_next_state_targets(tmp_path):
    path = tmp_path / "episode0.hdf5"
    with h5py.File(path, "w") as f:
        f["joint_action/vector"] = np.arange(3 * 14).reshape(3, 14)
        for camera in ("head_camera", "left_camera", "right_camera"):
            f[f"observation/{camera}/rgb"] = np.stack(
                [np.full((240, 320, 3), i, np.uint8) for i in range(3)]
            )
    frames = list(episode_frames(path, "task", lambda x: x))
    assert len(frames) == 2
    np.testing.assert_equal(frames[0]["observation.state"], np.arange(14))
    np.testing.assert_equal(frames[0]["action"], np.arange(14, 28))
    assert frames[1]["observation.images.image"].max() == 1
    with pytest.raises(ValueError):
        list(episode_frames(path, "", lambda x: x))


def test_xpolicy_already_aligned(tmp_path):
    path = tmp_path / "episode.hdf5"
    fields = (
        "left_arm_joint_states",
        "left_ee_joint_states",
        "right_arm_joint_states",
        "right_ee_joint_states",
    )
    with h5py.File(path, "w") as f:
        for group, offset in [("state", 0), ("action", 10)]:
            for i, (name, width) in enumerate(zip(fields, (6, 1, 6, 1), strict=True)):
                f[f"{group}/{name}"] = np.full((2, width), i + offset, np.float32)
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            f[f"vision/{camera}/colors"] = np.zeros((2, 240, 320, 3), np.uint8)
    frames = list(episode_frames(path, "task", lambda x: x))
    assert len(frames) == 2
    assert frames[0]["action"][6] == 11
    assert frames[0]["action"][13] == 13
