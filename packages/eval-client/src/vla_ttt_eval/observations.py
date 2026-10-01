"""Benchmark-to-wire conventions, before policy normalization."""

import numpy as np


def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def libero_observation(raw, task):
    quat = numpy(raw["robot0_eef_quat"]).astype(np.float32)
    w = np.clip(quat[3], -1, 1)
    den = np.sqrt(max(0, 1 - w * w))
    axis = quat[:3] * (2 * np.arccos(w) / den) if den > 1e-10 else np.zeros(3)
    return {
        "images": {
            "image": numpy(raw["agentview_image"])[::-1, ::-1].copy(),
            "image2": numpy(raw["robot0_eye_in_hand_image"])[::-1, ::-1].copy(),
        },
        "state": np.concatenate((raw["robot0_eef_pos"], axis, raw["robot0_gripper_qpos"])).astype(
            np.float32
        ),
        "task": task,
    }


def robomme_observation(raw, task):
    # Explicit memoryless baseline: conditioning video/history is not consumed.
    return {
        "images": {
            "image": numpy(raw["front_rgb_list"][-1]),
            "image2": numpy(raw["wrist_rgb_list"][-1]),
        },
        "state": np.concatenate(
            (
                numpy(raw["joint_state_list"][-1]).reshape(-1)[:7],
                numpy(raw["gripper_state_list"][-1]).reshape(-1)[:1],
            )
        ).astype(np.float32),
        "task": task,
    }
