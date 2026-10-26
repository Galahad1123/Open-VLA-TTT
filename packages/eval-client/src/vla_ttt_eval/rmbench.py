"""RMBench's ALOHA-AgileX joint-control contract (no policy dependencies)."""

import numpy as np

UPSTREAM_REVISION = "7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7"
DATA_REVISION = "5d8ff21038fba32ea549e9eee8497aaba9aab5cf"
TASKS = (
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
)
CAMERAS = dict(
    zip(("head_camera", "left_camera", "right_camera"), ("image", "image2", "image3"), strict=True)
)


def resolve_tasks(names):
    if names == ["all"]:
        return list(TASKS)
    if not names or len(names) != len(set(names)) or any(n not in TASKS for n in names):
        raise ValueError("Expected unique paper task names, or all")
    return list(names)


def validate_action(action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("Expected finite 14D absolute joint targets")
    return action


def canonical_observation(raw, instruction):
    state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("RMBench requires finite 14D ALOHA joint state")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Expected task-level instruction")
    images = {}
    for source, target in CAMERAS.items():
        value = raw["observation"][source]["rgb"]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.shape != (240, 320, 3)
        ):
            raise ValueError(f"Expected uint8 RGB 240x320 camera {source}")
        images[target] = value
    return {"state": state, "images": images, "task": instruction}
