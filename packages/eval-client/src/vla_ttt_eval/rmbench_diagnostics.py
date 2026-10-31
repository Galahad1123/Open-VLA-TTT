"""Read-only observations of the pinned RMBench scorer; never rescore a state."""

import json
import time

import numpy as np

from .rmbench import UPSTREAM_REVISION


class EpisodeDiagnostics:
    """Log every policy action and scorer transitions, restoring the scorer on exit."""

    FIELDS = (
        "stage_id",
        "press_cnt",
        "press_flag",
        "current_state_pointer",
        "fail_flag",
        "flag",
    )

    def __init__(self, env, path, *, task, episode, seed):
        self.env, self.path = env, path
        self.identity = dict(schema_version=1, task=task, episode=episode, seed=seed)
        self.frame = 0
        self.check_index = 0
        self.started = time.monotonic()

    def snapshot(self):
        env = self.env
        state = {
            "action_step": int(env.take_action_cnt),
            "step_limit": int(env.step_lim),
            "eval_success": bool(env.eval_success),
            "max_reward": np.asarray(env.max_reward).tolist(),
        }
        for name in self.FIELDS:
            if hasattr(env, name):
                state[name] = np.asarray(getattr(env, name)).tolist()
        if hasattr(env, "robot"):
            for side in ("left", "right"):
                name = side + "_gripper_val"
                if hasattr(env.robot, name):
                    state[name] = float(getattr(env.robot, name))
        if self.identity["task"] == "rearrange_blocks":
            b1, b2 = (np.asarray(b.get_pose().p) for b in (env.block1, env.block2))
            final = np.asarray(env.final_occupy_mat.get_pose().p)
            mid = np.asarray(env.mid_mat.get_pose().p)
            center = np.array([0.13, env.block_y_lim])
            errors = {
                "block1_to_final_xy": np.abs(b1[:2] - final[:2]).tolist(),
                "block2_to_original_xy": np.abs(b2[:2] - mid[:2]).tolist(),
                "block2_to_center_xy": np.abs(b2[:2] - center).tolist(),
            }
            button = float(env.get_current_button_value("button"))
            state["rearrange"] = {
                "block1_xyz": b1.tolist(),
                "block2_xyz": b2.tolist(),
                "final_mat_xyz": final.tolist(),
                "original_mat_xyz": mid.tolist(),
                "center_xy": center.tolist(),
                "button_joint": button,
                "position_errors": errors,
                # Instantaneous predicates, not a second implementation of the state machine.
                "conditions": {
                    "block1_at_final": all(v < 0.03 for v in errors["block1_to_final_xy"]),
                    "block2_at_original": all(v < 0.03 for v in errors["block2_to_original_xy"]),
                    "block2_at_center": all(v < 0.03 for v in errors["block2_to_center_xy"]),
                    "both_blocks_low": bool(b1[2] < 0.77 and b2[2] < 0.77),
                    "right_gripper_open": bool(env.robot.right_gripper_val > 0.8),
                    "button_below_threshold": button < -0.005,
                    "button_above_reset": button > -0.001,
                    "exactly_one_press": env.press_cnt == 1,
                },
            }
        elif self.identity["task"] == "put_back_block":
            pos = np.asarray(env.block.get_pose().p)
            target = np.asarray(env.target_pose)
            center = np.asarray(env.center_pose[:3])
            center_error = np.abs(pos[:2] - center[:2])
            target_error = np.abs(pos[:2] - target[:2])
            state["put_back_block"] = {
                "block_xyz": pos.tolist(),
                "original_xyz": target.tolist(),
                "center_xyz": center.tolist(),
                "center_error_xy": center_error.tolist(),
                "original_error_xy": target_error.tolist(),
                "button_joint": float(env.get_current_button_value("button")),
                "conditions": {
                    "block_at_center": bool(np.all(center_error < 0.04) and pos[2] < 0.77),
                    "block_at_original": bool(np.all(target_error < 0.03) and pos[2] < 0.77),
                    "right_gripper_open": bool(env.robot.right_gripper_val > 0.8),
                    "exactly_one_press": bool(env.press_cnt == 1),
                    "button_active": bool(env.press_flag),
                },
            }
        elif self.identity["task"] == "swap_blocks":
            b1, b2 = (np.asarray(b.get_pose().p) for b in (env.block1, env.block2))
            baskets = [np.asarray(b.get_pose().p) for b in env.baskets]
            target1 = baskets[env.with_block_basket_idx_2]
            target2 = baskets[env.with_block_basket_idx_1]
            error1, error2 = np.abs(b1[:2] - target1[:2]), np.abs(b2[:2] - target2[:2])
            state["swap_blocks"] = {
                "block1_xyz": b1.tolist(),
                "block2_xyz": b2.tolist(),
                "basket_xyz": [b.tolist() for b in baskets],
                "block1_target_xyz": target1.tolist(),
                "block2_target_xyz": target2.tolist(),
                "original_basket_indices": [
                    int(env.with_block_basket_idx_1),
                    int(env.with_block_basket_idx_2),
                ],
                "empty_basket_index": int(env.empty_basket_idx),
                "block1_error_xy": error1.tolist(),
                "block2_error_xy": error2.tolist(),
                "button_joint": float(env.get_current_button_value("button", env.button)),
                "conditions": {
                    "blocks_at_swapped_targets": bool(
                        np.all(error1 < [0.04, 0.06])
                        and np.all(error2 < [0.04, 0.06])
                        and b1[2] < 0.765
                        and b2[2] < 0.765
                    ),
                    "right_gripper_open": bool(env.robot.right_gripper_val > 0.8),
                    "exactly_one_press": bool(env.press_cnt == 1),
                    "button_active": bool(env.press_flag),
                    "valid_history": bool(env.flag),
                    # Mirror the pinned numeric test; do NOT call is_two_blocks (mutates flag).
                    "invalid_proximity_now": bool(
                        abs(b1[0] - b2[0]) < 0.03 and abs(b1[2] - b2[2]) < 0.01
                    ),
                },
            }
        return state

    def thresholds(self):
        common = dict(button_press_m=-0.005, button_reset_m=-0.001, right_gripper_open=0.8)
        task = self.identity["task"]
        if task == "rearrange_blocks":
            return dict(common, xy_tolerance_m=0.03, block_height_m=0.77)
        if task == "put_back_block":
            return dict(
                common,
                center_xy_tolerance_m=0.04,
                original_xy_tolerance_m=0.03,
                block_height_m=0.77,
            )
        if task == "swap_blocks":
            return dict(
                common,
                target_xy_tolerance_m=[0.04, 0.06],
                block_height_m=0.765,
                invalid_proximity_xz_m=[0.03, 0.01],
            )
        return None

    def write(self, event, **values):
        self.stream.write(
            json.dumps(
                {
                    **self.identity,
                    "event": event,
                    "elapsed_seconds": time.monotonic() - self.started,
                    **values,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.stream.flush()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x")
        self.original = self.env.check_success
        self.had_override = "check_success" in vars(self.env)
        self.override = vars(self.env).get("check_success")
        try:
            self.write(
                "start",
                upstream_revision=UPSTREAM_REVISION,
                video_frame=0,
                state=self.snapshot(),
                thresholds=self.thresholds(),
            )
            self.env.check_success = self._check
        except BaseException:
            self.stream.close()
            raise
        return self

    def _check(self, *args, **kwargs):
        before = self.snapshot()
        result = self.original(*args, **kwargs)  # Exactly one original call.
        self.check_index += 1
        after = self.snapshot()
        keys = (*self.FIELDS, "eval_success", "max_reward")
        if result or any(before.get(k) != after.get(k) for k in keys):
            self.write(
                "transition",
                check_index=self.check_index,
                previous_video_frame=self.frame,
                next_video_frame=self.frame + 1,
                scorer_result=bool(result),
                before=before,
                after=after,
            )
        return result

    def action(self, action, observed_state, *, frame):
        self.frame = frame
        self.write(
            "action",
            video_frame=frame,
            video_time_seconds=frame / 10,
            action=np.asarray(action).tolist(),
            observed_state=np.asarray(observed_state).tolist(),
            state=self.snapshot(),
        )

    def __exit__(self, exc_type, exc, tb):
        if self.had_override:
            self.env.check_success = self.override
        else:
            del self.env.check_success
        try:
            self.write(
                "error" if exc is not None else "end",
                video_frame=self.frame,
                state=self.snapshot(),
                error=repr(exc) if exc is not None else None,
            )
        finally:
            self.stream.close()
