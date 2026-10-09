"""RoboMME joint-angle, task-goal-only, memoryless baseline adapter."""

from .observations import robomme_observation

# Pinned robomme_benchmark c85e060 scripts/evaluation.py (not Challenge's 1500).
MME_VLA_MAX_STEPS = 1300
TASK_SUITES = {
    "counting": ["PickXtimes", "StopCube", "SwingXtimes", "BinFill"],
    "permanence": ["VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask"],
    "reference": ["VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight"],
    "imitation": ["InsertPeg", "MoveCube", "PatternLock", "RouteStick"],
}
TASKS = [task for suite in TASK_SUITES.values() for task in suite]


class RoboMMEEnvironment:
    def __init__(self, task, split, max_steps):
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        if task not in TASKS:
            raise ValueError(f"Unknown RoboMME task: {task}")
        self.builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=split,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        self.max_steps = max_steps
        self.env = None
        self.last_status = None

    def reset(self, episode, seed):
        # Official episode metadata fixes environment seeds; seed controls policy only.
        if (self.builder.env_id, episode) not in self.builder.metadata_index:
            raise ValueError("RoboMME episode index out of range or metadata missing")
        self.env_seed, self.difficulty = self.builder.resolve_episode(episode)
        if self.env_seed is None:
            raise ValueError("Official RoboMME episode seed is missing")
        self.last_status = None
        self.close()
        self.env = self.builder.make_env_for_episode(episode_idx=episode, max_steps=self.max_steps)
        raw, info = self.env.reset()
        # The official wrapper adds two to max_steps and consumes one during
        # reset. Follow its remaining budget, not a second, shorter outer cap.
        self.rollout_step_limit = int(
            self.env.max_steps_without_demonstration - self.env.steps_without_demonstration
        )
        self.task = info["task_goal"][0]
        return robomme_observation(raw, self.task)

    def step(self, action):
        raw, _, terminated, truncated, info = self.env.step(action)
        if info.get("status") == "error":
            raise RuntimeError(f"RoboMME simulation error: {info}")
        success = info.get("status") == "success"
        self.last_status = "timeout" if bool(truncated) else info.get("status")
        done = bool(terminated) or bool(truncated) or info.get("status") in {"fail", "timeout"}
        return robomme_observation(raw, self.task), success, done

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
