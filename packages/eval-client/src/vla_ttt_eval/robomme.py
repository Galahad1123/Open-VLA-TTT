"""RoboMME joint-angle, task-goal-only, memoryless baseline adapter."""

from .observations import robomme_observation


class RoboMMEEnvironment:
    def __init__(self, task, split, max_steps):
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self.builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=split,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        self.max_steps = max_steps
        self.env = None

    def reset(self, episode, seed):
        # Official episode metadata fixes environment seeds; seed controls policy only.
        if not 0 <= episode < self.builder.get_episode_num():
            raise ValueError("RoboMME episode index out of range or metadata missing")
        self.close()
        self.env = self.builder.make_env_for_episode(episode_idx=episode, max_steps=self.max_steps)
        raw, info = self.env.reset()
        self.task = info["task_goal"][0]
        return robomme_observation(raw, self.task)

    def step(self, action):
        raw, _, terminated, truncated, info = self.env.step(action)
        if info.get("status") == "error":
            raise RuntimeError(f"RoboMME simulation error: {info}")
        success = info.get("status") == "success"
        done = bool(terminated) or bool(truncated) or info.get("status") in {"fail", "timeout"}
        return robomme_observation(raw, self.task), success, done

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
