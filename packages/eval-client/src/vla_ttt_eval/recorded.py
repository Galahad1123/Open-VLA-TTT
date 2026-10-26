"""Record execution and diagnostic demo video without changing policy inputs."""

from .observations import robomme_observation
from .video import Video


def base_env(env):
    node = env.env
    for _ in range(12):
        if "task_list" in vars(node):
            return node
        if hasattr(node, "env"):
            node = node.env
        elif hasattr(node, "unwrapped") and node.unwrapped is not node:
            node = node.unwrapped
        else:
            break
    raise RuntimeError("Cannot find task progress instrumentation")


class Recorded:
    def __init__(self, env, obs, path):
        self.env, self.obs, self.path = env, obs, path
        self.transitions = []
        self.step_index = 0
        self.base = base_env(env)
        self.task_names = [x.get("name", "unknown") for x in self.base.task_list]
        self.demo_path = None
        raw = env._reset_observations
        # Diagnostic recording only: never send baseline demo frames to policy.
        if len(raw["front_rgb_list"]) > 1:
            self.demo_path = path.with_name(path.stem + "-demo.mp4")
            demo = Video(self.demo_path, 2)
            try:
                for i in range(0, len(raw["front_rgb_list"]) - 1, 5):
                    demo.capture(robomme_observation(raw, env.task, index=i))
            finally:
                demo.finish()
        self.capture_progress()
        self.video = Video(path, 10)
        try:
            self.video.capture(obs)
        except BaseException:
            self.video.finish()
            raise

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self, episode, seed):
        # Scene was reset exactly once, before outcome-independent eligibility check.
        return self.obs

    def capture_progress(self):
        index = int(getattr(self.base, "timestep", -1))
        name = self.task_names[index] if 0 <= index < len(self.task_names) else "all completed"
        state = dict(step=self.step_index, index=index, name=name)
        if not self.transitions or index != self.transitions[-1]["index"]:
            self.transitions.append(state)

    def step(self, action):
        result = self.env.step(action)
        self.step_index += 1
        self.video.capture(result[0])
        self.capture_progress()
        return result
