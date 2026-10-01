"""LIBERO adapter matching LeRobot 0.6.1's initial-state and camera conventions."""

from pathlib import Path

import numpy as np

from .observations import libero_observation

HORIZONS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class LiberoEnvironment:
    def __init__(self, suite, task_id):
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_suite = benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(int(task_id))
        self.task = task.language
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        # Official packaged, trusted initialization states only.
        self.initial_states = torch.load(
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
            weights_only=False,
        )

    def reset(self, episode, seed):
        if not 0 <= episode < len(self.initial_states):
            raise ValueError("LIBERO initial-state index out of range")
        self.env.seed(seed)
        self.env.reset()
        self.env.set_init_state(self.initial_states[episode])
        for _ in range(10):
            raw, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
        return libero_observation(raw, self.task)

    def step(self, action):
        raw, _, done, _ = self.env.step(action)
        success = bool(self.env.check_success())
        return libero_observation(raw, self.task), success, bool(done)

    def close(self):
        self.env.close()
