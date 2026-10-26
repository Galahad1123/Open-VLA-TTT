from types import SimpleNamespace as NS

import pytest
from vla_ttt_eval.subset import eligible, load_plan, summarize


@pytest.mark.parametrize(
    "task,env",
    [
        ("PickXtimes", NS(num_repeats=1)),
        ("SwingXtimes", NS(num_repeats=1)),
        ("VideoUnmask", NS(configs={"easy": {"pick": 1}}, difficulty="easy")),
        ("ButtonUnmask", NS(configs={"easy": {"pick": 1}}, difficulty="easy")),
        ("PickHighlight", NS(configs={"easy": {"pickup": 1}}, difficulty="easy")),
        (
            "BinFill",
            NS(red_cubes_target_number=1, blue_cubes_target_number=0, green_cubes_target_number=0),
        ),
    ],
)
def test_matching(task, env):
    assert eligible(task, env)


def test_multiple_repeats_excluded():
    assert not eligible("PickXtimes", NS(num_repeats=2))


def test_unknown_fails():
    with pytest.raises(ValueError):
        eligible("Unknown", NS())


def test_summary_excludes_errors_and_exclusions():
    s = summarize(
        [
            dict(type="episode", task="PickXtimes", success=True),
            dict(type="excluded", task="PickXtimes"),
            dict(type="error", task="PickXtimes"),
        ]
    )
    assert s["episodes"] == 1 and s["successes"] == 1
    assert s["errors"] == 1 and s["excluded"] == 1


def test_plan_rejects_global_ids(tmp_path):
    import json

    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(version=1, tasks={"PickXtimes": {"train_episodes": [508]}})))
    with pytest.raises(ValueError):
        load_plan(path)


def test_real_encoder_and_rollout_no_privileged_input(tmp_path):
    import shutil

    import numpy as np
    from vla_ttt_eval.recorded import Recorded
    from vla_ttt_eval.runner import run_episode

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    obs = {
        "images": {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "image2": np.zeros((256, 256, 3), dtype=np.uint8),
        }
    }

    class Env:
        def __init__(self):
            self.env = NS(task_list=[{"name": "privileged target"}], timestep=0)
            self._reset_observations = {"front_rgb_list": [None]}
            self.rollout_step_limit = 2

        def reset(self, *args):
            raise AssertionError("must not reset already selected scene")

        def step(self, action):
            self.env.timestep = 1
            return obs, True, True

    class Client:
        metadata = {}

        def reset(self, *args):
            pass

        def infer(self, value):
            assert value is obs
            assert "task_names" not in value
            return np.zeros(8)

    rec = Recorded(Env(), obs, tmp_path / "episode.mp4")
    try:
        result = run_episode(rec, Client(), "PickXtimes", 0, 1000, 2)
    finally:
        rec.video.finish()
    assert result["success"] and result["demo_writes"] == 0
    assert rec.video.frames == 2 and rec.path.stat().st_size > 0
    assert [x["index"] for x in rec.transitions] == [0, 1]
    with pytest.raises(FileExistsError):
        Recorded(Env(), obs, rec.path)


def test_cli_filters_before_policy_and_finishes_report(tmp_path, monkeypatch):
    import json
    import sys

    import vla_ttt_eval.subset as mod

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(version=1, tasks={"PickXtimes": {"train_episodes": [0]}})))
    output = tmp_path / "out"

    class Env:
        def __init__(self, *args):
            self.env = NS(task_list=[], num_repeats=2)
            self.env_seed = 1
            self.difficulty = "easy"

        def reset(self, ep, seed):
            return {}

        def close(self):
            pass

    class Client:
        metadata = {"implementation": "evo1_groot"}

        def __init__(self, *args, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def infer(self, *args):
            raise AssertionError("Excluded scene must not query policy")

    monkeypatch.setattr(mod, "RoboMMEEnvironment", Env)
    monkeypatch.setattr(mod, "PolicyClient", Client)
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(sys, "argv", ["subset", "--plan", str(plan), "--output", str(output)])
    mod.main()
    s = json.loads((output / "summary.json").read_text())
    assert (
        s["complete"] and s["episodes"] == 0 and s["excluded"] == 50 and s["success_rate"] is None
    )
    with pytest.raises(FileExistsError):
        mod.main()
