import json
from pathlib import Path

import pytest
import torch

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.training.schedulers import CosineFloorSchedulerConfig
from lerobot_policy_vla_ttt.training.train_task_subset import ScreenRecipe, adapt_policy


def test_floor_schedule_exact_endpoints():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-5)
    schedule = CosineFloorSchedulerConfig(num_warmup_steps=300, min_lr_ratio=0.1).build(opt, 3000)
    fn = schedule.lr_lambdas[0]
    assert fn(0) == 0 and fn(300) == 1
    assert fn(3000) == pytest.approx(0.1)
    assert fn(4000) == pytest.approx(0.1)
    assert fn(1650) == pytest.approx(0.55)


def test_new_recipe_defaults_preserve_old_behavior():
    r = ScreenRecipe(episodes=[508])
    assert r.action_representation == "absolute" and r.augmentation == "none"
    assert r.min_lr_ratio == 0


def test_delta_requires_fresh_or_matching_checkpoint():
    source = Evo1GrootConfig(chunk_size=20, n_action_steps=10)
    r = ScreenRecipe(
        episodes=[508],
        action_representation="delta_joint",
        augmentation="official_v1",
        min_lr_ratio=0.1,
    )
    with pytest.raises(ValueError, match="fresh|representation"):
        adapt_policy(source, r, "old")
    config = adapt_policy(source, r, "old", fresh=True)
    assert config.robomme_action_representation == "delta_joint"
    assert config.normalization_mapping["ACTION"].value == "QUANTILES"
    assert config.scheduler_min_lr_ratio == 0.1


def test_official_budgets_and_same_task_selection():
    root = Path(__file__).resolve().parents[2] / "configs/training/robomme-screen"
    for stage, steps, batch, warmup, save in [
        ("stage1", 3000, 12, 300, 1000),
        ("stage2", 30000, 8, 1000, 5000),
    ]:
        a = json.loads((root / f"multitask-six-official-{stage}.json").read_text())
        old = json.loads((root / f"multitask-six-{stage}.json").read_text())
        r = ScreenRecipe(**a)
        assert (r.steps, r.batch_size, r.warmup_steps, r.save_every) == (steps, batch, warmup, save)
        assert r.episodes == old["episodes"] and r.task_groups == old["task_groups"]
        assert r.lr == 1e-5 and r.min_lr_ratio == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_lr_ratio", float("nan")),
        ("min_lr_ratio", 1.1),
        ("action_representation", "bad"),
        ("augmentation", "bad"),
    ],
)
def test_bad_protocol_fields(field, value):
    with pytest.raises(ValueError):
        ScreenRecipe(**dict(episodes=[508], **{field: value}))


def test_launcher_new_variant(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/train_robomme_six_tasks.py"
    spec = importlib.util.spec_from_file_location("six_launcher_official", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    machine = tmp_path / "machine.toml"
    machine.write_text(f'[storage]\nruns_root = "{tmp_path}/outputs"\n')
    plan = module.build_plan(machine, "new-run", "official")
    assert plan[0]["steps"] == 3000 and plan[1]["steps"] == 30000
    assert str(plan[0]["checkpoint"]).endswith("checkpoints/003000/pretrained_model")
    assert str(plan[0]["checkpoint"]) in plan[1]["command"]


def test_scheduler_registered_in_fresh_plugin_process():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import lerobot_policy_vla_ttt
import draccus
from lerobot.optim.schedulers import LRSchedulerConfig
cfg = dict(type='vla_ttt_cosine_floor', num_warmup_steps=300, min_lr_ratio=.1)
s = draccus.decode(LRSchedulerConfig, cfg)
assert s.min_lr_ratio == .1
""",
        ],
        check=True,
    )
