import dataclasses

import pytest

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.training.train_task_subset import ScreenRecipe, adapt_policy


def test_init_comparison_uses_same_trainable_parameters_and_fresh_schedule():
    recipe = ScreenRecipe(episodes=[508, 512])
    configs = [
        adapt_policy(
            Evo1GrootConfig(training_stage=stage, chunk_size=20, n_action_steps=10),
            recipe,
            "/local/checkpoint",
        )
        for stage in ("stage1", "stage2")
    ]
    assert dataclasses.asdict(configs[0]) == dataclasses.asdict(configs[1])
    assert configs[0].training_stage == "stage2"
    assert configs[0].finetune_vlm and configs[0].finetune_action_head
    assert configs[0].scheduler_warmup_steps == recipe.warmup_steps
    assert configs[0].pretrained_path == "/local/checkpoint"


@pytest.mark.parametrize(
    "changes",
    [
        {"episodes": []},
        {"episodes": [1, 1]},
        {"episodes": [1600]},
        {"steps": 0},
        {"batch_size": 0},
        {"lr": float("nan")},
        {"warmup_steps": 2001},
        {"num_workers": -1},
    ],
)
def test_recipe_validation(changes):
    with pytest.raises(ValueError):
        ScreenRecipe(**({"episodes": [508]} | changes))


@pytest.mark.parametrize(
    "precision,dtype", [("fp16", "float16"), ("fp32", "float32"), ("bf16", "bfloat16")]
)
def test_explicit_precision_keeps_parameters_fp32(precision, dtype):
    c = adapt_policy(
        Evo1GrootConfig(chunk_size=20, n_action_steps=10),
        ScreenRecipe(episodes=[508]),
        "/checkpoint",
        precision=precision,
    )
    assert c.dtype == dtype and c.vlm_dtype == "float32"
    assert c.use_amp == (precision != "fp32")


def test_global_batch_is_divided_not_multiplied():
    from lerobot_policy_vla_ttt.training.train_task_subset import per_rank_batch

    assert per_rank_batch(12, 4) == 3
    assert per_rank_batch(12, 1) == 12
    for world in [0, 8]:
        with pytest.raises(ValueError):
            per_rank_batch(12, world)


def test_fp16_retry_does_not_advance_scheduler_or_rng():
    from types import SimpleNamespace

    import torch

    from lerobot_policy_vla_ttt.training.train_task_subset import successful_update

    acc = SimpleNamespace(optimizer_step_was_skipped=False)
    scheduler = SimpleNamespace(n=0)
    scheduler.step = lambda: setattr(scheduler, "n", scheduler.n + 1)
    samples = []

    def update(*args, accelerator, lr_scheduler, **kwargs):
        assert lr_scheduler is None
        samples.append(torch.rand(4))
        accelerator.optimizer_step_was_skipped = len(samples) == 1
        return SimpleNamespace(
            loss=SimpleNamespace(val=0.2),
            grad_norm=SimpleNamespace(val=float("inf") if len(samples) == 1 else 1.0),
        ), None

    tracker, _, skips = successful_update(update, accelerator=acc, lr_scheduler=scheduler)
    assert skips == 1 and scheduler.n == 1
    assert torch.equal(samples[0], samples[1])


def test_nonfinite_forward_is_not_retried():
    from types import SimpleNamespace

    from lerobot_policy_vla_ttt.training.train_task_subset import successful_update

    def update(**kwargs):
        return SimpleNamespace(
            loss=SimpleNamespace(val=float("nan")), grad_norm=SimpleNamespace(val=1.0)
        ), None

    with pytest.raises(RuntimeError, match="Nonfinite forward loss"):
        successful_update(update, accelerator=SimpleNamespace(optimizer_step_was_skipped=True))


def test_overflow_attempt_does_not_pollute_real_metric_tracker():
    from types import SimpleNamespace

    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    from lerobot_policy_vla_ttt.training.train_task_subset import successful_update

    tracker = MetricsTracker(
        12, 48, 2, {k: AverageMeter(k) for k in ["loss", "grad_norm", "update_s"]}
    )
    acc = SimpleNamespace(optimizer_step_was_skipped=False)
    attempts = []

    def update(metrics, accelerator, lr_scheduler):
        attempts.append(1)
        metrics.loss = 0.1
        metrics.grad_norm = float("inf") if len(attempts) == 1 else 2.0
        metrics.update_s = 0.01
        accelerator.optimizer_step_was_skipped = len(attempts) == 1
        return metrics, None

    result, _, skips = successful_update(update, tracker, accelerator=acc)
    assert result is tracker and skips == 1
    assert tracker.loss.count == 1 and tracker.grad_norm.avg == 2.0
    assert tracker.steps == 0
