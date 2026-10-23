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


def test_fresh_stage1_uses_only_template_config():
    recipe = ScreenRecipe(episodes=[508], training_stage="stage1")
    c = adapt_policy(
        Evo1GrootConfig(chunk_size=20, n_action_steps=10), recipe, "/old/policy", fresh=True
    )
    assert c.pretrained_path is None
    assert c.training_stage == "stage1" and not c.finetune_vlm
    assert not c.finetune_language_model and not c.finetune_vision_model
    assert c.finetune_action_head


def test_stage2_loads_new_stage1_and_unfreezes():
    c = adapt_policy(
        Evo1GrootConfig(chunk_size=20, n_action_steps=10),
        ScreenRecipe(episodes=[508], training_stage="stage2"),
        "/new/stage1",
    )
    assert c.pretrained_path == "/new/stage1" and c.finetune_vlm


@pytest.mark.parametrize(
    "groups", [{"a": [508], "b": [508, 512]}, {"a": [508]}, {"a": [], "b": [508, 512]}]
)
def test_balanced_groups_must_partition_selected_episodes(groups):
    with pytest.raises(ValueError):
        ScreenRecipe(episodes=[508, 512], sampling_mode="task_balanced", task_groups=groups)


def test_compact_frame_groups_do_not_use_original_frame_offsets():
    from types import SimpleNamespace

    from lerobot_policy_vla_ttt.training.train_task_subset import frame_groups_for_recipe

    recipe = ScreenRecipe(
        episodes=[200, 508],
        sampling_mode="task_balanced",
        task_groups={"button": [200], "pick": [508]},
    )
    dataset = SimpleNamespace(
        source_episode_ids=[200, 508],
        meta=SimpleNamespace(episodes={"dataset_from_index": [0, 3], "dataset_to_index": [3, 5]}),
    )
    g = frame_groups_for_recipe(dataset, recipe)
    assert list(g["button"]) == [0, 1, 2] and list(g["pick"]) == [3, 4]
