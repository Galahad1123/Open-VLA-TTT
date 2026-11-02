import pytest

from lerobot_policy_vla_ttt.training.episode_parallel import rmbench


def test_subset_keeps_global_offsets_and_groups_by_task_not_instruction():
    rows = [
        dict(
            episode_index=i,
            dataset_from_index=i * 100,
            dataset_to_index=(i + 1) * 100,
            tasks=[f"instruction{i}"],
        )
        for i in range(3)
    ]
    prep = dict(
        protocol="aloha-joint-next-state-v1",
        episodes=[
            dict(episode=0, task="rearrange_blocks", frames=100),
            dict(episode=1, task="swap_blocks", frames=100),
            dict(episode=2, task="put_back_block", frames=100),
        ],
    )
    episodes = rmbench.select_episodes(rows, prep, ["put_back_block", "swap_blocks"])
    assert [(e.index, e.task, e.start, e.end) for e in episodes] == [
        (1, "swap_blocks", 100, 200),
        (2, "put_back_block", 200, 300),
    ]
    with pytest.raises(ValueError):
        rmbench.select_episodes(rows, prep, ["missing"])
