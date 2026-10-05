import torch

from lerobot_policy_vla_ttt.models.robottt.modeling_robottt import RoboTTTPolicy
from lerobot_policy_vla_ttt.training.sequence import TrajectoryWindows, collate_windows
from lerobot_policy_vla_ttt.training.tbptt import train_sequence


def test_windows_never_cross_episodes_and_mark_padding():
    frames = [{"action": torch.full((3, 7), float(i)), "task": "move"} for i in range(7)]
    dataset = TrajectoryWindows(frames, [(0, 3), (3, 7)], length=3, stride=2)
    first, second = dataset[0], dataset[1]
    assert first["valid"].tolist() == [True, True, False]
    assert [x["action"][0, 0].item() for x in first["frames"]] == [0, 2, 2]
    assert [x["action"][0, 0].item() for x in second["frames"]] == [3, 5, 5]
    batch = collate_windows([first, second])
    assert batch["valid_mask"].shape == (2, 3)
    assert batch["frames"][0]["action"].shape == (2, 3, 7)


def frames():
    return [
        dict(
            action=torch.rand(2, 3, 7),
            **{
                "observation.state": torch.rand(2, 8),
                "observation.images.image": torch.rand(2, 3, 16, 16),
                "observation.images.image2": torch.rand(2, 3, 16, 16),
            },
            task=["a", "b"],
        )
        for _ in range(4)
    ]


def test_real_segment_backward_carries_values_and_excludes_padding(config):
    torch.manual_seed(12)
    policy = RoboTTTPolicy(config)
    data = frames()
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).bool()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    before = policy.model.action_head.memories[0].gate.detach().clone()
    result, memory = train_sequence(policy, data, optimizer, segment_length=2, valid_mask=valid)
    assert result["segments"] == 2
    assert result["supervised_scalars"] == 6 * 3 * 7
    assert memory[0].position.tolist() == [4, 2]
    assert all(w.grad_fn is None for state in memory for w in state.weights)
    assert not torch.equal(before, policy.model.action_head.memories[0].gate)


def test_full_segment_equals_explicit_unroll(config):
    torch.manual_seed(9)
    a, b = RoboTTTPolicy(config), RoboTTTPolicy(config)
    b.load_state_dict(a.state_dict())
    data = frames()
    torch.manual_seed(100)
    memory, numerator, count = None, 0, 0
    for frame in data:
        loss, n, memory = a.forward_step(frame, memory)
        numerator, count = numerator + loss, count + n
    (numerator / count).backward()
    expected = {k: p.grad.clone() for k, p in a.named_parameters() if p.grad is not None}
    torch.manual_seed(100)
    result, _ = train_sequence(b, data, torch.optim.SGD(b.parameters(), lr=0), segment_length=4)
    assert abs(result["loss"] - (numerator / count).item()) < 1e-6
    for k, p in b.named_parameters():
        if k in expected:
            torch.testing.assert_close(p.grad, expected[k])


def test_context_only_does_not_read_labels(config):
    policy = RoboTTTPolicy(config)
    frame = frames()[0]
    supervision = torch.zeros(2, dtype=torch.bool)
    torch.manual_seed(7)
    _, count, memory = policy.forward_step(frame, supervision_mask=supervision)
    frame["action"].fill_(999)
    torch.manual_seed(7)
    _, _, altered = policy.forward_step(frame, supervision_mask=supervision)
    assert count.item() == 0
    for a, b in zip(memory, altered, strict=True):
        for wa, wb in zip(a.weights, b.weights, strict=True):
            torch.testing.assert_close(wa, wb)


def test_training_runner_saves_reloadable_checkpoint(config, tmp_path):
    from lerobot_policy_vla_ttt.cli import make_pre_post_processors
    from lerobot_policy_vla_ttt.training.train_robottt import run_training

    policy = RoboTTTPolicy(config)
    stats = {
        key: {"min": -torch.ones(width), "max": torch.ones(width)}
        for key, width in [("action", 7), ("observation.state", 8)]
    }
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    data = frames()
    # Turn the raw B=2 frames into an indexable episode of unbatched observations.
    raw = [{k: v[0] for k, v in frame.items()} for frame in data]
    windows = TrajectoryWindows(raw, [(0, 4)], length=3, stride=1)
    run_training(
        policy,
        pre,
        post,
        windows,
        tmp_path,
        steps=2,
        batch_size=2,
        segment_length=2,
        learning_rate=1e-3,
        save_every=2,
    )
    restored = RoboTTTPolicy.from_pretrained(tmp_path / "step-00000002", local_files_only=True)
    assert restored.model.action_head.inference_state is None
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "step-00000002" / "optimizer.pt").is_file()
