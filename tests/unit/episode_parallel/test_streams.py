import json
from copy import deepcopy

import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import Episode, EpisodeSource
from lerobot_policy_vla_ttt.training.episode_parallel.engine import episode_rng
from lerobot_policy_vla_ttt.training.episode_parallel.runner import run

from .test_baseline import make_baseline
from .test_checkpoint import assert_control_equal, mixed_source


def stream_recipe(**kwargs):
    return Recipe(
        **(
            dict(
                training_mode="policy",
                sampling_mode="mixed_tasks",
                update_unit="chunk",
                steps=6,
                warmup_steps=1,
                minimum_lr_ratio=0,
                horizon=3,
                execution=2,
                global_batch_size=2,
            )
            | kwargs
        )
    )


def test_streams_continue_refill_and_seek_without_loading_frames():
    from lerobot_policy_vla_ttt.training.episode_parallel.streams import ChunkStreamSampler

    class NoReads:
        def __len__(self):
            return 14

        def __getitem__(self, i):
            raise AssertionError("sampler must not read data")

    source = EpisodeSource(
        NoReads(),
        [Episode(0, "a", 0, 1), Episode(1, "b", 1, 6), Episode(2, "c", 6, 14)],
        horizon=3,
        execution=2,
    )
    sampler = ChunkStreamSampler(source, seed=42, global_batch_size=2)
    previous = None
    history = []
    for step in range(20):
        batch = sampler.batch(step)
        assert len(batch) == 2
        assert len({source.episodes[s.episode_id].task for s in batch}) == 2
        if previous:
            for old, new in zip(previous, batch, strict=True):
                if old.decision + 1 < source.decision_count(old.episode_id):
                    assert (new.episode_id, new.decision, new.seed) == (
                        old.episode_id,
                        old.decision + 1,
                        old.seed,
                    )
                else:
                    assert new.decision == 0 and new.seed != old.seed
                assert new.flow_seed != old.flow_seed
        history.append(batch)
        previous = batch
    replay = ChunkStreamSampler(source, seed=42, global_batch_size=2)
    replay.seek(13)
    assert replay.batch(13) == history[13]
    with pytest.raises(ValueError):
        replay.batch(13)


def test_direct_decision_matches_event_path_and_tail_mask():
    source = mixed_source()
    for ep in source.episodes.values():
        supervised = [e.frame for e in source.events(ep.index, seed=7) if e.supervised]
        for i, expected in enumerate(supervised):
            actual = source.decision(ep.index, i, seed=7)
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    torch.testing.assert_close(expected[key], actual[key])
                else:
                    assert expected[key] == actual[key]
        with pytest.raises(ValueError):
            source.decision(ep.index, len(supervised), seed=7)


def test_chunk_batch_gradient_is_valid_element_weighted(monkeypatch):
    from lerobot_policy_vla_ttt.training.episode_parallel.streams import (
        ChunkStreamSampler,
        backward_chunks,
        prepare_chunks,
    )

    p = make_baseline(monkeypatch)
    q = deepcopy(p)
    source = mixed_source()
    sampler = ChunkStreamSampler(source, seed=12, global_batch_size=2)
    sampler.batch(0)
    samples = sampler.batch(1)
    prepared = prepare_chunks(p, source, samples, None)
    denominator = sum(n for _, _, n in prepared)
    assert len({n for _, _, n in prepared}) > 1
    metrics = backward_chunks(
        p, prepared, global_valid_count=denominator, backward=lambda x: x.backward()
    )
    losses = []
    for sample, frame, n in prepared:
        with episode_rng(sample.flow_seed, torch.device("cpu")), sdpa_kernel(SDPBackend.MATH):
            loss, _ = q(frame)
            losses.append(loss * n / denominator)
    sum(losses).backward()
    assert metrics["loss_sum"] / denominator == pytest.approx(float(sum(losses).detach()))
    assert metrics["decisions"] == 2 and metrics["supervised_scalars"] == denominator
    for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_stream_resume_mid_episode_matches_uninterrupted(monkeypatch, tmp_path):
    torch.manual_seed(12)
    whole = make_baseline(monkeypatch)
    split = deepcopy(whole)
    recipe = stream_recipe()
    common = dict(
        recipe=recipe, accelerator=Accelerator(cpu=True, mixed_precision="no"), identity={}
    )
    run(whole, mixed_source(), None, None, output=tmp_path / "whole", **common)
    run(split, mixed_source(), None, None, output=tmp_path / "split", stop_after=1, **common)
    restored = make_baseline(monkeypatch)
    run(
        restored,
        mixed_source(),
        None,
        None,
        output=tmp_path / "restored",
        resume=tmp_path / "split/step-00000001",
        **common,
    )
    torch.testing.assert_close(whole.state_dict(), restored.state_dict(), atol=0, rtol=0)
    a = torch.load(tmp_path / "whole/step-00000006/training.pt", weights_only=False)
    b = torch.load(tmp_path / "restored/step-00000006/training.pt", weights_only=False)
    assert_control_equal(a["control"], b["control"])
    assert a["totals"] == b["totals"]
    assert a["totals"]["decisions"] == 12
    rows = [json.loads(x) for x in (tmp_path / "whole/metrics.jsonl").read_text().splitlines()]
    assert all(r["decisions"] == 2 and len(r["stream_positions"]) == 2 for r in rows)


def test_stream_lr_matches_original_random_chunk_schedule(monkeypatch):
    from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig

    from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers

    recipe = stream_recipe(steps=8, warmup_steps=2)
    policy = make_baseline(monkeypatch)
    control = PhaseOptimizers(policy, recipe)
    reference = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=recipe.policy_lr)
    schedule = CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=2).build(reference, 8)
    for i in range(8):
        control.begin(i)
        assert control.optimizers["policy"].param_groups[0]["lr"] == pytest.approx(
            reference.param_groups[0]["lr"]
        )
        control.step()
        reference.step()
        schedule.step()


def test_stream_recipe_rejects_memory_and_same_task():
    with pytest.raises(ValueError, match="policy"):
        stream_recipe(training_mode="joint")
    with pytest.raises(ValueError, match="mixed_tasks"):
        stream_recipe(sampling_mode="same_task")
