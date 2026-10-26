import numpy as np
import torch

from lerobot_policy_vla_ttt.baselines import robomme_data


def test_delta_chunk_statistics_use_raw_anchor_repeat_tail_and_keep_gripper():
    # Two episodes: anchors [10, 20], [100], actions [12, 25], [103].
    state = np.repeat(np.array([[10.0], [20.0], [100.0]], dtype=np.float32), 8, axis=1)
    action = np.repeat(np.array([[12.0], [25.0], [103.0]], dtype=np.float32), 8, axis=1)
    stats = robomme_data.delta_chunk_statistics(state, action, [2, 1], 3)
    # Chunks [2,15,15], [5,5,5], [3,3,3]. No cross-episode 83 values.
    np.testing.assert_allclose(stats["action"]["q01"][:7], 2.08)
    np.testing.assert_allclose(stats["action"]["q99"][:7], 15)
    assert stats["action"]["q01"][7] == np.quantile([12, 25, 25, 25, 25, 25, 103, 103, 103], 0.01)
    np.testing.assert_allclose(stats["observation.state"]["q01"], 10.2)
    assert stats["action"]["count"].item() == 9
    np.testing.assert_array_equal(action[:, 0], [12, 25, 103])


def test_official_augmentation_uint8_reproducibility_and_wrist_geometry():
    from lerobot_policy_vla_ttt.baselines.robomme_transforms import OfficialRoboMMEAugmentation

    # Spatial edges expose accidental wrist rotation/crop; colors can change.
    image = torch.zeros(3, 256, 256, dtype=torch.uint8)
    image[:, 64:192, 64:192] = 200
    augment = OfficialRoboMMEAugmentation()
    torch.manual_seed(42)
    first = augment(image, wrist=True)
    torch.manual_seed(42)
    second = augment(image, wrist=True)
    assert torch.equal(first, second)
    assert first.shape == image.shape and first.dtype == torch.uint8
    assert torch.equal(first[:, 0, 0], first[:, 63, 63])
    assert not torch.equal(first[:, 63, 64], first[:, 64, 64])
    torch.manual_seed(42)
    external = augment(image, wrist=False)
    assert external.shape == image.shape and external.dtype == torch.uint8
    assert not torch.equal(external, first)
    assert image[:, 0, 0].sum() == 0


def test_official_rotation_always_runs_and_jitter_matches_pinned_hsv(monkeypatch):
    from lerobot_policy_vla_ttt.baselines import robomme_transforms as transforms

    image = torch.zeros(3, 256, 256, dtype=torch.uint8)
    image[0] = 128
    # Always apply jitter; fixed brightness +.2, contrast 0, hue 0.
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(0.1))
    draws = iter([0.2, 0.0, 0.0, 0.5])
    monkeypatch.setattr(transforms, "_uniform", lambda low, high: next(draws))
    result = transforms.OfficialRoboMMEAugmentation()(image, wrist=True)
    assert result[0, 0, 0].item() == 153
    assert result[1:, 0, 0].sum() == 0  # saturation remains 1 despite sampled .5

    # Nonwrist rotation must run even when jitter's p=.5 gate skips.
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(0.9))
    draws = []
    monkeypatch.setattr(transforms, "_uniform", lambda low, high: draws.append((low, high)) or 3.0)
    result = transforms.OfficialRoboMMEAugmentation()(image, wrist=False)
    assert (-5, 5) in draws
    assert result.sum() > 0


def test_full_delta_audit_excludes_demos_and_hash_is_portable(tmp_path):
    import json
    import shutil

    from test_robomme_control_data import raw_source

    # Reuse the source fixture builder, without importing it as a pytest fixture.
    root = raw_source.__wrapped__(tmp_path / "source")
    _, _, stats, report = robomme_data.inspect_source(root, "delta_joint", 3)
    np.testing.assert_array_equal(stats["action"]["q01"], [0, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(stats["observation.state"]["q99"], np.ones(8))
    assert stats["action"]["count"].item() == 4800
    assert report["normalization"]["episodes"] == 1600
    assert report["normalization"]["chunk_size"] == 3
    assert report["normalization"]["statistics"]["action"]["q01"] == [0, 0, 0, 0, 0, 0, 0, 1]
    json.dumps(report)  # Report can be persisted directly by the trainer.
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    _, _, _, moved = robomme_data.inspect_source(relocated, "delta_joint", 3)
    assert moved["normalization_sha256"] == report["normalization_sha256"]
