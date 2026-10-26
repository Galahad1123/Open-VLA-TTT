import numpy as np
import pytest
from vla_ttt_eval.rmbench import validate_action
from vla_ttt_eval.rmbench_video import H264Video


def test_joint_target_contract():
    assert validate_action(np.zeros(14)).shape == (14,)
    for bad in (np.zeros(8), np.full(14, np.nan), np.zeros((1, 14))):
        with pytest.raises(ValueError):
            validate_action(bad)


def test_h264_video(tmp_path):
    import json
    import subprocess

    path = tmp_path / "episode.mp4"
    with H264Video(path, (240, 320)) as writer:
        for _ in range(3):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]
        )
    )["streams"][0]
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)


def test_merge_rejects_duplicate_and_wrong_policy(tmp_path):
    import json

    from vla_ttt_eval.rmbench_runner import merge_results

    parts = [tmp_path / str(i) for i in range(2)]
    manifest = {
        "upstream_revision": "fixed",
        "protocol": "fixed",
        "seed": 0,
        "task": "put_back_block",
        "episodes": 2,
        "policy": {"checkpoint_sha256": "abc", "n_action_steps": 10},
    }
    for i, part in enumerate(parts):
        part.mkdir()
        (part / "manifest.json").write_text(json.dumps(manifest))
        (part / "episodes.jsonl").write_text(
            json.dumps(
                {"task": "put_back_block", "episode": i, "seed": 100000 + i, "success": i == 1}
            )
            + "\n"
        )
    assert merge_results(parts)["success_rate"] == 0.5
    with pytest.raises(ValueError):
        merge_results([parts[0], parts[0]])
    manifest["policy"]["checkpoint_sha256"] = "other"
    (parts[1] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge_results(parts)
