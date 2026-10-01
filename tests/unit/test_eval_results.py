import importlib
import json

import pytest


def test_summary_reports_task_success_and_marks_incomplete_runs(tmp_path):
    summary = importlib.import_module("vla_ttt_eval.results")
    path = tmp_path / "results.jsonl"
    rows = [
        {"type": "run", "config": {"tasks": ["A"], "episodes": 3}},
        {"type": "episode", "task": "A", "episode": 0, "success": True},
        {"type": "episode", "task": "A", "episode": 1, "success": False},
        {"type": "error", "task": "A", "episode": 2, "error": "sim error"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summary.summarize(path)
    assert result["complete"] is False
    assert result["errors"] == 1
    assert result["tasks"]["A"]["success_rate"] == 0.5
    assert result["tasks"]["A"]["completed"] == 2
    assert result["tasks"]["A"]["wilson95"] == pytest.approx([0.09453, 0.90547], abs=1e-4)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:3] + [rows[1]]))
    with pytest.raises(ValueError, match="Duplicate"):
        summary.summarize(path)
