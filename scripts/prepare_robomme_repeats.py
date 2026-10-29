"""Audit raw RoboMME repeat labels and generate paired portable recipes."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def repeat_count(prompt):
    match = re.search(r"\b(one|two|three|four|five) times\b", prompt)
    return WORDS[match[1]] if match else 1


def goal_count(task, goals):
    marker = "pick up the " if task == "PickXtimes" else "move to the top of the left-side target"
    nums = [
        ORDINALS[m[1]]
        for g in goals
        if g.startswith(marker)
        if (m := re.search(r"for the (first|second|third|fourth|fifth) time", g))
    ]
    if not nums:
        raise ValueError(f"No repeat goals for {task}")
    return max(nums)


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root
    tasks = {
        x["task_index"]: x["task"] for x in pq.read_table(root / "meta/tasks.parquet").to_pylist()
    }
    episodes = [
        e
        for f in sorted((root / "meta/episodes").rglob("*.parquet"))
        for e in pq.read_table(f).to_pylist()
    ]
    cache, groups, audit = {}, defaultdict(list), []
    for e in episodes:
        idx = e["episode_index"]
        task = {5: "PickXtimes", 7: "SwingXtimes"}.get(idx // 100)
        if task is None:
            continue
        key = e["data/chunk_index"], e["data/file_index"]
        if key not in cache:
            cache[key] = pq.read_table(
                root / f"data/chunk-{key[0]:03d}/file-{key[1]:03d}.parquet",
                columns=[
                    "episode_index",
                    "task_index",
                    "simple_subgoal",
                    "is_demo",
                    "exec_start_idx",
                ],
            ).to_pylist()
        rows = [x for x in cache[key] if x["episode_index"] == idx]
        prompts = {tasks[x["task_index"]] for x in rows}
        assert len(prompts) == 1 and len(rows) == e["length"], idx
        prompt = prompts.pop()
        count = repeat_count(prompt)
        assert count == goal_count(task, {x["simple_subgoal"] for x in rows}), idx
        assert not any(x["is_demo"] or x["exec_start_idx"] for x in rows), idx
        if count > 3:
            continue
        groups[f"{task}/{count}"].append(idx)
        audit.append(
            dict(episode=idx, task=task, repeats=count, length=e["length"], instruction=prompt)
        )
    groups = {k: sorted(groups[k]) for k in sorted(groups)}
    expected = [23, 25, 27, 23, 31, 46]
    assert [len(v) for v in groups.values()] == expected
    args.output.mkdir(parents=True, exist_ok=True)
    configroot = Path(__file__).resolve().parents[1] / "configs/training/robomme-screen"
    for kind in ["baseline", "aqmem"]:
        recipe = json.loads((configroot / f"multitask-six-frozen-{kind}-mb6-2k.json").read_text())
        recipe.update(task_groups=groups, diagnostics_every=50)
        (args.output / f"repeat-two-frozen-{kind}.json").write_text(
            json.dumps(recipe, indent=2) + "\n"
        )
    plan = {
        "version": 1,
        "description": "Two repeat tasks, 1-3 repeats; train diagnostic: first 10 per stratum.",
        "tasks": {
            t: {
                "repeats": [1, 2, 3],
                "train_episodes": sorted(
                    i % 100 for n in [1, 2, 3] for i in groups[f"{t}/{n}"][:10]
                ),
            }
            for t in ["PickXtimes", "SwingXtimes"]
        },
    }
    (args.output / "repeat-two-evaluation.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "repeat-two-data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in groups.items()}))


if __name__ == "__main__":
    main()
