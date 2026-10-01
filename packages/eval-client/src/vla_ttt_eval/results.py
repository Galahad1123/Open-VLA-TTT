"""Summarize one evaluation JSONL; partial/error runs are never labeled complete."""

import argparse
import json
import math
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("type") != "run":
        raise ValueError("Missing run header")
    config = rows[0]["config"]
    expected = len(config["tasks"]) * config["episodes"]
    tasks = {str(task): {"successes": 0, "completed": 0} for task in config["tasks"]}
    seen, errors = set(), 0
    for row in rows[1:]:
        if row["type"] == "error":
            errors += 1
            continue
        if row["type"] != "episode":
            raise ValueError("Unexpected result record")
        key = (row["task"], row["episode"])
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}")
        seen.add(key)
        if type(row["success"]) is not bool:
            raise ValueError("success must be boolean")
        item = tasks[row["task"]]
        item["completed"] += 1
        item["successes"] += int(row["success"])
    for item in tasks.values():
        n, successes = item["completed"], item["successes"]
        if not n:
            item.update(success_rate=None, wilson95=None)
            continue
        rate, z = successes / n, 1.96
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        item.update(success_rate=rate, wilson95=[max(0, center - half), min(1, center + half)])
    return {
        "complete": len(seen) == expected and errors == 0,
        "expected": expected,
        "completed": len(seen),
        "errors": errors,
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2))


if __name__ == "__main__":
    main()
