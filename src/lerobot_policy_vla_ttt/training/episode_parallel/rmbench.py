"""RMBench episode selection without renumbering the shared LeRobot dataset."""

from .data import Episode


def select_episodes(rows, preparation, tasks):
    if preparation.get("protocol") != "aloha-joint-next-state-v1":
        raise ValueError("Unsupported RMBench action protocol")
    mapping = {int(e["episode"]): e for e in preparation["episodes"]}
    if len(mapping) != len(preparation["episodes"]):
        raise ValueError("Duplicate preparation episode")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Provide distinct RMBench tasks")
    result = []
    for row in rows:
        index = int(row["episode_index"])
        item = mapping[index]
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if end - start != item["frames"]:
            raise ValueError("RMBench preparation length mismatch")
        if item["task"] in tasks:
            result.append(Episode(index, item["task"], start, end))
    if {e.task for e in result} != set(tasks):
        raise ValueError("Requested RMBench task has no episodes")
    return result
