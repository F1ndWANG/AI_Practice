"""Replay recorded allowlisted tool attempts and audit the resulting memory."""

import argparse
import json
from collections import Counter
from contextlib import closing
from pathlib import Path

from ai_practice.artifacts import sha256

from .runtime import TOOLS, Memory, execute
from .study import load_suite, summarize


def replay(output):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    task_file = output / "tasks.jsonl"
    if sha256(task_file) != summary["task_snapshot_sha256"]:
        raise ValueError("Recorded task snapshot changed")
    partitions = {split: dict(rows) for split, rows in load_suite(task_file).items()}
    phases = {
        "baseline": ("test",), "brs": ("test",), "brs_drs": ("test",),
        "broad_exploration": ("broad",), "deep_exploration": ("probe", "practice"),
    }
    broad, combined = Counter(), Counter()
    checked = Counter()
    for label, allowed in phases.items():
        path = output / f"{label}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_id = {}
        for split in allowed:
            by_id.update(partitions[split])
        recorded = [row["id"] for row in rows]
        required = partitions[allowed[0]]
        if len(set(recorded)) != len(recorded) or not set(required) <= set(recorded) <= set(by_id):
            raise ValueError(f"Missing, duplicate or unexpected task in {label}")
        for row in rows:
            task = by_id[row["id"]]
            spent = 0
            succeeded = False
            seen = set()
            budget = summary["test_budget_per_task"]
            if label == "deep_exploration" and row["id"] in partitions["probe"]:
                budget = min(2, budget)
            for attempt in row["attempts"]:
                recipe = tuple(attempt["recipe"])
                if (not recipe or recipe in seen or succeeded or len(recipe) > budget - spent
                        or any(tool not in TOOLS for tool in recipe)):
                    raise ValueError(f"Invalid recipe or budget in {label}/{row['id']}")
                seen.add(recipe)
                success, calls = execute(task, recipe)
                checks = lambda items: [(call["tool"], call.get("returncode"), call.get("error"))
                                        for call in items]
                if success != attempt["success"] or checks(calls) != checks(attempt["calls"]):
                    raise ValueError(f"Replay diverged in {label}/{row['id']}: {recipe}")
                tool_failed = any(call.get("error") or call.get("returncode") for call in calls)
                failure = None if success else "tool_error" if tool_failed else "output_mismatch"
                if attempt["failure"] != failure:
                    raise ValueError(f"Failure classification changed in {label}/{row['id']}")
                spent += len(calls)
                succeeded = success
                checked[label] += 1
                key = (task.family, recipe, success)
                if label == "broad_exploration":
                    broad[key] += 1
                if label in ("broad_exploration", "deep_exploration"):
                    combined[key] += 1
            if row["task"] != task.family or spent != row["tool_calls"] or succeeded != row["success"]:
                raise ValueError(f"Invalid outcome in {label}/{row['id']}")
        metrics = summary["variants"].get(label)
        if label in ("broad_exploration", "deep_exploration"):
            metrics = summary["exploration"]["brs" if label == "broad_exploration" else "drs"]
        if summarize(rows) != metrics:
            raise ValueError(f"Summary does not match the recorded trace for {label}")

    for name, expected in (("brs-experience.sqlite", broad), ("experience.sqlite", combined)):
        actual = Counter()
        with closing(Memory(output / name, frozen=True)) as memory:
            for family, recipe, wins, losses in memory.entries():
                if min(wins, losses) < 0 or not (wins or losses):
                    raise ValueError(f"Invalid observation counts in {name}")
                actual[(family, recipe, True)] = wins
                actual[(family, recipe, False)] = losses
        if +actual != expected:
            raise ValueError(f"Memory does not match the exploration trace: {name}")
    return {"verified_variants": list(phases), "replayed_attempts": dict(checked),
            "memory_snapshots_verified": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.output), indent=2))


if __name__ == "__main__":
    main()
