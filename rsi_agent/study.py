"""Run a matched, seeded exploration ablation on local Python/Bash tasks."""

import argparse
import json
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from ai_practice.artifacts import sha256

from .runtime import POLICIES, Memory, Task, consolidate, run_task


def cases(seed, split, per_family):
    rng = random.Random(f"{seed}:{split}")
    rows = []
    for index in range(per_family):
        words = [f"{split}{rng.randrange(1000):03d}" for _ in range(5)]
        # Broad practice uses clean inputs. Probe, repair and held-out cases add
        # whitespace and case variation that expose a fragile sort-only recipe.
        supplied = words if split == "broad" else [f" {word.upper()} " for word in words]
        source = "\n".join(supplied + supplied[:2]) + "\n"
        expected = "\n".join(sorted(set(words))) + "\n"
        rows.append((f"{split}:words:{index}", Task("words", source, expected)))
        numbers = [rng.randrange(-100, 100) for _ in range(6)] + [1000]
        presentation = "\n" if split == "broad" else "  "
        rows.append((f"{split}:sum:{index}",
                     Task("sum", presentation.join(map(str, numbers)) + "\n", f"{sum(numbers)}\n")))
    return rows


def load_suite(path):
    partitions = {split: [] for split in ("broad", "probe", "practice", "test")}
    ids = set()
    sources = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        fields = ("id", "split", "family", "source", "expected")
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in fields):
            raise ValueError(f"Invalid task on line {number}")
        case_id, split, family, source, expected = (row[key] for key in fields)
        if not case_id or case_id in ids or split not in partitions or family not in POLICIES:
            raise ValueError(f"Invalid ID, split or family on line {number}")
        key = (family, source)
        previous_split = sources.setdefault(key, split)
        if previous_split != split:
            raise ValueError(f"Task input reused across splits on line {number}")
        ids.add(case_id)
        partitions[split].append((case_id, Task(family, source, expected)))
    if not all(partitions.values()):
        raise ValueError("The suite needs broad, probe, practice and test tasks")
    families = {split: {task.family for _, task in cases}
                for split, cases in partitions.items()}
    if not families["probe"] <= families["practice"]:
        raise ValueError("Each probed family needs a practice task")
    return partitions


def execute_case(case, memory, budget, strategy="brs"):
    case_id, task = case
    recalled = memory.recall(task.family) if memory else []
    return {"id": case_id, **run_task(task, recalled, strategy, budget)}


def broad_wave(memory, tasks, budget, workers):
    snapshot = {task.family: memory.recall(task.family) for _, task in tasks}

    def actor(case):
        case_id, task = case
        return {"id": case_id, **run_task(task, snapshot[task.family], "brs", budget)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(actor, tasks))
    consolidate(memory, rows)
    return rows


def deep_practice(memory, probes, practice, budget):
    rows = []
    practice_by_family = {}
    for case in practice:
        practice_by_family.setdefault(case[1].family, []).append(case)
    next_index = Counter()
    for probe in probes:
        report = execute_case(probe, memory, min(budget, 2))
        rows.append({**report, "stage": "probe"})
        consolidate(memory, [report])
        if report["success"]:
            continue
        family = report["task"]
        candidates = practice_by_family[family]
        index = next_index[family]
        if index >= len(candidates):
            continue
        next_index[family] += 1
        repair = execute_case(candidates[index], memory, budget, "drs")
        rows.append({**repair, "stage": "practice", "trigger": report["id"]})
        consolidate(memory, [repair])
    return rows


def summarize(rows):
    return {
        "tasks": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "successful_memory_hits": sum(
            attempt["memory_hit"] and attempt["success"]
            for row in rows for attempt in row["attempts"]
        ),
        "failures": dict(Counter(attempt["failure"] for row in rows
                                for attempt in row["attempts"] if attempt["failure"])),
    }


def study(output, seed=42, per_family=8, budget=6, workers=4, suite=None):
    tasks = load_suite(suite) if suite else {
        phase: cases(seed, phase, count)
        for phase, count in (("broad", 2), ("probe", 1), ("practice", 1), ("test", per_family))
    }
    output.mkdir(parents=True, exist_ok=False)
    database = output / "experience.sqlite"
    held_out = tasks["test"]
    baseline = [execute_case(case, None, budget) for case in held_out]
    with closing(Memory(database, frozen=False)) as memory:
        broad = broad_wave(memory, tasks["broad"], budget, workers)
    snapshot = output / "brs-experience.sqlite"
    shutil.copyfile(database, snapshot)
    with closing(Memory(snapshot, frozen=True)) as memory:
        brs = [execute_case(case, memory, budget) for case in held_out]
    with closing(Memory(database, frozen=False)) as memory:
        deep = deep_practice(memory, tasks["probe"], tasks["practice"], budget)
    with closing(Memory(database, frozen=True)) as memory:
        brs_drs = [execute_case(case, memory, budget) for case in held_out]

    traces = {"baseline": baseline, "brs": brs, "brs_drs": brs_drs}
    for label, rows in {**traces, "broad_exploration": broad, "deep_exploration": deep}.items():
        (output / f"{label}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    summary = {"seed": seed, "test_budget_per_task": budget,
               "suite_sha256": sha256(suite) if suite else None,
               "test_tasks_by_family": dict(Counter(task.family for _, task in held_out)),
               "variants": {label: summarize(rows) for label, rows in traces.items()},
               "exploration_tool_calls": {"brs": sum(row["tool_calls"] for row in broad),
                                          "drs": sum(row["tool_calls"] for row in deep)}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suite", type=Path, help="Optional task suite JSONL")
    args = parser.parse_args()
    if min(args.per_family, args.budget, args.workers) < 1:
        parser.error("per-family, budget and workers must be positive")
    print(json.dumps(study(args.output, args.seed, args.per_family, args.budget, args.workers, args.suite), indent=2))


if __name__ == "__main__":
    main()
