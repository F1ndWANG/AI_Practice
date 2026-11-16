"""Run a matched, seeded exploration ablation on local Python/Bash tasks."""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from .runtime import Memory, Task, consolidate, run_task


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
    practice_by_family = {task.family: (case_id, task) for case_id, task in practice}
    for probe in probes:
        report = execute_case(probe, memory, 2)
        rows.append({**report, "stage": "probe"})
        consolidate(memory, [report])
        if report["success"]:
            continue
        repair = execute_case(practice_by_family[report["task"]], memory, budget, "drs")
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


def study(output, seed=42, per_family=8, budget=6, workers=4):
    output.mkdir(parents=True, exist_ok=False)
    database = output / "experience.sqlite"
    held_out = cases(seed, "test", per_family)
    baseline = [execute_case(case, None, budget) for case in held_out]
    with closing(Memory(database, frozen=False)) as memory:
        broad = broad_wave(memory, cases(seed, "broad", 2), budget, workers)
    with closing(Memory(database, frozen=True)) as memory:
        brs = [execute_case(case, memory, budget) for case in held_out]
    with closing(Memory(database, frozen=False)) as memory:
        deep = deep_practice(memory, cases(seed, "probe", 1), cases(seed, "practice", 1), budget)
    with closing(Memory(database, frozen=True)) as memory:
        brs_drs = [execute_case(case, memory, budget) for case in held_out]

    traces = {"baseline": baseline, "brs": brs, "brs_drs": brs_drs}
    for label, rows in {**traces, "broad_exploration": broad, "deep_exploration": deep}.items():
        (output / f"{label}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    summary = {"seed": seed, "test_budget_per_task": budget,
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
    args = parser.parse_args()
    if min(args.per_family, args.budget, args.workers) < 1:
        parser.error("per-family, budget and workers must be positive")
    print(json.dumps(study(args.output, args.seed, args.per_family, args.budget, args.workers), indent=2))


if __name__ == "__main__":
    main()
