"""Re-run held-out tasks at several call budgets using frozen Agent memories."""

import argparse
import json
import math
from contextlib import ExitStack, closing
from pathlib import Path

from ai_practice.artifacts import sha256, write_json

from .runtime import Memory
from .study import execute_case, load_suite, summarize


def sweep(study_dir, output, budgets):
    if not budgets or len(set(budgets)) != len(budgets) or min(budgets) < 1:
        raise ValueError("Budgets must be positive and distinct")
    parent = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    snapshot = study_dir / "tasks.jsonl"
    task_digest = sha256(snapshot)
    if task_digest != parent.get("task_snapshot_sha256"):
        raise ValueError("Study task snapshot checksum is missing or changed")
    tasks = load_suite(snapshot)["test"]
    files = {
        "brs": study_dir / "brs-experience.sqlite",
        "brs_drs": study_dir / "experience.sqlite",
    }
    digests = {name: sha256(path) for name, path in files.items()}
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    with ExitStack() as stack:
        states = {"baseline": None}
        states.update({name: stack.enter_context(closing(Memory(path, frozen=True)))
                       for name, path in files.items()})
        for budget in sorted(budgets):
            variants = {}
            for name, memory in states.items():
                rows = [execute_case(case, memory, budget) for case in tasks]
                variants[name] = summarize(rows)
                path = output / f"budget-{budget}-{name}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            base = variants["baseline"]
            for name, row in variants.items():
                exploration = parent["exploration_tool_calls"]["brs"] if name != "baseline" else 0
                if name == "brs_drs":
                    exploration += parent["exploration_tool_calls"]["drs"]
                saved = base["tool_calls"] - row["tool_calls"]
                comparable = row["success_rate"] >= base["success_rate"]
                row.update({"test_calls_saved": saved, "prior_exploration_calls": exploration,
                            "net_calls_saved": saved - exploration,
                            "estimated_break_even_tasks": (
                                math.ceil(exploration * len(tasks) / saved)
                                if name != "baseline" and comparable and row["success_rate"] > 0 and saved > 0 else None
                            )})
            results[str(budget)] = variants
    if any(sha256(path) != digests[name] for name, path in files.items()):
        raise ValueError("Memory snapshots changed during the budget sweep")
    report = {"kind": "agent_budget_sweep", "tasks": len(tasks), "task_snapshot_sha256": task_digest,
              "memory_sha256": digests, "planning_policy": "local", "budgets": results}
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    args = parser.parse_args()
    print(json.dumps(sweep(args.study, args.output, args.budgets), indent=2))


if __name__ == "__main__":
    main()
