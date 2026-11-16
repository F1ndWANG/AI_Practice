"""Measure the effect of false positive memories and a calibration replay guard."""

import argparse
import json
from contextlib import closing
from pathlib import Path

from ai_practice.artifacts import sha256, write_json

from .runtime import Memory, execute, run_task
from .study import load_suite, summarize


def stress(study_dir, output, *, budget=2, false_wins=64):
    if budget < 1 or false_wins < 1:
        raise ValueError("budget and false-wins must be positive")
    snapshot = study_dir / "tasks.jsonl"
    parent = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    if sha256(snapshot) != parent.get("task_snapshot_sha256"):
        raise ValueError("Missing or changed task snapshot")
    splits = load_suite(snapshot)
    source = study_dir / "experience.sqlite"
    original_digest = sha256(source)
    with closing(Memory(source, frozen=True)) as memory:
        families = sorted({task.family for _, task in splits["test"]})
        clean = {family: memory.recall(family) for family in families}

    anchors = {}
    for phase in ("probe", "practice", "broad"):
        for case_id, task in splits[phase]:
            anchors.setdefault(task.family, (case_id, task))
    if not set(families) <= anchors.keys():
        raise ValueError("Each held-out task family needs a calibration example outside test")

    polluted = {}
    guarded = {}
    calibration = []
    for family in families:
        entries = {recipe: (wins, losses) for recipe, wins, losses in clean[family]}
        bad_recipe = ("copy",)
        wins, losses = entries.get(bad_recipe, (0, 0))
        entries[bad_recipe] = (wins + false_wins, losses)
        ranked = sorted(((recipe, w, l) for recipe, (w, l) in entries.items()),
                        key=lambda row: (row[2] - row[1], json.dumps(row[0])))
        polluted[family] = ranked
        guarded[family] = []
        anchor_id, anchor = anchors[family]
        for recipe, successes, failures in ranked:
            if successes > failures:
                passed, calls = execute(anchor, recipe)
                calibration.append({"family": family, "anchor_id": anchor_id, "recipe": recipe,
                                    "passed": passed, "tool_calls": len(calls)})
                if not passed:
                    continue
            guarded[family].append((recipe, successes, failures))

    output.mkdir(parents=True, exist_ok=False)
    variants = {}
    for name, state in (("clean", clean), ("polluted", polluted), ("guarded", guarded)):
        rows = [{"id": case_id, **run_task(task, state[task.family], "brs", budget)}
                for case_id, task in splits["test"]]
        variants[name] = summarize(rows)
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    if sha256(source) != original_digest:
        raise ValueError("The source memory changed during the stress test")
    (output / "calibration.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in calibration), encoding="utf-8"
    )
    result = {"kind": "agent_memory_stress", "budget": budget, "false_wins_per_family": false_wins,
              "source_memory_sha256": original_digest, "variants": variants,
              "calibration_candidates": len(calibration),
              "calibration_rejected": sum(not row["passed"] for row in calibration),
              "calibration_tool_calls": sum(row["tool_calls"] for row in calibration)}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--false-wins", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(stress(args.study, args.output, budget=args.budget, false_wins=args.false_wins),
                     indent=2))


if __name__ == "__main__":
    main()
