"""Evaluate paired experiment regressions against explicit release thresholds."""

import argparse
import json
import math
from pathlib import Path

from vlm_posttraining.pipeline import read_jsonl, score_predictions, write_jsonl
from vlm_posttraining.compare import grounded_correct

from .artifacts import sha256, write_json


def indexed(rows):
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError(f"Repeated sample ID: {key}")
        result[key] = row
    if not result:
        raise ValueError("Evaluation input is empty")
    return result


def evaluate_pairs(base, candidate, metrics, *, max_regressions=None):
    if max_regressions is not None and max_regressions < 0:
        raise ValueError("max_regressions must be nonnegative")
    left, right = indexed(base), indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("Baseline and candidate must contain the same sample IDs")

    checks = {}
    changes = {}
    for name, (extract, max_drop) in metrics.items():
        if not math.isfinite(max_drop) or max_drop < 0:
            raise ValueError(f"{name} threshold must be finite and nonnegative")
        lower_is_better = name == "tool_calls"
        direction = -1 if lower_is_better else 1
        count = 0
        baseline_sum = candidate_sum = 0.0
        improved = regressed = 0
        for key, row in left.items():
            before, after = extract(row), extract(right[key])
            if before is None and after is None:
                continue
            if before is None or after is None or not (math.isfinite(before) and math.isfinite(after)):
                raise ValueError(f"Incomparable {name} values for {key}")
            count += 1
            baseline_sum += before
            candidate_sum += after
            if abs(after - before) > 1e-12:
                changes.setdefault(key, {"id": key})[name] = {"baseline": before, "candidate": after}
                improved += direction * (after - before) > 0
                regressed += direction * (after - before) < 0
        delta = ((candidate_sum - baseline_sum) / count) if count else None
        passed = (delta is not None and direction * delta >= -max_drop - 1e-12
                  and (max_regressions is None or regressed <= max_regressions or lower_is_better))
        checks[name] = {"baseline": baseline_sum / count if count else None,
                        "candidate": candidate_sum / count if count else None, "delta": delta,
                        "maximum_increase" if lower_is_better else "maximum_drop": max_drop, "samples": count,
                        "improved": improved, "regressed": regressed, "passed": passed}
    return {"passed": all(check["passed"] for check in checks.values()),
            "max_sample_regressions_per_metric": max_regressions,
            "samples": len(left), "checks": checks}, changes.values()


def save(output, report, details):
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "changes.jsonl", details)
    write_json(output / "summary.json", report)
    return report


AGENT_VARIANTS = ("baseline", "brs", "brs_drs")


def agent_inputs(run, variant):
    if variant not in AGENT_VARIANTS:
        raise ValueError(f"Unknown Agent variant: {variant}")
    spec = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    digest = sha256(run / "tasks.jsonl")
    if digest != spec.get("task_snapshot_sha256"):
        raise ValueError("Agent task snapshot checksum is missing or changed")
    traces = run / f"{variant}.jsonl"
    return spec, digest, traces


def gate_agent(base_run, candidate_run, output, *, baseline="baseline", candidate="brs_drs",
               max_success_drop=0.0, max_call_increase=0.0, max_regressions=None):
    old, left_digest, left = agent_inputs(base_run, baseline)
    new, right_digest, right = agent_inputs(candidate_run, candidate)
    if left_digest != right_digest or old["test_budget_per_task"] != new["test_budget_per_task"]:
        raise ValueError("Agent gate requires identical task snapshots and test budgets")
    from rsi_agent.study import load_suite

    tasks = {key: task.family for key, task in load_suite(base_run / "tasks.jsonl")["test"]}
    budget = old["test_budget_per_task"]
    traces = [list(read_jsonl(path)) for path in (left, right)]
    for rows in traces:
        if (set(indexed(rows)) != set(tasks) or any(
            row["task"] != tasks[row["id"]] or type(row["success"]) is not bool
            or type(row["tool_calls"]) is not int or not 0 <= row["tool_calls"] <= budget for row in rows
        )):
            raise ValueError("Agent traces do not match the held-out task snapshot or budget")
    report, details = evaluate_pairs(*traces, {
        "task_success": (lambda row: int(row["success"]), max_success_drop),
        "tool_calls": (lambda row: row["tool_calls"], max_call_increase),
    }, max_regressions=max_regressions)
    return save(output, {"kind": "regression_gate", "domain": "Agent",
                         "task_snapshot_sha256": left_digest, "budget": budget,
                         "baseline": {"variant": baseline, "trace_sha256": sha256(left)},
                         "candidate": {"variant": candidate, "trace_sha256": sha256(right)},
                         **report}, details)


def gate_vision(manifest, base, candidate, output, *, max_accuracy_drop=0.0,
                max_format_drop=0.0, max_grounded_drop=None, max_regressions=None):
    metrics = {
        "answer_accuracy": (lambda row: row["accuracy"], max_accuracy_drop),
        "format_acceptance": (lambda row: row["format"], max_format_drop),
    }
    if max_grounded_drop is not None:
        metrics["grounded_accuracy"] = (
            lambda row: int(grounded_correct(row)) if row["evidence"] is not None else None,
            max_grounded_drop,
        )
    report, details = evaluate_pairs(score_predictions(manifest, base), score_predictions(manifest, candidate),
                                     metrics, max_regressions=max_regressions)
    return save(output, {"kind": "regression_gate", "domain": "VLM",
                         "manifest_sha256": sha256(manifest), "baseline_sha256": sha256(base),
                         "candidate_sha256": sha256(candidate), **report}, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="domain", required=True)
    agent = commands.add_parser("agent")
    agent.add_argument("base_run", type=Path)
    agent.add_argument("candidate_run", type=Path)
    agent.add_argument("output", type=Path)
    agent.add_argument("--baseline", choices=AGENT_VARIANTS, default="baseline")
    agent.add_argument("--candidate", choices=AGENT_VARIANTS, default="brs_drs")
    agent.add_argument("--max-success-drop", type=float, default=0.0)
    agent.add_argument("--max-call-increase", type=float, default=0.0)
    vision = commands.add_parser("vision")
    vision.add_argument("manifest", type=Path)
    vision.add_argument("base", type=Path)
    vision.add_argument("candidate", type=Path)
    vision.add_argument("output", type=Path)
    vision.add_argument("--max-accuracy-drop", type=float, default=0.0)
    vision.add_argument("--max-format-drop", type=float, default=0.0)
    vision.add_argument("--max-grounded-drop", type=float)
    for command in (agent, vision):
        command.add_argument("--max-regressions", type=int)
    args = parser.parse_args()
    if args.max_regressions is not None and args.max_regressions < 0:
        parser.error("--max-regressions must be nonnegative")
    options = vars(args)
    action = gate_agent if options.pop("domain") == "agent" else gate_vision
    report = action(**options)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
