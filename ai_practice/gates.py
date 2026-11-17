"""Evaluate paired experiment regressions against explicit release thresholds."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from vlm_posttraining.pipeline import read_jsonl, score_predictions, write_jsonl
from vlm_posttraining.compare import grounded_correct

from .artifacts import artifact_index, sha256, write_json


def protocol_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def agent_baseline_digest(rows):
    fields = ("id", "task", "success", "tool_calls")
    return protocol_digest([{key: row[key] for key in fields}
                            for row in sorted(rows, key=lambda item: item["id"])])


def vision_reference_digest(manifest):
    records = []
    for row in read_jsonl(manifest):
        record = {key: value for key, value in row.items() if key != "image"}
        if record.get("image_sha256") is None:
            record["image_sha256"] = sha256((manifest.parent / row["image"]).resolve())
        records.append(record)
    return protocol_digest(sorted(records, key=lambda item: item["id"]))


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


def evaluate_slices(base, candidate, metrics, dimensions, minimum, max_regressions):
    if minimum < 1 or len(set(dimensions)) != len(dimensions):
        raise ValueError("Slice minimum must be positive and dimensions distinct")
    by_id = indexed(candidate)
    dimensions_report = {}
    for dimension in dimensions:
        buckets = defaultdict(list)
        for row in base:
            value = row.get(dimension)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Invalid slice value for {dimension}/{row['id']}")
            buckets[value].append(row)
        slices = []
        for value, left in sorted(buckets.items(), key=lambda item: (item[0] is None, item[0] or "")):
            right = [by_id[row["id"]] for row in left]
            outcome, _ = evaluate_pairs(left, right, metrics, max_regressions=max_regressions)
            for check in outcome["checks"].values():
                if value is None or check["samples"] < minimum:
                    check["status"] = "unavailable" if value is None else "insufficient"
                    check["passed"] = None
                else:
                    check["status"] = "passed" if check["passed"] else "failed"
            statuses = [check["status"] for check in outcome["checks"].values()]
            if "failed" in statuses:
                decision = "failed"
            elif "passed" in statuses:
                decision = "passed"
            else:
                decision = "unavailable" if value is None else "insufficient"
            slices.append({"value": value, "rows": len(left), "status": decision,
                           "checks": outcome["checks"]})
        checks = [check for item in slices for check in item["checks"].values()]
        assessed = sum(check["passed"] is not None for check in checks)
        dimensions_report[dimension] = {
            "passed": bool(assessed) and not any(check["passed"] is False for check in checks),
            "assessed_checks": assessed,
            "missing_labels": len(buckets.get(None, [])),
            "slices": slices,
        }
    return dimensions_report


def required_checks(slices, required, metrics):
    results = []
    for dimension, groups in required.items():
        if dimension not in slices:
            raise ValueError(f"Required dimension {dimension} must appear in slice_by")
        found = {group["value"]: group for group in slices[dimension]["slices"]}
        for label, names in groups.items():
            for metric in names:
                if metric not in metrics:
                    raise ValueError(f"Required slice metric {metric} is not configured")
                match = found.get(label)
                status = match["checks"][metric]["status"] if match else "missing"
                results.append({"dimension": dimension, "value": label, "metric": metric,
                                "status": status, "passed": status == "passed"})
    return results


def evaluate_gate(base, candidate, metrics, max_regressions, slice_by, min_slice_samples,
                  required_slices=None):
    report, details = evaluate_pairs(base, candidate, metrics, max_regressions=max_regressions)
    if not slice_by and not required_slices:
        return report, details
    slices = evaluate_slices(base, candidate, metrics, slice_by, min_slice_samples, max_regressions)
    coverage = required_checks(slices, required_slices or {}, metrics)
    by_id = indexed(base)
    changes = [
        {**row, "slices": {key: by_id[row["id"]].get(key) for key in slice_by}}
        for row in details
    ]
    return {**report, "overall_passed": report["passed"], "min_slice_samples": min_slice_samples,
            "passed": (report["passed"] and all(item["passed"] for item in slices.values())
                       and all(item["passed"] for item in coverage)),
            "by_slice": slices, "required_slice_checks": coverage}, changes


def save(output, report, details):
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "changes.jsonl", details)
    write_json(output / "summary.json", report)
    write_json(output / "receipt.json", {"schema": 1,
                                        "artifacts": artifact_index(output, ["summary.json", "changes.jsonl"])})
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
               max_success_drop=0.0, max_call_increase=0.0, max_regressions=None,
               slice_by=(), min_slice_samples=1, required_slices=None, policy=None):
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
    report, details = evaluate_gate(*traces, {
        "task_success": (lambda row: int(row["success"]), max_success_drop),
        "tool_calls": (lambda row: row["tool_calls"], max_call_increase),
    }, max_regressions, slice_by, min_slice_samples, required_slices)
    return save(output, {"kind": "regression_gate", "domain": "Agent",
                         "task_snapshot_sha256": left_digest, "budget": budget, "policy": policy,
                         "baseline": {"variant": baseline, "trace_sha256": sha256(left),
                                      "outcomes_sha256": agent_baseline_digest(traces[0])},
                         "candidate": {"variant": candidate, "trace_sha256": sha256(right)},
                         **report}, details)


def gate_vision(manifest, base, candidate, output, *, max_accuracy_drop=0.0,
                max_format_drop=0.0, max_grounded_drop=None, max_regressions=None,
                slice_by=(), min_slice_samples=1, required_slices=None, policy=None):
    metrics = {
        "answer_accuracy": (lambda row: row["accuracy"], max_accuracy_drop),
        "format_acceptance": (lambda row: row["format"], max_format_drop),
    }
    if max_grounded_drop is not None:
        metrics["grounded_accuracy"] = (
            lambda row: int(grounded_correct(row)) if row["evidence"] is not None else None,
            max_grounded_drop,
        )
    baseline_rows = score_predictions(manifest, base)
    report, details = evaluate_gate(baseline_rows, score_predictions(manifest, candidate),
                                    metrics, max_regressions, slice_by, min_slice_samples, required_slices)
    baseline_digest = protocol_digest([
        {key: row[key] for key in ("id", "completion")}
        for row in sorted(baseline_rows, key=lambda item: item["id"])
    ])
    return save(output, {"kind": "regression_gate", "domain": "VLM",
                         "manifest_sha256": sha256(manifest), "reference_protocol_sha256": vision_reference_digest(manifest),
                         "baseline_sha256": sha256(base), "baseline_completions_sha256": baseline_digest, "policy": policy,
                         "candidate_sha256": sha256(candidate), **report}, details)


POLICY_FIELDS = {
    "agent": {"max_success_drop", "max_call_increase"},
    "vision": {"max_accuracy_drop", "max_format_drop", "max_grounded_drop"},
}
POLICY_COMMON = {"max_regressions", "slice_by", "min_slice_samples", "required_slices"}


def unique_mapping(pairs):
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"Repeated JSON key: {key}")
        mapping[key] = value
    return mapping


def load_policy(path, domain):
    raw = path.read_bytes()
    mapping = json.loads(raw, object_pairs_hook=unique_mapping)
    if (not isinstance(mapping, dict) or set(mapping) != {"schema", "domain", "settings"}
            or type(mapping["schema"]) is not int or mapping["schema"] != 1 or mapping["domain"] != domain
            or not isinstance(mapping["settings"], dict)):
        raise ValueError(f"Expected a schema 1 policy for {domain}")
    settings = mapping["settings"]
    unexpected = set(settings) - POLICY_FIELDS[domain] - POLICY_COMMON
    if unexpected:
        raise ValueError(f"Unknown policy settings: {', '.join(sorted(unexpected))}")
    for name in POLICY_FIELDS[domain] & settings.keys():
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid policy threshold: {name}")
    for name, lower in (("max_regressions", 0), ("min_slice_samples", 1)):
        if name in settings and (type(settings[name]) is not int or settings[name] < lower):
            raise ValueError(f"Invalid policy setting: {name}")
    allowed = {"task"} if domain == "agent" else {"task", "dataset", "difficulty"}
    dimensions = settings.get("slice_by", [])
    if (not isinstance(dimensions, list) or any(type(key) is not str or key not in allowed for key in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("Invalid policy slice_by")
    required = settings.get("required_slices", {})
    if not isinstance(required, dict) or not required.keys() <= set(dimensions):
        raise ValueError("Required slice dimensions must appear in slice_by")
    expected_metrics = {"task_success", "tool_calls"} if domain == "agent" else {
        "answer_accuracy", "format_acceptance"
    }
    if "max_grounded_drop" in settings:
        expected_metrics.add("grounded_accuracy")
    for groups in required.values():
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Required slice groups must be a nonempty mapping")
        for label, metrics in groups.items():
            if (not label.strip() or not isinstance(metrics, list) or not metrics
                    or any(not isinstance(name, str) or name not in expected_metrics for name in metrics)
                    or len(set(metrics)) != len(metrics)):
                raise ValueError(f"Invalid required slice {label!r}")
    return {**settings, "policy": {"sha256": hashlib.sha256(raw).hexdigest(), **mapping}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="domain", required=True)
    agent = commands.add_parser("agent")
    agent.add_argument("base_run", type=Path)
    agent.add_argument("candidate_run", type=Path)
    agent.add_argument("output", type=Path)
    agent.add_argument("--baseline", choices=AGENT_VARIANTS, default="baseline")
    agent.add_argument("--candidate", choices=AGENT_VARIANTS, default="brs_drs")
    agent.add_argument("--max-success-drop", type=float, default=argparse.SUPPRESS)
    agent.add_argument("--max-call-increase", type=float, default=argparse.SUPPRESS)
    vision = commands.add_parser("vision")
    vision.add_argument("manifest", type=Path)
    vision.add_argument("base", type=Path)
    vision.add_argument("candidate", type=Path)
    vision.add_argument("output", type=Path)
    vision.add_argument("--max-accuracy-drop", type=float, default=argparse.SUPPRESS)
    vision.add_argument("--max-format-drop", type=float, default=argparse.SUPPRESS)
    vision.add_argument("--max-grounded-drop", type=float, default=argparse.SUPPRESS)
    agent.add_argument("--slice-by", choices=["task"], action="append", default=argparse.SUPPRESS)
    vision.add_argument("--slice-by", choices=["task", "difficulty", "dataset"], action="append", default=argparse.SUPPRESS)
    for command in (agent, vision):
        command.add_argument("--max-regressions", type=int, default=argparse.SUPPRESS)
        command.add_argument("--min-slice-samples", type=int, default=argparse.SUPPRESS)
        command.add_argument("--policy", type=Path)
    options = vars(parser.parse_args())
    domain = options.pop("domain")
    path = options.pop("policy")
    if path:
        if options.keys() & (POLICY_FIELDS[domain] | POLICY_COMMON):
            parser.error("--policy cannot be combined with inline threshold or slice flags")
        try:
            options.update(load_policy(path, domain))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    action = gate_agent if domain == "agent" else gate_vision
    report = action(**options)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
