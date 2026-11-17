"""Compare verified acceptance bundles against a common evaluation protocol."""

import argparse
import json
from pathlib import Path

from .artifacts import write_json
from .release import read, verify
from .report import percent, table


def metric_checks(gate):
    checks = {("overall", "", metric): result for metric, result in gate["checks"].items()}
    for dimension, grouping in gate.get("by_slice", {}).items():
        for group in grouping["slices"]:
            for metric, check in group["checks"].items():
                checks[(dimension, group["value"], metric)] = check
    return checks


def configuration(gate):
    def thresholds(check):
        return {key: check[key] for key in ("maximum_drop", "maximum_increase") if key in check}

    return {
        "policy_sha256": (gate.get("policy") or {}).get("sha256"),
        "maximum_regressions": gate.get("max_sample_regressions_per_metric"),
        "minimum_slice_samples": gate.get("min_slice_samples"),
        "thresholds": {metric: thresholds(check) for metric, check in gate["checks"].items()},
        "dimensions": sorted(gate.get("by_slice", {})),
        "required": sorted((item["dimension"], item["value"], item["metric"])
                           for item in gate.get("required_slice_checks", [])),
    }


def incompatible_reasons(domain, before, after):
    reasons = []
    if configuration(before) != configuration(after):
        reasons.append("policy_changed")
    if domain == "agent":
        if before.get("budget") != after.get("budget"):
            reasons.append("tool_budget_changed")
        if before.get("task_snapshot_sha256") != after.get("task_snapshot_sha256"):
            reasons.append("task_snapshot_changed")
        field = "outcomes_sha256" if all("outcomes_sha256" in row["baseline"] for row in (before, after)) else "trace_sha256"
        identity = [row["baseline"].get(field) for row in (before, after)]
        if before["baseline"].get("variant") != after["baseline"].get("variant"):
            reasons.append("baseline_variant_changed")
    else:
        field = "reference_protocol_sha256" if all("reference_protocol_sha256" in row for row in (before, after)) else "manifest_sha256"
        if not before.get(field) or before.get(field) != after.get(field):
            reasons.append("reference_changed")
        field = "baseline_completions_sha256" if all("baseline_completions_sha256" in row for row in (before, after)) else "baseline_sha256"
        identity = [row.get(field) for row in (before, after)]
    if not identity[0] or identity[0] != identity[1]:
        reasons.append("baseline_changed")
    left, right = metric_checks(before), metric_checks(after)
    if left.keys() != right.keys() or any(
        row["samples"] != right[key]["samples"] or row["baseline"] != right[key]["baseline"]
        for key, row in left.items() if key in right
    ):
        reasons.append("metric_support_changed")
    return reasons


def compare(previous, current):
    sources = [verify(path) for path in (previous, current)]
    domains = {}
    for domain in ("agent", "vision"):
        before, after = (read(path / domain / "summary.json") for path in (previous, current))
        reasons = incompatible_reasons(domain, before, after)
        rows = []
        if not reasons:
            next_checks = metric_checks(after)
            for (scope, group, metric), old in metric_checks(before).items():
                new = next_checks[scope, group, metric]
                left, right = old["candidate"], new["candidate"]
                delta = (right - left) if left is not None and right is not None else None
                improved = -delta if metric == "tool_calls" and delta is not None else delta
                measured = old["passed"] is not None and new["passed"] is not None
                rows.append({"scope": scope, "group": group, "metric": metric,
                             "samples": old["samples"], "before": left, "after": right,
                             "delta": delta, "regressed": bool(measured and improved is not None and improved < -1e-12)})
        domains[domain] = {"comparable": not reasons, "reasons": reasons, "checks": rows,
                           "previous_passed": before["passed"], "current_passed": after["passed"]}
    return {"kind": "release_comparison", "previous_revision": sources[0]["revision"],
            "current_revision": sources[1]["revision"],
            "comparable": all(item["comparable"] for item in domains.values()),
            "regressed": any(
                (item["previous_passed"] and not item["current_passed"])
                or any(check["regressed"] for check in item["checks"])
                for item in domains.values() if item["comparable"]
            ), "domains": domains}


def markdown(report):
    lines = ["# AI Practice 跨提交对比", "",
             f"前一提交  {report['previous_revision'] or '未填写'}", "",
             f"当前提交  {report['current_revision'] or '未填写'}", ""]
    for domain, result in report["domains"].items():
        lines.extend([f"## {domain}", ""])
        if not result["comparable"]:
            lines.extend(["不可比  " + "、".join(result["reasons"]), ""])
            continue
        rows = []
        for check in result["checks"]:
            number = percent
            if check["metric"] == "tool_calls":
                number = lambda value: "—" if value is None else f"{value:.3f}"
            rows.append((check["scope"], check["group"] or "—", check["metric"], check["samples"],
                         number(check["before"]), number(check["after"]), number(check["delta"]),
                         "退化" if check["regressed"] else "—"))
        lines.extend(table(("维度", "分组", "指标", "样本", "前值", "后值", "变化", "判定"), rows) + [""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    report = compare(args.previous, args.current)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", report)
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["comparable"]:
        raise SystemExit(2)
    if args.fail_on_regression and report["regressed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
