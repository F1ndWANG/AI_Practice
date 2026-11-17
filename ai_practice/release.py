"""Bundle both verified regression decisions into a portable release report."""

import argparse
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from .artifacts import artifact_index, verify_index, write_json
from .report import gate_lines


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_gate(source, destination, domain, include_changes):
    receipt = read(source / "receipt.json")
    required = {"summary.json", "changes.jsonl"}
    if receipt.get("schema") != 1 or set(receipt.get("artifacts", {})) != required:
        raise ValueError(f"Invalid gate receipt: {domain}")
    verify_index(source, receipt["artifacts"])
    destination.mkdir()
    files = sorted(required if include_changes else {"summary.json"})
    for name in files:
        shutil.copyfile(source / name, destination / name)
    verify_index(destination, {name: receipt["artifacts"][name] for name in files})
    result = read(destination / "summary.json")
    if result.get("kind") != "regression_gate" or result.get("domain") != domain:
        raise ValueError(f"Expected a {domain} regression gate")
    if type(result.get("passed")) is not bool:
        raise ValueError(f"Missing gate decision for {domain}")
    return result


def build(agent, vision, output, *, revision=None, include_changes=False, plan_sha256=None):
    if revision is not None and not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("revision must be a full lowercase Git SHA")
    if plan_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", plan_sha256):
        raise ValueError("Invalid acceptance plan SHA-256")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "bundle"
        staged.mkdir()
        agent_report = copy_gate(agent, staged / "agent", "Agent", include_changes)
        vision_report = copy_gate(vision, staged / "vision", "VLM", include_changes)
        result = {"kind": "acceptance_bundle", "schema": 1, "revision": revision,
                  "passed": agent_report["passed"] and vision_report["passed"],
                  "gates": {"agent": agent_report["passed"], "vision": vision_report["passed"]},
                  "changes_included": include_changes}
        if plan_sha256:
            result["plan_sha256"] = plan_sha256
        write_json(staged / "summary.json", result)
        lines = ["# AI Practice 交付验收", "", f"结果  {'通过' if result['passed'] else '失败'}", ""]
        if revision:
            lines.extend([f"Git 提交  `{revision}`", ""])
        if plan_sha256:
            lines.extend([f"验收清单 SHA-256  `{plan_sha256}`", ""])
        lines.extend(gate_lines(agent_report))
        lines.extend(gate_lines(vision_report))
        (staged / "report.md").write_text("\n".join(lines), encoding="utf-8")
        inputs = sorted(path.relative_to(staged) for path in staged.rglob("*") if path.is_file())
        write_json(staged / "receipt.json", {"schema": 1, "artifacts": artifact_index(staged, inputs)})
        staged.rename(output)
    return result


def verify(output):
    receipt = read(output / "receipt.json")
    if receipt.get("schema") != 1:
        raise ValueError("Unsupported bundle receipt")
    index = receipt.get("artifacts", {})
    verify_index(output, index)
    summary = read(output / "summary.json")
    if summary.get("kind") != "acceptance_bundle" or summary.get("schema") != 1:
        raise ValueError("Invalid acceptance bundle summary")
    expected = {"summary.json", "report.md", "agent/summary.json", "vision/summary.json"}
    if summary["changes_included"]:
        expected.update({"agent/changes.jsonl", "vision/changes.jsonl"})
    files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if set(index) != expected or files != expected | {"receipt.json"}:
        raise ValueError("Unexpected or missing acceptance bundle files")
    for section, domain in (("agent", "Agent"), ("vision", "VLM")):
        gate = read(output / section / "summary.json")
        if gate.get("domain") != domain or gate.get("passed") is not summary["gates"][section]:
            raise ValueError(f"Bundle result disagrees with {section} gate")
    if summary["passed"] is not all(summary["gates"].values()):
        raise ValueError("Bundle decision does not match its gates")
    return {"verified_files": len(index), "revision": summary["revision"], "passed": summary["passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("build")
    create.add_argument("agent", type=Path)
    create.add_argument("vision", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--revision")
    create.add_argument("--include-changes", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    result = (build(args.agent, args.vision, args.output, revision=args.revision,
                    include_changes=args.include_changes)
              if args.action == "build" else verify(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "build" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
