"""Run both acceptance gates, package their decisions, and optionally compare a prior run."""

import argparse
import json
from pathlib import Path

from .artifacts import write_json
from .compare_releases import compare, markdown
from .gates import AGENT_VARIANTS, gate_agent, gate_vision, load_policy
from .release import build, verify


def run(output, *, agent_base, agent_candidate, agent_policy, vision_manifest,
        vision_base, vision_candidate, vision_policy, agent_base_variant="baseline",
        agent_candidate_variant="brs_drs", revision=None, include_changes=False, previous=None, plan_sha256=None):
    output.mkdir(parents=True, exist_ok=False)
    status = {"kind": "acceptance_run", "schema": 1, "revision": revision, "stages": {},
              "passed": False, "exit_code": 2}
    if plan_sha256:
        status["plan_sha256"] = plan_sha256
    status_path = output / "status.json"
    write_json(status_path, status)

    def stage(name, operation):
        try:
            value = operation()
        except (OSError, ValueError, KeyError, TypeError) as error:
            status["stages"][name] = {"state": "error", "message": str(error)}
            write_json(status_path, status)
            return None
        status["stages"][name] = {"state": "passed" if value["passed"] else "failed"}
        write_json(status_path, status)
        return value

    agent = stage("agent", lambda: gate_agent(
        agent_base, agent_candidate, output / "agent-gate", baseline=agent_base_variant,
        candidate=agent_candidate_variant, **load_policy(agent_policy, "agent")
    ))
    vision = stage("vision", lambda: gate_vision(
        vision_manifest, vision_base, vision_candidate, output / "vision-gate",
        **load_policy(vision_policy, "vision")
    ))
    if agent is None or vision is None:
        status["stages"]["bundle"] = {"state": "skipped"}
        write_json(status_path, status)
        return status

    package = output / "bundle"

    def create_bundle():
        build(output / "agent-gate", output / "vision-gate", package,
              revision=revision, include_changes=include_changes, plan_sha256=plan_sha256)
        return verify(package)

    bundle = stage("bundle", create_bundle)
    if bundle is None:
        return status
    status["exit_code"] = 0 if bundle["passed"] else 1

    if previous is not None:
        def compare_bundle():
            report = compare(previous, package)
            destination = output / "comparison"
            destination.mkdir()
            write_json(destination / "summary.json", report)
            (destination / "report.md").write_text(markdown(report), encoding="utf-8")
            return {**report, "passed": report["comparable"] and not report["regressed"]}

        comparison = stage("comparison", compare_bundle)
        if comparison is None or not comparison["comparable"]:
            status["exit_code"] = 2
        elif not comparison["passed"]:
            status["exit_code"] = 1

    status["passed"] = status["exit_code"] == 0
    write_json(status_path, status)
    return status


REQUIRED_INPUTS = ("agent_base", "agent_candidate", "agent_policy", "vision_manifest",
                   "vision_base", "vision_candidate", "vision_policy")


def add_inputs(parser):
    for key in REQUIRED_INPUTS:
        option = key.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--agent-base-variant", choices=AGENT_VARIANTS, default="baseline")
    parser.add_argument("--agent-candidate-variant", choices=AGENT_VARIANTS, default="brs_drs")
    parser.add_argument("--revision")
    parser.add_argument("--include-changes", action="store_true")
    parser.add_argument("--previous", type=Path, help="Optional verified bundle from an earlier evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    add_inputs(parser)
    status = run(**vars(parser.parse_args()))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(status["exit_code"])


if __name__ == "__main__":
    main()
