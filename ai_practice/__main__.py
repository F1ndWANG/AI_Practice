"""Run and verify the CPU acceptance workflow for both projects."""

import argparse
import json
from pathlib import Path

from rsi_agent.study import study
from vlm_posttraining.compare import compare
from vlm_posttraining.fixtures import generate

from .artifacts import artifact_index, verify_index, write_json
from .report import render


def smoke(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    agent = study(output / "agent", seed=seed)
    fixtures = output / "fixtures"
    generated = generate(fixtures, seed)
    vision = compare(
        fixtures / "prepared/test.jsonl",
        {"repeated_answer": fixtures / "fixture-repeated_answer.jsonl",
         "oracle": fixtures / "fixture-oracle.jsonl"},
        output / "vision", seed=seed,
    )
    variants = agent["variants"]
    if not all(row["success_rate"] == 1.0 for row in variants.values()):
        raise RuntimeError("Local agent task failures")
    if not variants["brs_drs"]["tool_calls"] < variants["brs"]["tool_calls"] < variants["baseline"]["tool_calls"]:
        raise RuntimeError("Expected fewer test calls after both exploration stages")
    if (vision["variants"]["repeated_answer"]["counterfactual"]["both_answers_correct"] != 0
            or vision["variants"]["oracle"]["counterfactual"]["both_grounded_correct"] != 1):
        raise RuntimeError("Unexpected paired evaluation result")

    summary = {"seed": seed, "agent": variants, "synthetic_vision_samples": generated["input"],
               "vision_fixture_comparison": vision["paired"]}
    write_json(output / "summary.json", summary)
    files = sorted(path.relative_to(output) for path in output.rglob("*") if path.is_file())
    write_json(output / "receipt.json", {"schema": 1, "artifacts": artifact_index(output, files)})
    return summary


def verify(output):
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != 1 or "summary.json" not in receipt.get("artifacts", {}):
        raise ValueError("Unrecognized smoke receipt")
    verify_index(output, receipt["artifacts"])
    return {"verified": len(receipt["artifacts"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("smoke", help="Run both CPU experiments")
    run.add_argument("output", type=Path)
    run.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("verify", help="Verify the smoke run artifact hashes")
    check.add_argument("output", type=Path)
    report = commands.add_parser("report", help="Render an existing run as Markdown")
    report.add_argument("output", type=Path, help="Completed experiment directory")
    args = parser.parse_args()
    if args.command == "report":
        print(render(args.output))
        return
    result = smoke(args.output, args.seed) if args.command == "smoke" else verify(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
