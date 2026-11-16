"""Audit GRPO reward separation before spending GPU time on training."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from ai_practice.artifacts import sha256, write_json

from .pipeline import normalize, read_jsonl, reward_value, score, write_jsonl


def fixture_candidates(row):
    def output(answer, evidence):
        return json.dumps({"answer": answer, "evidence": evidence}, ensure_ascii=False)

    gold, evidence = row["answer"], row["evidence"]
    accepted = {normalize(value) for value in [gold, *row.get("answer_aliases", [])]}
    wrong = "[audit wrong answer]"
    while normalize(wrong) in accepted:
        wrong += "!"
    extra = "[audit unsupported evidence]"
    while normalize(extra) in {normalize(item) for item in evidence}:
        extra += "!"
    cases = [output(gold, evidence), output(gold, [*evidence, extra]),
             output(wrong, evidence), "invalid JSON"]
    if evidence:
        cases.append(output(gold, []))
    return cases


def inspect_groups(items):
    return {
        "groups": len(items),
        "candidates": sum(row["candidates"] for row in items),
        "zero_signal_groups": sum(row["reward_spread"] == 0.0 for row in items),
        "zero_signal_without_evidence": sum(row["spread_without_evidence"] == 0.0 for row in items),
        "answer_preference_inversions": sum(row["answer_preference_inverted"] for row in items),
        "answer_preference_ties": sum(row["answer_preference_tied"] for row in items),
        "groups_with_correct_and_wrong": sum(row["has_correct_and_wrong"] for row in items),
        "mean_reward_spread": mean(row["reward_spread"] for row in items),
        "malformed_candidates": sum(row["malformed"] for row in items),
    }


def audit(manifest, output, evidence_weight=0.5, candidates=None):
    if not math.isfinite(evidence_weight) or evidence_weight < 0:
        raise ValueError("Evidence weight must be finite and nonnegative")
    refs = {}
    for row in read_jsonl(manifest):
        if row["id"] in refs:
            raise ValueError(f"Repeated reference ID: {row['id']}")
        refs[row["id"]] = row
    if not refs:
        raise ValueError("Reference manifest is empty")
    if candidates is None:
        groups = {key: fixture_candidates(row) for key, row in refs.items()}
    else:
        groups = {}
        for row in read_jsonl(candidates):
            key, completions = row["id"], row["completions"]
            if key not in refs or key in groups:
                raise ValueError(f"Unknown or repeated candidate group: {key}")
            if not isinstance(completions, list) or len(completions) < 2 or any(
                not isinstance(item, str) for item in completions
            ):
                raise ValueError(f"Group {key} needs at least two text completions")
            groups[key] = completions
        if not groups:
            raise ValueError("No candidate groups supplied")

    by_slice = defaultdict(list)
    details = []
    for key, completions in groups.items():
        row = refs[key]
        metrics = [score(text, row["answer"], row["evidence"], row.get("answer_aliases", ()),
                         row.get("evidence_observed", True)) for text in completions]
        rewards = [reward_value(item, evidence_weight) for item in metrics]
        control = [reward_value(item, 0.0) for item in metrics]
        correct = [value for value, item in zip(rewards, metrics) if item["accuracy"]]
        wrong = [value for value, item in zip(rewards, metrics) if not item["accuracy"] and item["format"]]
        gap = min(correct) - max(wrong) if correct and wrong else None
        info = {"id": key, "task": row["task"], "evidence_observed": row.get("evidence_observed", True),
                "candidates": len(rewards), "reward_spread": max(rewards) - min(rewards),
                "spread_without_evidence": max(control) - min(control),
                "has_correct_and_wrong": gap is not None,
                "answer_preference_inverted": gap is not None and gap < -1e-12,
                "answer_preference_tied": gap is not None and abs(gap) <= 1e-12,
                "malformed": sum(not item["format"] for item in metrics),
                "scores": [{**item, "completion": text, "reward": value, "without_evidence": baseline}
                           for item, text, value, baseline in zip(metrics, completions, rewards, control)]}
        details.append(info)
        key = f"{row['task']}/{'evidence' if info['evidence_observed'] else 'answer_only'}"
        by_slice[key].append(info)

    report = {"kind": "vlm_reward_audit", "reference_sha256": sha256(manifest),
              "candidate_sha256": sha256(candidates) if candidates else None,
              "candidate_source": "external" if candidates else "synthetic_perturbations",
              "reference_samples": len(refs), "evidence_weight": evidence_weight,
              "overall": inspect_groups(details),
              "by_task_evidence": {key: inspect_groups(items) for key, items in sorted(by_slice.items())}}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "groups.jsonl", details)
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence-weight", type=float, default=0.5)
    parser.add_argument("--candidates", type=Path, help='Optional JSONL with id and completions')
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.output, args.evidence_weight, args.candidates), indent=2))


if __name__ == "__main__":
    main()
