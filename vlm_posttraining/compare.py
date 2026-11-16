"""Compare model predictions on one manifest with paired group-bootstrap intervals."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .pipeline import counterfactual_pairs, metrics, score_predictions, write_jsonl


def grounded_correct(row):
    return bool(row["accuracy"] and row["evidence"] == 1.0)


def paired_delta(base, candidate, rounds, seed, *, use_evidence=True):
    groups = defaultdict(list)
    improved = regressed = 0
    for left, right in zip(base, candidate):
        if use_evidence and left["evidence"] is None:
            continue
        change = (int(grounded_correct(right)) - int(grounded_correct(left)) if use_evidence
                  else int(right["accuracy"]) - int(left["accuracy"]))
        groups[left.get("split_group", left["source_group"])].append(change)
        improved += change == 1
        regressed += change == -1
    group_totals = [(sum(items), len(items)) for items in groups.values()]
    interval = None
    if len(group_totals) > 1:
        rng = random.Random(seed)
        bootstrap = []
        for _ in range(rounds):
            draw = rng.choices(group_totals, k=len(group_totals))
            bootstrap.append(sum(total for total, _ in draw) / sum(size for _, size in draw))
        bootstrap.sort()
        interval = [bootstrap[int(q * (rounds - 1))] for q in (0.025, 0.975)]
    eligible = sum(size for _, size in group_totals)
    key = "delta_grounded_accuracy" if use_evidence else "delta_accuracy"
    return {key: (improved - regressed) / eligible if eligible else None,
            "bootstrap_95": interval, "groups": len(groups), "samples": eligible,
            "improved": improved, "regressed": regressed}


def pair_metrics(rows):
    pairs = counterfactual_pairs(rows)
    if not pairs:
        return None
    annotated = [pair for pair in pairs if all(row["evidence"] is not None for row in pair)]
    return {
        "pairs": len(pairs), "evidence_annotated_pairs": len(annotated),
        "both_answers_correct": sum(all(row["accuracy"] for row in pair)
                                    for pair in pairs) / len(pairs),
        "both_grounded_correct": sum(all(map(grounded_correct, pair))
                                     for pair in annotated) / len(annotated) if annotated else None,
    }


def compare(manifest, predictions, output, rounds=1000, seed=42):
    if len(predictions) < 2 or rounds < 1:
        raise ValueError("Comparison requires at least two variants and a positive resample count")
    runs = {name: score_predictions(manifest, path) for name, path in predictions.items()}
    baseline = next(iter(runs))
    reference = runs[baseline]
    paired = {}
    reviews = []
    for name, rows in runs.items():
        if name == baseline:
            continue
        paired[name] = paired_delta(reference, rows, rounds, seed)
        paired[name]["answer_accuracy"] = paired_delta(reference, rows, rounds, seed, use_evidence=False)
        for left, right in zip(reference, rows):
            if not left["errors"] and not right["errors"]:
                continue
            reviews.append({
                **{key: left[key] for key in ("id", "image", "question", "answer",
                                              "task", "difficulty")},
                "dataset": left.get("dataset"),
                "answer_aliases": left.get("answer_aliases", []),
                "evidence": left["reference_evidence"],
                "evidence_observed": left.get("evidence_observed", True),
                "baseline": baseline, "candidate": name,
                "baseline_completion": left["completion"], "candidate_completion": right["completion"],
                "baseline_errors": left["errors"], "candidate_errors": right["errors"],
                "grounded_change": (None if left["evidence"] is None else
                                    int(grounded_correct(right)) - int(grounded_correct(left))),
            })
    variants = {}
    for name, rows in runs.items():
        slices = defaultdict(list)
        datasets = defaultdict(list)
        for row in rows:
            slices[f"{row['task']}/{row['difficulty']}"].append(row)
            if row.get("dataset"):
                datasets[row["dataset"]].append(row)
        overall = metrics(rows)
        eligible = overall["evidence_annotated"]
        variants[name] = {"overall": overall, "counterfactual": pair_metrics(rows),
                          "grounded_accuracy": sum(map(grounded_correct, rows)) / eligible if eligible else None,
                          "by_dataset": {key: metrics(items) for key, items in sorted(datasets.items())},
                          "by_task_difficulty": {key: metrics(items) for key, items in sorted(slices.items())}}
    report = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "baseline": baseline, "bootstrap_rounds": rounds, "seed": seed,
              "variants": variants, "paired": paired}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "review.jsonl", reviews)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH",
                        help="Repeat for each variant, with the baseline first")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    predictions = {}
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in predictions:
            parser.error("each prediction must have a distinct NAME=PATH")
        predictions[name] = Path(path)
    print(json.dumps(compare(args.manifest, predictions, args.output, args.rounds, args.seed), indent=2))


if __name__ == "__main__":
    main()
