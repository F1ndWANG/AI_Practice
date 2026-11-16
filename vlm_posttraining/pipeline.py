"""Manifest validation, image/source isolation, rewards and visual error analysis."""

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ai_practice.artifacts import sha256


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_images(prepared):
    checked = set()
    for split in ("train", "validation", "test"):
        for row in read_jsonl(prepared / f"{split}.jsonl"):
            pair = (row["image"], row["image_sha256"])
            if pair not in checked and sha256(Path(pair[0])) != pair[1]:
                raise ValueError(f"Image changed since preparation: {pair[0]}")
            checked.add(pair)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def evidence_set(values):
    return {normalize(value) for value in values}


def counterfactual_pairs(rows):
    pairs = defaultdict(list)
    for row in rows:
        label = row.get("pair_id")
        if label is None:
            continue
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid pair_id on {row['id']}")
        pairs[label].append(row)
    for label, items in pairs.items():
        if len(items) != 2:
            raise ValueError(f"Counterfactual pair {label} requires exactly two samples")
        a, b = items
        a_answers, b_answers = (
            {normalize(item) for item in [row["answer"], *row.get("answer_aliases", ())]}
            for row in items
        )
        if (a["question"] != b["question"] or a_answers & b_answers
                or a.get("split") != b.get("split")
                or a.get("split_group", a["source_group"]) != b.get("split_group", b["source_group"])
                or a.get("image_sha256", a["image"]) == b.get("image_sha256", b["image"])):
            raise ValueError(f"Invalid counterfactual pair: {label}")
    return list(pairs.values())


def prepare(source, output, seed):
    rows = []
    parents = {}
    partitions = ("train", "validation", "test")
    seen_ids = set()
    seen_samples = {}
    image_hashes = {}
    duplicate_count = 0

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for entry in read_jsonl(source):
        for field in ("id", "source_group", "task", "difficulty", "image", "question", "answer"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"Missing or invalid {field}: {entry.get('id')}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Repeated sample ID: {entry['id']}")
        seen_ids.add(entry["id"])
        if entry["task"] not in {"ocr", "docvqa", "visual_reasoning"}:
            raise ValueError(f"Unknown task: {entry['task']}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise ValueError(f"Invalid evidence: {entry['id']}")
        aliases = entry.get("answer_aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(item, str) and item.strip() for item in aliases):
            raise ValueError(f"Invalid answer_aliases: {entry['id']}")
        observed = entry.get("evidence_observed", True)
        if type(observed) is not bool or (not observed and evidence):
            raise ValueError(f"Invalid evidence_observed: {entry['id']}")
        split = entry.get("split")
        if split is not None and split not in partitions:
            raise ValueError(f"Unknown split: {split}")
        image = (source.parent / entry["image"]).resolve()
        if image not in image_hashes:
            if image.stat().st_size == 0:
                raise ValueError(f"Empty image: {image}")
            image_hashes[image] = sha256(image)
        digest = image_hashes[image]
        source_key = ("source", entry["source_group"])
        image_key = ("visual", digest)
        image_root, source_root = root(image_key), root(source_key)
        parents[max(image_root, source_root)] = min(image_root, source_root)
        key = (digest, normalize(entry["question"]))
        signature = (frozenset(map(normalize, [entry["answer"], *aliases])),
                     frozenset(evidence_set(evidence)), observed)
        duplicate = key in seen_samples
        if duplicate and seen_samples[key] != signature:
            raise ValueError(f"Conflicting annotation for {entry['id']}")
        duplicate_count += int(duplicate)
        seen_samples[key] = signature
        rows.append((source_key, duplicate, {**entry, "image": str(image), "image_sha256": digest}))

    explicit = {}
    for key, _, row in rows:
        group = root(key)
        if row.get("split"):
            prior = explicit.setdefault(group, row["split"])
            if prior != row["split"]:
                raise ValueError(f"Source/image leakage between {prior} and {row['split']}")
    by_split = {name: [] for name in partitions}
    for key, duplicate, row in rows:
        if duplicate:
            continue
        group = root(key)
        random_value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
        bucket = random_value % 100
        split = explicit.get(group, "train" if bucket < 80 else "validation" if bucket < 90 else "test")
        by_split[split].append({**row, "split": split, "split_group": group[1]})
    if not all(by_split.values()):
        raise ValueError("Every split needs data. Assign explicit splits or add independent source groups.")
    pairs = counterfactual_pairs(row for entries in by_split.values() for row in entries)
    pair_counts = Counter(pair[0]["split"] for pair in pairs)
    summary = {"input": len(rows), "duplicates_removed": duplicate_count, "splits": {}}
    for split, entries in by_split.items():
        write_jsonl(output / f"{split}.jsonl", entries)
        summary["splits"][split] = {
            "count": len(entries), "counterfactual_pairs": pair_counts[split],
            "task_difficulty": dict(Counter(
                f"{row['task']}/{row['difficulty']}" for row in entries
            )),
        }
    return summary


def score(completion, answer, evidence, aliases=(), evidence_observed=True):
    try:
        result = json.loads(completion)
    except (ValueError, TypeError):
        result = {}
    if not isinstance(result, dict) or set(result) != {"answer", "evidence"} or not (
        isinstance(result["answer"], str) and isinstance(result["evidence"], list)
        and all(isinstance(item, str) and item.strip() for item in result["evidence"])
    ):
        return {"accuracy": 0.0, "format": 0.0,
                "evidence": 0.0 if evidence_observed else None, "errors": ["invalid_format"]}
    correct = normalize(result["answer"]) in set(map(normalize, [answer, *aliases]))
    expected = evidence_set(evidence)
    predicted = evidence_set(result["evidence"])
    denom = len(expected) + len(predicted)
    errors = []
    if not correct:
        errors.append("answer_mismatch")
    if evidence_observed and predicted - expected:
        errors.append("unsupported_evidence")
    if evidence_observed and expected - predicted:
        errors.append("missing_evidence")
    evidence_f1 = (2 * len(expected & predicted) / denom if denom else 1.0)
    return {"accuracy": float(correct), "format": 1.0,
            "evidence": evidence_f1 if evidence_observed else None, "errors": errors}


def reward_value(scored, evidence_weight):
    return scored["accuracy"] + 0.2 * scored["format"] + evidence_weight * (scored["evidence"] or 0.0)


def score_predictions(manifest, predictions):
    references = list(read_jsonl(manifest))
    by_id = {}
    for row in read_jsonl(predictions):
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        by_id[row["id"]] = row
    ids = {row["id"] for row in references}
    if not references or len(ids) != len(references) or set(by_id) != ids:
        raise ValueError("Predictions must match the nonempty manifest exactly")
    scored = []
    for row in references:
        prediction = by_id[row["id"]]
        completion = prediction["completion"]
        scored.append({
            **row, "completion": completion, "reference_evidence": row["evidence"],
            "generation_seconds": prediction.get("generation_seconds"),
            "generated_tokens": prediction.get("generated_tokens"),
            **score(completion, row["answer"], row["evidence"],
                    row.get("answer_aliases", ()), row.get("evidence_observed", True)),
        })
    return scored


def evaluate(manifest, predictions, review):
    scored = score_predictions(manifest, predictions)
    write_jsonl(review, (row for row in scored if row["errors"]))
    task_slices = {task: [row for row in scored if row["task"] == task]
                   for task in sorted({row["task"] for row in scored})}
    return {"overall": metrics(scored), "by_task": {key: metrics(rows) for key, rows in task_slices.items()}}


def performance(rows):
    measured = []
    for row in rows:
        seconds, tokens = row.get("generation_seconds"), row.get("generated_tokens")
        if seconds is None and tokens is None:
            continue
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                or type(tokens) is not int or tokens < 0):
            raise ValueError(f"Invalid generation measurements: {row['id']}")
        measured.append((seconds, tokens))
    if not measured:
        return None
    times = sorted(seconds for seconds, _ in measured)
    total_tokens = sum(tokens for _, tokens in measured)
    return {"measured": len(measured), "p50_seconds": times[math.ceil(0.50 * len(times)) - 1],
            "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
            "generated_tokens_per_second": total_tokens / sum(times)}


def metrics(rows):
    observed = [row for row in rows if row["evidence"] is not None]
    parsed = [row for row in observed if row["format"]]
    return {
        "n": len(rows),
        "generation_performance": performance(rows),
        **{key: sum(row[key] for row in rows) / len(rows)
           for key in ("accuracy", "format")},
        "evidence_annotated": len(observed),
        "evidence": sum(row["evidence"] for row in observed) / len(observed) if observed else None,
        "unsupported_evidence_rate_among_valid": (
            sum("unsupported_evidence" in row["errors"] for row in parsed) / len(parsed)
            if parsed else None
        ),
        "errors": dict(Counter(error for row in rows for error in row["errors"])),
    }


def messages(question, evidence_observed=True):
    instruction = ("Each evidence item must be a short observable fact from this image."
                   if evidence_observed else "Provide the answer and use an empty evidence array.")
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": (
        f"{question}\nReturn exactly a JSON object with string answer and array evidence. " + instruction
    )}]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("manifest", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--seed", type=int, default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("predictions", type=Path)
    evaluation.add_argument("review", type=Path)
    args = parser.parse_args()
    result = (prepare(args.manifest, args.output, args.seed) if args.command == "prepare"
              else evaluate(args.manifest, args.predictions, args.review))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
