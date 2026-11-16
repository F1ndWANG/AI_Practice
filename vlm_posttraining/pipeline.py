"""Manifest validation, image/source isolation, rewards and visual error analysis."""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


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


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def evidence_set(values):
    return {normalize(value) for value in values}


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
        split = entry.get("split")
        if split is not None and split not in partitions:
            raise ValueError(f"Unknown split: {split}")
        image = (source.parent / entry["image"]).resolve()
        if image not in image_hashes:
            content = image.read_bytes()
            if not content:
                raise ValueError(f"Empty image: {image}")
            image_hashes[image] = hashlib.sha256(content).hexdigest()
        digest = image_hashes[image]
        source_key = ("source", entry["source_group"])
        image_key = ("image", digest)
        parents[root(image_key)] = root(source_key)
        key = (digest, normalize(entry["question"]))
        signature = (normalize(entry["answer"]), frozenset(evidence_set(evidence)))
        duplicate = key in seen_samples
        if duplicate and seen_samples[key] != signature:
            raise ValueError(f"Conflicting annotation for {entry['id']}")
        duplicate_count += int(duplicate)
        seen_samples[key] = signature
        rows.append((source_key, duplicate, {**entry, "image": str(image)}))

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
        by_split[split].append({**row, "split": split})
    if not all(by_split.values()):
        raise ValueError("Every split needs data. Assign explicit splits or add independent source groups.")
    summary = {"input": len(rows), "duplicates_removed": duplicate_count, "splits": {}}
    for split, entries in by_split.items():
        write_jsonl(output / f"{split}.jsonl", entries)
        summary["splits"][split] = {
            "count": len(entries), "task_difficulty": dict(Counter(
                f"{row['task']}/{row['difficulty']}" for row in entries
            )),
        }
    return summary


def score(completion, answer, evidence):
    try:
        result = json.loads(completion)
    except (ValueError, TypeError):
        result = {}
    if not isinstance(result, dict) or set(result) != {"answer", "evidence"} or not (
        isinstance(result["answer"], str) and isinstance(result["evidence"], list)
        and all(isinstance(item, str) and item.strip() for item in result["evidence"])
    ):
        return {"accuracy": 0.0, "format": 0.0, "evidence": 0.0, "errors": ["invalid_format"]}
    correct = normalize(result["answer"]) == normalize(answer)
    expected = evidence_set(evidence)
    predicted = evidence_set(result["evidence"])
    denom = len(expected) + len(predicted)
    errors = []
    if not correct:
        errors.append("answer_mismatch")
    if predicted - expected:
        errors.append("unsupported_evidence")
    if expected - predicted:
        errors.append("missing_evidence")
    return {"accuracy": float(correct), "format": 1.0,
            "evidence": 2 * len(expected & predicted) / denom if denom else 1.0,
            "errors": errors}


def evaluate(manifest, predictions, review):
    references = list(read_jsonl(manifest))
    by_id = {}
    for row in read_jsonl(predictions):
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction: {row['id']}")
        by_id[row["id"]] = row["completion"]
    if not references or set(by_id) != {row["id"] for row in references}:
        raise ValueError("Predictions must match the nonempty manifest exactly")
    scored = [
        {**row, "completion": by_id[row["id"]], **score(by_id[row["id"]], row["answer"], row["evidence"])}
        for row in references
    ]
    write_jsonl(review, (row for row in scored if row["errors"]))
    task_slices = {task: [row for row in scored if row["task"] == task]
                   for task in sorted({row["task"] for row in scored})}
    return {"overall": metrics(scored), "by_task": {key: metrics(rows) for key, rows in task_slices.items()}}


def metrics(rows):
    parsed = [row for row in rows if row["format"]]
    return {
        "n": len(rows),
        **{key: sum(row[key] for row in rows) / len(rows)
           for key in ("accuracy", "format", "evidence")},
        "unsupported_evidence_rate_among_valid": (
            sum("unsupported_evidence" in row["errors"] for row in parsed) / len(parsed)
            if parsed else None
        ),
        "errors": dict(Counter(error for row in rows for error in row["errors"])),
    }


def messages(question):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": (
        f"{question}\nReturn exactly a JSON object with string answer and array evidence. "
        "Each evidence item must be a short observable fact from this image."
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
