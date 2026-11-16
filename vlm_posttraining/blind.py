"""Build and verify an inference-only manifest bound to a private reference."""

import argparse
import hashlib
import json
from pathlib import Path

from ai_practice.artifacts import sha256, write_json

from .pipeline import read_jsonl, write_jsonl


PROMPT_FIELDS = ("id", "image", "question", "evidence_observed", "image_sha256")


def prompts(reference):
    seen = set()
    for row in read_jsonl(reference):
        if row["id"] in seen:
            raise ValueError(f"Repeated reference ID: {row['id']}")
        seen.add(row["id"])
        path = (reference.parent / row["image"]).resolve()
        yield {"id": row["id"], "image": str(path), "question": row["question"],
               "evidence_observed": row.get("evidence_observed", True),
               "image_sha256": row["image_sha256"] if "image_sha256" in row else sha256(path)}


def fingerprint(rows):
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build(reference, dispatch, receipt):
    rows = list(prompts(reference))
    if not rows:
        raise ValueError("Reference manifest is empty")
    resolved = {path.resolve() for path in (reference, dispatch, receipt)}
    if len(resolved) != 3:
        raise ValueError("Reference, inference manifest and receipt must be distinct files")
    binding = {"schema": 1, "sample_count": len(rows), "inference_sha256": fingerprint(rows),
               "reference_sha256": sha256(reference)}
    write_jsonl(dispatch, rows)
    write_json(receipt, binding)
    return binding


def verify(reference, dispatch, receipt):
    binding = json.loads(receipt.read_text(encoding="utf-8"))
    expected = list(prompts(reference))
    rows = list(read_jsonl(dispatch))
    if (binding.get("schema") != 1 or binding.get("sample_count") != len(expected)
            or binding.get("reference_sha256") != sha256(reference)
            or binding.get("inference_sha256") != fingerprint(rows)
            or rows != expected):
        raise ValueError("Blind inference manifest or reference changed since dispatch")
    checked = {}
    for row in rows:
        path = row["image"]
        if path not in checked:
            checked[path] = sha256(Path(path))
        if checked[path] != row["image_sha256"]:
            raise ValueError(f"Blind inference image changed: {path}")
    return {"verified_samples": len(rows), "verified_images": len(checked)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("reference", type=Path)
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    operation = build if args.action == "build" else verify
    print(json.dumps(operation(args.reference, args.dispatch, args.receipt), indent=2))


if __name__ == "__main__":
    main()
