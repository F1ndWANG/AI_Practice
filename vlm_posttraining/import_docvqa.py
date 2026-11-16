"""Convert licensed, locally available DocVQA Task 1 annotations to JSONL."""

import argparse
import json
from pathlib import Path

from .pipeline import normalize, write_jsonl


def convert(sources, images, output):
    images = images.resolve()
    rows = []
    ids = set()
    for split, annotations in sources:
        with annotations.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a nonempty DocVQA data array: {annotations}")
        for record in records:
            raw_id = record["questionId"]
            if type(raw_id) is not int and (not isinstance(raw_id, str) or not raw_id.strip()):
                raise ValueError(f"Invalid questionId in {annotations}")
            sample_id = f"docvqa:{raw_id}"
            if sample_id in ids:
                raise ValueError(f"Duplicate DocVQA questionId: {sample_id}")
            ids.add(sample_id)
            options = record.get("answers")
            if not isinstance(options, list) or not options or any(
                not isinstance(answer, str) or not normalize(answer) for answer in options
            ):
                raise ValueError(f"Missing answer labels for {sample_id}; use labeled annotations")
            unique = {}
            for answer in options:
                unique.setdefault(normalize(answer), answer.strip())
            answer, *aliases = unique.values()
            path = (images / record["image"]).resolve()
            if not path.is_relative_to(images) or not path.is_file():
                raise ValueError(f"Image missing or outside the image root: {record['image']}")
            document = record.get("ucsf_document_id") or record.get("docId")
            if type(document) not in (str, int) or not str(document).strip():
                raise ValueError(f"Missing document ID for {sample_id}")
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question for {sample_id}")
            entry = {
                "id": sample_id, "source_group": f"docvqa:{document}", "task": "docvqa",
                "difficulty": "unspecified", "image": str(path), "question": question.strip(),
                "answer": answer, "answer_aliases": aliases, "evidence": [], "evidence_observed": False,
            }
            if split != "auto":
                entry["split"] = split
            rows.append(entry)
    if not rows:
        raise ValueError("No questions to import")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_jsonl(output, rows)
    return {"questions": len(rows), "documents": len({row["source_group"] for row in rows}),
            "with_aliases": sum(bool(row["answer_aliases"]) for row in rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Root containing the annotation image paths")
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotations", action="append", required=True, metavar="SPLIT=FILE")
    args = parser.parse_args()
    sources = []
    for item in args.annotations:
        split, separator, path = item.partition("=")
        if not separator or not path or split not in {"auto", "train", "validation", "test"}:
            parser.error("Use auto, train, validation or test as the annotation split")
        sources.append((split, Path(path)))
    print(json.dumps(convert(sources, args.images, args.output), indent=2))


if __name__ == "__main__":
    main()
