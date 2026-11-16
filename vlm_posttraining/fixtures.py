"""Generate labeled image pairs for OCR, DocVQA and visual reasoning smoke tests."""

import argparse
import json
import random
from pathlib import Path

from .demo import png, write_png
from .pipeline import prepare, read_jsonl, write_jsonl

DIGITS = (
    ("111", "101", "101", "101", "111"), ("010", "110", "010", "010", "111"),
    ("111", "001", "111", "100", "111"), ("111", "001", "111", "001", "111"),
    ("101", "101", "111", "001", "001"), ("111", "100", "111", "001", "111"),
    ("111", "100", "111", "101", "111"), ("111", "001", "010", "010", "010"),
    ("111", "101", "111", "101", "111"), ("111", "101", "111", "001", "111"),
)


def ticket(path, top, bottom, offset):
    width, height = 128, 96
    rows = [bytearray(b"\xff" * (width * 3)) for _ in range(height)]
    for text, y in ((top, 10), (bottom, 54)):
        for index, digit in enumerate(text):
            for dy, line in enumerate(DIGITS[int(digit)]):
                for dx, mark in enumerate(line):
                    if mark == "1":
                        x = offset + index * 20 + dx * 4
                        for line_y in range(y + dy * 4, y + dy * 4 + 4):
                            rows[line_y][x * 3:(x + 4) * 3] = b"\x00" * 12
    write_png(path, width, height, b"".join(b"\x00" + row for row in rows))


def generate(output, seed=42):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    reference = []
    for index in range(12):
        split = "train" if index < 6 else "validation" if index < 9 else "test"
        for task in ("ocr", "docvqa", "visual_reasoning"):
            group = f"{task}-{index}"
            first, second = map(str, rng.sample(range(1000, 10000), 2))
            for side in range(2):
                sample_id = f"{group}-{side}"
                image = output / f"{sample_id}.png"
                value = (first, second)[side]
                if task == "visual_reasoning":
                    value, rgb = (("red", (215, 20, 40)), ("blue", (20, 40, 215)))[side]
                    png(image, rgb, index + 10)
                    question = "Is the square red or blue?"
                    evidence = f"{value} square"
                elif task == "ocr":
                    ticket(image, value, "", index + 5)
                    question = "What digits appear on the image?"
                    evidence = f"digits {value}"
                else:
                    ticket(image, str(index + 1000), value, index + 5)
                    question = "What is the number in the bottom row?"
                    evidence = f"bottom row {value}"
                reference.append({
                    "id": sample_id, "source_group": group, "pair_id": group,
                    "task": task, "difficulty": "two_rows" if task == "docvqa" else "easy",
                    "image": image.name, "question": question, "answer": value,
                    "evidence": [evidence], "split": split,
                })
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, reference)
    summary = prepare(manifest, output / "prepared", seed)
    test_rows = list(read_jsonl(output / "prepared/test.jsonl"))
    first_by_pair = {row["pair_id"]: row for row in reversed(test_rows)}
    for label, prediction_for in (
        ("repeated_answer", lambda row: first_by_pair[row["pair_id"]]),
        ("oracle", lambda row: row),
    ):
        write_jsonl(output / f"fixture-{label}.jsonl", (
            {"id": row["id"], "completion": json.dumps({
                key: prediction_for(row)[key] for key in ("answer", "evidence")
            })}
            for row in test_rows
        ))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.seed), indent=2))
