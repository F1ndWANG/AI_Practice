"""Generate synthetic, openly shareable vision samples for a CPU smoke run."""

import argparse
import json
import struct
import zlib
from pathlib import Path

from .pipeline import evaluate, prepare, read_jsonl, write_jsonl


def png(path, foreground, offset):
    width = height = 96
    white = bytes((255, 255, 255))
    color = bytes(foreground)
    pixels = b"".join(
        b"\x00" + b"".join(
            color if offset <= x < offset + 32 and 20 <= y < 52 else white
            for x in range(width)
        )
        for y in range(height)
    )

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def smoke(root):
    root.mkdir(parents=True, exist_ok=True)
    colors = {"red": (220, 30, 50), "green": (40, 160, 70), "blue": (45, 70, 220)}
    samples = []
    for index in range(12):
        color, rgb = list(colors.items())[index % 3]
        path = root / f"image-{index}.png"
        png(path, rgb, index + 10)
        split = "train" if index < 8 else "validation" if index < 10 else "test"
        samples.append({
            "id": str(index), "source_group": f"synthetic-{index}", "task": "visual_reasoning",
            "difficulty": "easy", "image": path.name, "question": "What color is the square?",
            "answer": color, "evidence": [f"{color} square"], "split": split,
        })
    samples.append({**samples[0], "id": "duplicate"})
    source = root / "manifest.jsonl"
    write_jsonl(source, samples)
    summary = prepare(source, root / "prepared", seed=42)
    test = root / "prepared/test.jsonl"
    predictions = root / "fixture-predictions.jsonl"
    write_jsonl(predictions, (
        {"id": row["id"], "completion": json.dumps({
            "answer": row["answer"],
            "evidence": row["evidence"] + (["purple triangle"] if index else []),
        })}
        for index, row in enumerate(read_jsonl(test))
    ))
    return {
        "prediction_source": "handwritten smoke fixture",
        "prepare": summary,
        "evaluate": evaluate(test, predictions, root / "review.jsonl"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(smoke(parser.parse_args().root), indent=2))
