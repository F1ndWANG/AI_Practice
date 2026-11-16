"""Collect candidate groups from a label-free inference manifest for reward auditing."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_practice.artifacts import sha256, write_json

from .blind import PROMPT_FIELDS, fingerprint
from .inference import DEFAULT_MODEL, VisualModel
from .pipeline import read_jsonl, write_jsonl


def read_inputs(dispatch, limit):
    rows = list(read_jsonl(dispatch))
    if not rows:
        raise ValueError("Blind inference manifest is empty")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(PROMPT_FIELDS):
            raise ValueError("Sampling requires a label-free manifest produced by blind build")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError(f"Invalid or repeated inference ID: {row['id']}")
        if (not isinstance(row["question"], str) or not row["question"].strip()
                or not isinstance(row["image"], str) or not row["image"]
                or type(row["evidence_observed"]) is not bool):
            raise ValueError(f"Invalid prompt data for {row['id']}")
        ids.add(row["id"])
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    selected = rows[:limit]
    checked = {}
    for row in selected:
        path = row["image"]
        if path not in checked:
            checked[path] = sha256(Path(path))
        if checked[path] != row["image_sha256"]:
            raise ValueError(f"Sampling image changed: {path}")
    return fingerprint(rows), selected


def collect(dispatch, output, *, model_id=DEFAULT_MODEL, adapter=None, device="cuda", count=4,
            seed=42, temperature=0.7, top_p=0.9, max_new_tokens=192, limit=None):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if count < 2 or max_new_tokens < 1 or not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Require ≥2 candidates, positive temperature/token budget, and top-p in (0, 1]")
    dispatch_digest, selected = read_inputs(dispatch, limit)
    adapter_weights = adapter / "adapter_model.safetensors" if adapter else None
    adapter_digest = sha256(adapter_weights) if adapter_weights else None
    visual = VisualModel(model_id, adapter, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "candidates"
        staged.mkdir()
        def samples():
            for item in selected:
                key = json.dumps([seed, item["id"]], ensure_ascii=False)
                group_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                generated = visual.sample(item, count=count, seed=group_seed, temperature=temperature,
                                          top_p=top_p, max_new_tokens=max_new_tokens)
                if len(generated) != count:
                    raise RuntimeError(f"Model returned an incomplete candidate group for {item['id']}")
                yield {"id": item["id"], "completions": [row["completion"] for row in generated],
                       "generation": [{key: row[key] for key in ("generation_seconds", "generated_tokens")}
                                      for row in generated]}
        target = staged / "candidates.jsonl"
        write_jsonl(target, samples())
        if adapter_weights and sha256(adapter_weights) != adapter_digest:
            raise RuntimeError("Adapter weights changed during candidate collection")
        metadata = {"schema": 1, "dispatch_sha256": dispatch_digest, "candidates_sha256": sha256(target),
                    "groups": len(selected), "count": count, "seed": seed, "temperature": temperature,
                    "top_p": top_p, "max_new_tokens": max_new_tokens, "model": model_id,
                    "adapter": str(adapter) if adapter else None, "adapter_sha256": adapter_digest,
                    "device": device}
        write_json(staged / "sampling.json", metadata)
        staged.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = collect(args.dispatch, args.output, model_id=args.model, adapter=args.adapter, device=args.device,
                     count=args.count, seed=args.seed, temperature=args.temperature, top_p=args.top_p,
                     max_new_tokens=args.max_new_tokens, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
