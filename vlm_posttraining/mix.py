"""Merge local data manifests and audit a shared train/validation/test assignment."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_practice.artifacts import artifact_index, sha256, verify_index, write_json

from .pipeline import prepare, read_jsonl, validate_images, write_jsonl


def mix(sources, output, seed=42):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not sources:
        raise ValueError("At least one input manifest is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    lineage = {}
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "mix"
        staged.mkdir()
        manifest = staged / "manifest.jsonl"

        def records():
            for name, source in sorted(sources.items()):
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                    raise ValueError(f"Invalid dataset name: {name}")
                checksum = sha256(source)
                count = 0
                group_map = {}
                for row in read_jsonl(source):
                    required = ("id", "source_group", "image")
                    if not isinstance(row, dict) or any(
                        not isinstance(row.get(key), str) or not row[key].strip() for key in required
                    ):
                        raise ValueError(f"Invalid ID, source group or image in {name}")
                    group = row["source_group"]
                    shared = row.get("shared_source_group")
                    if shared is not None and (not isinstance(shared, str) or not shared.strip()):
                        raise ValueError(f"Invalid shared source group in {name}")
                    mapped = f"global:{shared}" if shared is not None else f"dataset:{name}:{group}"
                    if group_map.setdefault(group, mapped) != mapped:
                        raise ValueError(f"Inconsistent shared source group for {name}:{group}")
                    count += 1
                    item = {**row, "dataset": name, "original_id": row["id"],
                            "id": f"{name}:{row['id']}", "source_group": mapped,
                            "image": str((source.parent / row["image"]).resolve())}
                    if row.get("pair_id") is not None:
                        if not isinstance(row["pair_id"], str) or not row["pair_id"].strip():
                            raise ValueError(f"Invalid pair identifier in {name}")
                        item["pair_id"] = f"{name}:{row['pair_id']}"
                    yield item
                if not count or sha256(source) != checksum:
                    raise ValueError(f"Input {name} is empty or changed during mixing")
                lineage[name] = {"path": str(source.resolve()), "sha256": checksum, "input_rows": count}

        write_jsonl(manifest, records())
        prepared = staged / "prepared"
        report = prepare(manifest, prepared, seed)
        retained = Counter()
        for split, stats in report["splits"].items():
            rows = list(read_jsonl(prepared / f"{split}.jsonl"))
            counts = Counter(row["dataset"] for row in rows)
            retained.update(counts)
            stats["by_dataset"] = dict(counts)
            stats["unannotated_evidence_by_dataset"] = dict(Counter(
                row["dataset"] for row in rows if not row.get("evidence_observed", True)
            ))
        for name, item in lineage.items():
            item["retained_rows"] = retained[name]
            item["duplicates_removed"] = item["input_rows"] - retained[name]
        paths = [manifest.relative_to(staged), *(
            path.relative_to(staged) for path in sorted(prepared.glob("*.jsonl"))
        )]
        result = {"kind": "multimodal_mix", "seed": seed, "sources": lineage, **report,
                  "artifacts": artifact_index(staged, paths)}
        write_json(staged / "audit.json", result)
        staged.rename(output)
    return result


def verify(output):
    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    artifacts = report.get("artifacts", {})
    expected = {"manifest.jsonl", *(f"prepared/{split}.jsonl"
                                   for split in ("train", "validation", "test"))}
    if report.get("kind") != "multimodal_mix" or set(artifacts) != expected:
        raise ValueError("Invalid dataset mix audit")
    verify_index(output, artifacts)
    validate_images(output / "prepared")
    return {"verified_manifests": len(artifacts), "retained_image_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=MANIFEST")
    parser.add_argument("--verify", action="store_true", help="Verify a previously generated mix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.verify == bool(args.source):
        parser.error("Provide one or more --source inputs or use --verify")
    if args.verify:
        print(json.dumps(verify(args.output), indent=2))
        return
    sources = {}
    for item in args.source:
        name, equals, manifest = item.partition("=")
        if not name or not equals or not manifest or name in sources:
            parser.error("Each source must have a distinct NAME=MANIFEST")
        sources[name] = Path(manifest)
    print(json.dumps(mix(sources, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
