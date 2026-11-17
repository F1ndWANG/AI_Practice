"""Freeze acceptance inputs and replay them with digest checks."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from .acceptance import REQUIRED_INPUTS, add_inputs, run
from .artifacts import sha256, write_json
from .gates import AGENT_VARIANTS, unique_mapping
from .release import verify
from vlm_posttraining.pipeline import read_jsonl


PATH_FIELDS = (*REQUIRED_INPUTS, "previous")
OPTION_FIELDS = ("agent_base_variant", "agent_candidate_variant", "revision", "include_changes")


def inputs_to_hash(options):
    paths = {key: options[key] for key in REQUIRED_INPUTS
             if key not in {"agent_base", "agent_candidate"}}
    for side in ("base", "candidate"):
        root = options[f"agent_{side}"]
        variant = options[f"agent_{side}_variant"]
        if variant not in AGENT_VARIANTS:
            raise ValueError(f"Unknown Agent variant: {variant}")
        for name in ("summary.json", "tasks.jsonl", f"{variant}.jsonl"):
            paths[f"agent_{side}/{name}"] = root / name
    manifest = options["vision_manifest"]
    for index, row in enumerate(read_jsonl(manifest)):
        if not row.get("image_sha256"):
            paths[f"vision_image/{index}"] = (manifest.parent / row["image"]).resolve()
    previous = options["previous"]
    if previous:
        verify(previous)
        paths["previous/receipt.json"] = previous / "receipt.json"
    return paths


def freeze(plan, options):
    if plan.exists():
        raise FileExistsError(plan)
    parent = plan.parent.resolve()
    options = {**options, **{key: options[key].resolve() if options[key] else None for key in PATH_FIELDS}}
    layout = {key: os.path.relpath(options[key], parent) if options[key] else None for key in PATH_FIELDS}
    digests = {key: sha256(path) for key, path in inputs_to_hash(options).items()}
    payload = {"schema": 1, "kind": "acceptance_plan", "paths": layout,
               "options": {key: options[key] for key in OPTION_FIELDS}, "digests": digests}
    write_json(plan, payload)
    return payload


def checked_options(plan):
    raw = plan.read_bytes()
    payload = json.loads(raw, object_pairs_hook=unique_mapping)
    if (not isinstance(payload, dict) or payload.get("schema") != 1
            or payload.get("kind") != "acceptance_plan"
            or not isinstance(payload.get("paths"), dict) or set(payload["paths"]) != set(PATH_FIELDS)
            or not isinstance(payload.get("options"), dict) or set(payload["options"]) != set(OPTION_FIELDS)):
        raise ValueError("Unsupported acceptance plan")
    options = dict(payload["options"])
    revision = options["revision"]
    if (type(options["include_changes"]) is not bool
            or (revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision)))):
        raise ValueError("Invalid acceptance plan options")
    for key, path in payload["paths"].items():
        if key == "previous" and path is None:
            options[key] = None
            continue
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"Expected a relative path for {key}")
        options[key] = (plan.parent / path).resolve()
    files = inputs_to_hash(options)
    if not isinstance(payload.get("digests"), dict) or set(payload["digests"]) != set(files):
        raise ValueError("Acceptance plan input inventory changed")
    for key, path in files.items():
        if sha256(path) != payload["digests"][key]:
            raise ValueError(f"Acceptance input changed: {key}")
    receipt = {"verified_inputs": len(files), "plan_sha256": hashlib.sha256(raw).hexdigest()}
    return options, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("freeze")
    create.add_argument("plan", type=Path)
    add_inputs(create)
    use = sub.add_parser("run")
    use.add_argument("plan", type=Path)
    use.add_argument("output", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "freeze":
            options = vars(args)
            path = options.pop("plan")
            options.pop("action")
            result = freeze(path, options)
        else:
            options, receipt = checked_options(args.plan)
            result = (run(args.output, **options, plan_sha256=receipt["plan_sha256"])
                      if args.action == "run" else receipt)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == "run":
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
