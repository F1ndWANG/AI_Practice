"""Merge a LoRA adapter and export a standalone Transformers model directory."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_practice.artifacts import sha256, write_json

from .inference import DEFAULT_MODEL


def export(model, adapter, output, dtype="bf16"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_digest = sha256(adapter_weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        weight_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        base = AutoModelForImageTextToText.from_pretrained(model, dtype=weight_dtype)
        merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload(safe_merge=True)
        merged.save_pretrained(staged, safe_serialization=True, max_shard_size="2GB")
        AutoProcessor.from_pretrained(model).save_pretrained(staged)
        if not (staged / "config.json").is_file() or not list(staged.glob("*.safetensors")):
            raise RuntimeError("Export did not produce a model config and safetensors weights")
        if sha256(adapter_weights) != adapter_digest:
            raise RuntimeError("The adapter changed during export")
        metadata = {"base_model": model, "adapter_sha256": adapter_digest,
                    "dtype": dtype, "format": "transformers-merged-lora"}
        write_json(staged / "export.json", metadata)
        staged.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    if not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("Adapter directory must contain adapter_model.safetensors")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    print(json.dumps(export(args.model, args.adapter, args.output, args.dtype), indent=2))


if __name__ == "__main__":
    main()
