"""Validate a multimodal dataset and optionally check local GPU readiness."""

import argparse
import json
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

from .pipeline import prepare, read_jsonl


def inspect_gpu(vllm):
    names = ("torch", "torchvision", "transformers", "datasets", "trl", "peft", "pillow")
    if vllm:
        names += ("vllm",)
    packages = {}
    missing = []
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError("Missing training packages: " + ", ".join(missing))
    import torch
    from datasets import Image as DatasetImage
    from torchvision.ops import nms
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOTrainer, SFTTrainer

    # Resolve lazy imports to surface mismatches before downloading weights.
    _ = (DatasetImage, nms, AutoModelForImageTextToText, AutoProcessor, GRPOTrainer, SFTTrainer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active GPU does not support BF16")
    device = torch.cuda.current_device()
    return {"packages": packages, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "memory_bytes": torch.cuda.get_device_properties(device).total_memory}


def inspect_dataset(source, seed, decode):
    formats = Counter()
    checked = set()
    with TemporaryDirectory() as directory:
        prepared = Path(directory)
        report = prepare(source, prepared, seed)
        for split in ("train", "validation", "test"):
            rows = list(read_jsonl(prepared / f"{split}.jsonl"))
            report["splits"][split]["source_groups"] = len({row["split_group"] for row in rows})
            report["splits"][split]["without_evidence"] = sum(not row["evidence"] for row in rows)
            report["splits"][split]["unannotated_evidence"] = sum(
                not row.get("evidence_observed", True) for row in rows
            )
            for row in rows:
                image_path = row["image"]
                if image_path in checked:
                    continue
                checked.add(image_path)
                if decode:
                    from PIL import Image, UnidentifiedImageError

                    try:
                        with Image.open(image_path) as image:
                            image.load()
                            formats[image.format] += 1
                    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                        raise ValueError(f"Unreadable image: {image_path}") from error
    return {**report, "unique_image_paths": len(checked),
            "decoded_image_formats": dict(formats) if decode else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-images", action="store_true", help="Decode every unique image with Pillow")
    parser.add_argument("--gpu", action="store_true", help="Check CUDA, BF16 and training imports")
    parser.add_argument("--vllm", action="store_true", help="Also check the vLLM package with --gpu")
    args = parser.parse_args()
    if args.vllm and not args.gpu:
        parser.error("--vllm requires --gpu")
    environment = inspect_gpu(args.vllm) if args.gpu else None
    dataset = inspect_dataset(args.manifest, args.seed, args.decode_images or args.gpu)
    print(json.dumps({"dataset": dataset, "gpu": environment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
