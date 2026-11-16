"""Train a Qwen2.5-VL LoRA adapter with SFT or continue it with GRPO."""

import argparse
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

from .pipeline import messages, read_jsonl, score, write_jsonl

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def multimodal_dataset(path, stage):
    from datasets import Dataset, Image

    rows = []
    for row in read_jsonl(path):
        if row.get("split") != "train":
            raise ValueError("Training requires a prepared train manifest")
        prompt = messages(row["question"])
        if stage == "sft":
            payload = json.dumps({key: row[key] for key in ("answer", "evidence")}, ensure_ascii=False)
            rows.append({"prompt": prompt, "image": row["image"],
                         "completion": [{"role": "assistant", "content": [{"type": "text", "text": payload}]}]})
        else:
            rows.append({"prompt": prompt, "image": row["image"],
                         "answer": row["answer"], "evidence": row["evidence"]})
    if not rows:
        raise ValueError("Training manifest is empty")
    return Dataset.from_list(rows).cast_column("image", Image())


def reward(completions, answer, evidence, *, evidence_weight=0.5, **kwargs):
    rewards = []
    for completion, expected_answer, expected_evidence in zip(completions, answer, evidence):
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        result = score(completion, expected_answer, expected_evidence)
        rewards.append(result["accuracy"] + 0.2 * result["format"] + evidence_weight * result["evidence"])
    return rewards


def train(args):
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA with BF16 support")
    dataset = multimodal_dataset(args.manifest, args.stage)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left" if args.stage == "grpo" else "right"
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    options = dict(output_dir=str(args.output), max_steps=args.steps, bf16=True, seed=args.seed,
                   report_to="none", logging_steps=1, gradient_checkpointing=True,
                   save_strategy="steps", save_steps=args.save_steps, save_total_limit=2)
    common = dict(train_dataset=dataset, processing_class=processor)
    if args.stage == "sft":
        config = SFTConfig(**options, per_device_train_batch_size=1, learning_rate=1e-4,
                           max_length=None, completion_only_loss=True)
        trainer = SFTTrainer(model=base, args=config, peft_config=lora, **common)
    else:
        policy = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
        config = GRPOConfig(**options, per_device_train_batch_size=args.generations,
                            num_generations=args.generations, learning_rate=5e-6, beta=0.0,
                            max_completion_length=192, use_vllm=args.vllm,
                            vllm_mode="colocate", vllm_gpu_memory_utilization=0.25)

        def visual_reward(completions, answer, evidence, **kwargs):
            return reward(completions, answer, evidence, evidence_weight=args.evidence_weight)

        trainer = GRPOTrainer(model=policy, args=config, reward_funcs=visual_reward, **common)

    if trainer.is_world_process_zero():
        args.output.mkdir(parents=True, exist_ok=True)
        fingerprint = {
            "stage": args.stage, "model": args.model,
            "sft_adapter": str(args.adapter.resolve()) if args.adapter else None,
            "sft_adapter_sha256": hashlib.sha256(
                (args.adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest() if args.adapter else None,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "samples": len(dataset), "seed": args.seed,
            "steps": args.steps, "generations": args.generations if args.stage == "grpo" else None,
            "evidence_weight": args.evidence_weight if args.stage == "grpo" else None,
            "vllm": args.vllm, "save_steps": args.save_steps,
            "versions": {name: version(name) for name in ("torch", "transformers", "trl", "peft")},
        }
        (args.output / "run.json").write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.output))
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output))


def predict(args):
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to("cuda").eval()

    def predictions():
        with torch.inference_mode():
            for row in read_jsonl(args.manifest):
                prompt = processor.apply_chat_template(messages(row["question"]), tokenize=False,
                                                       add_generation_prompt=True)
                with Image.open(row["image"]) as image:
                    inputs = processor(text=[prompt], images=[image.convert("RGB")],
                                       return_tensors="pt").to(model.device)
                torch.cuda.synchronize(model.device)
                started = time.perf_counter()
                response = model.generate(**inputs, do_sample=False, max_new_tokens=192)
                torch.cuda.synchronize(model.device)
                seconds = time.perf_counter() - started
                new_tokens = response[0, inputs.input_ids.shape[1]:]
                completion = processor.decode(new_tokens, skip_special_tokens=True)
                yield {"id": row["id"], "completion": completion, "generation_seconds": seconds,
                       "generated_tokens": new_tokens.numel()}

    write_jsonl(args.output, predictions())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sft", "grpo", "predict"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--evidence-weight", type=float, default=0.5)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--resume", type=Path, help="Trainer checkpoint directory for this stage")
    args = parser.parse_args()
    if args.steps < 1 or args.save_steps < 1:
        parser.error("steps and save-steps must be positive")
    if not math.isfinite(args.evidence_weight) or args.evidence_weight < 0:
        parser.error("evidence-weight must be finite and nonnegative")
    if args.resume and (args.stage == "predict" or not (args.resume / "trainer_state.json").is_file()):
        parser.error("--resume requires an existing training checkpoint directory")
    if args.stage != "grpo" and args.vllm:
        parser.error("--vllm applies to GRPO")
    if args.stage == "grpo" and (not args.adapter or args.generations < 2):
        parser.error("GRPO requires --adapter and at least two generations")
    if args.stage == "grpo" and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("--adapter must contain adapter_model.safetensors")
    if args.stage == "sft" and args.adapter:
        parser.error("SFT starts from --model")
    if args.stage == "predict":
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
