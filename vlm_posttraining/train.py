"""Train a Qwen2.5-VL LoRA adapter with SFT or continue it with GRPO."""

import argparse
import json
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


def reward(completions, answer, evidence, **kwargs):
    rewards = []
    for completion, expected_answer, expected_evidence in zip(completions, answer, evidence):
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        result = score(completion, expected_answer, expected_evidence)
        rewards.append(result["accuracy"] + 0.2 * result["format"] + 0.5 * result["evidence"])
    return rewards


def train(args):
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left" if args.stage == "grpo" else "right"
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    dataset = multimodal_dataset(args.manifest, args.stage)
    options = dict(output_dir=str(args.output), max_steps=args.steps, bf16=True, seed=args.seed,
                   report_to="none", logging_steps=1, gradient_checkpointing=True)
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
        trainer = GRPOTrainer(model=policy, args=config, reward_funcs=reward, **common)
    trainer.train()
    trainer.save_model(str(args.output))
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
                response = model.generate(**inputs, do_sample=False, max_new_tokens=192)
                completion = processor.decode(response[0, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)
                yield {"id": row["id"], "completion": completion}

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
    args = parser.parse_args()
    if args.stage == "grpo" and (not args.adapter or args.generations < 2):
        parser.error("GRPO requires --adapter and at least two generations")
    if args.stage == "sft" and args.adapter:
        parser.error("SFT starts from --model")
    if args.stage == "predict":
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
