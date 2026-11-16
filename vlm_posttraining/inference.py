"""Load a base, adapted, or merged model and ask a question about one image."""

import argparse
import json
import time
from pathlib import Path

from .pipeline import messages

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class VisualModel:
    def __init__(self, model, adapter=None, device="cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu for local inference.")
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA inference requires BF16 support")
        self.torch = torch
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        policy = AutoModelForImageTextToText.from_pretrained(model, dtype=dtype)
        if adapter:
            policy = PeftModel.from_pretrained(policy, str(adapter))
        self.model = policy.to(device).eval()

    def image_inputs(self, image_path, question, evidence_observed=True):
        from PIL import Image

        conversation = messages(question, evidence_observed)
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        with Image.open(image_path) as image:
            return self.processor(text=[prompt], images=[image.convert("RGB")],
                                  return_tensors="pt").to(self.model.device)

    def generate(self, inputs, **options):
        if self.device == "cuda":
            self.torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            response = self.model.generate(**inputs, return_dict_in_generate=False, **options)
        if self.device == "cuda":
            self.torch.cuda.synchronize(self.model.device)
        seconds = time.perf_counter() - started
        tokens = response[0, inputs.input_ids.shape[1]:]
        return {"completion": self.processor.decode(tokens, skip_special_tokens=True),
                "generation_seconds": seconds, "generated_tokens": tokens.numel()}


    def ask(self, image_path, question, evidence_observed=True):
        inputs = self.image_inputs(image_path, question, evidence_observed)
        return self.generate(inputs, do_sample=False, max_new_tokens=192)

    def sample(self, item, *, count, seed, temperature, top_p, max_new_tokens):
        inputs = self.image_inputs(item["image"], item["question"], item["evidence_observed"])
        self.torch.manual_seed(seed)
        return [self.generate(inputs, do_sample=True, num_beams=1, num_return_sequences=1,
                              temperature=temperature, top_p=top_p, top_k=0,
                              max_new_tokens=max_new_tokens) for _ in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--answer-only", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    result = VisualModel(args.model, args.adapter, args.device).ask(
        args.image, args.question, not args.answer_only
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
