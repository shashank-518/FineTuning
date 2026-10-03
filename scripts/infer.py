"""Chat with the fine-tuned badminton LoRA adapter (or base model)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system_prompt}\nUser: {question}\nAssistant:"


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Badminton LoRA inference")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA folder (defaults to config output_dir)",
    )
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--base-only", action="store_true", help="Skip LoRA adapter")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter = args.adapter or (ROOT / cfg["output_dir"])
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter) if adapter.exists() and not args.base_only else cfg["model_name"],
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )
    if not use_cuda:
        model = model.to("cpu")

    if not args.base_only:
        if not adapter.exists():
            raise FileNotFoundError(
                f"Adapter not found at {adapter}. Train first: python scripts/train.py"
            )
        model = PeftModel.from_pretrained(model, str(adapter))

    model.eval()
    prompt = build_prompt(tokenizer, cfg["system_prompt"], args.question)
    answer = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
