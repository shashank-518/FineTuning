"""
Fine-tune a small local instruct LLM on badminton Q&A with LoRA (PEFT).

Works on CPU (slow) or CUDA GPU. Uses the Unsloth Studio Python if available,
otherwise the current environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_example(row: dict, tokenizer, system_prompt: str) -> str:
    """Chat template when available; plain Instruct fallback otherwise."""
    user = row["instruction"].strip()
    assistant = row["output"].strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return (
        f"System: {system_prompt}\n"
        f"User: {user}\n"
        f"Assistant: {assistant}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Badminton LoRA fine-tune")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = (ROOT / cfg["data_path"]).resolve()
    output_dir = (ROOT / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_path)
    if not rows:
        print(f"No examples found in {data_path}", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} badminton examples from {data_path}")
    print(f"Base model: {cfg['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    print(f"Device: {'CUDA' if use_cuda else 'CPU'} | dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if not use_cuda:
        model = model.to("cpu")

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg["target_modules"]),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    system_prompt = cfg["system_prompt"]
    texts = [format_example(r, tokenizer, system_prompt) for r in rows]
    dataset = Dataset.from_dict({"text": texts})

    max_len = int(cfg["max_seq_length"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=2,
        bf16=False,
        fp16=use_cuda,
        optim="adamw_torch",
        report_to="none",
        seed=int(cfg["seed"]),
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model": cfg["model_name"],
        "data_path": str(data_path),
        "num_examples": len(rows),
        "output_dir": str(output_dir),
        "device": "cuda" if use_cuda else "cpu",
    }
    (output_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nDone. LoRA adapter saved to: {output_dir}")
    print("Run inference with: python scripts/infer.py --question \"What is a smash?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
