"""Shared model load + generate for CLI and web UI."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    path = path or (ROOT / "config.yaml")
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


class BadmintonModel:
    def __init__(self, cfg: dict | None = None, adapter: Path | None = None):
        self.cfg = cfg or load_config()
        self.adapter = Path(adapter) if adapter else (ROOT / self.cfg["output_dir"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        tok_src = (
            str(self.adapter)
            if self.adapter.exists()
            else self.cfg["model_name"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model_name"],
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            model = model.to("cpu")

        if not self.adapter.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {self.adapter}. Run: python scripts/train.py"
            )
        self.model = PeftModel.from_pretrained(model, str(self.adapter))
        self.model.eval()

    @torch.inference_mode()
    def ask(self, question: str) -> str:
        prompt = build_prompt(
            self.tokenizer, self.cfg["system_prompt"], question.strip()
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        temperature = float(self.cfg["temperature"])
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg["max_new_tokens"]),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=float(self.cfg["top_p"]),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return text.strip()
