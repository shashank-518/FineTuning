# Badminton Domain Fine-Tuning

Fine-tune a **small local instruct LLM** on badminton rules, technique, tactics, and equipment using **LoRA (PEFT)**.

Default model: [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (~360M params) — fits CPU and modest GPUs.

## Project layout

```
badminton-finetune/
  config.yaml              # model, LoRA, training hyperparams
  data/badminton_qa.jsonl  # ~100 badminton Q&A examples
  scripts/train.py         # LoRA fine-tune
  scripts/infer.py         # chat with the adapter
  scripts/app.py           # Shuttle web UI
  docs/                    # architecture + glossary PDF
  outputs/                 # saved LoRA weights
  requirements.txt
```

Architecture / development write-up / fine-tuning glossary PDF:

`docs/Badminton_FineTuning_Architecture_Guide.pdf`

Regenerate it with:

```powershell
python scripts/make_architecture_pdf.py
```

## Setup

From this folder, using your Unsloth Studio Python (recommended on this machine):

```powershell
cd C:\Users\shash\Documents\FineTuning\badminton-finetune
& 'C:\Users\shash\.unsloth\studio\unsloth_studio\Scripts\python.exe' -m pip install -r requirements.txt
```

Or any local venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python scripts/train.py
```

Quick smoke test (few steps only):

```powershell
python scripts/train.py --max-steps 10
```

Adapter saves to `outputs/badminton-smollm2-lora/`.

**Note:** This PC currently has **CPU-only PyTorch**. Training works but is slow. With an NVIDIA GPU + CUDA torch it will be much faster; change nothing in the scripts — CUDA is auto-detected.

## Web UI (recommended)

```powershell
python scripts/app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). First load pulls the model into memory (slow on CPU); each answer may take 20–60s.

## Infer (CLI)

```powershell
python scripts/infer.py --question "What is a jump smash?"
python scripts/infer.py --question "Explain doubles front-and-back formation"
```

Compare against the base model (no LoRA):

```powershell
python scripts/infer.py --base-only --question "What is a jump smash?"
```

## Swap to another small model

Edit `config.yaml`:

```yaml
model_name: Qwen/Qwen2.5-0.5B-Instruct
output_dir: outputs/badminton-qwen05-lora
```

Then re-run `python scripts/train.py`.

## Extend the dataset

Add lines to `data/badminton_qa.jsonl`:

```json
{"instruction": "Your question", "output": "Your answer"}
```

Keep answers factual and concise. Re-train after edits.

## Tips

- Start with the included ~100 examples to verify the pipeline, then grow the dataset (rules PDFs, coaching notes, FAQs).
- For stronger domain quality, aim for several hundred diverse Q&A pairs and 3–5 epochs.
- On CPU, keep `per_device_train_batch_size: 1` and use gradient accumulation (already set).
