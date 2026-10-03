"""Generate Architecture & Fine-Tuning Guide PDF for Shuttle / badminton-finetune."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Badminton_FineTuning_Architecture_Guide.pdf"
FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


class GuidePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Body", "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "Shuttle — Badminton Fine-Tuning Architecture & Glossary", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-14)
        self.set_font("Body", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}  |  {date.today().isoformat()}", align="C")

    def _reset_x(self):
        self.set_x(self.l_margin)

    def h1(self, text: str):
        self._reset_x()
        self.set_font("BodyBold", "", 18)
        self.set_text_color(15, 61, 46)
        self.multi_cell(0, 10, text)
        self._reset_x()
        self.ln(2)

    def h2(self, text: str):
        self.ln(3)
        self._reset_x()
        self.set_font("BodyBold", "", 14)
        self.set_text_color(20, 80, 60)
        self.multi_cell(0, 8, text)
        self._reset_x()
        self.ln(1)

    def h3(self, text: str):
        self.ln(2)
        self._reset_x()
        self.set_font("BodyBold", "", 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 7, text)
        self._reset_x()
        self.ln(1)

    def p(self, text: str):
        self._reset_x()
        self.set_font("Body", "", 10)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5.5, text)
        self._reset_x()
        self.ln(1.5)

    def bullet(self, text: str):
        self._reset_x()
        self.set_font("Body", "", 10)
        self.set_text_color(25, 25, 25)
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self._reset_x()
        self.ln(0.5)

    def code_block(self, text: str):
        self._reset_x()
        self.set_fill_color(240, 245, 242)
        self.set_font("Body", "", 8.5)
        self.set_text_color(20, 40, 30)
        self.multi_cell(0, 4.5, text, fill=True)
        self._reset_x()
        self.ln(2)

    def term(self, name: str, meaning: str):
        self._reset_x()
        self.set_font("BodyBold", "", 10)
        self.set_text_color(15, 61, 46)
        self.multi_cell(0, 5.5, name)
        self._reset_x()
        self.set_font("Body", "", 10)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5.5, meaning)
        self._reset_x()
        self.ln(1.5)


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = GuidePDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_font("Body", "", FONT)
    pdf.add_font("BodyBold", "", FONT_BOLD)
    pdf.add_page()

    # Title
    pdf.set_x(pdf.l_margin)
    pdf.set_font("BodyBold", "", 22)
    pdf.set_text_color(15, 61, 46)
    pdf.multi_cell(0, 11, "Shuttle: Badminton Domain Fine-Tuning")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("BodyBold", "", 14)
    pdf.multi_cell(0, 8, "Architecture, Development Story and Fine-Tuning Glossary")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Body", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0,
        6,
        f"Project path: FineTuning/badminton-finetune\n"
        f"Document date: {date.today().isoformat()}\n"
        f"Base model: HuggingFaceTB/SmolLM2-360M-Instruct  |  Method: LoRA (PEFT)",
    )
    pdf.set_x(pdf.l_margin)
    pdf.ln(4)

    # 1 Overview
    pdf.h1("1. Project Overview")
    pdf.p(
        "This project fine-tunes a small local instruction-following language model on "
        "badminton knowledge (rules, strokes, tactics, equipment, tournaments) and serves "
        "it through a browser chat UI named Shuttle. The goal was a complete, runnable "
        "pipeline on a CPU-only Windows machine: dataset -> LoRA training -> inference -> frontend."
    )
    pdf.bullet("Domain: Badminton coaching and rules Q&A")
    pdf.bullet("Model size: ~360M parameters (SmolLM2-360M-Instruct)")
    pdf.bullet("Adaptation method: LoRA adapters via Hugging Face PEFT (~1.2% trainable params)")
    pdf.bullet("UI: FastAPI backend + static HTML/CSS/JS frontend on http://127.0.0.1:7860")
    pdf.bullet("Dataset: 110 curated JSONL instruction/output pairs")

    # 2 Architecture
    pdf.h1("2. System Architecture")
    pdf.h2("2.1 High-level pipeline")
    pdf.code_block(
        "  [Badminton Q&A JSONL]\n"
        "            |\n"
        "            v\n"
        "  [config.yaml] -----> [scripts/train.py]\n"
        "                            |\n"
        "                            |  LoRA fine-tune (PEFT + Transformers Trainer)\n"
        "                            v\n"
        "               [outputs/badminton-smollm2-lora]\n"
        "                 (adapter weights + tokenizer)\n"
        "                            |\n"
        "              +-------------+-------------+\n"
        "              |                           |\n"
        "              v                           v\n"
        "     [scripts/infer.py]          [scripts/app.py]\n"
        "        CLI chat                    FastAPI server\n"
        "                                          |\n"
        "                                          v\n"
        "                               [frontend/ index + CSS + JS]\n"
        "                               Browser UI: Shuttle Coach"
    )
    pdf.h2("2.2 Component responsibilities")
    pdf.bullet(
        "data/badminton_qa.jsonl — supervised instruction dataset; each line is "
        '{"instruction": "...", "output": "..."}.'
    )
    pdf.bullet(
        "config.yaml — single source of truth for model name, LoRA ranks, epochs, "
        "batch size, learning rate, system prompt, and inference sampling."
    )
    pdf.bullet(
        "scripts/train.py — loads base model + tokenizer, wraps with LoRA, formats chat "
        "templates, runs Hugging Face Trainer, saves adapter to outputs/."
    )
    pdf.bullet(
        "scripts/runtime.py — shared BadmintonModel class used by web and (optionally) CLI: "
        "load base + LoRA, build chat prompt, generate answer."
    )
    pdf.bullet(
        "scripts/infer.py — command-line one-shot Q&A against the trained adapter."
    )
    pdf.bullet(
        "scripts/app.py — FastAPI lifespan loads the model once; POST /api/chat returns replies; "
        "serves frontend static files."
    )
    pdf.bullet(
        "frontend/ — Shuttle UI (court-green theme): brand, message log, composer, calls /api/chat."
    )

    pdf.h2("2.3 Runtime request flow (web)")
    pdf.code_block(
        "Browser  --GET /-->  FastAPI serves index.html\n"
        "Browser  --POST /api/chat {message}-->  BadmintonModel.ask()\n"
        "   1) Apply chat template (system + user)\n"
        "   2) Tokenize -> model.generate()\n"
        "   3) Decode new tokens -> JSON {reply}\n"
        "Browser renders assistant message in chat log"
    )
    pdf.p(
        "On this machine PyTorch is CPU-only (AMD GPU without ROCm mapping), so both training "
        "and generation run on CPU. First model load is slow; each answer may take tens of seconds."
    )

    pdf.h2("2.4 Training compute graph (conceptual)")
    pdf.code_block(
        "Base LLM weights (frozen)  +  LoRA low-rank matrices (trainable)\n"
        "        |                              |\n"
        "        +-------- forward pass --------+\n"
        "                     |\n"
        "              next-token loss\n"
        "                     |\n"
        "           backward updates ONLY LoRA A/B\n"
        "                     |\n"
        "           save adapter_model.safetensors"
    )

    # 3 Development story
    pdf.add_page()
    pdf.h1("3. How This Was Developed")
    pdf.h2("3.1 Context and constraints")
    pdf.p(
        "The workspace already contained an Unsloth Studio install under FineTuning/unsloth, "
        "with a managed Python environment at ~/.unsloth/studio/unsloth_studio. Hardware "
        "detection installed CPU-only PyTorch because the AMD iGPU arch was unknown. Therefore "
        "the badminton project was designed for a small instruct model + PEFT LoRA that can "
        "train on CPU, rather than a large GPU-only Unsloth CUDA recipe."
    )
    pdf.h2("3.2 Step-by-step development")
    pdf.h3("Step A — Scaffold the project")
    pdf.p(
        "Created badminton-finetune/ with folders for data, scripts, frontend, and outputs; "
        "added config.yaml, requirements.txt, and README so hyperparameters and run commands "
        "live in one place."
    )
    pdf.h3("Step B — Build a domain dataset")
    pdf.p(
        "Wrote ~110 badminton Q&A pairs covering court dimensions, scoring (rally to 21), "
        "serves, faults, strokes (clear, smash, drop, drive, lift), doubles formations, "
        "equipment (U weights, string tension), fitness, and major events (BWF, Thomas/Uber Cup). "
        "JSONL keeps each example independent and easy to extend."
    )
    pdf.h3("Step C — Training script")
    pdf.p(
        "Implemented scripts/train.py using Transformers + PEFT: load SmolLM2-360M-Instruct, "
        "attach LoRA on attention and MLP projections (q/k/v/o/gate/up/down), format examples "
        "with the model's chat template, and train with causal language-modeling loss "
        "(predict next tokens of the assistant answer). Defaults: 3 epochs, batch size 1, "
        "gradient accumulation 8, LoRA r=8, alpha=16, lr=2e-4."
    )
    pdf.h3("Step D — Validate with a smoke run, then full train")
    pdf.p(
        "First ran --max-steps 3 to verify downloads, tokenization, and checkpointing. Then "
        "ran the full 3-epoch job (~42 optimizer steps, ~18 minutes on CPU). Training loss "
        "fell from about 3.0 to about 1.75–2.2 average, indicating the adapter was learning "
        "the badminton phrasing patterns."
    )
    pdf.h3("Step E — Inference CLI")
    pdf.p(
        "Added scripts/infer.py to ask questions from the terminal and compare --base-only "
        "vs LoRA-adapted answers."
    )
    pdf.h3("Step F — Web frontend")
    pdf.p(
        "Because a UI is easier for demos, extracted shared loading into scripts/runtime.py, "
        "built FastAPI app.py with /api/chat, and designed the Shuttle frontend (Syne + Figtree, "
        "court-green palette, chat composer). Server loads the adapter once at startup and "
        "serves http://127.0.0.1:7860."
    )
    pdf.h3("Step G — Documentation PDF")
    pdf.p(
        "This PDF (generated by scripts/make_architecture_pdf.py) captures architecture, "
        "development rationale, and a glossary of fine-tuning terms for presentations and study."
    )

    pdf.h2("3.3 Design choices and why")
    pdf.bullet(
        "Small instruct model: fits RAM/CPU; good for learning the pipeline before scaling up."
    )
    pdf.bullet(
        "LoRA instead of full fine-tune: trains millions of params, not hundreds of millions; "
        "fast to save/share; base model stays reusable."
    )
    pdf.bullet(
        "PEFT/Transformers over Unsloth kernels: Unsloth shines on NVIDIA CUDA; this host is CPU/AMD."
    )
    pdf.bullet(
        "Chat template formatting: matches how instruct models expect system/user/assistant turns."
    )
    pdf.bullet(
        "Single config.yaml: avoids scattering hyperparameters across scripts."
    )

    pdf.h2("3.4 Project tree")
    pdf.code_block(
        "badminton-finetune/\n"
        "  config.yaml\n"
        "  requirements.txt\n"
        "  README.md\n"
        "  data/badminton_qa.jsonl\n"
        "  scripts/train.py\n"
        "  scripts/infer.py\n"
        "  scripts/runtime.py\n"
        "  scripts/app.py\n"
        "  scripts/make_architecture_pdf.py\n"
        "  frontend/index.html | styles.css | app.js\n"
        "  outputs/badminton-smollm2-lora/   # trained LoRA\n"
        "  docs/Badminton_FineTuning_Architecture_Guide.pdf"
    )

    # 4 How to run
    pdf.add_page()
    pdf.h1("4. How to Run")
    pdf.h2("Train")
    pdf.code_block(
        "cd C:\\Users\\shash\\Documents\\FineTuning\\badminton-finetune\n"
        "& 'C:\\Users\\shash\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe' "
        "scripts\\train.py"
    )
    pdf.h2("Web UI")
    pdf.code_block(
        "& 'C:\\Users\\shash\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe' "
        "scripts\\app.py\n"
        "Open http://127.0.0.1:7860"
    )
    pdf.h2("CLI ask")
    pdf.code_block(
        "& 'C:\\Users\\shash\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe' "
        "scripts\\infer.py --question \"What is a jump smash?\""
    )
    pdf.h2("Regenerate this PDF")
    pdf.code_block(
        "& 'C:\\Users\\shash\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe' "
        "scripts\\make_architecture_pdf.py"
    )

    # 5 Results
    pdf.h1("5. Observed Training Results")
    pdf.bullet("Examples: 110")
    pdf.bullet("Trainable params: ~4.34M / ~366M total (~1.19%)")
    pdf.bullet("Steps: 42 (3 epochs with grad accumulation)")
    pdf.bullet("Device: CPU, float32")
    pdf.bullet("Approx. wall time: ~18 minutes for full train")
    pdf.bullet("Loss trend: ~3.01 early -> ~1.75–1.91 late; average train_loss ~2.20")
    pdf.p(
        "Quality note: a 360M model with 110 examples learns domain style and many facts, "
        "but can still hallucinate details. Grow the dataset and/or use a larger model "
        "(e.g. Qwen2.5-0.5B/1.5B) for better accuracy."
    )

    # 6 Glossary
    pdf.add_page()
    pdf.h1("6. Fine-Tuning Terms Glossary")
    pdf.p(
        "Definitions below match how terms are used in this project and in common LLM fine-tuning practice."
    )

    terms = [
        (
            "Large Language Model (LLM)",
            "A neural network trained to predict the next token in text. After pretraining on "
            "huge corpora, it can be specialized with fine-tuning.",
        ),
        (
            "Base model / Pretrained model",
            "The starting checkpoint (here SmolLM2-360M-Instruct) whose general language "
            "ability we reuse instead of training from scratch.",
        ),
        (
            "Instruct model",
            "A model further tuned to follow user instructions in a chat style "
            "(system / user / assistant turns), not only raw text completion.",
        ),
        (
            "Fine-tuning",
            "Continued training on a smaller, task- or domain-specific dataset so the model "
            "behaves better for that domain (here: badminton Q&A).",
        ),
        (
            "Domain adaptation",
            "Fine-tuning aimed at a subject area (sports rules/coaching) rather than a "
            "generic skill like summarization.",
        ),
        (
            "Supervised Fine-Tuning (SFT)",
            "Training on labeled pairs (prompt -> desired answer). This project is classic SFT "
            "on instruction/output JSONL.",
        ),
        (
            "Instruction tuning",
            "SFT where examples look like user questions and assistant answers, teaching "
            "the model to respond helpfully to commands.",
        ),
        (
            "Parameter-Efficient Fine-Tuning (PEFT)",
            "Family of methods that update only a small subset of parameters (or small add-on "
            "modules) instead of all model weights. LoRA is a PEFT method; we use the "
            "Hugging Face peft library.",
        ),
        (
            "LoRA (Low-Rank Adaptation)",
            "Injects small trainable matrices (A and B) into linear layers. For a weight W, "
            "effective update is BA (low rank r). Base W stays frozen; only A/B train.",
        ),
        (
            "LoRA rank (r)",
            "Inner dimension of the low-rank update. Higher r = more capacity and VRAM; "
            "this project uses r=8.",
        ),
        (
            "LoRA alpha",
            "Scaling factor for the LoRA update (often effective scale alpha/r). We use alpha=16.",
        ),
        (
            "LoRA dropout",
            "Dropout on LoRA layers during training for regularization (0.05 here).",
        ),
        (
            "Target modules",
            "Which linear layers receive LoRA (q_proj, k_proj, v_proj, o_proj, gate/up/down_proj).",
        ),
        (
            "Adapter",
            "The saved LoRA weights (and config) living in outputs/badminton-smollm2-lora. "
            "At inference we load base model + adapter.",
        ),
        (
            "Full fine-tuning",
            "Updating all model parameters. More powerful but expensive; not used here.",
        ),
        (
            "Frozen weights",
            "Parameters that do not receive gradients. In LoRA, the original base weights are frozen.",
        ),
        (
            "Trainable parameters",
            "Parameters that do update. Here ~4.3M LoRA params (~1.2% of the model).",
        ),
        (
            "Token / Tokenization",
            "Text is split into tokens (subwords/ids) the model understands. The tokenizer "
            "converts text <-> token ids.",
        ),
        (
            "Chat template",
            "Model-specific formatting that wraps system/user/assistant messages with special "
            "markers before tokenization.",
        ),
        (
            "System prompt",
            "Standing instruction (e.g. 'You are a badminton coach...') prepended to each chat.",
        ),
        (
            "Context length / max_seq_length",
            "Maximum tokens per training example (512 in config). Longer text is truncated.",
        ),
        (
            "Causal Language Modeling (CLM)",
            "Loss that trains the model to predict each next token given previous tokens. "
            "Used for GPT-style fine-tuning.",
        ),
        (
            "Loss / Training loss",
            "Number measuring how wrong predictions are. Lower usually means better fit to "
            "the training answers (watch for overfitting).",
        ),
        (
            "Epoch",
            "One full pass over the training dataset. We used 3 epochs.",
        ),
        (
            "Step / Optimizer step",
            "One parameter update. With gradient accumulation, several mini-batches contribute "
            "before one step.",
        ),
        (
            "Batch size (per_device_train_batch_size)",
            "Examples processed together on one device per forward/backward. CPU default: 1.",
        ),
        (
            "Gradient accumulation",
            "Sum gradients over N mini-batches before updating (effective batch ~= batch * N). "
            "We use 8 to simulate a larger batch on low memory.",
        ),
        (
            "Learning rate",
            "How large each update is. Too high diverges; too low learns slowly. Config: 2e-4.",
        ),
        (
            "Warmup",
            "Gradually increase LR at the start for stability (warmup_ratio 0.05).",
        ),
        (
            "Weight decay",
            "Regularization that shrinks weights slightly to reduce overfitting (0.01).",
        ),
        (
            "Gradient checkpointing",
            "Trades compute for memory by recomputing activations in backward pass; enables "
            "larger models on limited RAM.",
        ),
        (
            "Checkpoint",
            "Saved mid-training snapshot (e.g. checkpoint-42) so you can resume or pick a best step.",
        ),
        (
            "Overfitting",
            "Memorizing the small dataset so the model repeats training answers but fails on "
            "new questions. Mitigate with more/diverse data and modest epochs.",
        ),
        (
            "Underfitting",
            "Model has not learned the domain well yet (high loss, vague answers). Fix with "
            "more steps/data or higher capacity.",
        ),
        (
            "Inference / Generation",
            "Using the trained model to produce answers without updating weights.",
        ),
        (
            "Sampling (temperature, top_p)",
            "Controls randomness: higher temperature = more creative/varied; top_p nucleus "
            "sampling limits to a probable token set. Config: temperature 0.7, top_p 0.9.",
        ),
        (
            "max_new_tokens",
            "Cap on how many tokens the model may generate in one answer (256).",
        ),
        (
            "Quantization (related term)",
            "Storing weights in lower precision (4-bit/8-bit) to save memory. Common with "
            "QLoRA on GPUs; not required for this small CPU float32 setup.",
        ),
        (
            "QLoRA (related term)",
            "LoRA on top of a quantized base model — popular for fine-tuning 7B+ models on "
            "a single GPU. Not used here because the 360M model fits in float32 on CPU.",
        ),
        (
            "RLHF / DPO (related terms)",
            "Preference-based alignment methods beyond SFT. Not used in this project; we only do SFT.",
        ),
        (
            "Evaluation / Eval set",
            "Held-out questions to measure quality. This starter project trains on all 110 "
            "examples; for production, split train/validation.",
        ),
        (
            "Hallucination",
            "Confident but wrong facts. Small models and tiny datasets are prone to this — "
            "verify critical rules against official BWF laws.",
        ),
        (
            "JSONL",
            "JSON Lines file format: one JSON object per line — convenient for streaming datasets.",
        ),
        (
            "Effective batch size",
            "per_device_batch * gradient_accumulation * num_gpus. Here: 1 * 8 * 1 = 8.",
        ),
        (
            "Device map / CUDA vs CPU",
            "Where tensors live. CUDA = NVIDIA GPU acceleration. This install used CPU because "
            "CUDA/ROCm was not available for the detected AMD graphics.",
        ),
    ]

    for name, meaning in terms:
        if pdf.get_y() > 260:
            pdf.add_page()
        pdf.term(name, meaning)

    # 7 Mapping terms to this repo
    pdf.add_page()
    pdf.h1("7. Term-to-Code Mapping in This Repo")
    pdf.bullet("Dataset (SFT examples) -> data/badminton_qa.jsonl")
    pdf.bullet("Hyperparameters -> config.yaml")
    pdf.bullet("LoRA r/alpha/dropout/target_modules -> config.yaml + LoraConfig in train.py")
    pdf.bullet("Trainer loop / epochs / LR -> TrainingArguments in train.py")
    pdf.bullet("Saved adapter -> outputs/badminton-smollm2-lora/")
    pdf.bullet("Inference generate() -> runtime.py BadmintonModel.ask()")
    pdf.bullet("API surface -> POST /api/chat in app.py")
    pdf.bullet("UI -> frontend/index.html + app.js")

    pdf.h1("8. Next Improvements")
    pdf.bullet("Split a validation set and track eval loss / manual rubric scores.")
    pdf.bullet("Expand dataset to 500–2000 high-quality badminton examples.")
    pdf.bullet("Try Qwen2.5-0.5B or 1.5B if quality is insufficient.")
    pdf.bullet("Add NVIDIA GPU + CUDA (or proper ROCm) for faster train/infer.")
    pdf.bullet("Optional: export merged model or GGUF for llama.cpp-style local chat.")
    pdf.bullet("Add citation-style answers grounded in a rules snippet store (RAG).")

    pdf.h1("9. Summary")
    pdf.p(
        "Shuttle demonstrates an end-to-end domain fine-tuning stack: curated badminton SFT "
        "data, LoRA adaptation of a small instruct LLM, CLI and FastAPI serving, and a branded "
        "web chat. It is intentionally small and CPU-runnable so the full lifecycle — data, "
        "train, evaluate-by-chat, document — can be learned and demoed without a large GPU cluster."
    )

    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
