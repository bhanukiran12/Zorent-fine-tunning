"""
Zorent WhatsApp LLM fine-tuning — single script for Kaggle.

Run ONLY as a shell script (never paste into a notebook cell):

    !wget -q https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
    !python kaggle_fine_tune.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Disable TF/Keras backends — required on Kaggle Colab base image
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Extras only — Kaggle already has transformers>=5.0.0, datasets, torch, pandas
# --no-deps avoids upgrading torch/pandas and breaking the runtime
PIP_PACKAGES = [
    "accelerate>=1.2.0",
    "bitsandbytes>=0.45.0",
    "huggingface_hub>=0.27.0",
    "peft>=0.15.0",
    "trl>=0.15.0",
]


def _packages_ok() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_packages() -> None:
    if _packages_ok():
        print("Fine-tuning packages already installed.")
        return
    print("Installing fine-tuning packages (--no-deps, safe for Kaggle)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            *PIP_PACKAGES,
        ],
    )
    print("Packages ready.")


def _bootstrap() -> None:
    """Parent launches a fresh subprocess; child installs + trains."""
    if os.environ.get("_ZORENT_BOOTSTRAPPED") == "1":
        _ensure_packages()
        return

    script = globals().get("__file__")
    if not script or not Path(script).exists():
        raise SystemExit(
            "\nDo NOT paste this file into a notebook cell.\n"
            "Run these two lines instead:\n\n"
            "  !wget -q https://raw.githubusercontent.com/bhanukiran12/"
            "Zorent-fine-tunning/main/kaggle_fine_tune.py\n"
            "  !python kaggle_fine_tune.py\n"
        )

    env = os.environ.copy()
    env["_ZORENT_BOOTSTRAPPED"] = "1"
    print("Starting clean Python process (avoids torch conflicts)...")
    subprocess.check_call([sys.executable, script], env=env)
    sys.exit(0)


_bootstrap()

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ── config ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TRAIN_SPLIT = 0.95
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

VALID_ROLES = {"system", "user", "assistant"}

DEFAULT_SYSTEM = (
    "You are a helpful WhatsApp business assistant. Handle customer queries "
    "related to sales, appointments, payments, and general information. "
    "Be friendly, concise, and professional. Respond in the same language "
    "the customer uses."
)


def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def _resolve_paths() -> tuple[Path, Path]:
    if _is_kaggle():
        output_dir = Path("/kaggle/working/zorent-whatsapp-lora")

        # 1) explicit path: DATASET_PATH=/kaggle/working/whatsapp_training_data.json python ...
        env_path = os.environ.get("DATASET_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path), output_dir

        # 2) downloaded next to script in /kaggle/working
        working_file = Path("/kaggle/working/whatsapp_training_data.json")
        if working_file.exists():
            return working_file, output_dir

        # 3) attached Kaggle dataset under /kaggle/input
        input_root = Path("/kaggle/input")
        candidates = list(input_root.rglob("whatsapp_training_data.json"))
        if candidates:
            return candidates[0], output_dir

        raise FileNotFoundError(
            "whatsapp_training_data.json not found.\n"
            "Easiest fix — run this in a cell BEFORE training:\n\n"
            "  !wget -q https://raw.githubusercontent.com/bhanukiran12/"
            "Zorent-fine-tunning/main/whatsapp_training_data.json\n"
        )
    else:
        dataset_path = Path(__file__).parent / "whatsapp_training_data.json"
        output_dir = Path(__file__).parent / "outputs" / "zorent-whatsapp-lora"

    output_dir.mkdir(parents=True, exist_ok=True)
    return dataset_path, output_dir


def normalize_messages(messages: list[dict]) -> list[dict] | None:
    cleaned: list[dict] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in VALID_ROLES or not content:
            continue
        cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2:
        return None
    if not any(m["role"] == "assistant" for m in cleaned):
        return None
    if not any(m["role"] == "user" for m in cleaned):
        return None
    return cleaned


def load_whatsapp_dataset(dataset_path: Path, train_split: float) -> Dataset:
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected JSON array of {messages: [...]} objects")

    records: list[dict] = []
    for item in raw:
        messages = normalize_messages(item.get("messages", []))
        if messages:
            records.append({"messages": messages})

    if not records:
        raise ValueError(f"No valid conversations in {dataset_path}")

    print(f"Valid conversations: {len(records)}")
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1 - train_split, seed=42)
    return split["train"]


def run_inference(model, tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()


def fine_tune() -> Path:
    dataset_path, output_dir = _resolve_paths()
    adapter_dir = output_dir / "final_adapter"

    print(f"Kaggle mode: {_is_kaggle()}")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")

    login(token=HF_TOKEN)

    train_dataset = load_whatsapp_dataset(dataset_path, TRAIN_SPLIT)
    print(f"Training samples: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    use_4bit = torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_steps=500,
            warmup_ratio=0.03,
            bf16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
            report_to="none",
            save_total_limit=2,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    print("Starting fine-tuning...")
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")

    test_prompt = "smartwatch gurinchi cheppagalaraaa?"
    print(f"\nTest prompt: {test_prompt}")
    print(f"Response: {run_inference(trainer.model, tokenizer, test_prompt)}")

    return adapter_dir


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: no GPU detected — training will be very slow.")
    fine_tune()
