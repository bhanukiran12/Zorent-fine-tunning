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
import time
from pathlib import Path

# Disable TF/Keras backends — required on Kaggle Colab base image
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

SCRIPT_VERSION = "2026-06-29-qwen3"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
GITHUB_RAW = (
    "https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/zorent_train.py"
)


def log(msg: str) -> None:
    print(f"[zorent] {msg}", flush=True)

# Extras only — Kaggle already has transformers>=5.0.0, datasets, torch, pandas
# --no-deps avoids upgrading torch/pandas and breaking the runtime
PIP_PACKAGES = [
    "accelerate>=1.2.0",
    "bitsandbytes>=0.45.0",
    "huggingface_hub>=0.27.0",
    "peft>=0.15.0",
    "trl>=0.15.0",
]


def _assert_fresh_script() -> None:
    script = globals().get("__file__")
    if not script or not Path(script).exists():
        return
    text = Path(script).read_text(encoding="utf-8")
    if "SFTConfig" not in text or "max_seq_length=MAX_SEQ_LENGTH" in text:
        raise SystemExit(
            f"Stale script downloaded (need {SCRIPT_VERSION}).\n"
            f"Run:\n  !wget -O zorent_train.py {GITHUB_RAW}\n"
            f"  !python -u zorent_train.py"
        )


def _packages_ok() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_packages() -> None:
    if _packages_ok():
        log("Fine-tuning packages already installed.")
        return
    log("Installing peft, trl, bitsandbytes, accelerate...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            *PIP_PACKAGES,
        ],
    )
    log("Packages installed.")


def _in_notebook_kernel() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


def _bootstrap() -> None:
    """Install deps. Re-exec only when pasted inside a notebook kernel."""
    if os.environ.get("_ZORENT_BOOTSTRAPPED") == "1":
        _ensure_packages()
        return

    # !python kaggle_fine_tune.py is already a clean process — run directly
    if not _in_notebook_kernel():
        os.environ["_ZORENT_BOOTSTRAPPED"] = "1"
        log("Starting Zorent WhatsApp fine-tuning...")
        log(f"Script version: {SCRIPT_VERSION} | model: {MODEL_NAME}")
        _ensure_packages()
        return

    script = globals().get("__file__")
    if not script or not Path(script).exists():
        raise SystemExit(
            "\nDo NOT paste this file into a notebook cell.\n"
            "Run these lines instead:\n\n"
            "  !wget -O kaggle_fine_tune.py https://raw.githubusercontent.com/"
            "bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py\n"
            "  !wget -O whatsapp_training_data.json https://raw.githubusercontent.com/"
            "bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json\n"
            "  !python -u kaggle_fine_tune.py\n"
        )

    env = os.environ.copy()
    env["_ZORENT_BOOTSTRAPPED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log("Restarting in clean process...")
    subprocess.check_call([sys.executable, "-u", script], env=env)
    sys.exit(0)


_bootstrap()
_assert_fresh_script()
log("Loading torch and training libraries (may take 1-2 min)...")
t0 = time.time()

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

log(f"Libraries loaded in {time.time() - t0:.0f}s")

# ── training config ──────────────────────────────────────────────────────────
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


def _get_hf_token() -> str:
    """Read HF token from env or Kaggle Secrets."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    if _is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN").strip()
            os.environ["HF_TOKEN"] = token
            log("Loaded HF_TOKEN from Kaggle Secrets.")
            return token
        except Exception as exc:
            log(f"Kaggle Secrets HF_TOKEN not found: {exc}")

    return ""


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
            "Upload it to /kaggle/working/ or wget it from GitHub.\n"
            "Note: private GitHub repos return 404 on Kaggle — make repo public\n"
            "or upload both files manually via Kaggle File → Upload.\n"
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


def _load_model_config(model_name: str, hf_token: str):
    """Patch Phi-3 rope_scaling for transformers 5.x if needed."""
    config = AutoConfig.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    rope = getattr(config, "rope_scaling", None)
    if isinstance(rope, dict) and rope and "type" not in rope:
        rope = dict(rope)
        rope["type"] = rope.get("rope_type", "longrope")
        config.rope_scaling = rope
    return config


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

    print(f"Valid conversations: {len(records)}", flush=True)
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
    hf_token = _get_hf_token()

    log(f"Script version: {SCRIPT_VERSION}")
    log(f"Model: {MODEL_NAME}")
    log(f"Kaggle mode: {_is_kaggle()}")
    log(f"GPU available: {__import__('torch').cuda.is_available()}")
    log(f"Dataset: {dataset_path}")
    log(f"Output:  {output_dir}")

    if not hf_token:
        raise SystemExit(
            "HF_TOKEN missing.\n"
            "1. Kaggle → Add-ons → Secrets → add name: HF_TOKEN, value: hf_...\n"
            "2. Re-run: !python -u kaggle_fine_tune.py\n"
            "Get token: https://huggingface.co/settings/tokens"
        )

    log("Logging in to Hugging Face...")
    login(token=hf_token)

    log("Loading WhatsApp dataset...")
    train_dataset = load_whatsapp_dataset(dataset_path, TRAIN_SPLIT)
    log(f"Training samples: {len(train_dataset)}")

    log(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=hf_token, trust_remote_code=True
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

    log(f"Loading model: {MODEL_NAME} (first run downloads ~6GB)...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = _load_model_config(MODEL_NAME, hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
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
        args=SFTConfig(
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
            max_length=MAX_SEQ_LENGTH,
            packing=False,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    log("Starting fine-tuning (this takes 30-60+ min on T4)...")
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log(f"DONE — saved adapter to {adapter_dir}")

    test_prompt = "smartwatch gurinchi cheppagalaraaa?"
    log(f"Test prompt: {test_prompt}")
    log(f"Response: {run_inference(trainer.model, tokenizer, test_prompt)}")

    return adapter_dir


if __name__ == "__main__":
    try:
        if not torch.cuda.is_available():
            log("WARNING: no GPU — training will be very slow. Enable GPU in Settings.")
        fine_tune()
        log("All finished successfully.")
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
