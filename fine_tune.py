"""
Fine-tune an LLM with LoRA/QLoRA on WhatsApp chat data.

Usage:
  python fine_tune.py
  python fine_tune.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
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

from hf_auth import HF_TOKEN
from prepare_data import prepare_whatsapp_dataset


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_fine_tune(cfg: dict, dataset_path: Path) -> Path:
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    model_name = cfg["model_name"]
    output_dir = Path(train_cfg["output_dir"])

    login(token=HF_TOKEN)

    print(f"Loading dataset: {dataset_path}")
    train_dataset: Dataset = prepare_whatsapp_dataset(
        dataset_path=dataset_path,
        train_split=train_cfg.get("train_split", 0.95),
    )
    print(f"Training samples: {len(train_dataset)}")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    bnb_config = None
    if train_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        bf16=train_cfg.get("bf16", True),
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=train_cfg["max_seq_length"],
        packing=False,
    )

    print("Starting fine-tuning on WhatsApp dataset...")
    trainer.train()

    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM on WhatsApp chat data"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        help="Override dataset_path from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = Path(args.dataset or cfg["dataset_path"])

    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not detected. QLoRA training needs a GPU; "
            "CPU training will be very slow."
        )

    run_fine_tune(cfg, dataset_path)


if __name__ == "__main__":
    main()
