"""Load WhatsApp chat JSON and convert to Hugging Face Dataset format."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

VALID_ROLES = {"system", "user", "assistant"}


def load_whatsapp_json(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of conversation objects")

    return data


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


def prepare_whatsapp_dataset(
    dataset_path: str | Path,
    train_split: float = 0.95,
) -> Dataset:
    raw = load_whatsapp_json(dataset_path)

    records: list[dict] = []
    for item in raw:
        messages = normalize_messages(item.get("messages", []))
        if messages:
            records.append({"messages": messages})

    if not records:
        raise ValueError(f"No valid conversations found in {dataset_path}")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1 - train_split, seed=42)
    return split["train"]
