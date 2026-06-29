"""Run inference with the fine-tuned WhatsApp LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hf_auth import HF_TOKEN

DEFAULT_SYSTEM = (
    "You are a helpful WhatsApp business assistant. Handle customer queries "
    "related to sales, appointments, payments, and general information. "
    "Be friendly, concise, and professional. Respond in the same language "
    "the customer uses."
)


def load_model(base_model: str, adapter_path: str):
    login(token=HF_TOKEN)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM,
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    input_len = inputs["input_ids"].shape[1]
    reply_ids = outputs[0][input_len:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument(
        "--adapter",
        default="./outputs/zorent-whatsapp-lora/final_adapter",
    )
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)
    print(generate(model, tokenizer, args.prompt))


if __name__ == "__main__":
    main()
