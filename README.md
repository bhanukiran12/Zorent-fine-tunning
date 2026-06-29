# Zorent-fine-tunning

Fine-tune a language model for **WhatsApp business assistant** conversations using `whatsapp_training_data.json` and Hugging Face LoRA/QLoRA.

## Dataset

`whatsapp_training_data.json` contains multi-turn chat examples:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Setup

```bash
pip install -r requirements.txt
```

Hugging Face token is configured in `hf_auth.py`.

## Train

```bash
python fine_tune.py
```

Custom dataset path:

```bash
python fine_tune.py --dataset ./whatsapp_training_data.json
```

Adapter weights are saved to `outputs/zorent-whatsapp-lora/final_adapter`.

## Inference

```bash
python inference.py --prompt "smartwatch gurinchi cheppagalaraaa?"
```

## Config

Edit `config.yaml` to change model, epochs, batch size, and `max_seq_length`.

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (recommended)
