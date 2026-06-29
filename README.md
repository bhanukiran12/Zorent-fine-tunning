# Zorent-fine-tunning

Single-script **Kaggle** fine-tuning for the WhatsApp business assistant LLM.

## Files

| File | Purpose |
|------|---------|
| `kaggle_fine_tune.py` | All-in-one: install deps, load data, train, test |
| `zorent_requirements.txt` | Fine-tuning extras only (supplement to official Kaggle image) |
| `whatsapp_training_data.json` | Multi-turn chat training data |

## Official Kaggle image (already included)

The Kaggle Docker / notebook runtime already ships:

- `transformers>=5.0.0`
- `datasets>=2.14.6`
- `torch`, `keras-nlp`, `torchtune`, etc.

`zorent_requirements.txt` only adds: `peft`, `trl`, `bitsandbytes`, `accelerate`, and related HF libs. It does **not** downgrade `transformers` or touch `protobuf`.

## Kaggle notebook

1. **Settings → Accelerator → GPU**
2. Attach `whatsapp_training_data.json` as a dataset
3. Clone repo and run:

```python
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

!git clone https://github.com/bhanukiran12/Zorent-fine-tunning.git
%cd Zorent-fine-tunning

!pip install -q -r zorent_requirements.txt
!python kaggle_fine_tune.py
```

Output: `/kaggle/working/zorent-whatsapp-lora/final_adapter`

## Kaggle Docker build

Append to the official `kaggle_requirements.txt` merge step:

```dockerfile
ADD zorent_requirements.txt /zorent_requirements.txt
RUN cat /zorent_requirements.txt >> /requirements.txt
```

## Troubleshooting

If you see `keras_nlp` or broken `transformers` imports:

1. **Session → Restart session**
2. Never run `pip install --force-reinstall transformers`
3. Only: `!pip install -q -r zorent_requirements.txt`
4. Then: `!python kaggle_fine_tune.py`

## Local run

```bash
pip install "transformers>=5.0.0" "datasets>=2.14.6" -r zorent_requirements.txt
python kaggle_fine_tune.py
```

## Dataset format

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
