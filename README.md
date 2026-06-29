# Zorent-fine-tunning

Single-script **Kaggle** fine-tuning for the WhatsApp business assistant LLM.

## Files

| File | Purpose |
|------|---------|
| `kaggle_fine_tune.py` | All-in-one: install deps, load data, train, test |
| `kaggle_requirements.txt` | Docker-safe requirements (no torch/tf/keras/protobuf) |
| `whatsapp_training_data.json` | Multi-turn chat training data |

## Kaggle notebook (standard GPU runtime)

1. **Settings → Accelerator → GPU**
2. Attach `whatsapp_training_data.json` as a dataset
3. Clone or upload this repo
4. Run:

```python
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
!pip install -q -r kaggle_requirements.txt
!python kaggle_fine_tune.py
```

Output: `/kaggle/working/zorent-whatsapp-lora/final_adapter`

## Kaggle Docker image

Compatible with the official Kaggle Colab-based Docker image (Python 3.12).

`kaggle_requirements.txt` only adds fine-tuning packages. It does **not** pin:
- `torch`, `tensorflow`, `keras`, `jax` (frozen from Colab base)
- `protobuf` (pinned to `5.29.5` by Kaggle Dockerfile)

Add to your Docker build:

```dockerfile
ADD kaggle_requirements.txt /kaggle_requirements.txt
RUN cat /kaggle_requirements.txt >> /requirements.txt
```

## Troubleshooting import errors

If you see `keras_nlp` or `transformers.tokenization_utils_tokenizers` errors:

1. **Session → Restart session**
2. Do **not** use `pip install --force-reinstall transformers`
3. Run only: `!pip install -q -r kaggle_requirements.txt`
4. Then: `!python kaggle_fine_tune.py`

## Local run

```bash
pip install -r kaggle_requirements.txt
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
