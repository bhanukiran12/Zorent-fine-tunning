# Zorent-fine-tunning

Single-script **Kaggle** fine-tuning for the WhatsApp business assistant LLM.

## Files

| File | Purpose |
|------|---------|
| `kaggle_fine_tune.py` | All-in-one: install deps, load data, train, test |
| `whatsapp_training_data.json` | Multi-turn chat training data |

## Kaggle setup

1. Create a new Kaggle notebook.
2. **Settings → Accelerator → GPU** (T4 x2 or better).
3. Add `whatsapp_training_data.json` as a **Kaggle dataset** and attach it.
4. (Optional) Add `HF_TOKEN` in **Notebook → Add-ons → Secrets**.
5. Upload `kaggle_fine_tune.py` or clone this repo.
6. Run in a notebook cell (**do not paste the script into a cell**):

```python
!python kaggle_fine_tune.py
```

The script auto-installs pinned package versions, restarts Python cleanly, then trains.

If you already pasted the old script into a cell and got import errors:
**Session → Restart session**, then use `!python kaggle_fine_tune.py` instead.

Output is saved to `/kaggle/working/zorent-whatsapp-lora/final_adapter`.

## Local run

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes huggingface_hub
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
