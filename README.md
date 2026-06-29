# Zorent-fine-tunning

## Kaggle — copy this ONE cell

```python
!wget -O kaggle_fine_tune.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
!wget -O whatsapp_training_data.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
!ls -lh kaggle_fine_tune.py whatsapp_training_data.json
!python -u kaggle_fine_tune.py
```

**Before running:** Settings → Accelerator → **GPU ON**

## What you should see

```
[zorent] Starting Zorent WhatsApp fine-tuning...
[zorent] Loading torch and training libraries (may take 1-2 min)...
[zorent] Libraries loaded in 45s
[zorent] Kaggle mode: True
[zorent] GPU available: True
[zorent] Dataset: /kaggle/working/whatsapp_training_data.json
...
[zorent] Starting fine-tuning (this takes 30-60+ min on T4)...
```

If you see **nothing for 1-2 minutes** — that is normal while torch loads.

If you see **nothing at all** — check:
1. GPU is enabled
2. `!ls -lh` shows both files downloaded (not 0 bytes)
3. Do **not** paste the script into a cell — only run `!python -u`

## Output

`/kaggle/working/zorent-whatsapp-lora/final_adapter`
