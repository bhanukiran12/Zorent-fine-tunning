# Zorent-fine-tunning

## Why wget shows 404?

Your GitHub repo is **private**. Kaggle cannot download from private repos.

**Fix (pick one):**

### Option A — Make repo public (easiest for wget)

1. Open https://github.com/bhanukiran12/Zorent-fine-tunning/settings
2. Scroll to **Danger Zone** → **Change repository visibility** → **Public**
3. Then run on Kaggle:

```python
!wget -O kaggle_fine_tune.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
!wget -O whatsapp_training_data.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
!ls -lh kaggle_fine_tune.py whatsapp_training_data.json
!python -u kaggle_fine_tune.py
```

### Option B — Upload files manually (keep repo private)

1. On your PC, copy `kaggle_fine_tune.py` and `whatsapp_training_data.json`
2. In Kaggle notebook: **File → Upload Notebook** or drag both files into `/kaggle/working/`
3. Run only:

```python
!ls -lh /kaggle/working/kaggle_fine_tune.py /kaggle/working/whatsapp_training_data.json
!python -u /kaggle/working/kaggle_fine_tune.py
```

**Before running:** Settings → Accelerator → **GPU ON**

**HF token:** Kaggle → **Add-ons → Secrets** → add `HF_TOKEN` = your token from https://huggingface.co/settings/tokens

The script reads it automatically — no need to paste the token in code.

## What you should see

```
[zorent] Starting Zorent WhatsApp fine-tuning...
[zorent] Loading torch and training libraries (may take 1-2 min)...
[zorent] GPU available: True
[zorent] Dataset: /kaggle/working/whatsapp_training_data.json
```

If `!ls -lh` shows **0 bytes** — download failed. Do not run training until files are > 0 bytes.

## Output

`/kaggle/working/zorent-whatsapp-lora/final_adapter`
