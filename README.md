# Zorent-fine-tunning

**One Python file. No folders. No imports from repo.**

## Kaggle — copy these 3 cells

**Cell 1** — download script + data:
```python
!wget -q https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
!wget -q https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
```

**Cell 2** — turn GPU on  
Settings → Accelerator → **GPU T4 x2** (or better)

**Cell 3** — train:
```python
!python kaggle_fine_tune.py
```

Done. Model saves to `/kaggle/working/zorent-whatsapp-lora/final_adapter`

---

## What you do NOT need

- No `git clone`
- No folder structure
- No `from something import ...` from your repo
- No separate Kaggle dataset (optional — script auto-finds JSON in `/kaggle/working/`)

## If it fails

1. **Session → Restart session**
2. Run cells 1 and 3 again
3. Do **not** paste `kaggle_fine_tune.py` code into a cell
