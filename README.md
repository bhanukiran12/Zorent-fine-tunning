# Zorent-fine-tunning

## ⚠️ Do NOT paste `kaggle_fine_tune.py` into a cell

That causes `IndentationError` and torch errors.  
Only copy the **3 lines below** (they download + run the file).

---

## Kaggle — copy ONLY this into one cell

```python
!wget -q -O kaggle_fine_tune.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
!wget -q -O whatsapp_training_data.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
!python kaggle_fine_tune.py
```

Before running: **Settings → Accelerator → GPU ON**

Output: `/kaggle/working/zorent-whatsapp-lora/final_adapter`

---

## Wrong vs right

| ❌ Wrong | ✅ Right |
|---------|---------|
| Copy-paste code from `kaggle_fine_tune.py` into a cell | Run `!python kaggle_fine_tune.py` |
| Paste only part of the file | Use the 3 `!wget` / `!python` lines above |

If you already pasted code: **Session → Restart session**, then run the 3 lines above.
