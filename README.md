# Zorent-fine-tunning

## Kaggle — copy this ONE cell

```python
!rm -f kaggle_fine_tune.py
!wget -O kaggle_fine_tune.py "https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py?t=$(date +%s)"
!grep -m1 "SCRIPT_VERSION" kaggle_fine_tune.py
!python -u kaggle_fine_tune.py
```

**Check:** you must see `2026-06-29-qwen` and `Model: Qwen/Qwen2.5-3B-Instruct` in output.  
If you see `Phi-3` — old file cached; run cell again.

**Before running:**
- Settings → Accelerator → **GPU ON**
- Add-ons → Secrets → `HF_TOKEN` = your Hugging Face token

## Output

`/kaggle/working/zorent-whatsapp-lora/final_adapter`
