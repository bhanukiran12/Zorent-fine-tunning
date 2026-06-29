# Zorent-fine-tunning

## Kaggle — use commit-pinned URL (avoids GitHub CDN cache)

```python
!rm -f zorent_train.py
!wget -O zorent_train.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/6a27209/zorent_train.py
!wget -O whatsapp_training_data.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
!head -30 zorent_train.py
!python -u zorent_train.py
```

Check `head` output shows `2026-06-29-qwen5` and **no** `_assert_fresh_script`.

**Before running:** GPU ON + Secrets → `HF_TOKEN`

## Output

`/kaggle/working/zorent-whatsapp-lora/final_adapter`
