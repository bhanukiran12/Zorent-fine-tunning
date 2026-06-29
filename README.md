# Zorent-fine-tunning

## Kaggle — use `zorent_train.py` (new filename, avoids CDN cache)

```python
!rm -f zorent_train.py kaggle_fine_tune.py
!wget -O zorent_train.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/zorent_train.py
!wget -O whatsapp_training_data.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_training_data.json
!grep -E "SCRIPT_VERSION|SFTConfig|max_length" zorent_train.py | head -3
!python -u zorent_train.py
```

**Must see:**
```
SCRIPT_VERSION = "2026-06-29-qwen3"
from trl import SFTConfig, SFTTrainer
max_length=MAX_SEQ_LENGTH,
```

**Before running:** GPU ON + Kaggle Secrets → `HF_TOKEN`

## Output

`/kaggle/working/zorent-whatsapp-lora/final_adapter`
