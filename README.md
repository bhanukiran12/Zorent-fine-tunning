# Zorent-fine-tunning

One file: **`kaggle_fine_tune.py`**

## Kaggle (run exactly like this)

```python
!wget -q https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/kaggle_fine_tune.py
!python kaggle_fine_tune.py
```

- Enable **GPU**
- Attach `whatsapp_training_data.json` as a dataset
- **Do not paste the script into a cell** — that breaks torch

If you already pasted it: **Session → Restart session**, then run the two lines above.

Output: `/kaggle/working/zorent-whatsapp-lora/final_adapter`
