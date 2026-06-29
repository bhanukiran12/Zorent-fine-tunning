# Zorent-fine-tunning

## Datasets

| File | Source |
|------|--------|
| `whatsapp_sheet8.json` | Sheet8 CSV — **EndUser** = user, **Agent** = assistant (16,264 chats) |
| `whatsapp_training_data.json` | Original WhatsApp business chats |

Convert CSV again:
```bash
python csv_to_json.py --csv "path/to/Sheet8.csv" --out whatsapp_sheet8.json
```

## Kaggle training

```python
!rm -f zorent_train.py
!wget -O zorent_train.py https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/zorent_train.py
!wget -O whatsapp_sheet8.json https://raw.githubusercontent.com/bhanukiran12/Zorent-fine-tunning/main/whatsapp_sheet8.json
!python -u zorent_train.py
```

Script auto-picks `whatsapp_sheet8.json` first, then `whatsapp_training_data.json`.

**GPU ON** + Kaggle Secrets → `HF_TOKEN`

## JSON format

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "student message"},
    {"role": "assistant", "content": "agent reply"}
  ]
}
```

CSV columns `participant_role_EndUser` → **user**, `participant_role_Agent` → **assistant**.  
Messages split by `|` become multi-turn chat bubbles.
