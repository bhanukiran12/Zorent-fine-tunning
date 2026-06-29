"""Convert Sheet8 CSV (user/agent columns) to WhatsApp training JSON."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SYSTEM_PROMPTS = [
    "You are a helpful placement support assistant for NxtWave students. "
    "Help with job applications, interview updates, company assessments, and placement queries. "
    "Be professional, empathetic, and concise.",
    "You are a placement support agent on WhatsApp. Assist students with company application status, "
    "assessments, interview scheduling, and job portal issues. Respond clearly and helpfully.",
    "You are a student placement support assistant. Answer questions about job applications, "
    "hiring updates, mock interviews, and company assessments. Match the student tone and language.",
]


def split_parts(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).split("|") if p.strip()]


def csv_to_json(csv_path: Path, out_path: Path) -> int:
    records: list[dict] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            user_parts = split_parts(row[0])
            assistant_parts = split_parts(row[1])
            if not user_parts or not assistant_parts:
                continue

            messages = [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}]
            n = max(len(user_parts), len(assistant_parts))
            for i in range(n):
                if i < len(user_parts):
                    messages.append({"role": "user", "content": user_parts[i]})
                if i < len(assistant_parts):
                    messages.append({"role": "assistant", "content": assistant_parts[i]})

            records.append({"messages": messages})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=r"c:\Users\Nxtwave\Downloads\Bhanu Kiran working - Sheet8.csv",
    )
    parser.add_argument(
        "--out",
        default="whatsapp_sheet8.json",
    )
    args = parser.parse_args()
    count = csv_to_json(Path(args.csv), Path(args.out))
    print(f"Saved {count} conversations to {args.out}")
