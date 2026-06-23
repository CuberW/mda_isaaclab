import json
import csv
from pathlib import Path

ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"
INPUT = LOG_DIR / "task_execution_log.json"
CSV_OUT = LOG_DIR / "task_execution_table.csv"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]

fields = [
    "task_id",
    "object_id",
    "class_name",
    "category",
    "target_bin",
    "recognition_confidence",
    "disturbance",
    "classify_correct",
    "grasp_success",
    "place_success",
    "target_correct",
    "duration_sec",
]

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in records:
        writer.writerow({k: r.get(k, "") for k in fields})

print(f"Saved CSV to: {CSV_OUT}")
