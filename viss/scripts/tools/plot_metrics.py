import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"
INPUT = LOG_DIR / "task_execution_log.json"
OUT = LOG_DIR / "metrics_bar_chart.png"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]

def rate(values):
    return sum(values) / len(values) if values else 0.0

metrics = {
    "Classification": rate([r["classify_correct"] for r in records]),
    "Grasp": rate([r["grasp_success"] for r in records]),
    "Placement": rate([r["place_success"] for r in records]),
    "Target Bin": rate([r["target_correct"] for r in records]),
    "End-to-End": rate([
        r["classify_correct"] and r["grasp_success"] and r["place_success"] and r["target_correct"]
        for r in records
    ]),
}

plt.figure(figsize=(8, 4.8))
plt.bar(metrics.keys(), [v * 100 for v in metrics.values()])
plt.ylim(0, 100)
plt.ylabel("Success Rate (%)")
plt.title("TrashBot Task Performance Metrics")

for i, v in enumerate(metrics.values()):
    plt.text(i, v * 100 + 2, f"{v * 100:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(OUT, dpi=200)
print(f"Saved chart to: {OUT}")
