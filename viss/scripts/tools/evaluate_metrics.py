import json
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
INPUT = ROOT / "data" / "logs" / "task_execution_log.json"


def rate(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]

    classification_accuracy = rate([r["classify_correct"] for r in records])
    grasp_success_rate = rate([r["grasp_success"] for r in records])
    placement_success_rate = rate([r["place_success"] for r in records])
    target_bin_accuracy = rate([r["target_correct"] for r in records])
    end_to_end_success_rate = rate([
        r["classify_correct"] and r["grasp_success"] and r["place_success"] and r["target_correct"]
        for r in records
    ])

    disturbance_records = [r for r in records if r["disturbance"] != "none"]
    disturbance_success_rate = rate([
        r["classify_correct"] and r["grasp_success"] and r["place_success"]
        for r in disturbance_records
    ])

    metrics = {
        "num_tasks": len(records),
        "classification_accuracy": round(classification_accuracy, 4),
        "grasp_success_rate": round(grasp_success_rate, 4),
        "placement_success_rate": round(placement_success_rate, 4),
        "target_bin_accuracy": round(target_bin_accuracy, 4),
        "end_to_end_success_rate": round(end_to_end_success_rate, 4),
        "disturbance_success_rate": round(disturbance_success_rate, 4),
        "average_duration_sec": round(sum(r["duration_sec"] for r in records) / len(records), 2),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
