import json
import yaml
from pathlib import Path
from datetime import datetime


ROOT = Path.home() / "trashbot_ws"
CLASS_CONFIG = ROOT / "config" / "trash_classes.yaml"
BIN_CONFIG = ROOT / "config" / "bins.yaml"
LOG_DIR = ROOT / "data" / "logs"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    classes = load_yaml(CLASS_CONFIG)["trash_classes"]
    bins = load_yaml(BIN_CONFIG)["bins"]

    tasks = []
    for object_id, item in classes.items():
        category = item["category"]
        bin_info = bins[category]

        task = {
            "object_id": object_id,
            "class_name": item["class_name"],
            "category": category,
            "confidence": item.get("confidence", 1.0),
            "target_bin": item["target_bin"],
            "target_position": bin_info["position"],
            "disturbance": item.get("disturbance", "none"),
            "status": "recognized"
        }
        tasks.append(task)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_objects": len(tasks),
        "tasks": tasks
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOG_DIR / "mock_perception_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
