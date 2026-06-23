import json
import random
from pathlib import Path
from datetime import datetime


ROOT = Path.home() / "trashbot_ws"
INPUT = ROOT / "data" / "logs" / "mock_perception_result.json"
OUTPUT = ROOT / "data" / "logs" / "task_execution_log.json"


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    random.seed(42)

    for idx, task in enumerate(data["tasks"], start=1):
        # 第一版先做可控模拟，后面接机器人控制后替换为真实结果
        classify_correct = True
        grasp_success = random.random() > 0.10
        place_success = grasp_success and random.random() > 0.05
        target_correct = place_success

        record = {
            "task_id": f"task_{idx:03d}",
            "object_id": task["object_id"],
            "class_name": task["class_name"],
            "category": task["category"],
            "target_bin": task["target_bin"],
            "recognition_confidence": task["confidence"],
            "disturbance": task["disturbance"],
            "classify_correct": classify_correct,
            "grasp_success": grasp_success,
            "place_success": place_success,
            "target_correct": target_correct,
            "state_sequence": [
                "SCAN",
                "RECOGNIZE",
                "PLAN_PICK",
                "PICK_SUCCESS" if grasp_success else "PICK_FAILED",
                "MOVE_TO_BIN" if grasp_success else "STOP",
                "PLACE_SUCCESS" if place_success else "PLACE_FAILED",
            ],
            "duration_sec": round(random.uniform(8.0, 18.0), 2)
        }
        records.append(record)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "records": records
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
