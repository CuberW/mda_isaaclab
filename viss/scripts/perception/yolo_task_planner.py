import argparse
import csv
import json
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
DEFAULT_INPUT = ROOT / "data" / "logs" / "yolo_seg_offline_result.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "logs"


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}

BIN_CN = {
    "bin_recyclable_blue": "蓝色可回收垃圾桶",
    "bin_kitchen_green": "绿色厨余垃圾桶",
    "bin_hazardous_red": "红色有害垃圾桶",
    "bin_other_gray": "灰色其他垃圾桶",
    "unknown": "未知垃圾桶",
}


def load_yolo_result(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_task_plan(yolo_result):
    detections = yolo_result.get("detections", [])

    tasks = []
    for idx, det in enumerate(detections, start=1):
        category = det.get("category", "unknown")
        target_bin = det.get("target_bin", "unknown")
        confidence = det.get("confidence", 0.0)
        centroid = det.get("centroid_px")
        bbox = det.get("bbox_xyxy")

        task = {
            "task_id": f"task_{idx:03d}",

            # 展示层字段：正式演示和表格只看这些
            "display_category": category,
            "display_category_cn": CATEGORY_CN.get(category, category),
            "target_bin": target_bin,
            "target_bin_cn": BIN_CN.get(target_bin, target_bin),
            "confidence": confidence,
            "centroid_px": centroid,
            "bbox_xyxy": bbox,

            # 内部追溯字段：不建议在演示表格里展示
            "raw_class_name": det.get("raw_class_name", det.get("class_name", "unknown")),
            "raw_object_id": det.get("object_id", "unknown"),

            # 任务规划状态
            "planning_status": "planned" if target_bin != "unknown" else "unplanned",
            "action": "pick_and_place",
            "expected_result": "send_to_target_bin",
        }
        tasks.append(task)

    return tasks


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, tasks):
    # CSV 只展示垃圾大类，不展示 YOLO 小类
    fieldnames = [
        "task_id",
        "display_category",
        "display_category_cn",
        "confidence",
        "target_bin",
        "target_bin_cn",
        "centroid_px",
        "planning_status",
        "action",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task in tasks:
            row = {k: task.get(k) for k in fieldnames}
            writer.writerow(row)


def summarize(tasks):
    total = len(tasks)
    planned = sum(1 for t in tasks if t["planning_status"] == "planned")

    category_count = {}
    bin_count = {}

    for t in tasks:
        category = t["display_category"]
        target_bin = t["target_bin"]
        category_count[category] = category_count.get(category, 0) + 1
        bin_count[target_bin] = bin_count.get(target_bin, 0) + 1

    return {
        "num_tasks": total,
        "num_planned_tasks": planned,
        "valid_planning_rate": round(planned / total, 4) if total > 0 else 0.0,
        "category_count": category_count,
        "target_bin_count": bin_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_result = load_yolo_result(input_path)
    tasks = build_task_plan(yolo_result)
    metrics = summarize(tasks)

    plan = {
        "source_image": yolo_result.get("image"),
        "source_model": yolo_result.get("model"),
        "num_tasks": len(tasks),
        "display_rule": "Task table displays garbage category only. Raw YOLO class is kept only for debug tracing.",
        "tasks": tasks,
        "metrics": metrics,
    }

    json_path = output_dir / "yolo_task_plan.json"
    csv_path = output_dir / "yolo_task_plan_table.csv"

    save_json(json_path, plan)
    save_csv(csv_path, tasks)

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()