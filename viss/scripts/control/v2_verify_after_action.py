import argparse
import json
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
DEFAULT_AFTER_YOLO = ROOT / "data" / "logs" / "yolo_seg_offline_result.json"
DEFAULT_PLAN = Path("/mnt/d/isaac_projects/v2_visual_task_plan.json")
DEFAULT_EXEC_RESULT = Path("/mnt/d/isaac_projects/v2_visual_execution_result.json")
DEFAULT_OUTPUT = ROOT / "data" / "logs" / "v2_visual_verify_result.json"


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_tasks_from_plan(plan):
    tasks = plan.get("tasks", [])

    count = {
        "total": len(tasks),
        "by_raw_class": {},
        "by_category": {},
        "by_target_bin": {},
    }

    for task in tasks:
        raw = task.get("raw_class_name", task.get("class_name", "unknown"))
        category = task.get("garbage_category", task.get("category", "unknown"))
        target_bin = task.get("target_bin", "unknown")

        count["by_raw_class"][raw] = count["by_raw_class"].get(raw, 0) + 1
        count["by_category"][category] = count["by_category"].get(category, 0) + 1
        count["by_target_bin"][target_bin] = count["by_target_bin"].get(target_bin, 0) + 1

    return count


def count_detections_from_yolo(yolo_result):
    detections = yolo_result.get("detections", [])

    count = {
        "total": len(detections),
        "by_raw_class": {},
        "by_category": {},
        "by_target_bin": {},
    }

    for det in detections:
        raw = det.get("raw_class_name", det.get("class_name", "unknown"))
        category = det.get("category", "unknown")
        target_bin = det.get("target_bin", "unknown")

        count["by_raw_class"][raw] = count["by_raw_class"].get(raw, 0) + 1
        count["by_category"][category] = count["by_category"].get(category, 0) + 1
        count["by_target_bin"][target_bin] = count["by_target_bin"].get(target_bin, 0) + 1

    return count


def verify_action(plan, exec_result, after_yolo):
    selected = plan.get("selected_task", {})

    raw_class = exec_result.get(
        "raw_class_name",
        selected.get("raw_class_name", selected.get("class_name", "unknown")),
    )
    category = exec_result.get(
        "garbage_category",
        selected.get("garbage_category", selected.get("category", "unknown")),
    )

    before_counts = count_tasks_from_plan(plan)
    after_counts = count_detections_from_yolo(after_yolo)

    before_total = int(before_counts.get("total", 0))
    after_total = int(after_counts.get("total", 0))

    before_raw = int(before_counts.get("by_raw_class", {}).get(raw_class, 0))
    after_raw = int(after_counts.get("by_raw_class", {}).get(raw_class, 0))

    before_category = int(before_counts.get("by_category", {}).get(category, 0))
    after_category = int(after_counts.get("by_category", {}).get(category, 0))

    vote_total_decreased = after_total < before_total
    vote_raw_class_decreased = after_raw < before_raw
    vote_category_decreased = after_category < before_category

    vote_score = (
        int(vote_total_decreased)
        + int(vote_raw_class_decreased)
        + int(vote_category_decreased)
    )

    verified = vote_score >= 2

    return {
        "mode": "V2_VISUAL_GRASP_RECOGNITION_VERIFY",
        "verified": verified,
        "vote_score": vote_score,

        "selected_raw_class": raw_class,
        "selected_category": category,
        "selected_category_cn": CATEGORY_CN.get(category, category),

        "before_counts": before_counts,
        "after_counts": after_counts,

        "before_total": before_total,
        "after_total": after_total,

        "before_raw_class_count": before_raw,
        "after_raw_class_count": after_raw,

        "before_category_count": before_category,
        "after_category_count": after_category,

        "vote_total_decreased": vote_total_decreased,
        "vote_raw_class_decreased": vote_raw_class_decreased,
        "vote_category_decreased": vote_category_decreased,

        "exec_status": exec_result.get("status"),
        "visual_world_xyz": exec_result.get("visual_world_xyz"),
        "attached_object_prim": exec_result.get("attached_object_prim"),
        "attach_distance_xy": exec_result.get("attach_distance_xy"),
        "attach_strict_match": exec_result.get("attach_strict_match"),

        "source_plan": str(DEFAULT_PLAN),
        "source_exec_result": str(DEFAULT_EXEC_RESULT),
        "source_after_yolo": str(DEFAULT_AFTER_YOLO),
        "after_image": after_yolo.get("image"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--exec-result", default=str(DEFAULT_EXEC_RESULT))
    parser.add_argument("--after-yolo", default=str(DEFAULT_AFTER_YOLO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    plan_path = Path(args.plan)
    exec_path = Path(args.exec_result)
    after_yolo_path = Path(args.after_yolo)
    output_path = Path(args.output)

    plan = load_json(plan_path)
    exec_result = load_json(exec_path)
    after_yolo = load_json(after_yolo_path)

    result = verify_action(plan, exec_result, after_yolo)
    save_json(output_path, result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output_path}")

    if result["verified"]:
        print("[OK] V2 视觉抓取后重新识别验证成功。")
    else:
        print("[WARN] V2 重新识别验证未通过。检查目标是否仍在相机视野内，或是否被 YOLO 再次检测到。")


if __name__ == "__main__":
    main()