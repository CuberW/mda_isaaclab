import argparse
import csv
import json
import time
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

# 参考旧智能垃圾桶代码中的 rcdict 思路：
# 将垃圾大类映射成执行编号，便于后续 Isaac Sim 机器人动作脚本调用。
CATEGORY_ACTION_CODE = {
    "recyclable": 1,
    "hazardous": 2,
    "kitchen": 3,
    "other": 4,
    "unknown": 0,
}

# 任务执行优先级：
# 有害、厨余优先；同类内部按置信度高的先执行。
CATEGORY_PRIORITY = {
    "hazardous": 1,
    "kitchen": 2,
    "recyclable": 3,
    "other": 4,
    "unknown": 9,
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def build_tasks(yolo_result):
    detections = yolo_result.get("detections", [])
    tasks = []

    for i, det in enumerate(detections, start=1):
        category = det.get("category", "unknown")
        target_bin = det.get("target_bin", "unknown")
        confidence = safe_float(det.get("confidence", 0.0))
        centroid = det.get("centroid_px")
        bbox = det.get("bbox_xyxy")

        task = {
            "task_id": f"task_{i:03d}",

            # 展示层字段：正式表格只显示这些
            "garbage_category": category,
            "garbage_category_cn": CATEGORY_CN.get(category, category),
            "target_bin": target_bin,
            "target_bin_cn": BIN_CN.get(target_bin, target_bin),
            "confidence": round(confidence, 4),
            "centroid_px": centroid,
            "bbox_xyxy": bbox,
            "action_code": CATEGORY_ACTION_CODE.get(category, 0),

            # 内部追溯字段：JSON 保留，不在正式 CSV 表格展示
            "raw_class_name": det.get("raw_class_name", det.get("class_name", "unknown")),
            "raw_object_id": det.get("object_id", "unknown"),

            # 规划状态
            "planning_status": "planned" if target_bin != "unknown" else "unplanned",
            "execution_status": "pending",
        }
        tasks.append(task)

    tasks.sort(
        key=lambda x: (
            CATEGORY_PRIORITY.get(x["garbage_category"], 9),
            -x["confidence"],
        )
    )

    # 重新编号，保证执行顺序清晰
    for i, task in enumerate(tasks, start=1):
        task["execution_order"] = i

    return tasks


def simulate_execution(tasks, default_success=True):
    """
    当前阶段是 Isaac Sim 逻辑执行前的任务闭环日志。
    默认所有已规划任务执行成功，后续接入 Isaac 动画脚本后可替换为真实执行结果。
    """
    execution_log = []

    for task in tasks:
        start_time = time.time()

        planned = task["planning_status"] == "planned"

        if not planned:
            grasp_success = False
            placement_success = False
            target_bin_correct = False
            end_to_end_success = False
            status = "failed_unplanned"
        else:
            grasp_success = bool(default_success)
            placement_success = bool(default_success)
            target_bin_correct = task["target_bin"] != "unknown"
            end_to_end_success = grasp_success and placement_success and target_bin_correct
            status = "success" if end_to_end_success else "failed"

        # 这里用固定阶段耗时，便于报告统计；后续可替换为 Isaac Sim 动画真实耗时
        approach_time = 1.2 if planned else 0.0
        grasp_time = 0.8 if planned else 0.0
        transport_time = 2.0 if planned else 0.0
        release_time = 0.6 if planned else 0.0
        duration_sec = approach_time + grasp_time + transport_time + release_time

        record = {
            "task_id": task["task_id"],
            "execution_order": task["execution_order"],

            # 正式展示字段
            "garbage_category": task["garbage_category"],
            "garbage_category_cn": task["garbage_category_cn"],
            "target_bin": task["target_bin"],
            "target_bin_cn": task["target_bin_cn"],
            "confidence": task["confidence"],
            "centroid_px": task["centroid_px"],
            "action_code": task["action_code"],

            # 内部追溯字段
            "raw_class_name": task["raw_class_name"],
            "raw_object_id": task["raw_object_id"],

            # 执行结果
            "planning_status": task["planning_status"],
            "execution_status": status,
            "grasp_success": grasp_success,
            "placement_success": placement_success,
            "target_bin_correct": target_bin_correct,
            "end_to_end_success": end_to_end_success,

            # 阶段耗时
            "approach_time_sec": round(approach_time, 2),
            "grasp_time_sec": round(grasp_time, 2),
            "transport_time_sec": round(transport_time, 2),
            "release_time_sec": round(release_time, 2),
            "duration_sec": round(duration_sec, 2),

            "start_timestamp": round(start_time, 3),
        }

        execution_log.append(record)

    return execution_log


def compute_metrics(execution_log):
    n = len(execution_log)
    if n == 0:
        return {
            "num_tasks": 0,
            "valid_planning_rate": 0.0,
            "grasp_success_rate": 0.0,
            "placement_success_rate": 0.0,
            "target_bin_accuracy": 0.0,
            "end_to_end_success_rate": 0.0,
            "average_confidence": 0.0,
            "average_duration_sec": 0.0,
            "category_count": {},
            "target_bin_count": {},
        }

    planned = sum(1 for x in execution_log if x["planning_status"] == "planned")
    grasp = sum(1 for x in execution_log if x["grasp_success"])
    placement = sum(1 for x in execution_log if x["placement_success"])
    target_ok = sum(1 for x in execution_log if x["target_bin_correct"])
    e2e = sum(1 for x in execution_log if x["end_to_end_success"])

    category_count = {}
    target_bin_count = {}

    for x in execution_log:
        category = x["garbage_category"]
        target_bin = x["target_bin"]

        category_count[category] = category_count.get(category, 0) + 1
        target_bin_count[target_bin] = target_bin_count.get(target_bin, 0) + 1

    return {
        "num_tasks": n,
        "num_planned_tasks": planned,
        "valid_planning_rate": round(planned / n, 4),
        "grasp_success_rate": round(grasp / n, 4),
        "placement_success_rate": round(placement / n, 4),
        "target_bin_accuracy": round(target_ok / n, 4),
        "end_to_end_success_rate": round(e2e / n, 4),
        "average_confidence": round(
            sum(x["confidence"] for x in execution_log) / n, 4
        ),
        "average_duration_sec": round(
            sum(x["duration_sec"] for x in execution_log) / n, 4
        ),
        "category_count": category_count,
        "target_bin_count": target_bin_count,
    }


def save_execution_csv(path: Path, execution_log):
    """
    正式表格：只显示垃圾大类，不显示 YOLO 小类。
    """
    fieldnames = [
        "execution_order",
        "task_id",
        "garbage_category_cn",
        "confidence",
        "target_bin_cn",
        "planning_status",
        "grasp_success",
        "placement_success",
        "target_bin_correct",
        "end_to_end_success",
        "duration_sec",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in execution_log:
            writer.writerow({k: row.get(k) for k in fieldnames})


def save_debug_csv(path: Path, execution_log):
    """
    调试表格：保留 YOLO 小类，用于证明小类到大类的映射链路。
    """
    fieldnames = [
        "execution_order",
        "task_id",
        "raw_class_name",
        "garbage_category",
        "garbage_category_cn",
        "target_bin",
        "target_bin_cn",
        "confidence",
        "centroid_px",
        "action_code",
        "execution_status",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in execution_log:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Simulate failed execution for testing. Default is all planned tasks success.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_result = load_json(input_path)
    tasks = build_tasks(yolo_result)
    execution_log = simulate_execution(tasks, default_success=(not args.fail))
    metrics = compute_metrics(execution_log)

    result = {
        "source_image": yolo_result.get("image"),
        "source_model": yolo_result.get("model"),
        "source_result": str(input_path),
        "note": "Formal display uses garbage category only. Raw YOLO class is kept for debug tracing.",
        "tasks": tasks,
        "execution_log": execution_log,
        "metrics": metrics,
    }

    json_path = output_dir / "yolo_task_execution_log.json"
    csv_path = output_dir / "yolo_task_execution_table.csv"
    debug_csv_path = output_dir / "yolo_task_execution_debug_table.csv"
    metrics_path = output_dir / "yolo_task_metrics.json"

    save_json(json_path, result)
    save_json(metrics_path, metrics)
    save_execution_csv(csv_path, execution_log)
    save_debug_csv(debug_csv_path, execution_log)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {debug_csv_path}")
    print(f"[SAVED] {metrics_path}")


if __name__ == "__main__":
    main()