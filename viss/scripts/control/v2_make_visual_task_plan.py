import argparse
import json
import time
import uuid
from pathlib import Path

import cv2


ROOT = Path.home() / "trashbot_ws"
DEFAULT_YOLO_RESULT = ROOT / "data" / "logs" / "yolo_seg_offline_result.json"
DEFAULT_OUTPUT = Path("/mnt/d/isaac_projects/v2_visual_task_plan.json")


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen":    "厨余垃圾",
    "hazardous":  "有害垃圾",
    "other":      "其他垃圾",
    "unknown":    "未知类别",
}

BIN_CN = {
    "bin_recyclable_blue": "蓝色可回收垃圾桶",
    "bin_kitchen_green":   "绿色厨余垃圾桶",
    "bin_hazardous_red":   "红色有害垃圾桶",
    "bin_other_gray":      "灰色其他垃圾桶",
    "unknown":             "未知垃圾桶",
}

ACTION_CODE = {
    "recyclable": 1,
    "hazardous":  2,
    "kitchen":    3,
    "other":      4,
    "unknown":    0,
}

CATEGORY_PRIORITY = {
    "hazardous":  1,
    "recyclable": 2,
    "other":      3,
    "kitchen":    4,
    "unknown":    9,
}

# qwen_first 合法 garbage_category 集合
QWEN_VALID_CATEGORIES = {"recyclable", "kitchen", "hazardous", "other"}

# qwen_first 合法 target_bin 集合
QWEN_VALID_BINS = {
    "bin_recyclable_blue",
    "bin_kitchen_green",
    "bin_hazardous_red",
    "bin_other_gray",
}

# V2.4 稳定闭环默认只跑这些类别
STABLE_RAW_CLASSES = {
    "battery",
    "battery1",
    "battery5",
    "drugbox",
    "drug",
    "drugbag",
    "capsule",
    "can",
    "bottle",
    "bottle2",
    "paper",
    "papercup",
}

# 厨余调试用，不建议主演示阶段默认开启
KITCHEN_DEBUG_RAW_CLASSES = {
    "potato",
    "potatocut",
    "rabbitcut",
    "mooli",
}

# 全部支持类别（YOLO 多类别模式）
ALL_RAW_CLASSES = {
    "potato",
    "rabbitcut",
    "battery1",
    "battery5",
    "bottle",
    "brick",
    "can",
    "china",
    "stone",
    "drug",
    "drugbag",
    "mooli",
    "battery",
    "bottle2",
    "drugbox",
    "papercup",
    "capsule",
    "potatocut",
    "paper",
}


def detect_qwen_first_mode(yolo_result: dict) -> bool:
    """
    判断输入 JSON 是否来自 qwen_first pipeline。
    满足以下任一条件则返回 True：
      - backend == "qwen_first_yolo11_roi"
      - pipeline == "qwen_first"
      - mode == "YOLO11_QWEN_FIRST_ROI_PERCEPTION"
    """
    backend = yolo_result.get("backend", "")
    pipeline = yolo_result.get("pipeline", "")
    mode = yolo_result.get("mode", "")

    return (
        backend == "qwen_first_yolo11_roi"
        or pipeline == "qwen_first"
        or mode == "YOLO11_QWEN_FIRST_ROI_PERCEPTION"
    )


def normalize_blocked_raw_for_profile(args, current_blocked):
    """
    profile=all 时默认不屏蔽任何 raw class。
    只有用户显式传入 --blocked-raw 非空字符串时才屏蔽。
    支持 --blocked-raw "" / none / __none__ 表示空屏蔽。
    """
    user_value = getattr(args, "blocked_raw", None)

    if user_value is None:
        if getattr(args, "profile", None) == "all":
            return set()
        return set(current_blocked or [])

    text = str(user_value).strip()

    if text in ["", "none", "None", "NONE", "__none__", "null", "NULL"]:
        return set()

    return {x.strip() for x in text.split(",") if x.strip()}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def get_image_shape(image_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]
    return int(w), int(h)


def get_allowed_raw_classes(profile: str):
    if profile == "stable":
        return set(STABLE_RAW_CLASSES)

    if profile == "stable_plus_kitchen":
        return set(STABLE_RAW_CLASSES) | set(KITCHEN_DEBUG_RAW_CLASSES)

    if profile == "kitchen_debug":
        return set(KITCHEN_DEBUG_RAW_CLASSES)

    if profile == "all":
        # profile=all: 所有已知 YOLO 类别均允许
        # qwen_first 模式下会走单独的 filter_detection_qwen_first()，不使用本函数
        return set(ALL_RAW_CLASSES)

    raise ValueError(f"Unknown profile: {profile}")


def get_centroid(det):
    centroid = det.get("centroid_px")

    if centroid is not None and len(centroid) == 2:
        return [float(centroid[0]), float(centroid[1])]

    bbox = det.get("bbox_xyxy")

    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [
        round((x1 + x2) / 2.0, 2),
        round((y1 + y2) / 2.0, 2),
    ]


def normalize_detection(det, original_index):
    """用于非 qwen_first 模式的标准化输出。"""
    category = det.get("category", "unknown")
    target_bin = det.get("target_bin", "unknown")
    raw_class = det.get("raw_class_name", det.get("class_name", "unknown"))
    centroid = get_centroid(det)

    return {
        "task_id": f"v2_task_{original_index:03d}",
        "raw_class_name": raw_class,
        "raw_object_id": det.get("object_id", "unknown"),
        "crop_path": det.get("crop_path"),
        "roi_crop_path": det.get("roi_crop_path"),
        "roi_crop_bbox_xyxy": det.get("roi_crop_bbox_xyxy"),
        "qwen_bbox_xyxy": det.get("qwen_bbox_xyxy"),
        "qwen_bbox_norm_xyxy": det.get("qwen_bbox_norm_xyxy"),
        "qwen_center_px": det.get("qwen_center_px"),
        "qwen_center_norm": det.get("qwen_center_norm"),
        "qwen_coarse_result": det.get("qwen_coarse_result"),
        "qwen_verify_result": det.get("qwen_verify_result"),
        "qwen_verify_skipped": det.get("qwen_verify_skipped"),
        "qwen_verify_confidence": det.get("qwen_verify_confidence"),

        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),

        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),

        "confidence": float(det.get("confidence", 0.0)),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "centroid_px": centroid,

        "action_code": ACTION_CODE.get(category, 0),
        "planning_status": "planned" if target_bin != "unknown" and centroid is not None else "unplanned",
    }


def normalize_detection_qwen_first(det, original_index):
    """
    qwen_first 模式专用标准化。
    保留 qwen_first 输出的所有几何字段：
    bottom_contact_px, bbox_area_ratio, polygon, object_id, source
    """
    garbage_category = det.get("garbage_category", "unknown")
    target_bin = det.get("target_bin", "unknown")
    raw_class = det.get("raw_class_name", "trash_object")
    centroid = get_centroid(det)

    return {
        "task_id": f"v2_task_{original_index:03d}",
        "raw_class_name": raw_class,
        "raw_object_id": det.get("raw_object_id", det.get("object_id", "unknown")),

        "garbage_category": garbage_category,
        "garbage_category_cn": CATEGORY_CN.get(garbage_category, garbage_category),

        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),

        "confidence": float(det.get("confidence", 0.0)),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "centroid_px": centroid,
        "bottom_contact_px": det.get("bottom_contact_px"),
        "bbox_area_ratio": det.get("bbox_area_ratio"),
        "polygon": det.get("polygon"),
        "object_id": det.get("object_id"),
        "crop_path": det.get("crop_path"),
        "roi_crop_path": det.get("roi_crop_path"),
        "roi_crop_bbox_xyxy": det.get("roi_crop_bbox_xyxy"),
        "qwen_bbox_xyxy": det.get("qwen_bbox_xyxy"),
        "qwen_bbox_norm_xyxy": det.get("qwen_bbox_norm_xyxy"),
        "qwen_center_px": det.get("qwen_center_px"),
        "qwen_center_norm": det.get("qwen_center_norm"),
        "qwen_coarse_result": det.get("qwen_coarse_result"),
        "qwen_verify_result": det.get("qwen_verify_result"),
        "qwen_verify_skipped": det.get("qwen_verify_skipped"),
        "qwen_verify_confidence": det.get("qwen_verify_confidence"),
        "source": det.get("source", "qwen_first"),

        "action_code": ACTION_CODE.get(garbage_category, 0),
        "planning_status": "planned" if target_bin != "unknown" and centroid is not None else "unplanned",
        "planner_accept_reason": "qwen_first_valid_category_and_geometry",
    }


def filter_detection(det, allowed_raw_classes, blocked_raw_classes, min_conf):
    """非 qwen_first 模式的过滤逻辑（保持原有行为）。"""
    raw_class = det.get("raw_class_name", det.get("class_name", "unknown"))
    conf = float(det.get("confidence", 0.0))

    if raw_class in blocked_raw_classes:
        return False, "blocked_raw_class"

    if raw_class not in allowed_raw_classes:
        return False, "not_in_profile"

    if conf < min_conf:
        return False, "low_confidence"

    centroid = get_centroid(det)
    if centroid is None:
        return False, "no_centroid"

    target_bin = det.get("target_bin", "unknown")
    if target_bin == "unknown":
        return False, "unknown_target_bin"

    return True, "accepted"


def filter_detection_qwen_first(det, blocked_raw_classes, min_conf):
    """
    qwen_first 兼容模式的过滤逻辑。
    不用 raw_class_name 做 profile 过滤，而是用 garbage_category + target_bin + 几何字段。
    """
    raw_class = det.get("raw_class_name", "trash_object")
    conf = float(det.get("confidence", 0.0))

    # blocked_raw_classes 仍然有效（动态屏蔽）
    if raw_class in blocked_raw_classes:
        return False, "blocked_raw_class"

    garbage_category = det.get("garbage_category", "unknown")
    target_bin = det.get("target_bin", "unknown")

    if garbage_category == "unknown" or garbage_category not in QWEN_VALID_CATEGORIES:
        return False, "unknown_category_or_bin"

    if target_bin == "unknown" or target_bin not in QWEN_VALID_BINS:
        return False, "unknown_category_or_bin"

    if conf < min_conf:
        return False, "low_confidence"

    # 几何字段检查
    if det.get("bbox_xyxy") is None:
        return False, "missing_geometry_fields"

    centroid = get_centroid(det)
    if centroid is None:
        return False, "missing_geometry_fields"

    if det.get("bottom_contact_px") is None:
        return False, "missing_geometry_fields"

    if det.get("bbox_area_ratio") is None:
        return False, "missing_geometry_fields"

    return True, "accepted"


def build_tasks(yolo_result, profile, min_conf, blocked_raw_classes):
    detections = yolo_result.get("detections", [])
    is_qwen_first = detect_qwen_first_mode(yolo_result)

    tasks = []
    rejected = []

    if is_qwen_first:
        # qwen_first 兼容模式：不做 raw_class profile 过滤
        for idx, det in enumerate(detections, start=1):
            raw_class = det.get("raw_class_name", "trash_object")
            accepted, reason = filter_detection_qwen_first(
                det=det,
                blocked_raw_classes=blocked_raw_classes,
                min_conf=min_conf,
            )

            if not accepted:
                rejected.append({
                    "original_index": idx,
                    "raw_class_name": raw_class,
                    "garbage_category": det.get("garbage_category", "unknown"),
                    "target_bin": det.get("target_bin", "unknown"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "reason": reason,
                })
                continue

            task = normalize_detection_qwen_first(det, idx)

            if task["planning_status"] == "planned":
                tasks.append(task)
            else:
                rejected.append({
                    "original_index": idx,
                    "raw_class_name": raw_class,
                    "garbage_category": det.get("garbage_category", "unknown"),
                    "target_bin": det.get("target_bin", "unknown"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "reason": "unplanned",
                })

    else:
        # 原有 YOLO 多类别模式
        allowed_raw_classes = get_allowed_raw_classes(profile)

        # profile=all 时：扩展 allowed 以包含所有可能类别，不再因 not_in_profile 拒绝
        # 注意：profile=all 只影响 raw_class 白名单检查，blocked_raw_classes 仍然有效
        if profile == "all":
            # profile=all 意味着不按 raw_class_name 过滤，跳过 not_in_profile 检查
            # 用一个永远为 True 的大集合替代
            allowed_raw_classes = None  # 下面 filter 时 None 表示全部允许

        for idx, det in enumerate(detections, start=1):
            raw_class = det.get("raw_class_name", det.get("class_name", "unknown"))
            conf = float(det.get("confidence", 0.0))

            # profile=all 时不做 raw_class 白名单检查
            if allowed_raw_classes is None:
                # 只检查 blocked、conf、centroid、target_bin
                if raw_class in blocked_raw_classes:
                    rejected.append({
                        "original_index": idx,
                        "raw_class_name": raw_class,
                        "confidence": conf,
                        "reason": "blocked_raw_class",
                    })
                    continue
                if conf < min_conf:
                    rejected.append({
                        "original_index": idx,
                        "raw_class_name": raw_class,
                        "confidence": conf,
                        "reason": "low_confidence",
                    })
                    continue
                centroid = get_centroid(det)
                if centroid is None:
                    rejected.append({
                        "original_index": idx,
                        "raw_class_name": raw_class,
                        "confidence": conf,
                        "reason": "no_centroid",
                    })
                    continue
                target_bin = det.get("target_bin", "unknown")
                if target_bin == "unknown":
                    rejected.append({
                        "original_index": idx,
                        "raw_class_name": raw_class,
                        "confidence": conf,
                        "reason": "unknown_target_bin",
                    })
                    continue
                accepted, reason = True, "accepted"
            else:
                accepted, reason = filter_detection(
                    det=det,
                    allowed_raw_classes=allowed_raw_classes,
                    blocked_raw_classes=blocked_raw_classes,
                    min_conf=min_conf,
                )

            if not accepted:
                rejected.append({
                    "original_index": idx,
                    "raw_class_name": raw_class,
                    "confidence": conf,
                    "reason": reason,
                })
                continue

            task = normalize_detection(det, idx)

            if task["planning_status"] == "planned":
                tasks.append(task)
            else:
                rejected.append({
                    "original_index": idx,
                    "raw_class_name": raw_class,
                    "confidence": conf,
                    "reason": "unplanned",
                })

    tasks.sort(
        key=lambda x: (
            CATEGORY_PRIORITY.get(x["garbage_category"], 9),
            -x["confidence"],
        )
    )

    for i, task in enumerate(tasks, start=1):
        task["execution_order"] = i

    return tasks, rejected, is_qwen_first


def choose_selected_task(tasks, select):
    if not tasks:
        return None

    if select == "confidence":
        sorted_tasks = sorted(tasks, key=lambda x: -x["confidence"])
        return sorted_tasks[0]

    if select == "priority":
        return tasks[0]

    raise ValueError(f"Unknown select mode: {select}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--yolo-json", dest="input", default=str(DEFAULT_YOLO_RESULT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--select", "--select-mode", dest="select",
        default="priority",
        choices=["priority", "confidence"],
        help="priority: category priority first; confidence: highest confidence first.",
    )
    parser.add_argument(
        "--profile",
        default="stable",
        choices=["stable", "stable_plus_kitchen", "kitchen_debug", "all"],
        help="stable is recommended for V2 closed-loop demo.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.60,
        help="Minimum confidence for task planning.",
    )
    parser.add_argument(
        "--blocked-raw",
        default="potato,potatocut,rabbitcut,mooli,brick,china,stone",
        help="Comma-separated raw classes to block.",
    )
    args = parser.parse_args()

    # PROFILE_ALL_BLOCK_FIX_APPLIED
    input_path = Path(args.input)
    output_path = Path(args.output)

    blocked_raw_classes = normalize_blocked_raw_for_profile(args, None)

    yolo_result = load_json(input_path)
    image_path = Path(yolo_result["image"])

    image_width, image_height = get_image_shape(image_path)

    tasks, rejected, is_qwen_first = build_tasks(
        yolo_result=yolo_result,
        profile=args.profile,
        min_conf=args.min_conf,
        blocked_raw_classes=blocked_raw_classes,
    )

    selected_task = choose_selected_task(tasks, args.select)

    plan = {
        "plan_id": f"v2_plan_{uuid.uuid4().hex[:8]}",
        "mode": "V2_4_VISUAL_PIXEL_TO_WORLD_STABLE_PROFILE",
        "created_timestamp": time.time(),

        "filter_profile": args.profile,
        "select_mode": args.select,
        "min_conf": args.min_conf,
        "blocked_raw_classes": sorted(list(blocked_raw_classes)),

        # 感知后端诊断字段
        "perception_backend": yolo_result.get("backend"),
        "perception_pipeline": yolo_result.get("pipeline"),
        "qwen_first_compatible": is_qwen_first,

        "source_image": str(image_path),
        "source_yolo_result": str(input_path),
        "source_model": yolo_result.get("model"),

        "image_width": image_width,
        "image_height": image_height,

        "num_tasks": len(tasks),
        "num_rejected": len(rejected),
        "selected_task": selected_task,
        "tasks": tasks,
        "rejected_detections": rejected,

        "note": (
            "V2.4 stable-profile planner. qwen_first mode uses garbage_category+target_bin "
            "filtering instead of raw_class profile. Use --profile all for all-class YOLO mode."
        ),
    }

    atomic_write_json(output_path, plan)

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output_path}")

    if is_qwen_first:
        print(f"[INFO] qwen_first compatible mode active. num_tasks={len(tasks)}, num_rejected={len(rejected)}")

    if selected_task is None:
        print("[WARN] No executable task after filtering.")
    else:
        print(
            "[SELECTED] "
            f"{selected_task['raw_class_name']} / "
            f"{selected_task.get('garbage_category', 'unknown')} / "
            f"{selected_task['target_bin']} / "
            f"conf={selected_task['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()