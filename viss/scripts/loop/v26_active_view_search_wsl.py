import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"

COMMAND_JSON = Path("/mnt/d/isaac_projects/v26_view_command.json")
RESULT_JSON = Path("/mnt/d/isaac_projects/v26_view_result.json")
YOLO_RESULT = LOG_DIR / "yolo_seg_offline_result.json"
SEARCH_LOG_JSON = LOG_DIR / "v26_active_view_search_log.json"


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


# 候选视角：不是一个固定点，而是机器人自动尝试多个观察位。
# 可以把它解释成“受限场景下的主动视角搜索策略”。
VIEW_CANDIDATES = [
    {
        "view_id": "front_mid",
        "robot_xyz": [-0.35, -0.88, 0.0],
        "look_at_xyz": [-0.65, -0.12, 0.30],
    },
    {
        "view_id": "front_left",
        "robot_xyz": [-0.70, -0.92, 0.0],
        "look_at_xyz": [-0.75, -0.10, 0.30],
    },
    {
        "view_id": "front_right",
        "robot_xyz": [0.00, -0.92, 0.0],
        "look_at_xyz": [-0.45, -0.10, 0.30],
    },
    {
        "view_id": "near_mid",
        "robot_xyz": [-0.45, -0.78, 0.0],
        "look_at_xyz": [-0.65, -0.08, 0.30],
    },
    {
        "view_id": "far_mid",
        "robot_xyz": [-0.25, -1.08, 0.0],
        "look_at_xyz": [-0.65, -0.12, 0.30],
    },
]


def run_bash(command: str, timeout_sec: int = None):
    print(f"[CMD] {command}")

    result = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )

    if result.stdout:
        print(result.stdout.strip())

    if result.stderr:
        print(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: code={result.returncode}, cmd={command}")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_safe(path: Path):
    try:
        return load_json(path)
    except Exception:
        return None


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def write_view_command(view):
    command_id = f"v26_cmd_{uuid.uuid4().hex[:8]}"

    command = {
        "mode": "V2_6_ACTIVE_VIEW_SEARCH_COMMAND",
        "command_id": command_id,
        "view_id": view["view_id"],
        "robot_xyz": view["robot_xyz"],
        "look_at_xyz": view["look_at_xyz"],
        "head_offset_xyz": [0.0, 0.0, 1.55],
        "move_frames": 90,
        "settle_frames": 50,
        "created_timestamp": time.time(),
    }

    save_json_atomic(COMMAND_JSON, command)

    print("[WRITE COMMAND]")
    print(json.dumps(command, ensure_ascii=False, indent=2))

    return command


def wait_view_result(command_id, timeout_sec):
    print(f"[WAIT VIEW RESULT] command_id={command_id}")

    start = time.time()

    while time.time() - start < timeout_sec:
        result = load_json_safe(RESULT_JSON)

        if result is not None and result.get("command_id") == command_id:
            print("[VIEW RESULT]")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result

        time.sleep(0.2)

    raise TimeoutError(f"Timeout waiting view result: {command_id}")


def collect_image(timeout_sec):
    cmd = (
        "source ~/use_ros2_isaac.sh && "
        "python3 ~/trashbot_ws/scripts/tools/collect_one_ros_image_fast.py "
        "--topic /trashbot/camera/rgb "
        f"--timeout {timeout_sec}"
    )

    run_bash(cmd, timeout_sec=timeout_sec + 10)


def run_yolo(conf):
    cmd = (
        "source ~/envs/yolo/bin/activate && "
        "python ~/trashbot_ws/scripts/perception/yolo_seg_offline.py "
        f"--conf {conf}"
    )

    run_bash(cmd, timeout_sec=180)

    if not YOLO_RESULT.exists():
        raise FileNotFoundError(f"YOLO result not found: {YOLO_RESULT}")

    return load_json(YOLO_RESULT)


def bbox_area_ratio(det, image_width, image_height):
    bbox = det.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = float(image_width * image_height)

    if total <= 0:
        return 0.0

    return area / total


def select_best_detection(yolo_result, min_conf, min_area_ratio):
    image_path = yolo_result.get("image")
    detections = yolo_result.get("detections", [])

    image_width = int(yolo_result.get("image_width", 480))
    image_height = int(yolo_result.get("image_height", 320))

    # 有些旧 yolo result 不写 image_width/image_height，兜底用 480x320
    candidates = []

    for det in detections:
        raw = det.get("raw_class_name", det.get("class_name", "unknown"))
        conf = float(det.get("confidence", 0.0))

        if raw not in STABLE_RAW_CLASSES:
            continue

        if conf < min_conf:
            continue

        area_ratio = bbox_area_ratio(det, image_width, image_height)

        if area_ratio < min_area_ratio:
            continue

        score = conf + min(area_ratio * 20.0, 1.0)

        item = {
            "raw_class_name": raw,
            "confidence": round(conf, 4),
            "area_ratio": round(area_ratio, 6),
            "score": round(score, 4),
            "category": det.get("category"),
            "target_bin": det.get("target_bin"),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "centroid_px": det.get("centroid_px"),
        }

        candidates.append(item)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "image": image_path,
        "num_detections": len(detections),
        "num_candidates": len(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
    }


def make_v2_plan(min_conf):
    cmd = (
        "python3 ~/trashbot_ws/scripts/control/v2_make_visual_task_plan.py "
        "--select confidence "
        "--profile stable "
        f"--min-conf {min_conf} "
        "--blocked-raw potato,potatocut,rabbitcut,mooli,brick,china,stone"
    )

    run_bash(cmd, timeout_sec=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--collect-timeout-sec", type=int, default=6)
    parser.add_argument("--view-timeout-sec", type=int, default=120)
    parser.add_argument("--max-views", type=int, default=5)
    args = parser.parse_args()

    records = []

    print("=" * 80)
    print("[START] V2.6 active view search WSL controller")
    print("[INFO] Isaac must be running v26_active_view_search_isaac_server.py")
    print("=" * 80)

    selected_view = None
    selected_detection = None

    for idx, view in enumerate(VIEW_CANDIDATES[:args.max_views], start=1):
        print("\n" + "=" * 80)
        print(f"[TRY VIEW {idx}] {view['view_id']}")
        print("=" * 80)

        command = write_view_command(view)
        view_result = wait_view_result(command["command_id"], args.view_timeout_sec)

        if view_result.get("status") != "success":
            records.append({
                "view_index": idx,
                "view": view,
                "command": command,
                "view_result": view_result,
                "detection_eval": None,
                "accepted": False,
                "reason": "view_move_failed",
            })
            continue

        collect_image(args.collect_timeout_sec)
        yolo_result = run_yolo(args.conf)

        detection_eval = select_best_detection(
            yolo_result=yolo_result,
            min_conf=args.conf,
            min_area_ratio=args.min_area_ratio,
        )

        accepted = detection_eval["best"] is not None

        record = {
            "view_index": idx,
            "view": view,
            "command": command,
            "view_result": view_result,
            "detection_eval": detection_eval,
            "accepted": accepted,
            "reason": "accepted" if accepted else "no_valid_detection",
        }

        records.append(record)

        print("[DETECTION EVAL]")
        print(json.dumps(detection_eval, ensure_ascii=False, indent=2))

        if accepted:
            selected_view = view
            selected_detection = detection_eval["best"]
            print("[ACCEPT VIEW]")
            print(json.dumps({
                "selected_view": selected_view,
                "selected_detection": selected_detection,
            }, ensure_ascii=False, indent=2))
            break

    summary = {
        "mode": "V2_6_ACTIVE_VIEW_SEARCH",
        "success": selected_view is not None,
        "selected_view": selected_view,
        "selected_detection": selected_detection,
        "records": records,
        "created_timestamp": time.time(),
    }

    save_json_atomic(SEARCH_LOG_JSON, summary)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVED] {SEARCH_LOG_JSON}")
    print("=" * 80)

    if selected_view is None:
        print("[FAILED] 没有找到可用视角。可以降低 --min-area-ratio 或增加候选视角。")
        return

    # 视角找到后，生成抓取任务 plan
    make_v2_plan(args.conf)

    print("[OK] 已自动找到有效视角，并生成 v2_visual_task_plan.json")
    print("[NEXT] Isaac 运行 v25_head_camera_single_target_executor_v2.py 执行单目标抓取。")


if __name__ == "__main__":
    main()