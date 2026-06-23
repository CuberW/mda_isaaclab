import argparse
import csv
import json
import subprocess
import time
import uuid
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"

VIEW_COMMAND_JSON = Path("/mnt/d/isaac_projects/v26_view_command.json")
VIEW_RESULT_JSON = Path("/mnt/d/isaac_projects/v26_view_result.json")

PLAN_JSON = Path("/mnt/d/isaac_projects/v2_visual_task_plan.json")
ACTION_RESULT_JSON = Path("/mnt/d/isaac_projects/v27_head_camera_action_result.json")

YOLO_RESULT = LOG_DIR / "yolo_seg_offline_result.json"

RUN_LOG_JSON = LOG_DIR / "v27_head_camera_all_trash_run_log.json"
RUN_LOG_CSV = LOG_DIR / "v27_head_camera_all_trash_run_table.csv"


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}


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
    command_id = f"v27_view_{uuid.uuid4().hex[:8]}"

    command = {
        "mode": "V2_7_VIEW_COMMAND",
        "command_id": command_id,
        "view_id": view["view_id"],
        "robot_xyz": view["robot_xyz"],
        "look_at_xyz": view["look_at_xyz"],
        "head_offset_xyz": [0.0, 0.0, 1.55],
        "move_frames": 90,
        "settle_frames": 50,
        "created_timestamp": time.time(),
    }

    save_json_atomic(VIEW_COMMAND_JSON, command)

    print("[WRITE VIEW COMMAND]")
    print(json.dumps(command, ensure_ascii=False, indent=2))

    return command


def wait_view_result(command_id, timeout_sec):
    print(f"[WAIT VIEW RESULT] {command_id}")

    start = time.time()

    while time.time() - start < timeout_sec:
        result = load_json_safe(VIEW_RESULT_JSON)

        if result is not None and result.get("command_id") == command_id:
            print(f"[VIEW RESULT] status={result.get('status')}")
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


def bbox_area_ratio(det, image_width=480, image_height=320):
    bbox = det.get("bbox_xyxy")

    if bbox is None or len(bbox) != 4:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = float(image_width * image_height)

    if total <= 0:
        return 0.0

    return area / total


def count_detections(yolo_result):
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


def evaluate_view_detections(yolo_result, min_conf, min_area_ratio, blocked_raw):
    detections = yolo_result.get("detections", [])

    candidates = []

    for det in detections:
        raw = det.get("raw_class_name", det.get("class_name", "unknown"))
        category = det.get("category", "unknown")
        conf = float(det.get("confidence", 0.0))

        if raw in blocked_raw:
            continue

        if conf < min_conf:
            continue

        area_ratio = bbox_area_ratio(det)

        if area_ratio < min_area_ratio:
            continue

        if det.get("target_bin", "unknown") == "unknown":
            continue

        score = conf + min(area_ratio * 20.0, 1.0)

        candidates.append({
            "raw_class_name": raw,
            "category": category,
            "target_bin": det.get("target_bin"),
            "confidence": round(conf, 4),
            "area_ratio": round(area_ratio, 6),
            "score": round(score, 4),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "centroid_px": det.get("centroid_px"),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "image": yolo_result.get("image"),
        "num_detections": len(detections),
        "num_candidates": len(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
    }


def make_plan(profile, min_conf, blocked_raw, selected_view, cycle_id):
    blocked_arg = ",".join(sorted(blocked_raw))

    cmd = (
        "python3 ~/trashbot_ws/scripts/control/v2_make_visual_task_plan.py "
        "--select confidence "
        f"--profile {profile} "
        f"--min-conf {min_conf} "
        f"--blocked-raw {blocked_arg}"
    )

    run_bash(cmd, timeout_sec=30)

    plan = load_json(PLAN_JSON)

    if plan.get("selected_task") is None:
        return plan

    plan["cycle_id"] = cycle_id
    plan["v26_selected_view"] = selected_view
    plan["mode"] = "V2_7_HEAD_CAMERA_ALL_TRASH_PLAN"

    save_json_atomic(PLAN_JSON, plan)

    return plan


def wait_action_result(plan_id, timeout_sec):
    print(f"[WAIT ACTION RESULT] plan_id={plan_id}")

    start = time.time()

    while time.time() - start < timeout_sec:
        result = load_json_safe(ACTION_RESULT_JSON)

        if result is not None and result.get("plan_id") == plan_id:
            print(f"[ACTION RESULT] status={result.get('status')}")
            return result

        time.sleep(0.2)

    raise TimeoutError(f"Timeout waiting action result: {plan_id}")


def verify_after_action(before_yolo, action_result, after_yolo):
    raw = action_result.get("raw_class_name", "unknown")
    category = action_result.get("garbage_category", "unknown")

    before_counts = count_detections(before_yolo)
    after_counts = count_detections(after_yolo)

    before_total = int(before_counts["total"])
    after_total = int(after_counts["total"])

    before_raw = int(before_counts["by_raw_class"].get(raw, 0))
    after_raw = int(after_counts["by_raw_class"].get(raw, 0))

    before_category = int(before_counts["by_category"].get(category, 0))
    after_category = int(after_counts["by_category"].get(category, 0))

    vote_total = after_total < before_total
    vote_raw = after_raw < before_raw
    vote_category = after_category < before_category

    vote_score = int(vote_total) + int(vote_raw) + int(vote_category)
    verified = vote_score >= 2

    return {
        "verified": verified,
        "vote_score": vote_score,
        "selected_raw_class": raw,
        "selected_category": category,
        "selected_category_cn": CATEGORY_CN.get(category, category),
        "before_total": before_total,
        "after_total": after_total,
        "before_raw_class_count": before_raw,
        "after_raw_class_count": after_raw,
        "before_category_count": before_category,
        "after_category_count": after_category,
        "vote_total_decreased": vote_total,
        "vote_raw_class_decreased": vote_raw,
        "vote_category_decreased": vote_category,
        "before_counts": before_counts,
        "after_counts": after_counts,
    }


def find_valid_view(args, blocked_raw):
    records = []

    for idx, view in enumerate(VIEW_CANDIDATES[:args.max_views_per_cycle], start=1):
        print("\n" + "-" * 80)
        print(f"[VIEW TRY] {idx}: {view['view_id']}")
        print("-" * 80)

        command = write_view_command(view)
        view_result = wait_view_result(command["command_id"], args.view_timeout_sec)

        if view_result.get("status") != "success":
            records.append({
                "view_index": idx,
                "view": view,
                "view_result": view_result,
                "accepted": False,
                "reason": "view_move_failed",
            })
            continue

        collect_image(args.collect_timeout_sec)
        yolo_result = run_yolo(args.conf)

        detection_eval = evaluate_view_detections(
            yolo_result=yolo_result,
            min_conf=args.conf,
            min_area_ratio=args.min_area_ratio,
            blocked_raw=blocked_raw,
        )

        accepted = detection_eval["best"] is not None

        record = {
            "view_index": idx,
            "view": view,
            "view_result": view_result,
            "yolo_image": yolo_result.get("image"),
            "detection_eval": detection_eval,
            "accepted": accepted,
            "reason": "accepted" if accepted else "no_valid_detection",
        }

        records.append(record)

        print("[DETECTION EVAL]")
        print(json.dumps(detection_eval, ensure_ascii=False, indent=2))

        if accepted:
            return view, yolo_result, detection_eval, records

    return None, None, None, records


def compute_metrics(records):
    n = len(records)

    if n == 0:
        return {
            "num_cycles": 0,
            "action_success_rate": 0.0,
            "verification_success_rate": 0.0,
            "average_attach_distance_xy": 0.0,
        }

    action_success = sum(1 for r in records if r.get("action_result", {}).get("status") == "success")
    verified = sum(1 for r in records if r.get("verification", {}).get("verified"))

    attach_values = [
        float(r.get("action_result", {}).get("attach_distance_xy"))
        for r in records
        if r.get("action_result", {}).get("attach_distance_xy") is not None
    ]

    return {
        "num_cycles": n,
        "action_success_rate": round(action_success / n, 4),
        "verification_success_rate": round(verified / n, 4),
        "average_attach_distance_xy": round(sum(attach_values) / len(attach_values), 4) if attach_values else 0.0,
    }


def save_logs(records, summary_extra=None):
    summary = {
        "mode": "V2_7_HEAD_CAMERA_ALL_TRASH_LOOP",
        "num_records": len(records),
        "metrics": compute_metrics(records),
        "records": records,
    }

    if summary_extra:
        summary.update(summary_extra)

    save_json_atomic(RUN_LOG_JSON, summary)

    fieldnames = [
        "cycle_id",
        "plan_id",
        "view_id",
        "raw_class_name",
        "category",
        "target_bin",
        "confidence",
        "action_status",
        "verified",
        "vote_score",
        "attach_distance_xy",
        "target_relative_to_robot_xyz",
    ]

    with open(RUN_LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            action = r.get("action_result", {})
            verify = r.get("verification", {})
            selected = r.get("selected_task", {})
            view = r.get("selected_view", {})

            writer.writerow({
                "cycle_id": r.get("cycle_id"),
                "plan_id": r.get("plan_id"),
                "view_id": view.get("view_id"),
                "raw_class_name": selected.get("raw_class_name"),
                "category": selected.get("garbage_category"),
                "target_bin": selected.get("target_bin"),
                "confidence": selected.get("confidence"),
                "action_status": action.get("status"),
                "verified": verify.get("verified"),
                "vote_score": verify.get("vote_score"),
                "attach_distance_xy": action.get("attach_distance_xy"),
                "target_relative_to_robot_xyz": action.get("target_relative_to_robot_xyz"),
            })

    print(f"[SAVED] {RUN_LOG_JSON}")
    print(f"[SAVED] {RUN_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--max-views-per-cycle", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--collect-timeout-sec", type=int, default=6)
    parser.add_argument("--view-timeout-sec", type=int, default=120)
    parser.add_argument("--action-timeout-sec", type=int, default=180)
    parser.add_argument("--profile", default="stable", choices=["stable", "stable_plus_kitchen", "all"])
    parser.add_argument(
        "--blocked-raw",
        default="potato,potatocut,rabbitcut,mooli,brick,china,stone",
        help="Comma-separated raw classes to block.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    blocked_raw = {x.strip() for x in args.blocked_raw.split(",") if x.strip()}

    records = []

    print("=" * 80)
    print("[START] V2.7 head camera all-trash loop")
    print("[INFO] Isaac must run both:")
    print("  1) v26_active_view_search_isaac_server.py")
    print("  2) v27_head_camera_action_server.py")
    print("=" * 80)

    for cycle_id in range(1, args.max_cycles + 1):
        print("\n" + "=" * 80)
        print(f"[CYCLE {cycle_id}]")
        print("=" * 80)

        cycle_start = time.time()

        selected_view, before_yolo, detection_eval, view_records = find_valid_view(args, blocked_raw)

        if selected_view is None:
            print("[DONE] No valid view / no valid target. Stop loop.")
            save_logs(records, {
                "stop_reason": "no_valid_target_found",
                "last_view_records": view_records,
            })
            break

        plan = make_plan(
            profile=args.profile,
            min_conf=args.conf,
            blocked_raw=blocked_raw,
            selected_view=selected_view,
            cycle_id=cycle_id,
        )

        if plan.get("selected_task") is None:
            print("[DONE] Plan has no selected task after filtering. Stop loop.")
            save_logs(records, {
                "stop_reason": "no_selected_task_after_filtering",
                "last_view_records": view_records,
            })
            break

        plan_id = plan["plan_id"]
        selected_task = plan["selected_task"]

        print("[SELECTED TASK]")
        print(json.dumps(selected_task, ensure_ascii=False, indent=2))

        action_result = wait_action_result(plan_id, args.action_timeout_sec)

        collect_image(args.collect_timeout_sec)
        after_yolo = run_yolo(args.conf)

        verification = verify_after_action(
            before_yolo=before_yolo,
            action_result=action_result,
            after_yolo=after_yolo,
        )

        print("[VERIFY]")
        print(json.dumps(verification, ensure_ascii=False, indent=2))

        record = {
            "cycle_id": cycle_id,
            "plan_id": plan_id,
            "selected_view": selected_view,
            "view_search_records": view_records,
            "selected_task": selected_task,
            "before_image": before_yolo.get("image"),
            "after_image": after_yolo.get("image"),
            "action_result": action_result,
            "verification": verification,
            "timing": {
                "cycle_sec": round(time.time() - cycle_start, 4),
            },
        }

        records.append(record)
        save_logs(records)

        action_ok = action_result.get("status") == "success"
        verify_ok = verification.get("verified") is True

        if args.stop_on_failure and (not action_ok or not verify_ok):
            print("[STOP] Failure detected and --stop-on-failure is set.")
            break

    save_logs(records)

    print("[DONE] V2.7 loop finished")
    print(json.dumps(compute_metrics(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()