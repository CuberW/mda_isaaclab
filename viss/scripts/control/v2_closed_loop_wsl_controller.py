import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"

PLAN_JSON = Path("/mnt/d/isaac_projects/v2_visual_task_plan.json")
ISAAC_RESULT_JSON = Path("/mnt/d/isaac_projects/v2_closed_loop_isaac_result.json")

YOLO_RESULT = LOG_DIR / "yolo_seg_offline_result.json"

RUN_LOG_JSON = LOG_DIR / "v2_closed_loop_run_log.json"
RUN_LOG_CSV = LOG_DIR / "v2_closed_loop_run_table.csv"


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}


def run_bash(command: str, timeout_sec: int = None):
    print(f"[CMD] {command}")
    start = time.time()

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

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: code={result.returncode}, cmd={command}")

    return elapsed


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def collect_image(timeout_sec: int):
    cmd = (
        "source ~/use_ros2_isaac.sh && "
        "python3 ~/trashbot_ws/scripts/tools/collect_one_ros_image_fast.py "
        "--topic /trashbot/camera/rgb "
        f"--timeout {timeout_sec}"
    )

    elapsed = run_bash(cmd, timeout_sec=timeout_sec + 5)
    return elapsed


def run_yolo(conf: float):
    cmd = (
        "source ~/envs/yolo/bin/activate && "
        "python ~/trashbot_ws/scripts/perception/yolo_seg_offline.py "
        f"--conf {conf}"
    )

    elapsed = run_bash(cmd, timeout_sec=180)

    if not YOLO_RESULT.exists():
        raise FileNotFoundError(f"YOLO result not found: {YOLO_RESULT}")

    yolo = load_json(YOLO_RESULT)

    return yolo, elapsed


def make_v2_plan(select: str, profile: str, blocked_raw: str):
    cmd = (
        "python3 ~/trashbot_ws/scripts/control/v2_make_visual_task_plan.py "
        f"--select {select} --profile {profile} --blocked-raw \"{blocked_raw}\""
    )

    elapsed = run_bash(cmd, timeout_sec=30)

    if not PLAN_JSON.exists():
        raise FileNotFoundError(f"Plan not found: {PLAN_JSON}")

    plan = load_json(PLAN_JSON)

    return plan, elapsed


def wait_for_isaac_result(plan_id: str, timeout_sec: int):
    print(f"[WAIT ISAAC] plan_id={plan_id}")
    start = time.time()

    while time.time() - start < timeout_sec:
        if ISAAC_RESULT_JSON.exists():
            try:
                result = load_json(ISAAC_RESULT_JSON)
            except Exception:
                time.sleep(0.2)
                continue

            if result.get("plan_id") == plan_id:
                elapsed = time.time() - start
                print(f"[ISAAC RESULT] status={result.get('status')}, elapsed={elapsed:.3f}s")
                return result, elapsed

        time.sleep(0.2)

    raise TimeoutError(f"Timeout waiting Isaac result for plan_id={plan_id}")


def count_plan_tasks(plan):
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


def count_yolo_detections(yolo_result):
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


def verify_after_action(plan, isaac_result, after_yolo):
    selected = plan.get("selected_task", {})

    raw_class = isaac_result.get(
        "raw_class_name",
        selected.get("raw_class_name", selected.get("class_name", "unknown")),
    )

    category = isaac_result.get(
        "garbage_category",
        selected.get("garbage_category", selected.get("category", "unknown")),
    )

    before_counts = count_plan_tasks(plan)
    after_counts = count_yolo_detections(after_yolo)

    before_total = int(before_counts.get("total", 0))
    after_total = int(after_counts.get("total", 0))

    before_raw = int(before_counts.get("by_raw_class", {}).get(raw_class, 0))
    after_raw = int(after_counts.get("by_raw_class", {}).get(raw_class, 0))

    before_category = int(before_counts.get("by_category", {}).get(category, 0))
    after_category = int(after_counts.get("by_category", {}).get(category, 0))

    vote_total = after_total < before_total
    vote_raw = after_raw < before_raw
    vote_category = after_category < before_category

    vote_score = int(vote_total) + int(vote_raw) + int(vote_category)
    verified = vote_score >= 2

    return {
        "verified": verified,
        "vote_score": vote_score,
        "selected_raw_class": raw_class,
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


def compute_metrics(records):
    n = len(records)

    if n == 0:
        return {
            "num_cycles": 0,
            "isaac_success_rate": 0.0,
            "verification_success_rate": 0.0,
            "average_attach_distance_xy": 0.0,
            "average_cycle_sec": 0.0,
        }

    isaac_success = sum(1 for r in records if r.get("isaac_result", {}).get("status") == "success")
    verified_count = sum(1 for r in records if r.get("verification", {}).get("verified"))

    attach_values = [
        float(r.get("isaac_result", {}).get("attach_distance_xy"))
        for r in records
        if r.get("isaac_result", {}).get("attach_distance_xy") is not None
    ]

    cycle_values = [
        float(r.get("timing", {}).get("cycle_sec", 0.0))
        for r in records
    ]

    return {
        "num_cycles": n,
        "isaac_success_rate": round(isaac_success / n, 4),
        "verification_success_rate": round(verified_count / n, 4),
        "average_attach_distance_xy": round(sum(attach_values) / len(attach_values), 4) if attach_values else 0.0,
        "average_cycle_sec": round(sum(cycle_values) / len(cycle_values), 4) if cycle_values else 0.0,
    }


def save_run_logs(records):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": "V2_4_VISUAL_CLOSED_LOOP_RUN",
        "num_records": len(records),
        "records": records,
        "metrics": compute_metrics(records),
    }

    save_json_atomic(RUN_LOG_JSON, summary)

    fieldnames = [
        "cycle_id",
        "plan_id",
        "raw_class_name",
        "category_cn",
        "target_bin",
        "isaac_status",
        "verified",
        "vote_score",
        "attach_distance_xy",
        "calibration_rmse_xy_m",
        "before_total",
        "after_total",
        "cycle_sec",
    ]

    with open(RUN_LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            isaac = r.get("isaac_result", {})
            verify = r.get("verification", {})
            selected = r.get("selected_task", {})
            timing = r.get("timing", {})

            writer.writerow({
                "cycle_id": r.get("cycle_id"),
                "plan_id": r.get("plan_id"),
                "raw_class_name": selected.get("raw_class_name"),
                "category_cn": selected.get("garbage_category_cn"),
                "target_bin": selected.get("target_bin"),
                "isaac_status": isaac.get("status"),
                "verified": verify.get("verified"),
                "vote_score": verify.get("vote_score"),
                "attach_distance_xy": isaac.get("attach_distance_xy"),
                "calibration_rmse_xy_m": isaac.get("calibration_rmse_xy_m"),
                "before_total": verify.get("before_total"),
                "after_total": verify.get("after_total"),
                "cycle_sec": timing.get("cycle_sec"),
            })

    print(f"[SAVED] {RUN_LOG_JSON}")
    print(f"[SAVED] {RUN_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--collect-timeout-sec", type=int, default=6)
    parser.add_argument("--isaac-timeout-sec", type=int, default=180)
    parser.add_argument("--select", default="priority", choices=["priority", "confidence"])
    parser.add_argument("--profile", default="stable", choices=["stable", "stable_plus_kitchen", "kitchen_debug", "all"])
    parser.add_argument("--blocked-raw", default="potato,potatocut,rabbitcut,mooli,brick,china,stone")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    print("[START] v2_closed_loop_wsl_controller.py")
    print("[INFO] 请确认 Isaac Sim 已运行 v2_closed_loop_isaac_executor.py")

    for cycle_id in range(1, args.max_cycles + 1):
        print("\n" + "=" * 80)
        print(f"[CYCLE {cycle_id}]")

        cycle_start = time.time()

        before_collect_sec = collect_image(args.collect_timeout_sec)
        before_yolo, before_yolo_sec = run_yolo(args.conf)

        plan, plan_sec = make_v2_plan(args.select, args.profile, args.blocked_raw)

        if plan.get("selected_task") is None or plan.get("num_tasks", 0) <= 0:
            print("[STOP] 当前没有可执行目标。")
            break

        plan["cycle_id"] = cycle_id
        save_json_atomic(PLAN_JSON, plan)

        plan_id = plan["plan_id"]
        selected_task = plan.get("selected_task")

        print("[SELECTED]")
        print(json.dumps(selected_task, ensure_ascii=False, indent=2))

        isaac_result, isaac_wait_sec = wait_for_isaac_result(
            plan_id=plan_id,
            timeout_sec=args.isaac_timeout_sec,
        )

        after_collect_sec = collect_image(args.collect_timeout_sec)
        after_yolo, after_yolo_sec = run_yolo(args.conf)

        verification = verify_after_action(
            plan=plan,
            isaac_result=isaac_result,
            after_yolo=after_yolo,
        )

        print("[VERIFY]")
        print(json.dumps(verification, ensure_ascii=False, indent=2))

        record = {
            "cycle_id": cycle_id,
            "plan_id": plan_id,
            "selected_task": selected_task,
            "source_image_before": plan.get("source_image"),
            "source_image_after": after_yolo.get("image"),
            "isaac_result": isaac_result,
            "verification": verification,
            "timing": {
                "before_collect_sec": round(before_collect_sec, 4),
                "before_yolo_sec": round(before_yolo_sec, 4),
                "plan_sec": round(plan_sec, 4),
                "isaac_wait_sec": round(isaac_wait_sec, 4),
                "after_collect_sec": round(after_collect_sec, 4),
                "after_yolo_sec": round(after_yolo_sec, 4),
                "cycle_sec": round(time.time() - cycle_start, 4),
            },
        }

        records.append(record)
        save_run_logs(records)

        if not verification["verified"]:
            print("[WARN] 本轮重新识别验证未通过，建议停止并检查。")
            break

    save_run_logs(records)

    print("[DONE] V2 closed loop finished")
    print(json.dumps(compute_metrics(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()