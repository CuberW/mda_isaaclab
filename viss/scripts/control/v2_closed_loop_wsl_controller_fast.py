#!/usr/bin/env python3
import argparse
import csv
import json
import select
import subprocess
import time
import uuid
from pathlib import Path

# Setup paths
ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"
IMAGE_ROOT = ROOT / "data" / "images"

YOLO_WORKER = ROOT / "scripts" / "perception" / "yolo_persistent_worker.py"
YOLO_PYTHON = Path.home() / "envs" / "yolo" / "bin" / "python"

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

class PersistentYoloClient:
    def __init__(self, conf: float):
        if not YOLO_PYTHON.exists():
            raise FileNotFoundError(f"YOLO python not found: {YOLO_PYTHON}")
        if not YOLO_WORKER.exists():
            raise FileNotFoundError(f"YOLO worker not found: {YOLO_WORKER}")

        self.conf = conf
        cmd = [
            str(YOLO_PYTHON),
            str(YOLO_WORKER),
            "--default-conf",
            str(conf),
        ]
        print(f"[START YOLO WORKER] {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Failed to open YOLO worker pipes")
        
        time.sleep(1.0)
        if self.process.poll() is not None:
            raise RuntimeError(f"YOLO worker exited early with code {self.process.returncode}")

    def infer(self, image_path: Path, request_id: str, save_overlay: bool, timeout_sec: int = 120):
        request = {
            "request_id": request_id,
            "image": str(image_path),
            "conf": self.conf,
            "save_overlay": save_overlay,
            "max_bbox_ratio": 0.35,
            "max_mask_ratio": 0.30,
        }
        line = json.dumps(request, ensure_ascii=False)
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

        start = time.time()
        while time.time() - start < timeout_sec:
            if self.process.poll() is not None:
                raise RuntimeError(f"YOLO worker exited with code {self.process.returncode}")

            ready, _, _ = select.select([self.process.stdout], [], [], 0.2)
            if not ready:
                continue

            response_line = self.process.stdout.readline()
            if not response_line:
                continue
            response_line = response_line.strip()
            if not response_line:
                continue

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                print(f"[WARN] non-json worker output: {response_line}")
                continue

            if response.get("request_id") != request_id:
                print(f"[WARN] skip stale response: {response}")
                continue

            if not response.get("ok"):
                raise RuntimeError(f"YOLO worker failed: {response}")

            return response["summary"]

        raise TimeoutError(f"YOLO worker timeout, request_id={request_id}")

    def stop(self):
        if self.process.poll() is not None:
            return
        try:
            request = {"cmd": "stop", "request_id": "stop"}
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()

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
    
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: code={result.returncode}, cmd={command}")
    return time.time() - start

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)

def find_latest_image():
    sessions = sorted([p for p in IMAGE_ROOT.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No image sessions found in: {IMAGE_ROOT}")
    latest = sessions[-1]
    images = sorted(list(latest.glob("*.jpg")) + list(latest.glob("*.png")))
    images = [p for p in images if "preview" not in p.name.lower()]
    if not images:
        raise FileNotFoundError(f"No images found in latest session: {latest}")
    return images[0]

def collect_one_image_fast(timeout_sec: int):
    before = None
    try:
        before = find_latest_image()
    except Exception:
        before = None

    cmd = (
        "source ~/use_ros2_isaac.sh && "
        "python3 ~/trashbot_ws/scripts/tools/collect_one_ros_image_fast.py "
        "--topic /trashbot/camera/rgb "
        f"--timeout {timeout_sec} --skip-frames 5"
    )
    elapsed = run_bash(cmd, timeout_sec=timeout_sec + 5)
    after = find_latest_image()
    if before is not None and after == before:
        print("[WARN] Latest image path did not change.")
    print(f"[IMAGE] {after}")
    return after, elapsed

def make_v2_plan(select: str, profile: str, blocked_raw: str):
    cmd = (
        "python3 ~/trashbot_ws/scripts/control/v2_make_visual_task_plan.py "
        f"--select {select} --profile {profile} --blocked-raw \"{blocked_raw}\""
    )
    elapsed = run_bash(cmd, timeout_sec=30)
    return load_json(PLAN_JSON), elapsed

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
    count = {"total": len(tasks), "by_raw_class": {}, "by_category": {}, "by_target_bin": {}}
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
    count = {"total": len(detections), "by_raw_class": {}, "by_category": {}, "by_target_bin": {}}
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
    raw_class = isaac_result.get("raw_class_name", selected.get("raw_class_name", "unknown"))
    category = isaac_result.get("garbage_category", selected.get("garbage_category", "unknown"))

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
    verified_count = sum(1 for r in records if r.get("verification", {}).get("verified") is True)
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
        "cycle_id", "plan_id", "raw_class_name", "category_cn", "target_bin",
        "isaac_status", "verified", "vote_score", "attach_distance_xy",
        "calibration_rmse_xy_m", "before_total", "after_total", "cycle_sec"
    ]
    with open(RUN_LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            isaac = r.get("isaac_result", {})
            verify = r.get("verification", {}) or {}
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
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--collect-timeout-sec", type=int, default=6)
    parser.add_argument("--isaac-timeout-sec", type=int, default=180)
    parser.add_argument("--select", default="priority", choices=["priority", "confidence"])
    parser.add_argument("--settle-sec", type=float, default=0.4)
    parser.add_argument("--profile", default="all", choices=["stable", "stable_plus_kitchen", "kitchen_debug", "all"])
    parser.add_argument("--blocked-raw", default="")# potato,potatocut,rabbitcut,mooli,brick,china,stone
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    
    print("[START] v2_closed_loop_wsl_controller_fast.py")
    print("[INFO] Please verify that v2_closed_loop_isaac_executor.py is running in Isaac Sim.")
    
    yolo_client = PersistentYoloClient(conf=args.conf)
    
    try:
        # Pre-capture first frame for Cycle 1 Before
        print("\n" + "-" * 50)
        print("[INIT] Capturing initial frame...")
        image_path, collect_sec = collect_one_image_fast(args.collect_timeout_sec)
        
        request_id = f"init_{uuid.uuid4().hex[:8]}"
        yolo_start = time.time()
        yolo_data = yolo_client.infer(
            image_path=image_path,
            request_id=request_id,
            save_overlay=True,
            timeout_sec=120,
        )
        yolo_sec = time.time() - yolo_start
        print(f"[INIT YOLO] num_detections={yolo_data.get('num_detections')}, time={yolo_sec:.3f}s")
        
        for cycle_id in range(1, args.max_cycles + 1):
            print("\n" + "=" * 80)
            print(f"[CYCLE {cycle_id}]")
            
            cycle_start_time = time.time()
            
            # 1. Planning
            plan, plan_sec = make_v2_plan(args.select, args.profile, args.blocked_raw)
            if plan.get("selected_task") is None or plan.get("num_tasks", 0) <= 0:
                print("[STOP] No targets available for planning. Finished.")
                break
                
            plan["cycle_id"] = cycle_id
            save_json_atomic(PLAN_JSON, plan)
            
            plan_id = plan["plan_id"]
            selected_task = plan.get("selected_task")
            print("[SELECTED]")
            print(json.dumps(selected_task, ensure_ascii=False, indent=2))
            
            # 2. Wait for execution
            isaac_result, isaac_wait_sec = wait_for_isaac_result(
                plan_id=plan_id,
                timeout_sec=args.isaac_timeout_sec,
            )
            time.sleep(args.settle_sec)
            
            # 3. Create record (After timing and Verification will be filled in the next step)
            record = {
                "cycle_id": cycle_id,
                "plan_id": plan_id,
                "selected_task": selected_task,
                "source_image_before": plan.get("source_image"),
                "source_image_after": None,
                "isaac_result": isaac_result,
                "verification": None,
                "timing": {
                    "before_collect_sec": round(collect_sec, 4),
                    "before_yolo_sec": round(yolo_sec, 4),
                    "plan_sec": round(plan_sec, 4),
                    "isaac_wait_sec": round(isaac_wait_sec, 4),
                    "after_collect_sec": 0.0,
                    "after_yolo_sec": 0.0,
                    "cycle_sec": 0.0,
                }
            }
            records.append(record)
            
            # 4. Capture next frame (serves as the Verification frame of current cycle, and the Before frame of next cycle)
            print("\n[NEXT STEP] Capturing next frame...")
            next_image_path, next_collect_sec = collect_one_image_fast(args.collect_timeout_sec)
            
            next_request_id = f"cycle_{cycle_id:03d}_{uuid.uuid4().hex[:8]}"
            next_yolo_start = time.time()
            next_yolo_data = yolo_client.infer(
                image_path=next_image_path,
                request_id=next_request_id,
                save_overlay=True,
                timeout_sec=120,
            )
            next_yolo_sec = time.time() - next_yolo_start
            
            # 5. Verify current cycle using the new frame
            verification = verify_after_action(plan, isaac_result, next_yolo_data)
            print("[VERIFY]")
            print(json.dumps(verification, ensure_ascii=False, indent=2))
            
            # 6. Update current cycle's record and save logs
            records[-1]["source_image_after"] = next_yolo_data.get("image")
            records[-1]["verification"] = verification
            records[-1]["timing"]["after_collect_sec"] = round(next_collect_sec, 4)
            records[-1]["timing"]["after_yolo_sec"] = round(next_yolo_sec, 4)
            records[-1]["timing"]["cycle_sec"] = round(time.time() - cycle_start_time, 4)
            
            save_run_logs(records)
            
            if not verification["verified"]:
                print("[WARN] Verification failed for this cycle. Stopping.")
                break
                
            # Prepare state variables for next iteration
            image_path = next_image_path
            collect_sec = next_collect_sec
            yolo_sec = next_yolo_sec
            yolo_data = next_yolo_data
            
        # Final output
        print("\n" + "=" * 80)
        print("[DONE] V2 Fast closed loop finished.")
        print(json.dumps(compute_metrics(records), ensure_ascii=False, indent=2))
        
    finally:
        yolo_client.stop()

if __name__ == "__main__":
    main()
