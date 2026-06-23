import argparse
import csv
import json
import select
import shlex
import subprocess
import time
import uuid
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"
IMAGE_ROOT = ROOT / "data" / "images"

YOLO_WORKER = ROOT / "scripts" / "perception" / "yolo_persistent_worker.py"
YOLO_PYTHON = Path.home() / "envs" / "yolo" / "bin" / "python"

PLAN_JSON = Path("/mnt/d/isaac_projects/closed_loop_task_plan.json")
ISAAC_RESULT_JSON = Path("/mnt/d/isaac_projects/closed_loop_isaac_result.json")

CLOSED_LOOP_LOG_JSON = LOG_DIR / "closed_loop_run_log.json"
CLOSED_LOOP_LOG_CSV = LOG_DIR / "closed_loop_run_table.csv"


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

ACTION_CODE = {
    "recyclable": 1,
    "hazardous": 2,
    "kitchen": 3,
    "other": 4,
    "unknown": 0,
}

CATEGORY_PRIORITY = {
    "hazardous": 1,
    "kitchen": 2,
    "recyclable": 3,
    "other": 4,
    "unknown": 9,
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

    def infer(
        self,
        image_path: Path,
        request_id: str,
        save_overlay: bool,
        timeout_sec: int = 120,
    ):
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
                raise RuntimeError(
                    f"YOLO worker exited with code {self.process.returncode}"
                )

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
            request = {
                "cmd": "stop",
                "request_id": "stop",
            }
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def run_bash(command: str, timeout_sec: int = None, allow_timeout: bool = False):
    print(f"[CMD] {command}")

    start = time.time()

    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        if allow_timeout:
            print(f"[TIMEOUT ALLOWED] {command}")
            return 124, e.stdout or "", e.stderr or "", time.time() - start
        raise

    if result.stdout:
        print(result.stdout.strip())

    if result.stderr:
        print(result.stderr.strip())

    return result.returncode, result.stdout, result.stderr, time.time() - start


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        f"--timeout {timeout_sec}"
    )

    code, stdout, stderr, elapsed = run_bash(
        cmd,
        timeout_sec=timeout_sec + 3,
        allow_timeout=False,
    )

    if code != 0:
        raise RuntimeError(f"Fast image collection failed with code {code}")

    after = find_latest_image()

    if before is not None and after == before:
        print("[WARN] 最新图片路径没有变化，继续使用最新图片。")

    print(f"[IMAGE] {after}")
    print(f"[TIMING] collect_sec={elapsed:.4f}")

    return after, elapsed


def normalize_detection(det, idx):
    category = det.get("category", "unknown")
    target_bin = det.get("target_bin", "unknown")
    raw_class = det.get("raw_class_name", det.get("class_name", "unknown"))

    return {
        "task_id": f"closed_task_{idx:03d}",
        "raw_class_name": raw_class,
        "raw_object_id": det.get("object_id", "unknown"),
        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),
        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),
        "confidence": float(det.get("confidence", 0.0)),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "centroid_px": det.get("centroid_px"),
        "action_code": ACTION_CODE.get(category, 0),
        "planning_status": "planned" if target_bin != "unknown" else "unplanned",
    }


def build_tasks_from_yolo(yolo_data):
    detections = yolo_data.get("detections", [])
    tasks = []

    for idx, det in enumerate(detections, start=1):
        task = normalize_detection(det, idx)

        if task["planning_status"] == "planned":
            tasks.append(task)

    tasks.sort(
        key=lambda x: (
            CATEGORY_PRIORITY.get(x["garbage_category"], 9),
            -x["confidence"],
        )
    )

    for i, task in enumerate(tasks, start=1):
        task["execution_order"] = i

    return tasks


def count_detections(tasks):
    count = {
        "total": len(tasks),
        "by_raw_class": {},
        "by_category": {},
        "by_target_bin": {},
    }

    for task in tasks:
        raw = task.get("raw_class_name", "unknown")
        category = task.get("garbage_category", "unknown")
        target_bin = task.get("target_bin", "unknown")

        count["by_raw_class"][raw] = count["by_raw_class"].get(raw, 0) + 1
        count["by_category"][category] = count["by_category"].get(category, 0) + 1
        count["by_target_bin"][target_bin] = count["by_target_bin"].get(target_bin, 0) + 1

    return count


def verify_previous_action(previous_result, current_counts):
    if previous_result is None:
        return None

    before_counts = previous_result.get("before_counts", {})
    raw_class = previous_result.get("raw_class_name", "unknown")
    category = previous_result.get("garbage_category", "unknown")

    before_total = int(before_counts.get("total", 0))
    after_total = int(current_counts.get("total", 0))

    before_raw = int(before_counts.get("by_raw_class", {}).get(raw_class, 0))
    after_raw = int(current_counts.get("by_raw_class", {}).get(raw_class, 0))

    before_category = int(before_counts.get("by_category", {}).get(category, 0))
    after_category = int(current_counts.get("by_category", {}).get(category, 0))

    vote_total = after_total < before_total
    vote_raw = after_raw < before_raw
    vote_category = after_category < before_category

    vote_score = int(vote_total) + int(vote_raw) + int(vote_category)
    verified = vote_score >= 2

    return {
        "verified": verified,
        "vote_score": vote_score,
        "vote_total_decreased": vote_total,
        "vote_raw_class_decreased": vote_raw,
        "vote_category_decreased": vote_category,
        "raw_class_name": raw_class,
        "garbage_category": category,
        "before_total": before_total,
        "after_total": after_total,
        "before_raw_class_count": before_raw,
        "after_raw_class_count": after_raw,
        "before_category_count": before_category,
        "after_category_count": after_category,
        "previous_plan_id": previous_result.get("plan_id"),
        "previous_task_id": previous_result.get("task_id"),
        "previous_action_status": previous_result.get("action_status"),
    }


def build_plan(cycle_id, yolo_data, tasks, previous_verification):
    plan_id = f"plan_{cycle_id:03d}_{uuid.uuid4().hex[:8]}"
    before_counts = count_detections(tasks)
    selected_task = tasks[0] if tasks else None

    return {
        "plan_id": plan_id,
        "cycle_id": cycle_id,
        "created_timestamp": time.time(),
        "source_image": yolo_data.get("image"),
        "source_model": yolo_data.get("model"),
        "num_detections": len(tasks),
        "before_counts": before_counts,
        "previous_verification": previous_verification,
        "selected_task": selected_task,
        "tasks": tasks,
        "note": "Fast closed loop: persistent YOLO worker, one-frame ROS capture, Isaac executes selected_task only.",
    }


def wait_for_isaac_result(plan_id: str, timeout_sec: int):
    start = time.time()

    print(f"[WAIT ISAAC] plan_id={plan_id}")

    while time.time() - start < timeout_sec:
        if ISAAC_RESULT_JSON.exists():
            try:
                result = load_json(ISAAC_RESULT_JSON)
            except Exception:
                time.sleep(0.2)
                continue

            if result.get("plan_id") == plan_id:
                print(f"[ISAAC RESULT] {result.get('action_status')}")
                print(f"[TIMING] isaac_wait_sec={time.time() - start:.4f}")
                return result, time.time() - start

        time.sleep(0.2)

    raise TimeoutError(f"Timeout waiting Isaac result for plan_id={plan_id}")


def compute_closed_loop_metrics(records):
    executed = [r for r in records if r.get("selected_task") is not None]
    n = len(executed)

    if n == 0:
        return {
            "num_executed_actions": 0,
            "isaac_action_completion_rate": 0.0,
            "num_verified_actions": 0,
            "verification_success_rate": 0.0,
            "average_collect_sec": 0.0,
            "average_yolo_sec": 0.0,
            "average_isaac_wait_sec": 0.0,
        }

    action_completed = 0
    verified = 0
    verified_count = 0

    collect_times = []
    yolo_times = []
    isaac_times = []

    for r in executed:
        isaac_result = r.get("isaac_result") or {}

        if isaac_result.get("action_completed"):
            action_completed += 1

        verification = r.get("next_cycle_verification")

        if verification is not None:
            verified_count += 1

            if verification.get("verified"):
                verified += 1

        timing = r.get("timing", {})

        if "collect_sec" in timing:
            collect_times.append(float(timing["collect_sec"]))

        if "yolo_total_sec" in timing:
            yolo_times.append(float(timing["yolo_total_sec"]))

        if "isaac_wait_sec" in timing:
            isaac_times.append(float(timing["isaac_wait_sec"]))

    return {
        "num_executed_actions": n,
        "isaac_action_completion_rate": round(action_completed / n, 4),
        "num_verified_actions": verified_count,
        "verification_success_rate": round(verified / verified_count, 4) if verified_count > 0 else 0.0,
        "average_collect_sec": round(sum(collect_times) / len(collect_times), 4) if collect_times else 0.0,
        "average_yolo_sec": round(sum(yolo_times) / len(yolo_times), 4) if yolo_times else 0.0,
        "average_isaac_wait_sec": round(sum(isaac_times) / len(isaac_times), 4) if isaac_times else 0.0,
    }


def save_closed_loop_logs(records):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_cycles": len(records),
        "records": records,
        "metrics": compute_closed_loop_metrics(records),
    }

    atomic_write_json(CLOSED_LOOP_LOG_JSON, summary)

    fieldnames = [
        "cycle_id",
        "plan_id",
        "selected_raw_class",
        "selected_category_cn",
        "target_bin_cn",
        "isaac_action_status",
        "verification_status",
        "vote_score",
        "before_total",
        "after_total",
        "collect_sec",
        "yolo_total_sec",
        "isaac_wait_sec",
    ]

    with open(CLOSED_LOOP_LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            verification = record.get("next_cycle_verification") or {}
            selected_task = record.get("selected_task") or {}
            isaac_result = record.get("isaac_result") or {}
            timing = record.get("timing", {})

            writer.writerow({
                "cycle_id": record.get("cycle_id"),
                "plan_id": record.get("plan_id"),
                "selected_raw_class": selected_task.get("raw_class_name"),
                "selected_category_cn": selected_task.get("garbage_category_cn"),
                "target_bin_cn": selected_task.get("target_bin_cn"),
                "isaac_action_status": isaac_result.get("action_status"),
                "verification_status": verification.get("verified"),
                "vote_score": verification.get("vote_score"),
                "before_total": verification.get("before_total"),
                "after_total": verification.get("after_total"),
                "collect_sec": timing.get("collect_sec"),
                "yolo_total_sec": timing.get("yolo_total_sec"),
                "isaac_wait_sec": timing.get("isaac_wait_sec"),
            })

    print(f"[SAVED] {CLOSED_LOOP_LOG_JSON}")
    print(f"[SAVED] {CLOSED_LOOP_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--collect-timeout-sec", type=int, default=6)
    parser.add_argument("--isaac-timeout-sec", type=int, default=180)
    parser.add_argument("--save-overlay-every", type=int, default=0)
    parser.add_argument("--settle-sec", type=float, default=0.4)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.parent.mkdir(parents=True, exist_ok=True)

    records = []
    pending_previous_result = None

    print("[START] closed_loop_wsl_controller_fast.py")
    print("[INFO] Isaac Sim 里要先运行 closed_loop_isaac_executor.py")
    print("[INFO] YOLO 模型只会加载一次，第一轮较慢，后续会明显变快。")

    yolo_client = PersistentYoloClient(conf=args.conf)

    try:
        for cycle_id in range(1, args.max_cycles + 1):
            print("\n" + "=" * 80)
            print(f"[CYCLE {cycle_id}] fast collect -> persistent yolo -> plan -> isaac -> verify next")

            cycle_start = time.time()

            image_path, collect_sec = collect_one_image_fast(
                timeout_sec=args.collect_timeout_sec,
            )

            save_overlay = (
                args.save_overlay_every > 0
                and cycle_id % args.save_overlay_every == 0
            )

            request_id = f"cycle_{cycle_id:03d}_{uuid.uuid4().hex[:8]}"

            yolo_start = time.time()

            yolo_data = yolo_client.infer(
                image_path=image_path,
                request_id=request_id,
                save_overlay=save_overlay,
                timeout_sec=120,
            )

            yolo_total_sec = time.time() - yolo_start

            print(f"[YOLO] num_detections={yolo_data.get('num_detections')}")
            print(f"[TIMING] yolo_total_sec={yolo_total_sec:.4f}")

            tasks = build_tasks_from_yolo(yolo_data)
            current_counts = count_detections(tasks)

            previous_verification = verify_previous_action(
                pending_previous_result,
                current_counts,
            )

            if records and previous_verification is not None:
                records[-1]["next_cycle_verification"] = previous_verification
                print("[VERIFY PREVIOUS]")
                print(json.dumps(previous_verification, ensure_ascii=False, indent=2))

            if not tasks:
                print("[STOP] 当前画面没有可规划垃圾，闭环结束。")
                break

            plan = build_plan(
                cycle_id=cycle_id,
                yolo_data=yolo_data,
                tasks=tasks,
                previous_verification=previous_verification,
            )

            selected_task = plan.get("selected_task")

            print("[SELECTED TASK]")
            print(json.dumps(selected_task, ensure_ascii=False, indent=2))

            atomic_write_json(PLAN_JSON, plan)
            print(f"[SAVED PLAN] {PLAN_JSON}")

            isaac_result, isaac_wait_sec = wait_for_isaac_result(
                plan_id=plan["plan_id"],
                timeout_sec=args.isaac_timeout_sec,
            )

            time.sleep(args.settle_sec)

            record = {
                "cycle_id": cycle_id,
                "plan_id": plan["plan_id"],
                "source_image": plan.get("source_image"),
                "num_detections_before_action": plan.get("num_detections"),
                "before_counts": plan.get("before_counts"),
                "selected_task": selected_task,
                "isaac_result": isaac_result,
                "next_cycle_verification": None,
                "timing": {
                    "collect_sec": round(collect_sec, 4),
                    "yolo_total_sec": round(yolo_total_sec, 4),
                    "isaac_wait_sec": round(isaac_wait_sec, 4),
                    "cycle_sec": round(time.time() - cycle_start, 4),
                },
            }

            records.append(record)
            pending_previous_result = isaac_result

            save_closed_loop_logs(records)

        if pending_previous_result is not None and records:
            print("\n" + "=" * 80)
            print("[FINAL VERIFY] 再采一帧验证最后一次动作")

            image_path, collect_sec = collect_one_image_fast(
                timeout_sec=args.collect_timeout_sec,
            )

            request_id = f"final_verify_{uuid.uuid4().hex[:8]}"

            yolo_start = time.time()

            yolo_data = yolo_client.infer(
                image_path=image_path,
                request_id=request_id,
                save_overlay=False,
                timeout_sec=120,
            )

            yolo_total_sec = time.time() - yolo_start

            tasks = build_tasks_from_yolo(yolo_data)
            current_counts = count_detections(tasks)

            final_verification = verify_previous_action(
                pending_previous_result,
                current_counts,
            )

            records[-1]["next_cycle_verification"] = final_verification

            print("[FINAL VERIFY RESULT]")
            print(json.dumps(final_verification, ensure_ascii=False, indent=2))

            save_closed_loop_logs(records)

        print("[DONE] fast closed-loop run finished")
        print(json.dumps(compute_closed_loop_metrics(records), ensure_ascii=False, indent=2))

    finally:
        yolo_client.stop()


if __name__ == "__main__":
    main()