import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path.home() / "trashbot_ws"
LOG_DIR = ROOT / "data" / "logs"
IMAGE_DIR = ROOT / "data" / "images"

global_yolo_proc = None
global_qwen_proc = None
global_pending_verification = None


YOLO_PYTHON = Path.home() / "envs" / "yolo" / "bin" / "python"
YOLO_WORKER_SCRIPT = ROOT / "scripts" / "perception" / "v27_yoloe_persistent_worker.py"
YOLO_REQUEST_JSON = LOG_DIR / "v27_yolo_worker_request.json"
YOLO_RESPONSE_JSON = LOG_DIR / "v27_yolo_worker_response.json"
YOLO_READY_JSON = LOG_DIR / "v27_yolo_worker_ready.json"

QWEN_WORKER_SCRIPT = ROOT / "scripts" / "perception" / "v27_qwen_perception_worker.py"
QWEN_WORKER_REQUEST_JSON  = LOG_DIR / "v27_qwen_worker_request.json"
QWEN_WORKER_RESPONSE_JSON = LOG_DIR / "v27_qwen_worker_response.json"
QWEN_WORKER_READY_JSON    = LOG_DIR / "v27_qwen_worker_ready.json"


VIEW_COMMAND_JSON = Path("/mnt/d/isaac_projects/v26_view_command.json")
VIEW_RESULT_JSON = Path("/mnt/d/isaac_projects/v26_view_result.json")
PLAN_JSON = Path("/mnt/d/isaac_projects/v2_visual_task_plan.json")
ACTION_RESULT_JSON = Path("/mnt/d/isaac_projects/v27_head_camera_action_result.json")

RUN_LOG_JSON = LOG_DIR / "v27_head_camera_all_trash_fast_run_log.json"
RUN_LOG_CSV = LOG_DIR / "v27_head_camera_all_trash_fast_run_table.csv"


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


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for i in range(50):
        try:
            tmp.replace(path)
            return
        except (PermissionError, OSError) as e:
            if i == 49:
                print(f"[ERROR] Failed to save {path} after 50 retries: {repr(e)}")
                raise
            time.sleep(0.02)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_safe(path: Path):
    try:
        if not path.exists():
            return None
        return load_json(path)
    except Exception:
        return None


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


def clean_ipc_files():
    for p in [
        VIEW_COMMAND_JSON,
        VIEW_RESULT_JSON,
        PLAN_JSON,
        ACTION_RESULT_JSON,
        YOLO_REQUEST_JSON,
        YOLO_RESPONSE_JSON,
        YOLO_READY_JSON,
    ]:
        try:
            if p.exists():
                p.unlink()
                print(f"[DELETE] {p}")
        except Exception as e:
            print(f"[WARN] failed delete {p}: {repr(e)}")


def wait_json_field(path: Path, field: str, value, timeout_sec: float):
    start = time.time()
    while time.time() - start < timeout_sec:
        data = load_json_safe(path)
        if data is not None and data.get(field) == value:
            return data
        time.sleep(0.03)
    raise TimeoutError(f"Timeout waiting {path} where {field}={value}")


def start_yolo_worker(args, backend_override=None):
    """启动 YOLOe 常驻 worker（original 模式），或 Qwen Perception 常驻 worker（qwen_first/yolo_first）。"""
    global global_qwen_proc
    backend = backend_override or getattr(args, "perception_backend", "original")

    if backend in ["qwen_first", "yolo_first", "yolo11_qwen"]:
        # 启动 qwen perception 常驻 worker
        for p in [QWEN_WORKER_REQUEST_JSON, QWEN_WORKER_RESPONSE_JSON, QWEN_WORKER_READY_JSON]:
            if p.exists():
                p.unlink()

        model_path = getattr(args, "qwen_first_model", None) or getattr(args, "model", None)
        if not model_path:
            raise ValueError("--qwen-first-model is required for qwen_first backend")

        cmd = [
            str(YOLO_PYTHON),
            str(QWEN_WORKER_SCRIPT),
            "--model", str(model_path),
            "--request-json",  str(QWEN_WORKER_REQUEST_JSON),
            "--response-json", str(QWEN_WORKER_RESPONSE_JSON),
            "--ready-json",    str(QWEN_WORKER_READY_JSON),
        ]

        print("[START QWEN PERCEPTION WORKER]")
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd)
        global_qwen_proc = proc

        timeout = getattr(args, "yolo_ready_timeout_sec", 60)
        start = time.time()
        while time.time() - start < timeout:
            ready = load_json_safe(QWEN_WORKER_READY_JSON)
            if ready and ready.get("status") == "ready":
                print(f"[QWEN WORKER READY] loaded in {ready.get('model_load_sec', '?')}s")
                return proc  # proc 由调用方保存到 global_yolo_proc
            if proc.poll() is not None:
                raise RuntimeError(f"Qwen worker exited early: code={proc.returncode}")
            time.sleep(0.2)

        raise TimeoutError("Timeout waiting for Qwen perception worker ready.")

    # --- original: YOLOe 常驻 worker ---
    if not YOLO_PYTHON.exists():
        raise FileNotFoundError(f"YOLO python not found: {YOLO_PYTHON}")

    for p in [YOLO_REQUEST_JSON, YOLO_RESPONSE_JSON, YOLO_READY_JSON]:
        if p.exists():
            p.unlink()

    cmd = [
        str(YOLO_PYTHON),
        str(YOLO_WORKER_SCRIPT),
        "--request-json", str(YOLO_REQUEST_JSON),
        "--response-json", str(YOLO_RESPONSE_JSON),
        "--ready-json", str(YOLO_READY_JSON),
        "--model", str(args.model),
        "--imgsz", str(args.imgsz),
        "--device", args.yolo_device,
        "--max-det", str(args.max_det),
        "--max-area-ratio", str(args.max_area_ratio),
    ]

    print("[START YOLOE WORKER]")
    print(" ".join(cmd))

    proc = subprocess.Popen(cmd)

    start = time.time()
    while time.time() - start < args.yolo_ready_timeout_sec:
        ready = load_json_safe(YOLO_READY_JSON)
        if ready is not None and ready.get("status") == "ready":
            print("[YOLOE READY]")
            print(json.dumps(ready, ensure_ascii=False, indent=2))
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"YOLO worker exited early: code={proc.returncode}")
        time.sleep(0.1)

    raise TimeoutError("Timeout waiting YOLO worker ready.")



def stop_yolo_worker(proc):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    # 尝试通过 shutdown 命令优雅退出（两种 worker 都支持）
    for rj in [YOLO_REQUEST_JSON, QWEN_WORKER_REQUEST_JSON]:
        try:
            save_json_atomic(rj, {
                "request_id": f"shutdown_{uuid.uuid4().hex[:8]}",
                "command": "shutdown",
                "timestamp": time.time(),
            })
        except Exception:
            pass

    try:
        proc.wait(timeout=5)
        print("[WORKER STOPPED]")
    except Exception:
        proc.terminate()
        print("[WORKER TERMINATED]")



def run_yolo_fast(image_path: Path, conf: float, args, save_overlay=False, backend=None):
    if backend is None:
        backend = getattr(args, "perception_backend", "original")
    if backend in ["yolo11_qwen", "qwen_first", "yolo_first"]:
        t0 = time.time()
        use_conf = getattr(args, "qwen_first_conf", conf)
        pipeline = "qwen_first" if backend == "qwen_first" else "yolo_first"

        # --- 优先走常驻 worker（YOLO 不重复加载）---
        worker_ready = load_json_safe(QWEN_WORKER_READY_JSON)
        if worker_ready and worker_ready.get("status") == "ready":
            request_id = f"v27_qwen_{uuid.uuid4().hex[:8]}"
            req = {
                "request_id": request_id,
                "command": "infer",
                "image_path": str(image_path),
                "pipeline": pipeline,
                "conf": float(use_conf),
                "roi_expand": float(getattr(args, "qwen_first_roi_expand", 2.0)),
                "verify_mode": str(getattr(args, "qwen_first_verify_mode", "top1")),
                "max_qwen_candidates": int(getattr(args, "qwen_first_max_candidates", 5)),
                "max_roi_refine": int(getattr(args, "qwen_first_max_roi_refine", 3)),
                "vis_mode": "planner",
                "save_vis": bool(save_overlay or getattr(args, "qwen_first_save_vis", False)),
                "qwen_verify_workers": int(getattr(args, "qwen_verify_workers", 4)),
                "min_area_ratio": float(getattr(args, "min_area_ratio", 0.0005)),
                "max_area_ratio": float(getattr(args, "max_area_ratio", 0.60)),
                "timestamp": time.time(),
            }
            print(f"[QWEN WORKER IPC] request_id={request_id} image={image_path.name}")
            t_ipc_sent = time.time()
            save_json_atomic(QWEN_WORKER_REQUEST_JSON, req)

            timeout = getattr(args, "qwen_first_timeout", 120)
            response = wait_json_field(
                QWEN_WORKER_RESPONSE_JSON,
                "request_id",
                request_id,
                timeout_sec=timeout,
            )
            t_ipc_done = time.time()
            worker_pickup_latency = response.get("timestamp", t_ipc_done) - t_ipc_sent
            inference_sec = response.get("elapsed_sec", t_ipc_done - t_ipc_sent)
            print(f"[QWEN WORKER IPC] request_id={request_id} done "
                  f"total={t_ipc_done-t_ipc_sent:.2f}s "
                  f"pickup_latency={worker_pickup_latency:.2f}s "
                  f"inference={inference_sec:.2f}s")

            if response.get("status") != "success":
                raise RuntimeError(f"Qwen worker error: {response.get('error', response)}")

            result_data = response["result"]
            if "inference_sec" not in result_data:
                result_data["inference_sec"] = round(time.time() - t0, 4)
            return result_data

        # --- fallback: subprocess（worker 未启动时兼容旧行为）---
        print("[QWEN WORKER] not ready, falling back to subprocess")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "perception" / "yolo11_qwen_perception_offline.py"),
            "--pipeline", pipeline,
            "--image", str(image_path),
            "--model", str(getattr(args, "qwen_first_model", args.model)),
            "--conf", str(use_conf),
            "--roi-expand", str(getattr(args, "qwen_first_roi_expand", 2.0)),
            "--verify-mode", str(getattr(args, "qwen_first_verify_mode", "top1")),
            "--max-qwen-candidates", str(getattr(args, "qwen_first_max_candidates", 5)),
            "--max-roi-refine", str(getattr(args, "qwen_first_max_roi_refine", 3)),
            "--vis-mode", "planner",
            "--qwen-verify-workers", str(getattr(args, "qwen_verify_workers", 4)),
        ]
        if save_overlay or getattr(args, "qwen_first_save_vis", False):
            cmd.append("--save-vis")

        print(f"[QWEN FIRST CMD] {' '.join(cmd)}")
        result_code = subprocess.run(cmd)
        if result_code.returncode != 0:
            raise RuntimeError(f"yolo11_qwen_perception_offline.py failed: exit {result_code.returncode}")

        result_json_path = ROOT / "data" / "logs" / "yolo_seg_offline_result.json"
        if not result_json_path.exists():
            raise FileNotFoundError(f"Perception result JSON not found: {result_json_path}")
        with open(result_json_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        if "inference_sec" not in result_data:
            result_data["inference_sec"] = round(time.time() - t0, 4)
        return result_data

    request_id = f"v27_yolo_{uuid.uuid4().hex[:8]}"

    save_json_atomic(YOLO_REQUEST_JSON, {
        "request_id": request_id,
        "command": "infer",
        "image_path": str(image_path),
        "conf": float(conf),
        "imgsz": int(args.imgsz),
        "max_det": int(args.max_det),
        "max_area_ratio": float(args.max_area_ratio),
        "save_overlay": bool(save_overlay),
        "timestamp": time.time(),
    })

    response = wait_json_field(
        YOLO_RESPONSE_JSON,
        "request_id",
        request_id,
        timeout_sec=args.yolo_infer_timeout_sec,
    )

    if response.get("status") != "success":
        raise RuntimeError(json.dumps(response, ensure_ascii=False, indent=2))

    return response["result"]


def latest_fast_image(after_time=None):
    candidates = list(IMAGE_DIR.glob("*_fast/frame_000001.jpg"))
    if after_time is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= after_time - 1.0]

    if not candidates:
        candidates = list(IMAGE_DIR.glob("*_fast/frame_000001.jpg"))

    if not candidates:
        raise FileNotFoundError("No *_fast/frame_000001.jpg found.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_image(timeout_sec):
    """
    Robust ROS image collection.

    First try with the user-provided timeout. If Isaac/ROS is temporarily slow,
    retry once with a longer timeout instead of crashing the whole V2.7 loop.
    """
    t0 = time.time()

    timeouts = [float(timeout_sec), max(6.0, float(timeout_sec) * 3.0)]

    last_error = None

    for idx, one_timeout in enumerate(timeouts, start=1):
        cmd = (
            "source ~/use_ros2_isaac.sh && "
            "python3 ~/trashbot_ws/scripts/tools/collect_one_ros_image_fast.py "
            "--topic /trashbot/camera/rgb "
            f"--timeout {one_timeout}"
        )

        try:
            print(f"[COLLECT TRY {idx}] timeout={one_timeout}")
            run_bash(cmd, timeout_sec=int(one_timeout) + 10)
            img = latest_fast_image(after_time=t0)
            print(f"[IMAGE] {img}")
            return img

        except Exception as e:
            last_error = e
            print(f"[COLLECT WARN] failed on try {idx}: {repr(e)}")
            time.sleep(0.5)

    raise RuntimeError(f"failed to collect image after retries: {repr(last_error)}")


def write_view_command(view, args):
    command_id = f"v27_view_{uuid.uuid4().hex[:8]}"

    command = {
        "mode": "V2_7_VIEW_COMMAND",
        "command_id": command_id,
        "view_id": view["view_id"],
        "robot_xyz": view["robot_xyz"],
        "look_at_xyz": view["look_at_xyz"],
        "head_offset_xyz": [0.0, 0.0, 1.55],
        "move_frames": int(args.move_frames),
        "settle_frames": int(args.settle_frames),
        "created_timestamp": time.time(),
    }

    save_json_atomic(VIEW_COMMAND_JSON, command)
    print(f"[VIEW COMMAND] {command_id} {view['view_id']}")
    return command


def wait_view_result(command_id, timeout_sec):
    print(f"[WAIT VIEW RESULT] {command_id}")
    result = wait_json_field(VIEW_RESULT_JSON, "command_id", command_id, timeout_sec)
    print(f"[VIEW RESULT] status={result.get('status')}")
    return result


def bbox_area_ratio(det, image_width=480, image_height=320):
    bbox = det.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(image_width * image_height))


def evaluate_detections(yolo_result, min_conf, min_area_ratio, blocked_raw):
    image_width = int(yolo_result.get("image_width", 480))
    image_height = int(yolo_result.get("image_height", 320))

    candidates = []

    for det in yolo_result.get("detections", []):
        raw = det.get("raw_class_name", det.get("class_name", "unknown"))
        conf = float(det.get("confidence", 0.0))
        target_bin = det.get("target_bin", "unknown")

        if raw in blocked_raw:
            continue
        if conf < min_conf:
            continue
        if target_bin == "unknown":
            continue

        area_ratio = bbox_area_ratio(det, image_width, image_height)
        if area_ratio < min_area_ratio:
            continue

        score = conf + min(area_ratio * 20.0, 1.0)

        candidates.append({
            "raw_class_name": raw,
            "category": det.get("category", "unknown"),
            "target_bin": target_bin,
            "confidence": round(conf, 4),
            "area_ratio": round(area_ratio, 6),
            "score": round(score, 4),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "centroid_px": det.get("centroid_px"),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    return {
        "num_detections": len(yolo_result.get("detections", [])),
        "num_candidates": len(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
    }


def count_detections(yolo_result):
    out = {
        "total": 0,
        "by_raw_class": {},
        "by_category": {},
    }

    for det in yolo_result.get("detections", []):
        raw = det.get("raw_class_name", det.get("class_name", "unknown"))
        category = det.get("category", "unknown")

        out["total"] += 1
        out["by_raw_class"][raw] = out["by_raw_class"].get(raw, 0) + 1
        out["by_category"][category] = out["by_category"].get(category, 0) + 1

    return out


def make_plan(args, blocked_raw, selected_view, cycle_id):
    blocked_arg = ",".join(sorted(blocked_raw))

    cmd = (
        "python3 ~/trashbot_ws/scripts/control/v2_make_visual_task_plan.py "
        "--select confidence "
        f"--profile {args.profile} "
        f"--min-conf {args.conf}"
    )

    if blocked_arg:
        cmd += f" --blocked-raw {blocked_arg}"
    else:
        # Explicitly override planner's default blocked list.
        cmd += " --blocked-raw ''"

    run_bash(cmd, timeout_sec=30)

    plan = load_json(PLAN_JSON)

    if plan.get("selected_task") is not None:
        plan["cycle_id"] = cycle_id
        plan["mode"] = "V2_7_HEAD_CAMERA_ALL_TRASH_FAST_PLAN"
        plan["v26_selected_view"] = selected_view
        save_json_atomic(PLAN_JSON, plan)

    return plan


def build_planner_no_task_record(cycle_id, cycle_t0, image_path, before_yolo, plan, perception_meta):
    """生成 planner_no_task 的 cycle record，用于 planner_ready=True 但 num_tasks=0 的情况。"""
    rejected = plan.get("rejected_detections", [])
    rejected_reasons = list({r.get("reason", "unknown") for r in rejected})

    record = {
        "cycle_id": cycle_id,
        "plan_id": plan.get("plan_id"),
        "selected_view": None,
        "view_search_records": [],
        "selected_task": None,
        "before_image": str(image_path) if image_path else None,
        "after_image": None,
        "before_yolo_inference_sec": before_yolo.get("inference_sec") if before_yolo else 0.0,
        "after_yolo_inference_sec": None,
        "action_result": {
            "status": "planner_no_task",
            "raw_class_name": "none",
            "garbage_category": "none",
        },
        "verification": {
            "verified": False,
            "vote_score": 0,
            "reason": "planner_filtered_all_detections",
        },
        "planner_num_tasks": plan.get("num_tasks", 0),
        "planner_num_rejected": plan.get("num_rejected", 0),
        "planner_rejected_reasons": rejected_reasons,
        "planner_qwen_first_compatible": plan.get("qwen_first_compatible", False),
        "cycle_sec": round(time.time() - cycle_t0, 4),
    }
    record.update(perception_meta)
    return record


def wait_action_result(plan_id, timeout_sec):
    print(f"[WAIT ACTION RESULT] {plan_id}")
    result = wait_json_field(ACTION_RESULT_JSON, "plan_id", plan_id, timeout_sec)
    print(f"[ACTION RESULT] status={result.get('status')}")
    return result


def verify_action(before_yolo, action_result, after_yolo, perception_backend="original"):
    """
    验证单次抳取动作是否成功。

    qwen_first 模式：
      - 主投票： action_result.status=="success" && action_completed==true && object_hidden_after_drop==true
        三条同时成立↛ verified=True，reason="action_result_hidden_after_drop"
      - 辅助投票（不改变主投票结果）：
          vote_total: after_total < before_total
          vote_category: after garbage_category 数量 < before（不再用 raw_class_name）
      - vote_raw 始终为 False（trash_object 在 after YOLO 中同样不可靠）

    original 模式：保持旧行为（raw_class / category / total 三票）。
    """
    raw = action_result.get("raw_class_name", "unknown")
    category = action_result.get("garbage_category", "unknown")

    before = count_detections(before_yolo)
    after = count_detections(after_yolo)

    is_qwen_first = (
        perception_backend in ("qwen_first", "yolo_first")
        or raw == "trash_object"
    )

    if is_qwen_first:
        # 主投票：Isaac 实际执行并隐藏了物体
        action_hidden = (
            action_result.get("status") == "success"
            and action_result.get("action_completed") is True
            and action_result.get("object_hidden_after_drop") is True
        )

        # 辅助投票：不改变主投票结果，仅供记录
        vote_total = after["total"] < before["total"]

        # qwen_first 用 garbage_category 计数（before/after U perception 都用该字段）
        # count_detections 的 by_category key 取自 det["category"]，qwen_first JSON 写的是 "category": "trash_object"
        # 所以这里用 garbage_category 直接匹配 by_category（两者可能不一致）也没有问题：
        # Isaac 刷新后 before/after 都用同一个 perception backend，所以笔数减少就是正向证据
        vote_category = (
            after["by_category"].get(category, 0)
            < before["by_category"].get(category, 0)
        )

        aux_score = int(vote_total) + int(vote_category)

        if action_hidden:
            verified = True
            reason = "action_result_hidden_after_drop"
        else:
            # 主投票未通过，依靠辅助投票（需刖2票）
            verified = aux_score >= 2
            reason = "aux_votes_only"

        return {
            "verified": verified,
            "reason": reason,
            "qwen_first_mode": True,
            "action_hidden_primary_vote": action_hidden,
            "aux_vote_score": aux_score,
            "vote_total_decreased": vote_total,
            "vote_category_decreased": vote_category,
            "vote_raw_class_decreased": False,  # 不再统计 trash_object
            "selected_raw_class": raw,
            "selected_category": category,
            "before_total": before["total"],
            "after_total": after["total"],
            "before_category_count": before["by_category"].get(category, 0),
            "after_category_count": after["by_category"].get(category, 0),
        }

    # --- 原有模式（original / YOLO 多类别）---
    vote_total = after["total"] < before["total"]
    vote_raw = after["by_raw_class"].get(raw, 0) < before["by_raw_class"].get(raw, 0)
    vote_category = after["by_category"].get(category, 0) < before["by_category"].get(category, 0)

    vote_score = int(vote_total) + int(vote_raw) + int(vote_category)

    return {
        "verified": vote_score >= 2,
        "vote_score": vote_score,
        "selected_raw_class": raw,
        "selected_category": category,
        "before_total": before["total"],
        "after_total": after["total"],
        "before_raw_class_count": before["by_raw_class"].get(raw, 0),
        "after_raw_class_count": after["by_raw_class"].get(raw, 0),
        "before_category_count": before["by_category"].get(category, 0),
        "after_category_count": after["by_category"].get(category, 0),
        "vote_total_decreased": vote_total,
        "vote_raw_class_decreased": vote_raw,
        "vote_category_decreased": vote_category,
    }


def resolve_pending_verification(current_view_id, yolo_result, records):
    global global_pending_verification
    if global_pending_verification is None:
        return

    # Check if the view is the same
    target_cycle_idx = global_pending_verification["cycle_index_to_update"]
    if target_cycle_idx >= len(records):
        # Safety check
        global_pending_verification = None
        return

    before_yolo = global_pending_verification["before_yolo"]
    action_result = global_pending_verification["action_result"]
    prev_view_id = global_pending_verification["view_id"]
    backend = global_pending_verification["perception_backend"]

    # If the view is the same and we have a valid yolo_result, do full verification
    if current_view_id is not None and current_view_id == prev_view_id and yolo_result is not None:
        verification = verify_action(
            before_yolo,
            action_result,
            yolo_result,
            perception_backend=backend
        )
        print(f"[VERIFY DEFERRED] Resolved verification for cycle index {target_cycle_idx} at view {current_view_id}: verified={verification.get('verified')}")
    else:
        # If the view changed or loop ended, verify using action_hidden only (fallback verification)
        action_hidden = (
            action_result.get("status") == "success"
            and action_result.get("action_completed") is True
            and action_result.get("object_hidden_after_drop") is True
        )
        if action_hidden:
            verified = True
            reason = "action_result_hidden_after_drop_view_changed_or_ended"
        else:
            verified = False
            reason = "action_failed_view_changed_or_ended"

        verification = {
            "verified": verified,
            "reason": reason,
            "qwen_first_mode": True,
            "action_hidden_primary_vote": action_hidden,
            "aux_vote_score": 0,
            "vote_total_decreased": False,
            "vote_category_decreased": False,
            "vote_raw_class_decreased": False,
            "selected_raw_class": action_result.get("raw_class_name", "unknown"),
            "selected_category": action_result.get("garbage_category", "unknown"),
        }
        print(f"[VERIFY DEFERRED] Fallback verification for cycle index {target_cycle_idx} due to view change/end (prev={prev_view_id}, current={current_view_id}): verified={verified}")

    # Update the record in records list
    records[target_cycle_idx]["verification"] = verification
    # Save the updated logs
    save_logs(records)

    # Clear pending
    global_pending_verification = None


def run_qwen_first_perception(image_path: Path, args, pipeline: str = "qwen_first"):
    """
    Run qwen_first / yolo_first perception using persistent worker, falling back to subprocess.
    Returns (success: bool, result_data: dict or None, reason: str, elapsed_sec: float)
    """
    t0 = time.time()

    # --- 优先走常驻 worker（YOLO 不重复加载）---
    worker_ready = load_json_safe(QWEN_WORKER_READY_JSON)
    if worker_ready and worker_ready.get("status") == "ready":
        request_id = f"v27_qwen_{uuid.uuid4().hex[:8]}"
        req = {
            "request_id": request_id,
            "command": "infer",
            "image_path": str(image_path),
            "pipeline": pipeline,
            "conf": float(args.qwen_first_conf),
            "roi_expand": float(args.qwen_first_roi_expand),
            "verify_mode": str(args.qwen_first_verify_mode),
            "max_qwen_candidates": int(args.qwen_first_max_candidates),
            "max_roi_refine": int(args.qwen_first_max_roi_refine),
            "vis_mode": "planner",
            "save_vis": bool(args.qwen_first_save_vis),
            "qwen_verify_workers": int(getattr(args, "qwen_verify_workers", 4)),
            "min_area_ratio": float(getattr(args, "min_area_ratio", 0.0005)),
            "max_area_ratio": float(getattr(args, "max_area_ratio", 0.60)),
            "timestamp": time.time(),
        }
        print(f"[QWEN WORKER IPC] request_id={request_id} image={image_path.name}")
        t_ipc_sent = time.time()
        
        try:
            if QWEN_WORKER_RESPONSE_JSON.exists():
                QWEN_WORKER_RESPONSE_JSON.unlink()

            save_json_atomic(QWEN_WORKER_REQUEST_JSON, req)

            timeout = getattr(args, "qwen_first_timeout", 90)
            response = wait_json_field(
                QWEN_WORKER_RESPONSE_JSON,
                "request_id",
                request_id,
                timeout_sec=timeout,
            )
            t_ipc_done = time.time()
            worker_pickup_latency = response.get("timestamp", t_ipc_done) - t_ipc_sent
            inference_sec = response.get("elapsed_sec", t_ipc_done - t_ipc_sent)
            print(f"[QWEN WORKER IPC] request_id={request_id} done "
                  f"total={t_ipc_done-t_ipc_sent:.2f}s "
                  f"pickup_latency={worker_pickup_latency:.2f}s "
                  f"inference={inference_sec:.2f}s")

            if response.get("status") == "success":
                result_data = response["result"]
                elapsed = time.time() - t0
                if "inference_sec" not in result_data:
                    result_data["inference_sec"] = elapsed
                return True, result_data, "success", elapsed
            else:
                print(f"[QWEN WORKER ERROR] Worker failed: {response.get('error')}. Falling back to subprocess...")
        except Exception as e:
            print(f"[QWEN WORKER EXCEPTION] IPC error: {repr(e)}. Falling back to subprocess...")

    # --- Subprocess fallback ---
    print("[QWEN WORKER] not ready or failed, falling back to subprocess")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "perception" / "yolo11_qwen_perception_offline.py"),
        "--pipeline", pipeline,
        "--image", str(image_path),
        "--model", str(args.qwen_first_model),
        "--conf", str(args.qwen_first_conf),
        "--roi-expand", str(args.qwen_first_roi_expand),
        "--verify-mode", str(args.qwen_first_verify_mode),
        "--max-qwen-candidates", str(args.qwen_first_max_candidates),
        "--max-roi-refine", str(args.qwen_first_max_roi_refine),
        "--vis-mode", "planner",
        "--qwen-verify-workers", str(getattr(args, "qwen_verify_workers", 4)),
    ]
    if args.qwen_first_save_vis:
        cmd.append("--save-vis")

    print(f"[QWEN FIRST CMD] {' '.join(cmd)}")

    proc = None
    stdout, stderr = "", ""
    returncode = None
    elapsed = 0.0
    reason = "success"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(timeout=args.qwen_first_timeout)
        returncode = proc.returncode
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[QWEN FIRST TIMEOUT] Exceeded {args.qwen_first_timeout}s")
        if proc:
            proc.kill()
            try:
                stdout, stderr = proc.communicate()
            except Exception:
                pass
        return False, None, "qwen_first_timeout", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return False, None, f"qwen_first_exception: {repr(e)}", elapsed

    stdout_lines = stdout.splitlines() if stdout else []
    stderr_lines = stderr.splitlines() if stderr else []

    print("="*40 + " Qwen First stdout (last 80 lines) " + "="*40)
    for line in stdout_lines[-80:]:
        print(line)
    print("="*40 + " Qwen First stderr (last 80 lines) " + "="*40)
    for line in stderr_lines[-80:]:
        print(line)
    print("="*100)

    print(f"[QWEN FIRST SUBPROCESS DONE] returncode={returncode}, elapsed={elapsed:.2f}s")

    if returncode != 0:
        return False, None, "qwen_first_subprocess_failed", elapsed

    result_json_path = ROOT / "data" / "logs" / "yolo_seg_offline_result.json"
    if not result_json_path.exists():
        return False, None, "qwen_first_output_json_missing", elapsed

    try:
        with open(result_json_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        return True, result_data, "success", elapsed
    except Exception as e:
        return False, None, f"qwen_first_json_parse_error: {repr(e)}", elapsed


def validate_planner_ready(result_data: dict) -> bool:
    if not result_data:
        return False
    if not result_data.get("planner_ready"):
        return False
    detections = result_data.get("detections", [])
    if not detections:
        return False
    if result_data.get("num_detections", 0) <= 0:
        return False

    required_keys = [
        "raw_class_name", "garbage_category", "target_bin", "confidence",
        "bbox_xyxy", "centroid_px", "bottom_contact_px", "bbox_area_ratio", "polygon"
    ]

    for det in detections:
        if det.get("approach_required") is True:
            return False
        for key in required_keys:
            if det.get(key) is None:
                return False

    return True


def get_perception_result(image_path: Path, args):
    """
    Get perception result for the given image path, implementing backend choice,
    subprocess execution, validation, logging, and fallback strategies.
    Returns (result_data, meta_info: dict)
    """
    backend = args.perception_backend

    meta = {
        "perception_backend": backend,
        "qwen_first_cmd": "none",
        "qwen_first_elapsed_sec": 0.0,
        "qwen_first_returncode": 0,
        "qwen_first_planner_ready": False,
        "qwen_first_num_detections": 0,
        "qwen_first_num_approach_candidates": 0,
        "qwen_first_num_rejected": 0,
        "qwen_first_verify_mode": args.qwen_first_verify_mode,
        "qwen_first_timing": {},
        "qwen_first_fallback_used": False,
        "qwen_first_fallback_reason": "none"
    }

    if backend not in ["qwen_first", "yolo_first"]:
        t0 = time.time()
        res = run_yolo_fast(image_path, args.conf, args, args.save_overlay)
        res_dets = res.get("detections", [])
        meta.update({
            "qwen_first_planner_ready": True,
            "qwen_first_num_detections": len(res_dets),
            "qwen_first_elapsed_sec": time.time() - t0,
        })
        return res, meta

    pipeline = "qwen_first" if backend == "qwen_first" else "yolo_first"

    success, res_data, reason, elapsed = run_qwen_first_perception(image_path, args, pipeline)

    meta["qwen_first_elapsed_sec"] = elapsed
    meta["qwen_first_cmd"] = f"pipeline={pipeline} conf={args.qwen_first_conf} model={Path(args.qwen_first_model).name}"

    if reason == "qwen_first_timeout":
        meta["qwen_first_returncode"] = -9
    elif reason == "qwen_first_subprocess_failed":
        meta["qwen_first_returncode"] = 1
    elif success:
        meta["qwen_first_returncode"] = 0

    is_ready = False
    if success and res_data:
        is_ready = validate_planner_ready(res_data)
        meta["qwen_first_planner_ready"] = is_ready
        meta["qwen_first_num_detections"] = res_data.get("num_detections", 0)
        meta["qwen_first_num_approach_candidates"] = res_data.get("num_approach_candidates", 0)
        meta["qwen_first_num_rejected"] = res_data.get("num_rejected", 0)
        meta["qwen_first_timing"] = {
            "qwen_coarse_sec": res_data.get("qwen_coarse_sec", 0.0),
            "yolo_roi_total_sec": res_data.get("yolo_roi_total_sec", 0.0),
            "qwen_verify_total_sec": res_data.get("qwen_verify_total_sec", 0.0),
            "total_sec": res_data.get("total_sec", 0.0),
            "yolo_roi_calls": res_data.get("yolo_roi_calls", 0),
            "qwen_verify_calls": res_data.get("num_qwen_verify_calls", 0),
            "qwen_verify_skipped": res_data.get("num_qwen_verify_skipped", 0),
        }

    if is_ready:
        return res_data, meta

    fallback = args.qwen_first_fallback
    meta["qwen_first_fallback_used"] = True
    meta["qwen_first_fallback_reason"] = reason if not success else "planner_ready_false"

    print(f"[PERCEPTION WARNING] Primary backend {backend} failed or planner_ready is False (reason: {meta['qwen_first_fallback_reason']}). Executing fallback: {fallback}")

    if fallback == "none":
        empty_res = {
            "detections": [],
            "approach_candidates": res_data.get("approach_candidates", []) if res_data else [],
            "rejected_detections": res_data.get("rejected_detections", []) if res_data else [],
            "planner_ready": False,
            "num_detections": 0,
            "inference_sec": elapsed,
        }
        return empty_res, meta

    elif fallback == "original":
        global global_yolo_proc
        if global_yolo_proc is not None and args.perception_backend in ["qwen_first", "yolo_first", "yolo11_qwen"]:
            print("[PERCEPTION FALLBACK] Stopping Qwen persistent worker for YOLOE fallback...")
            stop_yolo_worker(global_yolo_proc)
            global_yolo_proc = None

        if global_yolo_proc is None:
            print("[PERCEPTION FALLBACK] Starting YOLOE persistent worker for fallback...")
            try:
                global_yolo_proc = start_yolo_worker(args, backend_override="original")
            except Exception as e:
                print(f"[PERCEPTION FALLBACK ERROR] Failed to start YOLOE worker: {repr(e)}")
                empty_res = {
                    "detections": [],
                    "approach_candidates": res_data.get("approach_candidates", []) if res_data else [],
                    "rejected_detections": res_data.get("rejected_detections", []) if res_data else [],
                    "planner_ready": False,
                    "num_detections": 0,
                    "inference_sec": elapsed,
                }
                return empty_res, meta

        res = run_yolo_fast(image_path, args.conf, args, args.save_overlay, backend="original")
        return res, meta

    elif fallback == "yolo_first":
        print("[PERCEPTION FALLBACK] Running yolo_first fallback...")
        fb_success, fb_res_data, fb_reason, fb_elapsed = run_qwen_first_perception(image_path, args, "yolo_first")
        meta["qwen_first_elapsed_sec"] += fb_elapsed
        if fb_success and fb_res_data:
            return fb_res_data, meta
        else:
            empty_res = {
                "detections": [],
                "approach_candidates": [],
                "rejected_detections": [],
                "planner_ready": False,
                "num_detections": 0,
                "inference_sec": elapsed + fb_elapsed,
            }
            return empty_res, meta

    return res_data, meta


def find_valid_view(args, blocked_raw, records=None):
    view_records = []

    for idx, view in enumerate(VIEW_CANDIDATES[:args.max_views_per_cycle], start=1):
        print("\n" + "-" * 80)
        print(f"[VIEW TRY] {idx}: {view['view_id']}")
        print("-" * 80)

        command = write_view_command(view, args)
        view_result = wait_view_result(command["command_id"], args.view_timeout_sec)

        if view_result.get("status") != "success":
            view_records.append({
                "view_index": idx,
                "view": view,
                "view_result": view_result,
                "accepted": False,
                "reason": "view_move_failed",
            })
            continue

        image_path = collect_image(args.collect_timeout_sec)
        yolo_result, perception_meta = get_perception_result(image_path, args)

        if records is not None:
            resolve_pending_verification(view["view_id"], yolo_result, records)

        num_dets = yolo_result.get("num_detections", 0)
        num_apps = yolo_result.get("num_approach_candidates", 0)
        planner_ready = yolo_result.get("planner_ready", False)

        if not planner_ready and num_dets == 0 and num_apps > 0:
            record = {
                "view_index": idx,
                "view": view,
                "view_result": view_result,
                "image_path": str(image_path),
                "yolo_inference_sec": yolo_result.get("inference_sec"),
                "detection_eval": {"num_detections": 0, "num_candidates": 0, "best": None, "candidates": []},
                "accepted": False,
                "reason": "need_approach_but_not_implemented",
                "yolo_result": yolo_result,
                "perception_meta": perception_meta,
            }
            view_records.append(record)
            return None, image_path, yolo_result, None, view_records, perception_meta, "need_approach_but_not_implemented"

        if not planner_ready and not yolo_result.get("success", True):
            record = {
                "view_index": idx,
                "view": view,
                "view_result": view_result,
                "image_path": str(image_path),
                "yolo_inference_sec": yolo_result.get("inference_sec"),
                "detection_eval": {"num_detections": 0, "num_candidates": 0, "best": None, "candidates": []},
                "accepted": False,
                "reason": "perception_not_ready",
                "yolo_result": yolo_result,
                "perception_meta": perception_meta,
            }
            view_records.append(record)
            return None, image_path, yolo_result, None, view_records, perception_meta, "perception_not_ready"

        detection_eval = evaluate_detections(
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
            "image_path": str(image_path),
            "yolo_inference_sec": yolo_result.get("inference_sec"),
            "detection_eval": detection_eval,
            "accepted": accepted,
            "reason": "accepted" if accepted else "no_valid_detection",
            "yolo_result": yolo_result,
            "perception_meta": perception_meta,
        }

        view_records.append(record)

        print("[DETECTION EVAL]")
        print(json.dumps(detection_eval, ensure_ascii=False, indent=2))

        if accepted:
            return view, image_path, yolo_result, detection_eval, view_records, perception_meta, "accepted"

    return None, None, None, None, view_records, {}, "no_valid_target_found"


def compute_metrics(records):
    if not records:
        return {
            "num_cycles": 0,
            "action_success_rate": 0.0,
            "verification_success_rate": 0.0,
            "average_attach_distance_xy": 0.0,
            "average_yolo_inference_sec": 0.0,
        }

    action_ok = sum(1 for r in records if r.get("action_result", {}).get("status") == "success")
    verify_ok = sum(1 for r in records if r.get("verification", {}).get("verified"))

    attach = [
        float(r["action_result"]["attach_distance_xy"])
        for r in records
        if r.get("action_result", {}).get("attach_distance_xy") is not None
    ]

    yolo_secs = [
        float(r.get("before_yolo_inference_sec", 0.0))
        for r in records
        if r.get("before_yolo_inference_sec") is not None
    ]

    return {
        "num_cycles": len(records),
        "action_success_rate": round(action_ok / len(records), 4),
        "verification_success_rate": round(verify_ok / len(records), 4),
        "average_attach_distance_xy": round(sum(attach) / len(attach), 4) if attach else 0.0,
        "average_yolo_inference_sec": round(sum(yolo_secs) / len(yolo_secs), 4) if yolo_secs else 0.0,
    }


def save_logs(records, extra=None):
    summary = {
        "mode": "V2_7_HEAD_CAMERA_ALL_TRASH_FAST_LOOP",
        "metrics": compute_metrics(records),
        "records": records,
    }

    if extra:
        summary.update(extra)

    save_json_atomic(RUN_LOG_JSON, summary)

    with open(RUN_LOG_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
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
            "before_yolo_inference_sec",
            "after_yolo_inference_sec",
            "cycle_sec",
            "perception_backend",
            "qwen_first_elapsed_sec",
            "qwen_first_returncode",
            "qwen_first_planner_ready",
            "qwen_first_num_detections",
            "qwen_first_num_approach_candidates",
            "qwen_first_num_rejected",
            "qwen_first_verify_mode",
            "qwen_first_fallback_used",
            "qwen_first_fallback_reason",
        ], extrasaction='ignore')
        writer.writeheader()

        for r in records:
            task = r.get("selected_task", {}) or {}
            action = r.get("action_result", {}) or {}
            verify = r.get("verification", {}) or {}
            view = r.get("selected_view", {}) or {}

            writer.writerow({
                "cycle_id": r.get("cycle_id"),
                "plan_id": r.get("plan_id"),
                "view_id": view.get("view_id") if isinstance(view, dict) else "none",
                "raw_class_name": task.get("raw_class_name"),
                "category": task.get("garbage_category", task.get("category")),
                "target_bin": task.get("target_bin"),
                "confidence": task.get("confidence"),
                "action_status": action.get("status"),
                "verified": verify.get("verified"),
                "vote_score": verify.get("vote_score"),
                "attach_distance_xy": action.get("attach_distance_xy"),
                "before_yolo_inference_sec": r.get("before_yolo_inference_sec"),
                "after_yolo_inference_sec": r.get("after_yolo_inference_sec"),
                "cycle_sec": r.get("cycle_sec"),
                "perception_backend": r.get("perception_backend"),
                "qwen_first_elapsed_sec": r.get("qwen_first_elapsed_sec"),
                "qwen_first_returncode": r.get("qwen_first_returncode"),
                "qwen_first_planner_ready": r.get("qwen_first_planner_ready"),
                "qwen_first_num_detections": r.get("qwen_first_num_detections"),
                "qwen_first_num_approach_candidates": r.get("qwen_first_num_approach_candidates"),
                "qwen_first_num_rejected": r.get("qwen_first_num_rejected"),
                "qwen_first_verify_mode": r.get("qwen_first_verify_mode"),
                "qwen_first_fallback_used": r.get("qwen_first_fallback_used"),
                "qwen_first_fallback_reason": r.get("qwen_first_fallback_reason"),
            })

    print(f"[SAVED] {RUN_LOG_JSON}")
    print(f"[SAVED] {RUN_LOG_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--max-views-per-cycle", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--collect-timeout-sec", type=int, default=2)
    parser.add_argument("--view-timeout-sec", type=int, default=60)
    parser.add_argument("--action-timeout-sec", type=int, default=150)
    parser.add_argument("--profile", default="all", choices=["stable", "stable_plus_kitchen", "all"])
    parser.add_argument("--blocked-raw", default="")
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo11s-seg-best.pt"))
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--yolo-ready-timeout-sec", type=int, default=60)
    parser.add_argument("--yolo-infer-timeout-sec", type=int, default=30)
    parser.add_argument("--move-frames", type=int, default=45)
    parser.add_argument("--settle-frames", type=int, default=15)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--perception-backend", default="original",
                        choices=["original", "yolo_first", "qwen_first", "yoloe", "yolo11_qwen"])
    parser.add_argument("--qwen-first-model", default=str(ROOT / "models" / "yolo11s-seg-best.pt"))
    parser.add_argument("--qwen-first-conf", type=float, default=0.15)
    parser.add_argument("--qwen-first-roi-expand", type=float, default=2.0)
    parser.add_argument("--qwen-first-verify-mode", default="top1",
                        choices=["all", "planner_only", "top1", "none"])
    parser.add_argument("--qwen-first-max-candidates", type=int, default=5)
    parser.add_argument("--qwen-first-max-roi-refine", type=int, default=3)
    parser.add_argument("--qwen-first-timeout", type=int, default=90)
    parser.add_argument("--qwen-first-save-vis", action="store_true", default=False)
    parser.add_argument("--qwen-first-fallback", default="original",
                        choices=["none", "original", "yolo_first"])
    parser.add_argument("--qwen-verify-workers", type=int, default=4)
    args = parser.parse_args()

    if args.perception_backend == "yoloe":
        args.perception_backend = "original"
    elif args.perception_backend == "yolo11_qwen":
        args.perception_backend = "qwen_first"

    blocked_raw = {x.strip() for x in args.blocked_raw.split(",") if x.strip()}

    clean_ipc_files()

    global global_yolo_proc
    global_yolo_proc = None
    records = []

    try:
        global_yolo_proc = start_yolo_worker(args)

        print("=" * 80)
        print("[START] V2.7 fast all-trash loop")
        print("[INFO] Isaac must be running:")
        print("  1) v26_active_view_search_isaac_server.py")
        print("  2) v27_head_camera_action_server.py")
        print("=" * 80)

        for cycle_id in range(1, args.max_cycles + 1):
            cycle_t0 = time.time()

            print("\n" + "=" * 80)
            print(f"[CYCLE {cycle_id}]")
            print("=" * 80)

            (selected_view, image_path, before_yolo, detection_eval, 
             view_records, perception_meta, outcome) = find_valid_view(args, blocked_raw, records)

            if not perception_meta:
                perception_meta = {
                    "perception_backend": args.perception_backend,
                    "qwen_first_cmd": "none",
                    "qwen_first_elapsed_sec": 0.0,
                    "qwen_first_returncode": 0,
                    "qwen_first_planner_ready": False,
                    "qwen_first_num_detections": 0,
                    "qwen_first_num_approach_candidates": 0,
                    "qwen_first_num_rejected": 0,
                    "qwen_first_verify_mode": getattr(args, "qwen_first_verify_mode", "top1"),
                    "qwen_first_timing": {},
                    "qwen_first_fallback_used": False,
                    "qwen_first_fallback_reason": "none"
                }

            if outcome == "need_approach_but_not_implemented":
                print("[CYCLE] need_approach_but_not_implemented. Skipping grasping in this cycle.")
                record = {
                    "cycle_id": cycle_id,
                    "plan_id": None,
                    "selected_view": None,
                    "view_search_records": view_records,
                    "selected_task": None,
                    "before_image": str(image_path) if image_path else None,
                    "after_image": None,
                    "before_yolo_inference_sec": before_yolo.get("inference_sec") if before_yolo else 0.0,
                    "after_yolo_inference_sec": None,
                    "action_result": {
                        "status": "need_approach_but_not_implemented",
                        "raw_class_name": "none",
                        "garbage_category": "none"
                    },
                    "verification": {
                        "verified": False,
                        "vote_score": 0,
                        "reason": "need_approach_but_not_implemented"
                    },
                    "cycle_sec": round(time.time() - cycle_t0, 4),
                }
                record.update(perception_meta)
                records.append(record)
                save_logs(records)
                continue

            elif outcome == "perception_not_ready":
                print("[CYCLE] perception_not_ready. Skipping grasping in this cycle.")
                record = {
                    "cycle_id": cycle_id,
                    "plan_id": None,
                    "selected_view": None,
                    "view_search_records": view_records,
                    "selected_task": None,
                    "before_image": str(image_path) if image_path else None,
                    "after_image": None,
                    "before_yolo_inference_sec": before_yolo.get("inference_sec") if before_yolo else 0.0,
                    "after_yolo_inference_sec": None,
                    "action_result": {
                        "status": "perception_not_ready",
                        "raw_class_name": "none",
                        "garbage_category": "none"
                    },
                    "verification": {
                        "verified": False,
                        "vote_score": 0,
                        "reason": "perception_not_ready"
                    },
                    "cycle_sec": round(time.time() - cycle_t0, 4),
                }
                record.update(perception_meta)
                records.append(record)
                save_logs(records)
                continue

            elif selected_view is None:
                print("[DONE] no valid target found.")
                save_logs(records, {
                    "stop_reason": "no_valid_target_found",
                    "last_view_records": view_records,
                })
                break

            plan = make_plan(args, blocked_raw, selected_view, cycle_id)

            if plan.get("selected_task") is None:
                print("[WARN] planner returned num_tasks=0 after perception succeeded.")
                print(f"[PLANNER] num_tasks={plan.get('num_tasks',0)}, "
                      f"num_rejected={plan.get('num_rejected',0)}, "
                      f"qwen_first_compatible={plan.get('qwen_first_compatible',False)}")
                rejected_summary = [
                    {"raw": r.get("raw_class_name"), "reason": r.get("reason")}
                    for r in plan.get("rejected_detections", [])
                ]
                print(f"[PLANNER REJECTED] {json.dumps(rejected_summary, ensure_ascii=False)}")

                no_task_record = build_planner_no_task_record(
                    cycle_id=cycle_id,
                    cycle_t0=cycle_t0,
                    image_path=image_path,
                    before_yolo=before_yolo,
                    plan=plan,
                    perception_meta=perception_meta,
                )
                records.append(no_task_record)
                save_logs(records)

                if args.stop_on_failure:
                    print("[STOP] stop_on_failure=True, stopping after planner_no_task.")
                    break
                # 不 break，继续下一个 cycle（可能换视角后能找到可抓取目标）
                continue

            plan_id = plan["plan_id"]
            selected_task = plan["selected_task"]

            print("[SELECTED TASK]")
            print(json.dumps(selected_task, ensure_ascii=False, indent=2))

            action_result = wait_action_result(plan_id, args.action_timeout_sec)

            action_ok = action_result.get("status") == "success"

            if action_ok:
                # Defer verification to the next cycle's first perception call to eliminate redundant Qwen/YOLO inference
                global global_pending_verification
                global_pending_verification = {
                    "cycle_index_to_update": len(records),  # The index of the record we are about to append
                    "before_yolo": before_yolo,
                    "action_result": action_result,
                    "view_id": selected_view["view_id"] if isinstance(selected_view, dict) else None,
                    "perception_backend": args.perception_backend,
                }

                # Create a placeholder verification dict
                verification = {
                    "verified": True,  # Assume success for metric logging, will be updated in next cycle
                    "pending": True,
                    "reason": "pending_next_cycle_verification",
                    "action_status": action_result.get("status"),
                    "selected_raw_class": action_result.get("raw_class_name"),
                    "selected_category": action_result.get("garbage_category"),
                }
                after_image_str = None
                after_yolo_infer = None

            else:
                verification = {
                    "verified": False,
                    "vote_score": 0,
                    "reason": "action_failed_skip_after_verification",
                    "action_status": action_result.get("status"),
                    "selected_raw_class": action_result.get("raw_class_name"),
                    "selected_category": action_result.get("garbage_category"),
                }
                after_image_str = None
                after_yolo_infer = None

                failed_raw = action_result.get("raw_class_name")
                failed_status = action_result.get("status")

                if failed_raw and failed_status == "failed_no_matching_object":
                    blocked_raw.add(failed_raw)
                    print(f"[DYNAMIC BLOCK] raw_class={failed_raw} because no matching visible object")
                else:
                    print(f"[NO BLOCK ON FAILURE] raw_class={failed_raw}, status={failed_status}")

            print("[VERIFY]")
            print(json.dumps(verification, ensure_ascii=False, indent=2))

            record = {
                "cycle_id": cycle_id,
                "plan_id": plan_id,
                "selected_view": selected_view,
                "view_search_records": view_records,
                "selected_task": selected_task,
                "before_image": str(image_path),
                "after_image": after_image_str,
                "before_yolo_inference_sec": before_yolo.get("inference_sec"),
                "after_yolo_inference_sec": after_yolo_infer,
                "action_result": action_result,
                "verification": verification,
                "cycle_sec": round(time.time() - cycle_t0, 4),
            }
            record.update(perception_meta)

            records.append(record)
            save_logs(records)

            action_ok = action_result.get("status") == "success"
            verify_ok = verification.get("verified") is True

            if args.stop_on_failure and (not action_ok or not verify_ok):
                print("[STOP] failure detected.")
                break

        # Resolve any leftover deferred verification before exit
        if global_pending_verification is not None:
            resolve_pending_verification(None, None, records)

        save_logs(records)
        print("[DONE]")
        print(json.dumps(compute_metrics(records), ensure_ascii=False, indent=2))

    finally:
        stop_yolo_worker(global_yolo_proc)


if __name__ == "__main__":
    main()
