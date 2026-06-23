#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v27_qwen_perception_worker.py
=============================
qwen_first / yolo_first 感知常驻进程。

在主进程生命期内只加载一次 YOLO 模型（节省 ~10s 冷启动）。
通过 JSON IPC 文件与 v27_head_camera_all_trash_loop_fast.py 通信：

  请求文件：~/trashbot_ws/data/logs/v27_qwen_worker_request.json
  响应文件：~/trashbot_ws/data/logs/v27_qwen_worker_response.json
  就绪文件：~/trashbot_ws/data/logs/v27_qwen_worker_ready.json

请求 JSON 格式：
  {
    "request_id": "v27_qwen_xxx",
    "image_path": "/path/to/frame.jpg",
    "pipeline": "qwen_first",          # or "yolo_first"
    "conf": 0.15,
    "roi_expand": 2.0,
    "verify_mode": "top1",
    "max_qwen_candidates": 5,
    "max_roi_refine": 3,
    "vis_mode": "planner",
    "save_vis": false
  }

响应 JSON 格式：
  {
    "request_id": "...",
    "status": "success" | "failed",
    "result": { ... },   # 与 yolo_seg_offline_result.json 相同结构
    "error": "...",      # 仅 failed 时
    "elapsed_sec": ...
  }

输出 JSON 路径与单次脚本完全一致：
  ~/trashbot_ws/data/logs/yolo_seg_offline_result.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 必须在导入 yolo11_qwen_perception_offline 模块中的函数之前加载 .env
# ---------------------------------------------------------------------------
_ENV_PATH = Path.home() / "trashbot_ws" / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if _line.startswith("export "):
            _line = _line[7:].strip()
        if "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip(); _v = _v.strip()
        if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ('"', "'"):
            _v = _v[1:-1]
        if _k and _k not in os.environ:
            os.environ[_k] = _v

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 从 yolo11_qwen_perception_offline 导入所有需要的函数
# 这样逻辑完全复用，不需要维护两份代码
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path.home() / "trashbot_ws" / "scripts" / "perception"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from yolo11_qwen_perception_offline import (
    run_qwen_first,
    run_yolo_first,
    check_planner_ready,
    save_json_atomic,
    save_summary_text,
    draw_overlay_qwen_first,
    draw_overlay_qwen,
    _get_qwen_config,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OVERLAY,
    DEFAULT_QWEN_RAW_LOG,
    ROOT,
)

LOG_DIR = ROOT / "data" / "logs"

WORKER_REQUEST_JSON = LOG_DIR / "v27_qwen_worker_request.json"
WORKER_RESPONSE_JSON = LOG_DIR / "v27_qwen_worker_response.json"
WORKER_READY_JSON = LOG_DIR / "v27_qwen_worker_ready.json"


def load_json_safe(path: Path):
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def make_fake_args(req: dict):
    """把请求 JSON 里的参数包装成 argparse.Namespace，供现有函数使用。"""
    class _Args:
        pass
    a = _Args()
    a.pipeline        = req.get("pipeline", "qwen_first")
    a.conf            = float(req.get("conf", 0.15))
    a.roi_expand      = float(req.get("roi_expand", 2.0))
    a.verify_mode     = req.get("verify_mode", "top1")
    a.max_qwen_candidates = int(req.get("max_qwen_candidates", 5))
    a.max_roi_refine  = int(req.get("max_roi_refine", 3))
    a.vis_mode        = req.get("vis_mode", "planner")
    a.save_vis        = bool(req.get("save_vis", False))
    a.qwen_verify_workers = int(req.get("qwen_verify_workers", 4))
    a.include_approach_in_detections = False
    a.allow_unknown   = False
    a.qwen_coarse_conf  = float(req.get("qwen_coarse_conf", 0.55))
    a.qwen_verify_conf  = float(req.get("qwen_verify_conf", 0.50))
    a.min_area_ratio  = float(req.get("min_area_ratio", 0.0005))
    a.max_area_ratio  = float(req.get("max_area_ratio", 0.60))
    return a


def normalize_rejected(rejected_detections):
    for r in rejected_detections:
        ex_reason = (r.get("reason") or r.get("reject_reason")
                     or r.get("rejected_reason") or r.get("status")
                     or r.get("error_reason") or "unknown_rejection_reason")
        if ex_reason in ("invalid_qwen_bbox", "invalid_qwen_bbox_norm"):
            mapped = "invalid_qwen_bbox_norm"
        elif ex_reason in ("failed_yolo_refine", "no_yolo_detection_in_roi",
                           "yolo_boxes_below_area_threshold"):
            mapped = "failed_yolo_refine"
        elif ex_reason == "qwen_verify_not_graspable":
            mapped = "qwen_verify_not_graspable"
        elif ex_reason == "duplicate_after_yolo_refine":
            mapped = "duplicate_after_yolo_refine"
        else:
            mapped = ex_reason
        r["reason"] = mapped
        r.setdefault("reject_reason", mapped)


def run_one_request(model, req: dict):
    """
    处理一次感知请求。
    model: 已加载的 YOLO 实例（在进程生命期内只创建一次）
    req:   请求 JSON dict
    返回 (summary dict, elapsed_sec)
    """
    t0 = time.time()
    image_path = Path(req["image_path"]).expanduser().resolve()
    args = make_fake_args(req)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    orig_h, orig_w = img.shape[:2]
    t_after_load = time.time()
    print(f"[WORKER STAGE] image_load={t_after_load-t0:.3f}s")

    qwen_candidates_for_vis = []

    if args.pipeline == "yolo_first":
        crop_dir = ROOT / "data" / "crops" / "qwen" / timestamp
        crop_dir.mkdir(parents=True, exist_ok=True)
        detections, rejected_detections, infer_sec = run_yolo_first(
            image_path, model, args, img, orig_w, orig_h, crop_dir)

        # approach split
        approach_candidates, final_detections = [], []
        for d in detections:
            if d.get("approach_required", False):
                d["not_sent_to_planner_reason"] = "approach_required"
                approach_candidates.append(d)
            else:
                final_detections.append(d)
        detections = final_detections

        normalize_rejected(rejected_detections)

        summary = {
            "mode": "YOLO11_SEG_QWEN_PERCEPTION",
            "backend": "yolo11_seg_qwen",
            "pipeline": "yolo_first",
            "requested_pipeline": "yolo_first",
            "executed_pipeline": "yolo_first",
            "fallback_used": False,
            "fallback_reason": "none",
            "model": req.get("model_path", ""),
            "qwen_model": os.environ.get("QWEN_MODEL", "qwen3-vl-flash"),
            "image": str(image_path),
            "source_image": str(image_path),
            "image_width": int(orig_w),
            "image_height": int(orig_h),
            "created_timestamp": time.time(),
            "inference_sec": round(infer_sec, 4),
            "planner_ready": check_planner_ready(detections),
            "num_detections": len(detections),
            "num_approach_candidates": len(approach_candidates),
            "num_rejected": len(rejected_detections),
            "verify_mode": args.verify_mode,
            "num_qwen_verify_calls": len(detections) + len(approach_candidates),
            "num_qwen_verify_skipped": 0,
            "qwen_verify_workers": 1,
            "qwen_verify_parallel": False,
            "qwen_coarse_sec": 0.0,
            "yolo_roi_total_sec": 0.0,
            "qwen_verify_total_sec": 0.0,
            "yolo_roi_calls": 0,
            "detections": detections,
            "approach_candidates": approach_candidates,
            "rejected_detections": rejected_detections,
        }

    else:  # qwen_first
        print(f"[WORKER STAGE] pre_qwen_first_setup={time.time()-t0:.3f}s")
        _get_qwen_config()
        try:
            DEFAULT_QWEN_RAW_LOG.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_QWEN_RAW_LOG.write_text(
                f"# Qwen raw responses — run started at {datetime.now().isoformat()}\n",
                encoding="utf-8")
        except Exception:
            pass

        (detections, rejected_detections, infer_sec,
         qwen_candidates_for_vis, fb_meta) = run_qwen_first(
            image_path, model, args, img, orig_w, orig_h, timestamp)

        # approach split
        approach_candidates, final_detections = [], []
        for d in detections:
            if d.get("approach_required", False):
                d["not_sent_to_planner_reason"] = "approach_required"
                approach_candidates.append(d)
            else:
                final_detections.append(d)
        detections = final_detections

        normalize_rejected(rejected_detections)

        executed_pl = fb_meta.get("executed_pipeline", "qwen_first")
        fallback_used = fb_meta.get("fallback_used", False)
        fallback_reason = fb_meta.get("fallback_reason", "none")

        if executed_pl == "yolo_first":
            _mode = "YOLO11_SEG_QWEN_PERCEPTION"
            _backend = "yolo11_seg_qwen"
        else:
            _mode = "YOLO11_QWEN_FIRST_ROI_PERCEPTION"
            _backend = "qwen_first_yolo11_roi"

        summary = {
            "mode": _mode,
            "backend": _backend,
            "pipeline": executed_pl,
            "requested_pipeline": args.pipeline,
            "executed_pipeline": executed_pl,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "model": req.get("model_path", ""),
            "qwen_model": os.environ.get("QWEN_MODEL", "qwen3-vl-flash"),
            "image": str(image_path),
            "source_image": str(image_path),
            "image_width": int(orig_w),
            "image_height": int(orig_h),
            "created_timestamp": time.time(),
            "inference_sec": round(infer_sec, 4),
            "planner_ready": check_planner_ready(detections),
            "num_qwen_candidates": len(qwen_candidates_for_vis),
            "num_detections": len(detections),
            "num_approach_candidates": len(approach_candidates),
            "num_rejected": len(rejected_detections),
            "verify_mode": fb_meta.get("verify_mode", args.verify_mode),
            "num_qwen_verify_calls": fb_meta.get("num_qwen_verify_calls", 0),
            "num_qwen_verify_skipped": fb_meta.get("num_qwen_verify_skipped", 0),
            "qwen_verify_workers": args.qwen_verify_workers if executed_pl == "qwen_first" else 1,
            "qwen_verify_parallel": False,
            "qwen_coarse_sec": fb_meta.get("qwen_coarse_sec", 0.0) if executed_pl == "qwen_first" else 0.0,
            "yolo_roi_total_sec": fb_meta.get("yolo_roi_total_sec", 0.0) if executed_pl == "qwen_first" else 0.0,
            "qwen_verify_total_sec": fb_meta.get("qwen_verify_total_sec", 0.0) if executed_pl == "qwen_first" else 0.0,
            "qwen_verify_wall_sec": fb_meta.get("qwen_verify_wall_sec", 0.0) if executed_pl == "qwen_first" else 0.0,
            "yolo_roi_calls": fb_meta.get("yolo_roi_calls", 0) if executed_pl == "qwen_first" else 0,
            "detections": detections,
            "approach_candidates": approach_candidates,
            "rejected_detections": rejected_detections,
        }

        # timing fields for v27
        summary["total_sec"] = summary["inference_sec"]

        if args.save_vis:
            if executed_pl == "yolo_first":
                draw_overlay_qwen(image_path, detections + approach_candidates, DEFAULT_OVERLAY)
            else:
                if args.vis_mode == "debug":
                    overlay_path = ROOT / "data" / "logs" / "yolo11_qwen_overlay_debug_latest.jpg"
                else:
                    overlay_path = ROOT / "data" / "logs" / "yolo11_qwen_overlay_latest.jpg"
                draw_overlay_qwen_first(
                    image_path, detections, rejected_detections,
                    qwen_candidates_for_vis, overlay_path,
                    approach_candidates=approach_candidates,
                    vis_mode=args.vis_mode)

    # 写输出 JSON（与单次脚本路径完全一致）
    summary["timing_breakdown"] = {
        "image_load_sec": round(t_after_load - t0, 4),
        "pipeline_sec": round(summary.get("inference_sec", 0.0), 4),
        "qwen_coarse_sec": round(summary.get("qwen_coarse_sec", 0.0), 4),
        "yolo_roi_total_sec": round(summary.get("yolo_roi_total_sec", 0.0), 4),
        "qwen_verify_total_sec": round(summary.get("qwen_verify_total_sec", 0.0), 4),
        "qwen_verify_wall_sec": round(summary.get("qwen_verify_wall_sec", 0.0), 4),
        "yolo_roi_calls": summary.get("yolo_roi_calls", 0),
        "qwen_verify_calls": summary.get("num_qwen_verify_calls", 0),
    }
    stage_now = time.time()
    print(f"[WORKER STAGE] pipeline_total={summary.get('inference_sec', 0.0):.3f}s "
          f"coarse={summary.get('qwen_coarse_sec', 0.0):.3f}s "
          f"yolo_roi={summary.get('yolo_roi_total_sec', 0.0):.3f}s "
          f"qwen_verify={summary.get('qwen_verify_total_sec', 0.0):.3f}s "
          f"overhead={stage_now-t0-summary.get('inference_sec', 0.0):.3f}s")
    save_json_atomic(DEFAULT_OUTPUT_JSON, summary)

    elapsed = round(time.time() - t0, 4)
    print(f"[WORKER] done request_id={req.get('request_id', '?')} "
          f"pipeline={args.pipeline} dets={len(detections)} elapsed={elapsed}s")
    return summary, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--request-json",  default=str(WORKER_REQUEST_JSON))
    parser.add_argument("--response-json", default=str(WORKER_RESPONSE_JSON))
    parser.add_argument("--ready-json",    default=str(WORKER_READY_JSON))
    parser.add_argument("--poll-interval", type=float, default=0.01,
                        help="Poll interval in seconds (default 0.01)")
    args_main = parser.parse_args()

    request_json  = Path(args_main.request_json).expanduser()
    response_json = Path(args_main.response_json).expanduser()
    ready_json    = Path(args_main.ready_json).expanduser()

    model_path = Path(args_main.model).expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print("=" * 80)
    print("[START] v27 Qwen Perception persistent worker")
    print(f"[MODEL] {model_path}")
    print(f"[REQUEST ] {request_json}")
    print(f"[RESPONSE] {response_json}")
    print("=" * 80)

    t_load = time.time()
    model = YOLO(str(model_path))
    load_sec = round(time.time() - t_load, 3)
    print(f"[READY] YOLO loaded in {load_sec}s")

    ready_data = {
        "status": "ready",
        "backend": "qwen_perception_worker",
        "model": str(model_path),
        "model_load_sec": load_sec,
        "timestamp": time.time(),
    }
    save_json_atomic(ready_json, ready_data)

    processed_ids = set()

    while True:
        req = load_json_safe(request_json)

        if not req:
            time.sleep(args_main.poll_interval)
            continue

        cmd = req.get("command", "infer")

        if cmd == "shutdown":
            print("[SHUTDOWN] received shutdown command. Exiting.")
            break

        request_id = req.get("request_id") or str(request_json.stat().st_mtime_ns)

        if request_id in processed_ids:
            time.sleep(args_main.poll_interval)
            continue

        processed_ids.add(request_id)

        if "image_path" not in req:
            print(f"[IGNORE] request {request_id}: no image_path")
            time.sleep(args_main.poll_interval)
            continue

        # 注入 model_path 供 summary 记录
        req.setdefault("model_path", str(model_path))

        t0 = time.time()
        try:
            summary, elapsed = run_one_request(model, req)
            response = {
                "request_id": request_id,
                "status": "success",
                "result": summary,
                "elapsed_sec": elapsed,
                "timestamp": time.time(),
            }
        except Exception as e:
            elapsed = round(time.time() - t0, 4)
            print(f"[ERROR] request {request_id}: {repr(e)}")
            import traceback
            traceback.print_exc()
            response = {
                "request_id": request_id,
                "status": "failed",
                "error": repr(e),
                "elapsed_sec": elapsed,
                "timestamp": time.time(),
            }

        save_json_atomic(response_json, response)
        time.sleep(args_main.poll_interval)


if __name__ == "__main__":
    main()
