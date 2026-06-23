import argparse
import json
import time
import os
import sys
from pathlib import Path

# Remove local workspace paths to force importing from environment site-packages
for path in list(sys.path):
    if 'ultralytics-main' in path or path == '':
        sys.path.remove(path)

import cv2
import numpy as np
import torch

# Register safe globals for PyTorch 2.6+ compatibility
try:
    from ultralytics.nn.tasks import YOLOESegModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([YOLOESegModel])
        print("Registered YOLOESegModel as safe global.")
except Exception as e:
    print("Could not register safe globals:", e)

from ultralytics import YOLO

DEFAULT_OUTPUT_JSON = Path.home() / "trashbot_ws/data/logs/yolo_seg_offline_result.json"
DEFAULT_OVERLAY = Path.home() / "trashbot_ws/data/logs/yolo_seg_overlay.jpg"


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json_safe(path: Path):
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_request_image(req):
    for key in ["image", "image_path", "source_image", "path"]:
        v = req.get(key)
        if v:
            return Path(v).expanduser()
    return None


def polygon_from_mask(mask_xy):
    if mask_xy is None:
        return []

    pts = []
    try:
        arr = np.asarray(mask_xy, dtype=float)
        for x, y in arr:
            pts.append([round(float(x), 3), round(float(y), 3)])
    except Exception:
        return []

    return pts


def compute_bottom_contact(polygon, bbox_xyxy):
    # bbox_xyxy is [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox_xyxy
    if not polygon or len(polygon) < 3:
        # Fallback to midpoint of bottom edge of bbox
        return [round(float((x1 + x2) / 2.0), 3), round(float(y2), 3)]
    
    pts = np.array(polygon, dtype=float)
    y_coords = pts[:, 1]
    y_max = np.max(y_coords)
    
    # Filter points whose y coordinate is within 5 pixels of the absolute maximum y
    threshold = 5.0
    bottom_pts = pts[y_coords >= (y_max - threshold)]
    
    if len(bottom_pts) == 0:
        return [round(float((x1 + x2) / 2.0), 3), round(float(y2), 3)]
        
    # Average the x coordinates of these points
    bottom_x = np.mean(bottom_pts[:, 0])
    
    return [round(float(bottom_x), 3), round(float(y_max), 3)]


def draw_overlay(image_path, detections, overlay_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox_xyxy"]]
        label = f'{d["raw_class_name"]}:{d["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), img)


def run_yoloe(model, image_path, args, request_id, conf):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]

    t0 = time.time()

    results = model.predict(
        source=str(image_path),
        imgsz=int(args.imgsz),
        conf=float(conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=args.device if args.device != "auto" else None,
        half=bool(args.half),
        verbose=False,
    )

    infer_sec = time.time() - t0

    r = results[0]
    detections = []

    boxes = r.boxes
    masks = r.masks

    if boxes is not None:
        for i, box in enumerate(boxes):
            conf_i = float(box.conf[0].item()) if box.conf is not None else float(conf)

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            x1, y1, x2, y2 = xyxy

            x1 = max(0.0, min(w - 1.0, x1))
            x2 = max(0.0, min(w - 1.0, x2))
            y1 = max(0.0, min(h - 1.0, y1))
            y2 = max(0.0, min(h - 1.0, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, w * h)

            if area_ratio > float(args.max_area_ratio):
                continue

            polygon = []
            if masks is not None and getattr(masks, "xy", None) is not None and i < len(masks.xy):
                polygon = polygon_from_mask(masks.xy[i])

            # Calculate bottom contact point from mask polygon or bbox fallback
            bottom_contact_px = compute_bottom_contact(polygon, [x1, y1, x2, y2])

            det = {
                "object_id": f"yoloe_det_{i + 1:03d}",
                "raw_class_name": "trash_object",
                "class_name": "trash_object",
                "display_name": "trash_object",
                "category": "trash_object",
                "garbage_category": "unknown",
                "target_bin": "unknown",
                "confidence": round(conf_i, 4),
                "bbox_xyxy": [
                    round(float(x1), 3),
                    round(float(y1), 3),
                    round(float(x2), 3),
                    round(float(y2), 3),
                ],
                "centroid_px": [
                    round(float((x1 + x2) / 2.0), 3),
                    round(float((y1 + y2) / 2.0), 3),
                ],
                "bottom_contact_px": bottom_contact_px,
                "bbox_area_ratio": round(float(area_ratio), 6),
                "polygon": polygon,
                "source": "trained_yoloe26s_trash_object_seg",
            }

            detections.append(det)

    result = {
        "mode": "YOLOE_26S_SEG_PERCEPTION",
        "backend": "yoloe",
        "request_id": request_id,
        "model": args.model,
        "image": str(image_path),
        "image_path": str(image_path),
        "image_width": int(w),
        "image_height": int(h),
        "created_timestamp": time.time(),
        "inference_sec": round(float(infer_sec), 4),
        "num_detections": len(detections),
        "detections": detections,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    parser.add_argument("--ready-json", required=True)

    parser.add_argument("--model", default=str(Path.home() / "trashbot_ws/runs/yoloe_trash_object_seg/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.6)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY))
    args = parser.parse_args()

    request_json = Path(args.request_json).expanduser()
    response_json = Path(args.response_json).expanduser()
    ready_json = Path(args.ready_json).expanduser()
    output_json = Path(args.output_json).expanduser()
    overlay_path = Path(args.overlay).expanduser()

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Disable AMP/half to avoid c10::Half != float mismatch
    args.half = False

    print("=" * 80)
    print("[START] V2.7 YOLOE persistent worker (Fine-tuned Single-Class Mode)")
    print("[MODEL]", args.model)
    print("[DEVICE]", args.device, "half=", args.half)
    print("[IMGSZ]", args.imgsz)
    print("=" * 80)

    t_load = time.time()
    model = YOLO(args.model)
    load_sec = time.time() - t_load

    ready = {
        "status": "ready",
        "backend": "yoloe",
        "model": args.model,
        "device": args.device,
        "half": bool(args.half),
        "imgsz": args.imgsz,
        "model_load_sec": round(load_sec, 4),
        "timestamp": time.time(),
    }

    save_json_atomic(ready_json, ready)
    print("[READY] model loaded in", round(load_sec, 4), "s")

    processed_request_ids = set()

    while True:
        req = load_json_safe(request_json)

        if not req:
            time.sleep(0.05)
            continue

        request_id = req.get("request_id") or req.get("id") or req.get("plan_id") or str(request_json.stat().st_mtime_ns)

        if request_id in processed_request_ids:
            time.sleep(0.05)
            continue

        processed_request_ids.add(request_id)

        try:
            image_path = get_request_image(req)
            if image_path is None:
                response = {
                    "status": "ignored",
                    "request_id": request_id,
                    "backend": "yoloe",
                    "reason": "request json has no image path",
                    "timestamp": time.time(),
                }
                save_json_atomic(response_json, response)
                print(f"[IGNORE INVALID REQUEST] {request_id}: no image path")
                time.sleep(0.05)
                continue

            conf = float(req.get("conf", req.get("confidence", 0.25)))
            save_overlay = bool(req.get("save_overlay", False))

            result = run_yoloe(
                model=model,
                image_path=image_path,
                args=args,
                request_id=request_id,
                conf=conf,
            )

            save_json_atomic(output_json, result)

            if save_overlay:
                draw_overlay(image_path, result["detections"], overlay_path)

            response = {
                "status": "success",
                "request_id": request_id,
                "backend": "yoloe",
                "result": result,
                "timestamp": time.time(),
            }

            save_json_atomic(response_json, response)

            print(
                f"[YOLOE-seg] {request_id} det={len(result['detections'])} "
                f"infer={result['inference_sec']}s image={image_path.name}"
            )

        except Exception as e:
            response = {
                "status": "failed",
                "request_id": request_id,
                "error": repr(e),
                "timestamp": time.time(),
            }
            save_json_atomic(response_json, response)
            print("[ERROR]", repr(e))

        time.sleep(0.05)


if __name__ == "__main__":
    main()