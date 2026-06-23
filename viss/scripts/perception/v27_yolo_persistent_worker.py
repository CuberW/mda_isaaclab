import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path.home() / "trashbot_ws"
DEFAULT_MODEL = ROOT / "models" / "best_seg.pt"
DEFAULT_CONFIG = ROOT / "config" / "trash_classes.yaml"
LOG_DIR = ROOT / "data" / "logs"

STANDARD_RESULT_JSON = LOG_DIR / "yolo_seg_offline_result.json"
STANDARD_OVERLAY_JPG = LOG_DIR / "yolo_seg_overlay.jpg"


FALLBACK_CLASS_MAP = {
    "potato": ("kitchen", "bin_kitchen_green"),
    "rabbitcut": ("kitchen", "bin_kitchen_green"),
    "potatocut": ("kitchen", "bin_kitchen_green"),
    "mooli": ("kitchen", "bin_kitchen_green"),

    "battery": ("hazardous", "bin_hazardous_red"),
    "battery1": ("hazardous", "bin_hazardous_red"),
    "battery5": ("hazardous", "bin_hazardous_red"),
    "drug": ("hazardous", "bin_hazardous_red"),
    "drugbag": ("hazardous", "bin_hazardous_red"),
    "drugbox": ("hazardous", "bin_hazardous_red"),
    "capsule": ("hazardous", "bin_hazardous_red"),

    "bottle": ("recyclable", "bin_recyclable_blue"),
    "bottle2": ("recyclable", "bin_recyclable_blue"),
    "can": ("recyclable", "bin_recyclable_blue"),
    "paper": ("recyclable", "bin_recyclable_blue"),

    "brick": ("other", "bin_other_gray"),
    "china": ("other", "bin_other_gray"),
    "stone": ("other", "bin_other_gray"),
    "papercup": ("recyclable", "bin_recyclable_blue"),
}


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}


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


def load_class_map(config_path: Path):
    mapping = dict(FALLBACK_CLASS_MAP)

    try:
        import yaml

        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            classes = data.get("trash_classes", {})
            for raw, item in classes.items():
                category = item.get("category", "unknown")
                target_bin = item.get("target_bin", "unknown")
                mapping[raw] = (category, target_bin)

    except Exception as e:
        print(f"[WARN] failed to load class config, use fallback: {repr(e)}")

    return mapping


def polygon_centroid(poly):
    if poly is None or len(poly) == 0:
        return None

    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    return [float(arr[:, 0].mean()), float(arr[:, 1].mean())]


def draw_overlay(image_bgr, detections):
    out = image_bgr.copy()

    color_by_category = {
        "recyclable": (255, 128, 0),
        "kitchen": (0, 180, 0),
        "hazardous": (0, 0, 255),
        "other": (128, 128, 128),
        "unknown": (255, 255, 255),
    }

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        category = det.get("category", "unknown")
        color = color_by_category.get(category, (255, 255, 255))
        label = f"{category} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if det.get("polygon"):
            pts = np.asarray(det["polygon"], dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cv2.polylines(out, [pts], True, color, 2)

        tx = max(0, x1)
        ty = max(18, y1 - 6)
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def infer_once(model, image_path: Path, class_map, conf, imgsz, device, half, max_det, max_area_ratio, save_overlay):
    t0 = time.time()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    image_height, image_width = image_bgr.shape[:2]

    results = model.predict(
        source=str(image_path),
        conf=float(conf),
        imgsz=int(imgsz),
        device=device,
        half=bool(half),
        max_det=int(max_det),
        verbose=False,
        retina_masks=False,
    )

    r = results[0]
    names = r.names

    detections = []

    boxes = r.boxes
    masks_xy = None

    if getattr(r, "masks", None) is not None and r.masks is not None:
        try:
            masks_xy = r.masks.xy
        except Exception:
            masks_xy = None

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            raw_class_name = str(names.get(cls_id, cls_id))
            category, target_bin = class_map.get(raw_class_name, ("unknown", "unknown"))

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            x1, y1, x2, y2 = xyxy

            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = bbox_area / max(1.0, float(image_width * image_height))

            if area_ratio > max_area_ratio:
                continue

            polygon = []
            centroid = None

            if masks_xy is not None and i < len(masks_xy):
                try:
                    poly = np.asarray(masks_xy[i], dtype=np.float32)
                    if poly.ndim == 2 and poly.shape[0] >= 3:
                        polygon = [[round(float(x), 2), round(float(y), 2)] for x, y in poly.tolist()]
                        centroid = polygon_centroid(poly)
                except Exception:
                    polygon = []
                    centroid = None

            if centroid is None:
                centroid = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            det = {
                "object_id": f"{raw_class_name}_{len(detections):02d}",
                "class_id": cls_id,
                "class_name": raw_class_name,
                "raw_class_name": raw_class_name,
                "display_name": category,
                "category": category,
                "garbage_category": category,
                "garbage_category_cn": CATEGORY_CN.get(category, category),
                "target_bin": target_bin,
                "confidence": round(score, 4),
                "bbox_xyxy": [round(float(v), 2) for v in xyxy],
                "bbox_area_ratio": round(float(area_ratio), 6),
                "centroid_px": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
                "polygon": polygon,
            }

            detections.append(det)

    result_json = {
        "mode": "V27_YOLO_PERSISTENT_RESULT",
        "image": str(image_path),
        "source_image": str(image_path),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "model_path": str(DEFAULT_MODEL),
        "conf_threshold": float(conf),
        "imgsz": int(imgsz),
        "device": str(device),
        "half": bool(half),
        "num_detections": len(detections),
        "detections": detections,
        "inference_sec": round(time.time() - t0, 4),
        "timestamp": time.time(),
    }

    save_json_atomic(STANDARD_RESULT_JSON, result_json)

    if save_overlay:
        overlay = draw_overlay(image_bgr, detections)
        STANDARD_OVERLAY_JPG.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(STANDARD_OVERLAY_JPG), overlay)
        result_json["overlay_path"] = str(STANDARD_OVERLAY_JPG)

    return result_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    parser.add_argument("--ready-json", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--poll-sec", type=float, default=0.05)
    args = parser.parse_args()

    request_json = Path(args.request_json)
    response_json = Path(args.response_json)
    ready_json = Path(args.ready_json)
    model_path = Path(args.model)
    config_path = Path(args.config)

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    half = bool(torch.cuda.is_available() and str(device) != "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("[START] V2.7 YOLO persistent worker")
    print(f"[MODEL] {model_path}")
    print(f"[DEVICE] {device}, half={half}")
    print(f"[IMGSZ] {args.imgsz}")
    print("=" * 80)

    class_map = load_class_map(config_path)

    t_load = time.time()
    model = YOLO(str(model_path))
    load_sec = time.time() - t_load

    save_json_atomic(ready_json, {
        "status": "ready",
        "model": str(model_path),
        "device": str(device),
        "half": bool(half),
        "imgsz": int(args.imgsz),
        "model_load_sec": round(load_sec, 4),
        "timestamp": time.time(),
    })

    print(f"[READY] model loaded in {load_sec:.3f}s")

    last_request_id = None

    while True:
        req = load_json_safe(request_json)

        if req is None:
            time.sleep(args.poll_sec)
            continue

        request_id = req.get("request_id")
        if not request_id or request_id == last_request_id:
            time.sleep(args.poll_sec)
            continue

        last_request_id = request_id

        if req.get("command") == "shutdown":
            save_json_atomic(response_json, {
                "request_id": request_id,
                "status": "shutdown",
                "timestamp": time.time(),
            })
            print("[SHUTDOWN]")
            break

        try:
            image_path = Path(req["image_path"])
            conf = float(req.get("conf", 0.50))
            save_overlay = bool(req.get("save_overlay", False))

            result = infer_once(
                model=model,
                image_path=image_path,
                class_map=class_map,
                conf=conf,
                imgsz=int(req.get("imgsz", args.imgsz)),
                device=device,
                half=half,
                max_det=int(req.get("max_det", args.max_det)),
                max_area_ratio=float(req.get("max_area_ratio", args.max_area_ratio)),
                save_overlay=save_overlay,
            )

            save_json_atomic(response_json, {
                "request_id": request_id,
                "status": "success",
                "result": result,
                "standard_result_json": str(STANDARD_RESULT_JSON),
                "standard_overlay_jpg": str(STANDARD_OVERLAY_JPG) if save_overlay else None,
                "timestamp": time.time(),
            })

            print(
                f"[YOLO] {request_id} det={result['num_detections']} "
                f"infer={result['inference_sec']}s image={image_path.name}"
            )

        except Exception as e:
            save_json_atomic(response_json, {
                "request_id": request_id,
                "status": "failed_exception",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time(),
            })
            print(f"[FAILED] {request_id}: {repr(e)}")


if __name__ == "__main__":
    main()
