import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path.home() / "trashbot_ws"
DEFAULT_MODEL = ROOT / "models" / "best_seg.pt"
DEFAULT_CONFIG = ROOT / "config" / "trash_classes.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "logs"


CATEGORY_DISPLAY_NAME = {
    "recyclable": "recyclable",
    "kitchen": "kitchen",
    "hazardous": "hazardous",
    "other": "other",
    "unknown": "unknown",
}

CATEGORY_COLOR = {
    "recyclable": (255, 80, 80),
    "kitchen": (80, 220, 80),
    "hazardous": (80, 80, 255),
    "other": (180, 80, 255),
    "unknown": (160, 160, 160),
}


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def load_class_mapping(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mapping = {}

    for object_id, item in data["trash_classes"].items():
        class_name = item["class_name"]
        mapping[class_name] = {
            "object_id": object_id,
            "category": item["category"],
            "target_bin": item["target_bin"],
            "disturbance": item.get("disturbance", "none"),
        }

    return mapping


def polygon_centroid(poly):
    if poly is None or len(poly) < 3:
        return None

    pts = np.array(poly, dtype=np.float32)
    m = cv2.moments(pts)

    if abs(m["m00"]) < 1e-6:
        return None

    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])

    return [round(cx, 2), round(cy, 2)]


def draw_label(image, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    label_x1 = int(x1)
    label_y1 = int(y1) - text_h - baseline - 6

    if label_y1 < 0:
        label_y1 = int(y1) + 4

    label_x2 = label_x1 + text_w + 8
    label_y2 = label_y1 + text_h + baseline + 6

    h, w = image.shape[:2]

    label_x1 = max(0, min(label_x1, w - 1))
    label_x2 = max(0, min(label_x2, w - 1))
    label_y1 = max(0, min(label_y1, h - 1))
    label_y2 = max(0, min(label_y2, h - 1))

    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)

    text_org = (label_x1 + 4, label_y2 - baseline - 3)

    cv2.putText(
        image,
        text,
        text_org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(image_path: Path, detections):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    overlay = image.copy()

    for det in detections:
        category = det.get("category", "unknown")
        color = CATEGORY_COLOR.get(category, CATEGORY_COLOR["unknown"])

        polygon = det.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

        x1, y1, x2, y2 = det["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{det.get('display_name', category)} {det.get('confidence', 0.0):.2f}"
        draw_label(overlay, label, x1, y1, color)

    return overlay


class YoloPersistentWorker:
    def __init__(self, model_path: Path, config_path: Path, output_dir: Path):
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log(f"[LOAD MODEL] {self.model_path}")
        start = time.time()

        self.class_mapping = load_class_mapping(self.config_path)
        self.model = YOLO(str(self.model_path))

        log(f"[MODEL READY] elapsed={time.time() - start:.3f}s")

    def infer_one(
        self,
        request_id: str,
        image_path: Path,
        conf: float,
        save_overlay: bool,
        max_bbox_ratio: float,
        max_mask_ratio: float,
    ):
        start_total = time.time()

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        start_infer = time.time()

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )

        infer_sec = time.time() - start_infer

        result = results[0]
        names = result.names
        boxes = result.boxes
        masks = result.masks

        image_h, image_w = result.orig_shape
        image_area = image_h * image_w

        detections = []

        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                confidence = float(box.conf[0])

                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                xyxy = [round(float(v), 2) for v in xyxy]

                bbox_w = xyxy[2] - xyxy[0]
                bbox_h = xyxy[3] - xyxy[1]
                bbox_area = bbox_w * bbox_h

                if max_bbox_ratio > 0 and bbox_area > image_area * max_bbox_ratio:
                    log(
                        f"[FILTER] too large bbox: {cls_name}, "
                        f"ratio={bbox_area / image_area:.3f}"
                    )
                    continue

                polygon = None
                centroid = None
                mask_area = None

                if masks is not None and masks.xy is not None and i < len(masks.xy):
                    polygon_np = masks.xy[i]
                    polygon = [
                        [round(float(x), 2), round(float(y), 2)]
                        for x, y in polygon_np
                    ]

                    centroid = polygon_centroid(polygon_np)
                    mask_area = float(
                        cv2.contourArea(np.array(polygon_np, dtype=np.float32))
                    )

                    if max_mask_ratio > 0 and mask_area > image_area * max_mask_ratio:
                        log(
                            f"[FILTER] too large mask: {cls_name}, "
                            f"ratio={mask_area / image_area:.3f}"
                        )
                        continue

                mapped = self.class_mapping.get(cls_name)

                if mapped:
                    category = mapped["category"]
                    target_bin = mapped["target_bin"]
                    object_id = mapped["object_id"]
                    disturbance = mapped["disturbance"]
                else:
                    category = "unknown"
                    target_bin = "unknown"
                    object_id = "unknown"
                    disturbance = "none"

                display_name = CATEGORY_DISPLAY_NAME.get(category, category)

                detections.append({
                    "det_id": f"det_{len(detections) + 1:03d}",
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "raw_class_name": cls_name,
                    "display_name": display_name,
                    "category": category,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": xyxy,
                    "centroid_px": centroid,
                    "mask_area_px": round(mask_area, 2) if mask_area is not None else None,
                    "polygon": polygon,
                    "target_bin": target_bin,
                    "object_id": object_id,
                    "disturbance": disturbance,
                })

        summary = {
            "request_id": request_id,
            "image": str(image_path),
            "model": str(self.model_path),
            "config": str(self.config_path),
            "confidence_threshold": conf,
            "num_detections": len(detections),
            "detections": detections,
            "timing": {
                "infer_sec": round(infer_sec, 4),
                "total_sec": round(time.time() - start_total, 4),
            },
        }

        json_path = self.output_dir / "yolo_seg_offline_result.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        overlay_path = None

        if save_overlay:
            overlay = draw_overlay(image_path, detections)
            overlay_path = self.output_dir / "yolo_seg_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)

        return {
            "request_id": request_id,
            "ok": True,
            "json_path": str(json_path),
            "overlay_path": str(overlay_path) if overlay_path else None,
            "summary": summary,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--default-conf", type=float, default=0.60)
    args = parser.parse_args()

    worker = YoloPersistentWorker(
        model_path=Path(args.model),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
    )

    log("[WORKER READY] waiting JSON lines on stdin")

    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            request = json.loads(line)

            if request.get("cmd") == "stop":
                response = {
                    "request_id": request.get("request_id", "stop"),
                    "ok": True,
                    "message": "worker stopped",
                }
                print(json.dumps(response, ensure_ascii=False), flush=True)
                break

            request_id = request.get("request_id", f"req_{int(time.time() * 1000)}")
            image_path = Path(request["image"])
            conf = float(request.get("conf", args.default_conf))
            save_overlay = bool(request.get("save_overlay", False))

            max_bbox_ratio = float(request.get("max_bbox_ratio", 0.35))
            max_mask_ratio = float(request.get("max_mask_ratio", 0.30))

            response = worker.infer_one(
                request_id=request_id,
                image_path=image_path,
                conf=conf,
                save_overlay=save_overlay,
                max_bbox_ratio=max_bbox_ratio,
                max_mask_ratio=max_mask_ratio,
            )

        except Exception as e:
            response = {
                "request_id": "unknown",
                "ok": False,
                "error": repr(e),
            }

        print(json.dumps(response, ensure_ascii=False), flush=True)

    log("[WORKER EXIT]")


if __name__ == "__main__":
    main()