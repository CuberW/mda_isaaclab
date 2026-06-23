import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path.home() / "trashbot_ws"
DEFAULT_MODEL = ROOT / "models" / "best_seg.pt"
DEFAULT_CONFIG = ROOT / "config" / "trash_classes.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "logs"


# OpenCV 默认字体不稳定支持中文，所以 overlay 图中先用英文大类名
CATEGORY_DISPLAY_NAME = {
    "recyclable": "recyclable",
    "kitchen": "kitchen",
    "hazardous": "hazardous",
    "other": "other",
    "unknown": "unknown",
}

# BGR 颜色，用于不同大类显示
CATEGORY_COLOR = {
    "recyclable": (255, 80, 80),
    "kitchen": (80, 220, 80),
    "hazardous": (80, 80, 255),
    "other": (180, 80, 255),
    "unknown": (160, 160, 160),
}


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


def find_latest_image():
    image_root = ROOT / "data" / "images"
    sessions = sorted([p for p in image_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError("No image sessions found in ~/trashbot_ws/data/images")

    latest = sessions[-1]
    images = sorted(list(latest.glob("*.jpg")) + list(latest.glob("*.png")))
    images = [p for p in images if "preview" not in p.name.lower()]
    if not images:
        raise FileNotFoundError(f"No image files found in {latest}")

    return images[0]


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
    """
    在图像上绘制标签。
    注意：cv2.putText 对中文支持不好，所以这里使用英文大类名。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    label_x1 = int(x1)
    label_y1 = int(y1) - text_h - baseline - 6

    # 如果框太靠近上边界，就把标签画到框内下方
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


def draw_detection_overlay(image_path: Path, detections):
    """
    自定义绘制 overlay。
    不使用 result.plot()，避免自动显示 YOLO 原始小类名。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    overlay = image.copy()

    # 先画半透明 mask
    mask_layer = image.copy()
    for det in detections:
        category = det.get("category", "unknown")
        color = CATEGORY_COLOR.get(category, CATEGORY_COLOR["unknown"])

        polygon = det.get("polygon")
        if polygon is None or len(polygon) < 3:
            continue

        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_layer, [pts], color)

    alpha = 0.28
    overlay = cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)

    # 再画框、轮廓和大类标签
    for det in detections:
        category = det.get("category", "unknown")
        display_name = det.get("display_name", category)
        conf = det.get("confidence", 0.0)
        color = CATEGORY_COLOR.get(category, CATEGORY_COLOR["unknown"])

        x1, y1, x2, y2 = det["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        polygon = det.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

        label = f"{display_name} {conf:.2f}"
        draw_label(overlay, label, x1, y1, color)

        centroid = det.get("centroid_px")
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(overlay, (cx, cy), 3, color, -1)

    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model_path = Path(args.model)
    config_path = Path(args.config)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image) if args.image else find_latest_image()

    print(f"[MODEL] {model_path}")
    print(f"[IMAGE] {image_path}")

    class_mapping = load_class_mapping(config_path)
    model = YOLO(str(model_path))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        verbose=False,
    )

    result = results[0]
    names = result.names

    detections = []

    boxes = result.boxes
    masks = result.masks

    image_h, image_w = result.orig_shape
    image_area = image_h * image_w

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])

            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            xyxy = [round(float(v), 2) for v in xyxy]

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

            mapped = class_mapping.get(cls_name)

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

            bbox_w = xyxy[2] - xyxy[0]
            bbox_h = xyxy[3] - xyxy[1]
            bbox_area = bbox_w * bbox_h

            # 只保留原来的“大面积异常过滤”，不做小类白名单过滤
            if bbox_area > image_area * 0.35:
                print(
                    f"[FILTER] too large bbox: {cls_name}, "
                    f"category={category}, area_ratio={bbox_area / image_area:.2f}"
                )
                continue

            if mask_area is not None and mask_area > image_area * 0.30:
                print(
                    f"[FILTER] too large mask: {cls_name}, "
                    f"category={category}, area_ratio={mask_area / image_area:.2f}"
                )
                continue

            detections.append({
                "det_id": f"det_{len(detections) + 1:03d}",

                # YOLO 原始小类：保留在 JSON 中，方便调试和追溯
                "class_id": cls_id,
                "class_name": cls_name,
                "raw_class_name": cls_name,

                # 展示层只用大类
                "display_name": display_name,
                "category": category,

                "confidence": round(conf, 4),
                "bbox_xyxy": xyxy,
                "centroid_px": centroid,
                "mask_area_px": round(mask_area, 2) if mask_area is not None else None,
                "polygon": polygon,

                "target_bin": target_bin,
                "object_id": object_id,
                "disturbance": disturbance,
            })

    summary = {
        "image": str(image_path),
        "model": str(model_path),
        "config": str(config_path),
        "confidence_threshold": args.conf,
        "num_detections": len(detections),
        "display_rule": "Overlay displays garbage category only; raw YOLO class is kept in JSON.",
        "detections": detections,
    }

    json_path = output_dir / "yolo_seg_offline_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    overlay = draw_detection_overlay(image_path, detections)
    overlay_path = output_dir / "yolo_seg_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {overlay_path}")


if __name__ == "__main__":
    main()