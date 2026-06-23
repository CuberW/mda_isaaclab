#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np


YOLO_JSON = Path.home() / "trashbot_ws/data/logs/yolo_seg_offline_result.json"
OUT_IMG = Path.home() / "trashbot_ws/data/logs/yoloe_seg_overlay_latest.jpg"
OUT_TXT = Path.home() / "trashbot_ws/data/logs/yoloe_seg_overlay_latest.txt"


def color_for_name(name):
    table = {
        "battery": (0, 0, 255),
        "drugbox": (255, 0, 255),
        "bottle": (255, 128, 0),
        "bottle2": (255, 128, 0),
        "can": (0, 255, 255),
        "paper": (255, 255, 0),
        "papercup": (0, 255, 0),
        "potato": (128, 255, 0),
        "potatocut": (128, 255, 0),
    }
    return table.get(str(name), (255, 255, 255))


def draw_transparent_poly(img, pts, color, alpha=0.35):
    if len(pts) < 3:
        return img

    overlay = img.copy()
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts_np], color)
    cv2.polylines(overlay, [pts_np], True, color, 2)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def main():
    if not YOLO_JSON.exists():
        raise FileNotFoundError(YOLO_JSON)

    data = json.load(open(YOLO_JSON, "r", encoding="utf-8"))

    image_path = (
        data.get("image")
        or data.get("image_path")
        or data.get("source_image")
    )

    if not image_path:
        raise RuntimeError("YOLO json has no image/image_path/source_image")

    image_path = Path(image_path).expanduser()

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    lines = []
    lines.append(f"json: {YOLO_JSON}")
    lines.append(f"image: {image_path}")
    lines.append(f"out: {OUT_IMG}")
    lines.append(f"num_detections: {len(data.get('detections', []))}")
    lines.append("")

    for i, d in enumerate(data.get("detections", []), 1):
        raw = d.get("raw_class_name", "unknown")
        label = d.get("prompt_label", raw)
        conf = float(d.get("confidence", 0.0))
        bbox = d.get("bbox_xyxy")
        centroid = d.get("centroid_px")
        polygon = d.get("polygon", [])

        color = color_for_name(raw)

        # 画 mask polygon
        if polygon and len(polygon) >= 3:
            pts = [[int(round(x)), int(round(y))] for x, y in polygon]
            img = draw_transparent_poly(img, pts, color, alpha=0.35)

        # 画 bbox
        if bbox:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            text = f"{i}:{raw} {conf:.2f}"
            cv2.putText(
                img,
                text,
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

        # 画中心点
        if centroid:
            u, v = [int(round(x)) for x in centroid]
            cv2.circle(img, (u, v), 4, color, -1)
            cv2.circle(img, (u, v), 7, color, 1)

        lines.append(
            f"{i}. raw={raw}, prompt={label}, conf={conf:.4f}, "
            f"bbox={bbox}, centroid={centroid}, polygon_pts={len(polygon)}"
        )

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_IMG), img)

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("[SAVED_IMAGE]", OUT_IMG)
    print("[SAVED_TXT]", OUT_TXT)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
