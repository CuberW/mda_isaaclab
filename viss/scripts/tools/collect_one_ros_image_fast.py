import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


ROOT = Path.home() / "trashbot_ws"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "images"


def ros_image_to_cv2(msg: Image):
    """
    不依赖 cv_bridge，直接把 sensor_msgs/Image 转成 OpenCV BGR 图像。
    支持 rgb8 / bgr8 / rgba8 / bgra8 / mono8 / 8UC3。
    """
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ["rgb8", "bgr8", "8uc3"]:
        channels = 3
        rows = data.reshape((height, step))
        img = rows[:, :width * channels].reshape((height, width, channels))

        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img.copy()

    if encoding in ["rgba8", "bgra8"]:
        channels = 4
        rows = data.reshape((height, step))
        img = rows[:, :width * channels].reshape((height, width, channels))

        if encoding == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img.copy()

    if encoding in ["mono8", "8uc1"]:
        channels = 1
        rows = data.reshape((height, step))
        img = rows[:, :width * channels].reshape((height, width))
        return img.copy()

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class OneImageCollector(Node):
    def __init__(self, topic: str, skip_frames: int = 5):
        super().__init__("trashbot_one_image_collector")
        self.topic = topic
        self.received_msg = None
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10,
        )

    def image_callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count > self.skip_frames:
            self.received_msg = msg


def save_metadata(session_dir: Path, image_path: Path, topic: str, msg: Image, elapsed_sec: float):
    metadata_path = session_dir / "metadata.csv"

    with open(metadata_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "topic",
                "width",
                "height",
                "encoding",
                "step",
                "elapsed_sec",
                "timestamp",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "filename": image_path.name,
            "topic": topic,
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": msg.encoding,
            "step": int(msg.step),
            "elapsed_sec": round(elapsed_sec, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/trashbot/camera/rgb")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--skip-frames", type=int, default=5)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S_fast")
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    image_path = session_dir / "frame_000001.jpg"

    rclpy.init(args=None)
    node = OneImageCollector(args.topic, skip_frames=args.skip_frames)

    start_time = time.time()

    try:
        while rclpy.ok():
            elapsed = time.time() - start_time

            if elapsed > args.timeout:
                raise TimeoutError(
                    f"Timeout waiting for image on topic {args.topic}, timeout={args.timeout}s"
                )

            rclpy.spin_once(node, timeout_sec=0.05)

            if node.received_msg is not None:
                msg = node.received_msg
                image = ros_image_to_cv2(msg)

                ok = cv2.imwrite(
                    str(image_path),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
                )

                if not ok:
                    raise RuntimeError(f"Failed to save image: {image_path}")

                save_metadata(
                    session_dir=session_dir,
                    image_path=image_path,
                    topic=args.topic,
                    msg=msg,
                    elapsed_sec=time.time() - start_time,
                )

                print(f"[SAVED_IMAGE] {image_path}")
                print(f"[ELAPSED_SEC] {time.time() - start_time:.4f}")
                return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)