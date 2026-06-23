import csv
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
from pathlib import Path


class CameraDatasetCollector(Node):
    def __init__(self):
        super().__init__("camera_dataset_collector")

        self.topic = "/trashbot/camera/rgb"
        self.bridge = CvBridge()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path.home() / "trashbot_ws" / "data" / "images" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "metadata.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["saved_id", "filename", "ros_sec", "ros_nanosec"])
        self.csv_file.flush()

        self.saved_count = 0
        self.max_saved = 20

        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.callback,
            10
        )

        self.get_logger().info(f"Listening: {self.topic}")
        self.get_logger().info(f"Saving to: {self.output_dir}")

    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            self.saved_count += 1
            filename = f"frame_{self.saved_count:06d}.png"
            path = self.output_dir / filename

            cv2.imwrite(str(path), image)

            self.writer.writerow([
                self.saved_count,
                filename,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec
            ])
            self.csv_file.flush()

            self.get_logger().info(f"Saved {filename}")

            if self.saved_count >= self.max_saved:
                self.get_logger().info("Collection finished.")
                self.csv_file.close()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(str(e))


def main():
    rclpy.init()
    node = CameraDatasetCollector()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
