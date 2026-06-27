#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Path configuration
VIEW_CMD_JSON = os.environ.get("VIEW_CMD_JSON", "isaac_projects/v26_view_command.json")
VIEW_RESULT_JSON = os.environ.get("VIEW_RESULT_JSON", "isaac_projects/v26_view_result.json")
VISUAL_PLAN_JSON = os.environ.get("VISUAL_PLAN_JSON", "isaac_projects/v2_visual_task_plan.json")
ACTION_RESULT_JSON = os.environ.get("ACTION_RESULT_JSON", "isaac_projects/v27_head_camera_action_result.json")
DUMMY_IMAGE_PATH = os.environ.get("DUMMY_IMAGE_PATH", "data/images/mock_frame.jpg")

class MockIsaacSim(Node):
    def __init__(self):
        super().__init__("mock_isaac_sim")
        self.publisher_ = self.create_publisher(Image, "/trashbot/camera/rgb", 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10 Hz
        
        self.last_view_cmd_id = None
        self.last_action_plan_id = None
        
        # Load mock image
        if os.path.exists(DUMMY_IMAGE_PATH):
            self.get_logger().info(f"Loaded dummy image: {DUMMY_IMAGE_PATH}")
            self.cv_image = cv2.imread(DUMMY_IMAGE_PATH)
        else:
            self.get_logger().warn(f"Dummy image not found: {DUMMY_IMAGE_PATH}, generating random image...")
            self.cv_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
        self.image_msg = self.cv2_to_imgmsg(self.cv_image)
        self.get_logger().info("Mock Isaac Sim running...")

    def cv2_to_imgmsg(self, cv_img):
        # Convert OpenCV BGR to Image Message
        msg = Image()
        msg.height = cv_img.shape[0]
        msg.width = cv_img.shape[1]
        msg.encoding = "bgr8"
        msg.step = cv_img.shape[1] * 3
        msg.data = cv_img.tobytes()
        msg.header = Header()
        return msg

    def timer_callback(self):
        self.image_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.image_msg)
        
        # 1. Watch view command
        if os.path.exists(VIEW_CMD_JSON):
            try:
                with open(VIEW_CMD_JSON, 'r', encoding='utf-8') as f:
                    cmd_data = json.load(f)
                
                command_id = cmd_data.get("command_id")
                view_id = cmd_data.get("view_id")
                
                if command_id and command_id != self.last_view_cmd_id:
                    self.last_view_cmd_id = command_id
                    self.get_logger().info(f"[MOCK VIEW] Received view command: {view_id} (ID: {command_id})")
                    
                    # Remove old result if any to trigger fresh wait
                    if os.path.exists(VIEW_RESULT_JSON):
                        try:
                            os.remove(VIEW_RESULT_JSON)
                        except Exception:
                            pass
                            
                    time.sleep(0.5)
                    result_data = {
                        "status": "success",
                        "command_id": command_id,
                        "view_id": view_id
                    }
                    with open(VIEW_RESULT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f)
                    self.get_logger().info(f"[MOCK VIEW] Wrote view result: {VIEW_RESULT_JSON}")
            except Exception as e:
                self.get_logger().error(f"Error handling view command: {e}")

        # 2. Watch action plan
        if os.path.exists(VISUAL_PLAN_JSON):
            try:
                with open(VISUAL_PLAN_JSON, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)
                
                plan_id = plan_data.get("plan_id")
                selected_task = plan_data.get("selected_task")
                
                if plan_id and plan_id != self.last_action_plan_id:
                    self.last_action_plan_id = plan_id
                    raw_name = selected_task.get("raw_class_name") if selected_task else "unknown"
                    self.get_logger().info(f"[MOCK ACTION] Received pick plan for: {raw_name} (ID: {plan_id})")
                    
                    # Remove old action result
                    if os.path.exists(ACTION_RESULT_JSON):
                        try:
                            os.remove(ACTION_RESULT_JSON)
                        except Exception:
                            pass
                            
                    time.sleep(1.0)
                    action_data = {
                        "status": "success",
                        "plan_id": plan_id,
                        "action": "grasp",
                        "target": raw_name
                    }
                    with open(ACTION_RESULT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(action_data, f)
                    self.get_logger().info(f"[MOCK ACTION] Wrote action result: {ACTION_RESULT_JSON}")
            except Exception as e:
                self.get_logger().error(f"Error handling action command: {e}")

def main(args=None):
    rclpy.init(args=args)
    mock_sim = MockIsaacSim()
    # Clean up preexisting status files to avoid false positives
    for path in [VIEW_CMD_JSON, VIEW_RESULT_JSON, VISUAL_PLAN_JSON, ACTION_RESULT_JSON]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up preexisting file: {path}")
            
    try:
        rclpy.spin(mock_sim)
    except KeyboardInterrupt:
        pass
    finally:
        mock_sim.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
