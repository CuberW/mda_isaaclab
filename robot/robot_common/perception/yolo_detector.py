"""
YOLO-based object detectors: YOLOv8 (closed-set) and YOLO-World (open-vocabulary).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from robot_common.infra.config import PROJECT_ROOT
from robot_common.infra.logging import logger


class YOLODetector:
    """YOLOv8 detector for closed-set detection (garbage classification, etc.)."""

    def __init__(self, model_name: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "cuda"):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device if self._cuda_available() else "cpu"
        self._model = None

        from robot_common.perception import DetectionResult

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            model_path = PROJECT_ROOT / self.model_name
            if not model_path.exists():
                logger.info(f"Downloading YOLO model: {self.model_name}")
            self._model = YOLO(str(model_path))
            logger.info(f"YOLOv8 model loaded: {self.model_name}")
        except ImportError:
            logger.error("ultralytics not installed. pip install ultralytics")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model = None

    def detect(self, image: np.ndarray,
               text_prompts: List[str] = None) -> List:
        """Run YOLOv8 detection. text_prompts ignored for closed-set."""
        self._load_model()
        if self._model is None:
            return []

        from robot_common.perception import DetectionResult

        results = self._model(image, conf=self.confidence_threshold, verbose=False)
        detections = []

        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

                    detections.append(DetectionResult(
                        class_name=cls_name, confidence=conf,
                        bbox=bbox, id=i,
                    ))

        return detections


class YOLOWorldDetector:
    """YOLO-World detector for open-vocabulary detection."""

    def __init__(self, model_path: str = "",
                 confidence_threshold: float = 0.3,
                 device: str = "cuda"):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device if self._cuda_available() else "cpu"
        self._model = None

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOWorld
            model_file = PROJECT_ROOT / "yolov8s-worldv2.pt"
            if model_file.exists():
                self._model = YOLOWorld(str(model_file))
            elif self.model_path:
                self._model = YOLOWorld(self.model_path)
            else:
                self._model = YOLOWorld("yolov8s-worldv2.pt")
            logger.info("YOLO-World model loaded")
        except ImportError:
            logger.error("ultralytics not installed")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO-World: {e}")
            self._model = None

    def detect(self, image: np.ndarray,
               text_prompts: List[str] = None) -> List:
        """Run YOLO-World detection with text prompts."""
        self._load_model()
        if self._model is None:
            return []

        from robot_common.perception import DetectionResult

        if text_prompts:
            self._model.set_classes(text_prompts)

        results = self._model(image, conf=self.confidence_threshold, verbose=False)
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = text_prompts[cls_id] if text_prompts and cls_id < len(text_prompts) else f"obj_{cls_id}"
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                detections.append(DetectionResult(
                    class_name=cls_name, confidence=conf,
                    bbox=bbox, id=i,
                ))

        return detections
