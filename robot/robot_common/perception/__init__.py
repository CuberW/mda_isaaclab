"""
Perception Hub - Unified interface for detection, segmentation, 3D localization.

Supports:
  - Open-vocabulary detection: GroundingDINO, YOLO-World, OWL-ViT
  - Closed-set detection: YOLOv8
  - Instance segmentation: SAM, SAM2
  - 3D localization: RGB-D → point cloud → 3D coordinates
  - Multi-camera management with TF transforms
  - Classification: CLIP, VLM
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from robot_common.infra.config import PerceptionConfig, CameraConfig, MODELS_DIR
from robot_common.infra.logging import logger


# ── Object detection result ──────────────────────────────────
class DetectionResult:
    """Single detection result."""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int],
                 mask: Optional[np.ndarray] = None, position_3d: Optional[np.ndarray] = None,
                 id: int = 0):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox          # (x1, y1, x2, y2) in pixels
        self.mask = mask          # Binary mask (H, W)
        self.position_3d = position_3d  # (x, y, z) in world/camera frame
        self.id = id

    def __repr__(self):
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"bbox={self.bbox}, pos={self.position_3d})")


# ── Perception Hub ───────────────────────────────────────────
class PerceptionHub:
    """Unified perception interface. Switches between detector types."""

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.detector = None
        self.segmentor = None
        self.classifier = None
        self._init_detector()
        self._init_segmentor()
        self._init_classifier()
        logger.info(f"PerceptionHub initialized: detector={config.detector}, "
                    f"segmentor={config.segmentor}")

    def _init_detector(self):
        """Initialize object detector based on config."""
        det = self.config.detector
        if det == "yolov8":
            from robot_common.perception.yolo_detector import YOLODetector
            self.detector = YOLODetector(
                model_name="yolov8n.pt",
                confidence_threshold=self.config.confidence_threshold,
            )
        elif det == "yolo_world":
            from robot_common.perception.yolo_detector import YOLOWorldDetector
            self.detector = YOLOWorldDetector(
                model_path=str(MODELS_DIR / "yolo-world-light"),
                confidence_threshold=self.config.confidence_threshold,
            )
        elif det == "grounding_dino":
            from robot_common.perception.grounding_dino import GroundingDINODetector
            self.detector = GroundingDINODetector(
                model_path=str(MODELS_DIR / "grounding-dino-base"),
                confidence_threshold=self.config.confidence_threshold,
            )
        else:
            logger.warning(f"Unknown detector: {det}, no detector loaded")

    def _init_segmentor(self):
        """Initialize segmentor."""
        seg = self.config.segmentor
        if seg == "sam":
            from robot_common.perception.sam_segmentor import SAMSegmentor
            self.segmentor = SAMSegmentor(
                model_path=str(MODELS_DIR / "sam-vit-b"),
                allow_bbox_fallback=False,
            )
        elif seg == "sam2":
            logger.info("SAM2 configured (lazy load)")
        elif seg == "none":
            pass
        else:
            logger.warning(f"Unknown segmentor: {seg}")

    def _init_classifier(self):
        """Initialize classifier for second-pass confirmation."""
        from robot_common.perception.classifier import CLIPClassifier
        try:
            self.classifier = CLIPClassifier()
        except Exception as e:
            logger.warning(f"CLIP classifier not available: {e}")

    def detect(self, image: np.ndarray, text_prompts: List[str] = None,
               depth: np.ndarray = None) -> List[DetectionResult]:
        """Run detection on image. Returns list of DetectionResult."""
        if self.detector is None:
            logger.error("No detector loaded")
            return []

        results = self.detector.detect(image, text_prompts or [])

        # Compute 3D positions if depth is available
        if depth is not None:
            for r in results:
                if r.bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in r.bbox]
                    h, w = depth.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    roi = depth[y1:y2, x1:x2]
                    valid = np.isfinite(roi) & (roi > 1e-4)
                    if r.mask is not None and np.asarray(r.mask).shape == depth.shape:
                        valid &= np.asarray(r.mask[y1:y2, x1:x2]).astype(bool)
                    if not np.any(valid):
                        continue
                    ys, xs = np.nonzero(valid)
                    z_vals = roi[valid]
                    z_med = float(np.median(z_vals))
                    keep = np.abs(z_vals - z_med) < max(0.03, 0.08 * z_med)
                    if np.any(keep):
                        xs = xs[keep]
                        ys = ys[keep]
                        z_vals = z_vals[keep]
                    cx = float(x1 + np.mean(xs))
                    cy = float(y1 + np.mean(ys))
                    z = float(np.median(z_vals))
                    fx = fy = 500.0
                    img_cx = depth.shape[1] / 2
                    img_cy = depth.shape[0] / 2
                    x = (cx - img_cx) * z / fx
                    y = (cy - img_cy) * z / fy
                    r.position_3d = np.array([x, y, z], dtype=float)

        return results

    def segment(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Run segmentation on image region."""
        if self.segmentor is None:
            return None
        return self.segmentor.segment(image, bbox)

    def classify(self, image_crop: np.ndarray, candidate_labels: List[str]) -> Tuple[str, float]:
        """Second-pass classification of an image crop."""
        if self.classifier is None:
            return candidate_labels[0] if candidate_labels else ("unknown", 0.0)
        return self.classifier.classify(image_crop, candidate_labels)

    def detect_and_segment(self, image: np.ndarray, text_prompts: List[str],
                           depth: np.ndarray = None) -> List[DetectionResult]:
        """Full pipeline: detect + segment + 3D localize."""
        results = self.detect(image, text_prompts, depth)
        if self.segmentor and results:
            for r in results:
                if r.bbox is not None:
                    r.mask = self.segment(image, r.bbox)
            if depth is not None:
                self._localize_3d(results, depth)
        return results

    def _localize_3d(self, results: List[DetectionResult], depth: np.ndarray):
        """Recompute 3D positions after masks are available."""
        for r in results:
            if r.bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            h, w = depth.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = depth[y1:y2, x1:x2]
            valid = np.isfinite(roi) & (roi > 1e-4)
            if r.mask is not None and np.asarray(r.mask).shape == depth.shape:
                valid &= np.asarray(r.mask[y1:y2, x1:x2]).astype(bool)
            if not np.any(valid):
                continue
            ys, xs = np.nonzero(valid)
            z_vals = roi[valid]
            z_med = float(np.median(z_vals))
            keep = np.abs(z_vals - z_med) < max(0.03, 0.08 * z_med)
            if np.any(keep):
                xs = xs[keep]
                ys = ys[keep]
                z_vals = z_vals[keep]
            cx = float(x1 + np.mean(xs))
            cy = float(y1 + np.mean(ys))
            z = float(np.median(z_vals))
            fx = fy = 500.0
            img_cx = depth.shape[1] / 2
            img_cy = depth.shape[0] / 2
            r.position_3d = np.array([
                (cx - img_cx) * z / fx,
                (cy - img_cy) * z / fy,
                z,
            ], dtype=float)
