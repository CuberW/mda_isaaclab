"""
GroundingDINO detector - open vocabulary object detection.

Uses HuggingFace transformers pipeline for GroundingDINO.
Supports text-prompted zero-shot detection.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from robot_common.infra.config import MODELS_DIR
from robot_common.infra.logging import logger


class GroundingDINODetector:
    """Open-vocabulary object detection with GroundingDINO."""

    def __init__(self, model_path: str = "",
                 confidence_threshold: float = 0.3,
                 device: str = "cuda"):
        self.model_path = model_path or str(MODELS_DIR / "grounding-dino-base")
        self.confidence_threshold = confidence_threshold
        self.device = device if self._cuda_available() else "cpu"
        self._model = None
        self._processor = None

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
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch

            model_id = "IDEA-Research/grounding-dino-base"
            # Try local first
            local_path = Path(self.model_path)
            if local_path.exists() and (local_path / "config.json").exists():
                model_id = str(local_path)

            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self._model.eval()
            logger.info(f"GroundingDINO loaded: {model_id}")
        except ImportError:
            logger.error("transformers not installed. pip install transformers")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")

    def detect(self, image: np.ndarray,
               text_prompts: List[str] = None) -> List:
        """Run GroundingDINO detection with text prompt.

        Args:
            image: RGB image (H, W, 3), uint8
            text_prompts: List of text queries, e.g. ["banana", "apple"]

        Returns:
            List of DetectionResult
        """
        self._load_model()
        if self._model is None or self._processor is None:
            logger.error("GroundingDINO model not loaded")
            return []

        from robot_common.perception import DetectionResult
        import torch
        from PIL import Image

        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Build text prompt
        if not text_prompts:
            text_prompts = ["object"]
        text = ". ".join(text_prompts) + "."

        # Inference
        inputs = self._processor(images=pil_image, text=text, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Parse results (handle different API versions)
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]],
            )
        except TypeError:
            # Older transformers API
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                target_sizes=[pil_image.size[::-1]],
            )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
                score_val = float(score.cpu() if hasattr(score, 'cpu') else score)
                bbox = (int(box_np[0]), int(box_np[1]), int(box_np[2]), int(box_np[3]))
                detections.append(DetectionResult(
                    class_name=str(label),
                    confidence=score_val,
                    bbox=bbox,
                    id=i,
                ))

        return detections
