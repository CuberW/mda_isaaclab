"""
CLIP-based classifier for second-pass confirmation of detections.

Used in Task 3.19: after YOLOv8 detects garbage, CLIP confirms the category.
"""

from typing import List, Tuple, Optional

import numpy as np

from robot_common.infra.logging import logger


class CLIPClassifier:
    """CLIP-based zero-shot classifier for image crops."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device if self._cuda_available() else "cpu"
        self._model = None
        self._processor = None
        self._labels_cache = None
        self._label_embeddings = None

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
            from transformers import CLIPProcessor, CLIPModel
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self._model.eval()
            logger.info(f"CLIP loaded: {self.model_name}")
        except ImportError:
            logger.error("transformers not installed")
        except Exception as e:
            logger.warning(f"CLIP not available: {e}")
            self._model = None
            self._processor = None

    def classify(self, image: np.ndarray,
                 candidate_labels: List[str]) -> Tuple[str, float]:
        """Classify image against candidate labels.

        Args:
            image: RGB image crop (H, W, 3), uint8
            candidate_labels: List of category names

        Returns:
            (best_label, confidence)
        """
        if not candidate_labels:
            return ("unknown", 0.0)

        self._load_model()
        if self._model is None:
            return (candidate_labels[0], 0.5)

        try:
            import torch
            from PIL import Image

            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            inputs = self._processor(
                text=candidate_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits_per_image = outputs.logits_per_image  # (1, n_labels)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            best_idx = int(probs.argmax())
            return (candidate_labels[best_idx], float(probs[best_idx]))

        except Exception as e:
            logger.warning(f"CLIP classification failed: {e}")
            return (candidate_labels[0], 0.5)

    def encode_text(self, labels: List[str]) -> Optional[np.ndarray]:
        """Pre-compute text embeddings for fast classification."""
        self._load_model()
        if self._model is None:
            return None

        try:
            import torch
            inputs = self._processor(text=labels, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            return text_features.cpu().numpy()
        except Exception as e:
            logger.warning(f"CLIP text encoding failed: {e}")
            return None
