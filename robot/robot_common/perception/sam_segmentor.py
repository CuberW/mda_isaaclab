"""
SAM (Segment Anything Model) segmentor for instance segmentation.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from robot_common.infra.config import MODELS_DIR
from robot_common.infra.logging import logger


class SAMSegmentor:
    """Segment Anything Model for instance segmentation from bounding boxes."""

    def __init__(self, model_path: str = "", device: str = "cuda",
                 allow_bbox_fallback: bool = False):
        self.model_path = model_path or str(MODELS_DIR / "sam-vit-b")
        self.device = device if self._cuda_available() else "cpu"
        self.allow_bbox_fallback = allow_bbox_fallback
        self._model = None
        self._processor = None
        self._predictor = None
        self._backend = ""

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
            import torch

            local_path = Path(self.model_path)
            checkpoint = local_path / "sam_vit_b_01ec64.pth"
            if checkpoint.exists():
                from segment_anything import SamPredictor, sam_model_registry

                self._model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
                if self.device == "cuda":
                    self._model = self._model.to("cuda")
                self._model.eval()
                self._predictor = SamPredictor(self._model)
                self._backend = "segment_anything"
                logger.info(f"SAM loaded: {checkpoint}")
                return

            from transformers import SamModel, SamProcessor

            model_id = "facebook/sam-vit-base"
            if local_path.exists() and (local_path / "config.json").exists():
                model_id = str(local_path)

            self._model = SamModel.from_pretrained(model_id)
            self._processor = SamProcessor.from_pretrained(model_id)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self._model.eval()
            self._backend = "transformers"
            logger.info(f"SAM loaded: {model_id}")
        except ImportError:
            logger.error("transformers not installed")
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")

    def _bbox_mask(self, image: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return mask

    def segment(self, image: np.ndarray,
                bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Segment object from bounding box.

        Args:
            image: RGB image (H, W, 3)
            bbox: (x1, y1, x2, y2)

        Returns:
            Binary mask (H, W)
        """
        self._load_model()
        if self._model is None:
            if self.allow_bbox_fallback:
                return self._bbox_mask(image, bbox)
            raise RuntimeError("SAM model is required; bbox-mask fallback is disabled")

        try:
            from PIL import Image

            if self._backend == "segment_anything":
                self._predictor.set_image(image)
                box = np.asarray(bbox, dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=box,
                    multimask_output=True,
                )
                if len(masks) == 0:
                    return None
                best_idx = int(np.argmax(scores))
                return masks[best_idx].astype(np.uint8)

            import torch

            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Prepare input box in SAM format [x1, y1, x2, y2]
            input_boxes = [[[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]]]

            inputs = self._processor(pil_image, input_boxes=input_boxes, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )[0]

            if len(masks) > 0:
                mask = masks[0, 0].cpu().numpy() > 0.5
                return mask.astype(np.uint8)

        except Exception as e:
            if self.allow_bbox_fallback:
                logger.warning(f"SAM segmentation failed, using bbox fallback: {e}")
                return self._bbox_mask(image, bbox)
            raise RuntimeError(f"SAM segmentation failed: {e}") from e

        return None

    def segment_from_points(self, image: np.ndarray,
                            points: list, labels: list) -> Optional[np.ndarray]:
        """Segment from point prompts."""
        self._load_model()
        if self._model is None:
            return None

        try:
            import torch
            from PIL import Image

            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            input_points = [[points]]
            input_labels = [[labels]]

            inputs = self._processor(pil_image,
                                     input_points=input_points,
                                     input_labels=input_labels,
                                     return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )[0]

            if len(masks) > 0:
                # Return highest confidence mask
                iou_scores = outputs.iou_scores[0, 0]
                best_idx = iou_scores.argmax().item() if iou_scores.numel() > 0 else 0
                mask = masks[best_idx, 0].cpu().numpy() > 0.5
                return mask.astype(np.uint8)

        except Exception as e:
            logger.warning(f"SAM point segmentation failed: {e}")

        return None
