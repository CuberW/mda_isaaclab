"""Live multi-panel debug view for robot task execution.

The panel is intentionally observer-only: it renders images and text from the
running pipeline, but never feeds data back into perception, planning, or
control. This keeps debugging visibility from changing task behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from robot_common.infra.logging import logger
from robot_common.infra.visualization import draw_detections

try:
    import cv2
    HAS_CV2 = True
except Exception:  # pragma: no cover - depends on local GUI/runtime
    cv2 = None
    HAS_CV2 = False


@dataclass
class LiveDebugPanel:
    """OpenCV-based runtime dashboard for perception/planning/control signals."""

    enabled: bool = False
    window_name: str = "3.19 live debug"
    size: tuple[int, int] = (1280, 720)
    last_status: dict[str, Any] = field(default_factory=dict)
    _warned: bool = False
    _closed: bool = False

    def close(self) -> None:
        if HAS_CV2 and self.enabled and not self._closed:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._closed = True

    def update(
        self,
        *,
        stage: str,
        env: Any | None = None,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        detections: Iterable[Any] | None = None,
        mask: np.ndarray | None = None,
        overview_camera: str = "overview_cam",
        status: dict[str, Any] | None = None,
    ) -> None:
        """Render one debug frame if enabled.

        Args:
            stage: Current module/state label.
            env: Optional MuJoCo env used to render an overview camera.
            rgb/depth/detections/mask: Latest perception outputs.
            status: Coordinates, ReKep waypoints, IK error, contact details, etc.
        """
        if not self.enabled or self._closed:
            return
        if not HAS_CV2:
            if not self._warned:
                logger.warning("OpenCV is not available; live debug panel disabled")
                self._warned = True
            self.enabled = False
            return

        try:
            merged = dict(self.last_status)
            if status:
                merged.update(status)
            merged["stage"] = stage
            self.last_status = merged

            w, h = self.size
            panel_w, panel_h = w // 2, h // 2
            det_img = self._rgb_panel(rgb, detections, (panel_w, panel_h), "camera / detections")
            aux_img = self._aux_panel(depth, mask, (panel_w, panel_h), "depth / mask")
            overview = self._overview_panel(env, overview_camera, (panel_w, panel_h))
            text = self._status_panel(merged, (panel_w, panel_h))
            canvas = np.vstack([np.hstack([det_img, aux_img]), np.hstack([overview, text])])
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.close()
        except Exception as exc:
            if not self._warned:
                logger.warning(f"Live debug panel disabled after render failure: {exc}")
                self._warned = True
            self.enabled = False

    def _rgb_panel(self, rgb, detections, size: tuple[int, int], title: str) -> np.ndarray:
        if rgb is None:
            img = self._blank(size, "no rgb frame")
        else:
            frame = np.asarray(rgb)
            if detections:
                frame = draw_detections(frame.astype(np.uint8), list(detections))
            img = self._resize_rgb(frame, size)
        return self._title(img, title)

    def _aux_panel(self, depth, mask, size: tuple[int, int], title: str) -> np.ndarray:
        if mask is not None:
            arr = np.asarray(mask).astype(np.uint8) * 255
            img = cv2.cvtColor(self._resize_gray(arr, size), cv2.COLOR_GRAY2BGR)
        elif depth is not None:
            d = np.asarray(depth, dtype=float)
            valid = np.isfinite(d)
            if np.any(valid):
                lo, hi = np.percentile(d[valid], [5, 95])
                denom = max(float(hi - lo), 1e-6)
                gray = np.clip((d - lo) / denom * 255.0, 0, 255).astype(np.uint8)
            else:
                gray = np.zeros(d.shape[:2], dtype=np.uint8)
            img = cv2.applyColorMap(self._resize_gray(gray, size), cv2.COLORMAP_TURBO)
        else:
            img = self._blank(size, "no depth / mask")
        return self._title(img, title)

    def _overview_panel(self, env, camera: str, size: tuple[int, int]) -> np.ndarray:
        if env is None:
            return self._title(self._blank(size, "no env"), "overview")
        try:
            return self._title(self._resize_rgb(env.render(camera), size), camera)
        except Exception:
            return self._title(self._blank(size, f"{camera} unavailable"), camera)

    def _status_panel(self, status: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = (22, 24, 28)
        lines = self._status_lines(status)
        y = 28
        for i, line in enumerate(lines[:18]):
            color = (235, 235, 235)
            if i == 0:
                color = (80, 220, 255)
            elif "fail" in line.lower() or "false" in line.lower():
                color = (80, 120, 255)
            elif "ok" in line.lower() or "true" in line.lower():
                color = (120, 230, 120)
            cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y += 26
        return img

    def _status_lines(self, status: dict[str, Any]) -> list[str]:
        order = [
            "stage", "target_body", "detected_class", "category", "object_world",
            "grasp_world", "rekep_hover", "rekep_grasp", "rekep_lift",
            "base_goal", "ik_source", "pregrasp_ik_error", "grasp_ik_error",
            "motion_source", "finger_span", "pinch_quality", "contact_bodies",
            "delivery", "failure",
        ]
        lines = []
        for key in order:
            if key in status:
                lines.append(f"{key}: {self._fmt(status[key])}")
        for key in sorted(k for k in status if k not in order):
            lines.append(f"{key}: {self._fmt(status[key])}")
        return lines or ["stage: waiting"]

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=3, suppress_small=True)
        if isinstance(value, (list, tuple)):
            if len(value) <= 6 and all(isinstance(x, (int, float, np.floating)) for x in value):
                return "[" + ", ".join(f"{float(x):.3f}" for x in value) + "]"
            return str(value)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        frame = np.asarray(rgb)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = cv2.resize(frame[:, :, :3], size)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _resize_gray(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(gray)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.resize(arr, size)

    @staticmethod
    def _blank(size: tuple[int, int], text: str) -> np.ndarray:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[:] = (35, 35, 35)
        cv2.putText(img, text, (24, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _title(img: np.ndarray, title: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out
