"""Debug artifact writers for perception, frame, and trajectory inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from robot_common.infra.logging import logger
from robot_common.infra.visualization import draw_detections

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _jsonable(value: Any) -> Any:
    """Convert numpy-heavy values into JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class EpisodeDebugWriter:
    """Writes one folder of debug artifacts for a task episode."""

    def __init__(self, task_id: str, episode_id: int, enabled: bool = True,
                 root: str | Path = "results/debug"):
        self.enabled = enabled
        self.task_id = task_id
        self.episode_id = int(episode_id)
        self.root = Path(root) / f"task_{task_id}_ep{self.episode_id}"
        self.events: dict[str, Any] = {}
        self.images: list[str] = []
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def save_image(self, name: str, image: np.ndarray,
                   detections: Optional[Iterable[Any]] = None) -> Optional[Path]:
        """Save an RGB image, optionally with detection overlays."""
        if not self.enabled or not HAS_PIL:
            return None
        img = np.asarray(image)
        if img.size == 0:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if detections is not None:
            img = draw_detections(img, list(detections))
        path = self.root / f"{name}.png"
        Image.fromarray(img).save(path)
        self.images.append(path.name)
        return path

    def add(self, key: str, value: Any) -> None:
        if self.enabled:
            self.events[key] = _jsonable(value)

    def append(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.events.setdefault(key, [])
        self.events[key].append(_jsonable(value))

    def camera_snapshot(self, env, camera_name: str) -> dict:
        """Capture camera intrinsics/extrinsics and basic frame sanity data."""
        pos, rot = env.get_camera_pose(camera_name)
        intr = env.get_camera_intrinsics(camera_name)
        sample_world = pos + rot @ np.array([0.0, 0.0, -1.0])
        round_trip = env.camera_point_to_world(
            env.world_point_to_camera(sample_world, camera_name),
            camera_name,
        )
        return {
            "camera": camera_name,
            "position_world": pos,
            "rotation_world_from_mujoco_camera": rot,
            "intrinsics": intr,
            "round_trip_error_m": float(np.linalg.norm(round_trip - sample_world)),
        }

    def detections_snapshot(self, env, camera_name: str, detections: Iterable[Any],
                            body_aliases: Optional[dict[str, str]] = None) -> list[dict]:
        """Serialize detections with camera/world coordinates and body errors."""
        rows = []
        for det in detections:
            row = {
                "label": getattr(det, "class_name", ""),
                "confidence": float(getattr(det, "confidence", 0.0) or 0.0),
                "bbox": list(getattr(det, "bbox", []) or []),
            }
            p_cam = getattr(det, "position_3d", None)
            if p_cam is not None:
                p_cam = np.asarray(p_cam, dtype=float)
                p_world = env.camera_point_to_world(p_cam, camera_name)
                row["point_camera"] = p_cam
                row["point_world"] = p_world
                row["projected_pixel"] = env.project_world_point(p_world, camera_name)["pixel"]
                if body_aliases:
                    body = body_aliases.get(row["label"], "")
                    row["matched_body"] = body
                    if body in getattr(env, "_body_names", []):
                        body_pos = env.get_body_position(body)
                        row["body_world"] = body_pos
                        row["body_error_m"] = float(np.linalg.norm(p_world - body_pos))
            rows.append(_jsonable(row))
        return rows

    def write(self) -> Optional[Path]:
        if not self.enabled:
            return None
        path = self.root / "debug.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(self.events), f, indent=2, ensure_ascii=False)
        readme = self.root / "README.txt"
        lines = [
            f"Task {self.task_id} episode {self.episode_id}",
            "",
            "Images:",
        ]
        lines.extend(f"- {name}" for name in self.images)
        lines.extend([
            "",
            "Debug keys:",
        ])
        lines.extend(f"- {key}" for key in sorted(self.events.keys()))
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Debug artifacts saved: {path}")
        return path
