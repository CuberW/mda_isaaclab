"""Optional wrapper around the AnyGrasp SDK.

The SDK is licensed and ships binary modules, so this wrapper keeps imports lazy
and fails with actionable errors when the local checkout is incomplete.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates


IDENTITY_GRASP_TO_TCP = np.eye(4, dtype=np.float32)


def parse_homogeneous_matrix(value: str | None, *, name: str) -> np.ndarray:
    if value is None or not str(value).strip():
        matrix = IDENTITY_GRASP_TO_TCP.copy()
    else:
        parts = [float(item.strip()) for item in str(value).split(",") if item.strip()]
        if len(parts) != 16:
            raise ValueError(f"{name} must contain 16 comma-separated row-major values.")
        matrix = np.asarray(parts, dtype=np.float32).reshape(4, 4)
    validate_homogeneous_matrix(matrix, name=name)
    return matrix.astype(np.float32)


def validate_homogeneous_matrix(matrix: np.ndarray, *, name: str) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    orth_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float32)))
    det = float(np.linalg.det(rotation))
    if orth_error > 1e-3:
        raise ValueError(f"{name} rotation is not orthonormal enough; error={orth_error:.6f}.")
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"{name} rotation determinant must be +1; det={det:.6f}.")
    translation_norm = float(np.linalg.norm(matrix[:3, 3]))
    if translation_norm > 0.25:
        raise ValueError(f"{name} translation is unexpectedly large for a gripper-frame calibration: {translation_norm:.3f} m.")


def calibration_metadata(matrix: np.ndarray) -> dict[str, Any]:
    rotation = np.asarray(matrix[:3, :3], dtype=np.float32)
    return {
        "matrix": np.asarray(matrix, dtype=np.float32).tolist(),
        "rotation_det": float(np.linalg.det(rotation)),
        "orthonormal_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3, dtype=np.float32))),
        "translation_norm_m": float(np.linalg.norm(matrix[:3, 3])),
        "frame_convention": {
            "grasp_group_x_axis": "gripper depth/approach axis",
            "grasp_group_y_axis": "gripper opening width axis",
            "grasp_group_z_axis": "finger height axis",
            "kuavo_wrist_tcp_offset_axis": "-Z from right wrist to fingertip center after task319 inline gripper mount",
            "kuavo_gripper_local_tcp_offset_axis": "+X in the attached gripper_base frame",
        },
    }


class AnyGraspWrapper:
    """Lazy AnyGrasp SDK loader returning candidates in the camera frame."""

    def __init__(
        self,
        repo_root: Path,
        checkpoint_path: Path,
        *,
        max_gripper_width: float = 0.07,
        gripper_height: float = 0.03,
        top_down_grasp: bool = False,
        apply_object_mask: bool = True,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        grasp_to_tcp: np.ndarray | None = None,
        workspace_padding_m: float = 0.03,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.max_gripper_width = float(max(0.0, min(max_gripper_width, 0.10)))
        self.gripper_height = float(gripper_height)
        self.top_down_grasp = bool(top_down_grasp)
        self.apply_object_mask = bool(apply_object_mask)
        self.dense_grasp = bool(dense_grasp)
        self.collision_detection = bool(collision_detection)
        self.grasp_to_tcp = IDENTITY_GRASP_TO_TCP.copy() if grasp_to_tcp is None else np.asarray(grasp_to_tcp, dtype=np.float32)
        validate_homogeneous_matrix(self.grasp_to_tcp, name="anygrasp_grasp_to_tcp")
        self.workspace_padding_m = float(workspace_padding_m)
        self._detector = None

    def _add_paths(self) -> None:
        paths = [
            self.repo_root / "grasp_detection",
            self.repo_root / "pointnet2",
            self.repo_root,
            self.repo_root.parent / "graspnetAPI",
        ]
        for path in paths:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def load(self):
        if self._detector is not None:
            return self._detector
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"AnyGrasp SDK repo not found: {self.repo_root}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AnyGrasp checkpoint not found: {self.checkpoint_path}")
        self._add_paths()
        try:
            from gsnet import AnyGrasp  # type: ignore
        except Exception as exc:
            raise ImportError(
                "AnyGrasp SDK module 'gsnet' is unavailable. Clone graspnet/anygrasp_sdk, "
                "install its dependencies, copy the Python-version-matched gsnet/lib_cxx binaries into "
                "grasp_detection, provide OpenSSL 1.1, and place the license folder as required by the SDK."
            ) from exc
        cfg = Namespace(
            checkpoint_path=str(self.checkpoint_path),
            max_gripper_width=self.max_gripper_width,
            gripper_height=self.gripper_height,
            top_down_grasp=self.top_down_grasp,
            debug=False,
        )
        detector = AnyGrasp(cfg)
        detector.load_net()
        self._detector = detector
        return detector

    def detect(self, rgb: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray, workspace_mask: np.ndarray) -> GraspCandidates:
        detector = self.load()
        height, width = depth_m.shape[:2]
        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise ValueError("RGB and depth shapes do not match for AnyGrasp input.")
        xmap, ymap = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        points_z = depth_m.astype(np.float32)
        points_x = (xmap - float(intrinsics[0, 2])) / float(intrinsics[0, 0]) * points_z
        points_y = (ymap - float(intrinsics[1, 2])) / float(intrinsics[1, 1]) * points_z
        points = np.stack([points_x, points_y, points_z], axis=-1)
        valid = workspace_mask.astype(bool) & np.isfinite(points_z) & (points_z > 0.0)
        points = points[valid].astype(np.float32)
        if points.shape[0] == 0:
            return _empty_candidates()
        colors = rgb[..., :3].astype(np.float32)
        if colors.size and float(np.max(colors)) > 1.0:
            colors = colors / 255.0
        colors = colors[valid].astype(np.float32)
        lims = _workspace_lims(points, padding=self.workspace_padding_m)
        grasp_group, _ = detector.get_grasp(
            points,
            colors,
            lims=lims,
            apply_object_mask=self.apply_object_mask,
            dense_grasp=self.dense_grasp,
            collision_detection=self.collision_detection,
        )
        if len(grasp_group) == 0:
            return _empty_candidates()
        grasp_group = grasp_group.nms().sort_by_score()
        translations = np.asarray(grasp_group.translations, dtype=np.float32)
        rotations = np.asarray(grasp_group.rotation_matrices, dtype=np.float32)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(grasp_group), axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = translations
        poses = poses @ self.grasp_to_tcp[None, :, :]
        centers_2d = _project_points(poses[:, :3, 3], intrinsics)
        return GraspCandidates(
            poses=poses.astype(np.float32),
            scores=np.asarray(grasp_group.scores, dtype=np.float32),
            widths=np.asarray(grasp_group.widths, dtype=np.float32),
            centers_2d=centers_2d,
        )


def _workspace_lims(points: np.ndarray, *, padding: float) -> list[float]:
    lo = np.percentile(points, 1.0, axis=0) - padding
    hi = np.percentile(points, 99.0, axis=0) + padding
    lo[2] = max(0.0, lo[2])
    hi[2] = max(hi[2], lo[2] + 0.02)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]


def _empty_candidates() -> GraspCandidates:
    return GraspCandidates(
        poses=np.empty((0, 4, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        widths=np.empty((0,), dtype=np.float32),
        centers_2d=np.empty((0, 2), dtype=np.float32),
    )


def _project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points_camera[:, 2], 1e-6)
    u = intrinsics[0, 0] * points_camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)
