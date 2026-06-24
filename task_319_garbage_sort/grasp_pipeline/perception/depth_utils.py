"""RGB-D geometry helpers for IsaacLab camera frames."""

from __future__ import annotations

import numpy as np


def pixel_to_camera(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray | None:
    """Back-project one pixel into the camera frame."""

    if not np.isfinite(depth_m) or depth_m <= 0.0:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None
    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    return np.array([x, y, depth_m], dtype=np.float32)


def pixel_to_world(
    u: float,
    v: float,
    depth_image_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> np.ndarray | None:
    """Back-project one depth pixel into the world frame."""

    height, width = depth_image_m.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= width or vi < 0 or vi >= height:
        return None
    point_camera = pixel_to_camera(u, v, float(depth_image_m[vi, ui]), intrinsics)
    if point_camera is None:
        return None
    point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0], dtype=np.float32)
    point_world = t_camera_to_world @ point_h
    if point_world[3] == 0.0:
        return None
    return (point_world[:3] / point_world[3]).astype(np.float32)


def masked_depth_center_world(
    mask: np.ndarray,
    depth_image_m: np.ndarray,
    intrinsics: np.ndarray,
    t_camera_to_world: np.ndarray,
) -> np.ndarray | None:
    """Estimate an object's world center from valid depth pixels inside a mask."""

    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    depths = depth_image_m[ys, xs]
    valid = np.isfinite(depths) & (depths > 0.0)
    if not np.any(valid):
        return None
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    depths = depths[valid].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    points_camera = np.stack(((xs - cx) * depths / fx, (ys - cy) * depths / fy, depths), axis=1)
    points_h = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1), dtype=np.float32)], axis=1)
    points_world = (t_camera_to_world @ points_h.T).T[:, :3]
    return np.median(points_world, axis=0).astype(np.float32)
