"""Mask-based filtering for full-image grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates


@dataclass(slots=True)
class MaskFilterResult:
    candidates: GraspCandidates
    indices: np.ndarray
    metadata: dict


def erode_mask(mask: np.ndarray, margin: int = 5) -> np.ndarray:
    """Binary erosion implemented with NumPy to avoid a hard OpenCV dependency."""

    mask = mask.astype(bool)
    if margin <= 1:
        return mask
    pad = margin // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(margin):
        for dx in range(margin):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return eroded


def _take_candidates(candidates: GraspCandidates, indices: np.ndarray) -> GraspCandidates:
    indices = np.asarray(indices, dtype=np.int64)
    return GraspCandidates(
        poses=candidates.poses[indices],
        scores=candidates.scores[indices],
        widths=candidates.widths[indices],
        centers_2d=candidates.centers_2d[indices],
        depths=candidates.depths[indices] if candidates.depths is not None else None,
    )


def _indices_inside_mask(candidates: GraspCandidates, mask: np.ndarray) -> np.ndarray:
    valid_indices: list[int] = []
    height, width = mask.shape[:2]
    for idx, (u, v) in enumerate(candidates.centers_2d):
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if 0 <= ui < width and 0 <= vi < height and mask[vi, ui]:
            valid_indices.append(idx)
    return np.asarray(valid_indices, dtype=np.int64)


def _unique_preserve_order(*index_groups: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    ordered: list[int] = []
    for group in index_groups:
        for raw_idx in np.asarray(group, dtype=np.int64):
            idx = int(raw_idx)
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
    return np.asarray(ordered, dtype=np.int64)


def _nearest_distances(points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    cloud = np.asarray(cloud, dtype=np.float32)
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(cloud) == 0:
        return np.full((len(points),), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(cloud).query(points, k=1, workers=-1)
        return distances.astype(np.float32)
    except Exception:
        min_dist_sq = np.full((len(points),), np.inf, dtype=np.float32)
        chunk = 4096
        for start in range(0, len(cloud), chunk):
            cloud_chunk = cloud[start : start + chunk]
            diff = points[:, None, :] - cloud_chunk[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            min_dist_sq = np.minimum(min_dist_sq, np.min(dist_sq, axis=1).astype(np.float32))
        return np.sqrt(min_dist_sq).astype(np.float32)


def filter_grasps_by_mask_relaxed(
    candidates: GraspCandidates,
    object_mask: np.ndarray,
    *,
    margin: int = 5,
    candidate_points_world: np.ndarray | None = None,
    object_points_world: np.ndarray | None = None,
    min_filtered_grasps: int = 8,
    distance_threshold_m: float = 0.02,
) -> MaskFilterResult:
    object_mask = object_mask.astype(bool)
    eroded = erode_mask(object_mask, margin)
    strict_indices = _indices_inside_mask(candidates, eroded)
    raw_indices = np.empty((0,), dtype=np.int64)
    distance_indices = np.empty((0,), dtype=np.int64)
    distances = np.full((len(candidates.scores),), np.nan, dtype=np.float32)
    mode = "strict_2d_mask"

    if len(strict_indices) < min_filtered_grasps:
        raw_indices = _indices_inside_mask(candidates, object_mask)
        if len(raw_indices) > len(strict_indices):
            mode = "raw_2d_mask"

    combined_2d = _unique_preserve_order(strict_indices, raw_indices)
    if (
        len(combined_2d) < min_filtered_grasps
        and candidate_points_world is not None
        and object_points_world is not None
        and distance_threshold_m > 0.0
    ):
        distances = _nearest_distances(candidate_points_world, object_points_world)
        distance_indices = np.flatnonzero(distances <= float(distance_threshold_m)).astype(np.int64)
        if len(distance_indices) > 0:
            mode = "distance_relaxed_3d"

    indices = _unique_preserve_order(strict_indices, raw_indices, distance_indices)
    metadata = {
        "mask_filter_mode": mode,
        "mask_margin_px": int(margin),
        "min_filtered_grasps": int(min_filtered_grasps),
        "mask_distance_threshold_m": float(distance_threshold_m),
        "strict_2d_count": int(len(strict_indices)),
        "raw_2d_count": int(len(raw_indices)),
        "distance_filtered_count": int(len(distance_indices)),
        "filtered_count": int(len(indices)),
    }
    if np.any(np.isfinite(distances)):
        finite = distances[np.isfinite(distances)]
        metadata["candidate_distance_min_m"] = float(np.min(finite))
        metadata["candidate_distance_median_m"] = float(np.median(finite))
    return MaskFilterResult(candidates=_take_candidates(candidates, indices), indices=indices, metadata=metadata)


def filter_grasps_by_mask(candidates: GraspCandidates, object_mask: np.ndarray, margin: int = 5) -> GraspCandidates:
    return filter_grasps_by_mask_relaxed(candidates, object_mask, margin=margin, min_filtered_grasps=0).candidates


def clamp_width(width: float, min_width: float = 0.0, max_width: float = 0.120) -> float:
    return float(np.clip(width, min_width, max_width))
