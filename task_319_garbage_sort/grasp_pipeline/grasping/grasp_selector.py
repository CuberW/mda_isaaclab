"""Select a physically reasonable target-object grasp."""

from __future__ import annotations

import numpy as np

from task_319_garbage_sort.grasp_pipeline.grasping.mask_filter import clamp_width
from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates, PerceivedObject, SelectedGrasp


def rank_grasps(
    candidates: GraspCandidates,
    target: PerceivedObject,
    *,
    t_camera_to_world: np.ndarray | None = None,
    table_surface_z: float = 0.81,
    max_width_m: float = 0.120,
    top_k: int = 20,
    source: str = "graspnet",
) -> list[SelectedGrasp]:
    if len(candidates.scores) == 0:
        return []
    order = np.argsort(candidates.scores)[::-1][: min(top_k, len(candidates.scores))]
    target_center = target.center_3d
    scored: list[tuple[float, int, SelectedGrasp]] = []
    for rank, idx in enumerate(order):
        pose = np.array(candidates.poses[idx], dtype=np.float32)
        pose_world = t_camera_to_world @ pose if t_camera_to_world is not None else pose
        if pose_world[2, 3] < table_surface_z + 0.015:
            continue
        # GraspNet/GraspGroup convention: local +X is the gripper depth/approach axis,
        # +Y is jaw opening width, and +Z is finger height.
        approach_dir = pose_world[:3, 0]
        approach_norm = np.linalg.norm(approach_dir)
        if approach_norm > 0.0:
            approach_dir = approach_dir / approach_norm
        top_down_bonus = max(0.0, float(np.dot(approach_dir, np.array([0.0, 0.0, -1.0]))))
        centrality = 0.0
        center_distance = None
        if target_center is not None:
            center_distance = float(np.linalg.norm(pose_world[:3, 3] - target_center))
            centrality = max(0.0, 1.0 - center_distance / 0.15)
        combined = float(candidates.scores[idx]) * 0.62 + top_down_bonus * 0.28 + centrality * 0.10
        metadata = {
            "candidate_index": int(idx),
            "candidate_rank_by_score": int(rank),
            "combined_score": combined,
            "top_down_bonus": top_down_bonus,
            "approach_axis_local": "+X",
            "approach_axis_world": approach_dir.astype(float).tolist(),
            "centrality": centrality,
            "target_center_distance_m": center_distance,
        }
        if candidates.depths is not None:
            metadata["graspnet_depth_m"] = float(candidates.depths[idx])
        grasp = SelectedGrasp(
            pose_world=pose_world,
            width=clamp_width(float(candidates.widths[idx]), max_width=max_width_m),
            score=float(candidates.scores[idx]),
            center_2d=(float(candidates.centers_2d[idx, 0]), float(candidates.centers_2d[idx, 1])),
            target=target,
            source=source,
            metadata=metadata,
        )
        scored.append((combined, -rank, grasp))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for rank, (_, _, grasp) in enumerate(scored):
        grasp.metadata = dict(grasp.metadata or {})
        grasp.metadata["rank_after_scoring"] = int(rank)
    return [item[2] for item in scored]


def select_best_grasp(
    candidates: GraspCandidates,
    target: PerceivedObject,
    *,
    t_camera_to_world: np.ndarray | None = None,
    table_surface_z: float = 0.81,
    max_width_m: float = 0.120,
    top_k: int = 20,
    source: str = "graspnet",
) -> SelectedGrasp | None:
    ranked = rank_grasps(
        candidates,
        target,
        t_camera_to_world=t_camera_to_world,
        table_surface_z=table_surface_z,
        max_width_m=max_width_m,
        top_k=top_k,
        source=source,
    )
    return ranked[0] if ranked else None
