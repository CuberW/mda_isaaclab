"""Thin wrapper around the local graspnet-baseline checkout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from task_319_garbage_sort.grasp_pipeline.grasping.anygrasp_wrapper import IDENTITY_GRASP_TO_TCP, validate_homogeneous_matrix
from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates


class GraspNetWrapper:
    """Lazy GraspNet baseline loader.

    The heavy dependencies are imported only when inference is requested so that
    unit tests for filtering and control can run without CUDA/Open3D.
    """

    def __init__(
        self,
        repo_root: Path,
        checkpoint_path: Path,
        *,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
        score_thresh: float = 0.0,
        force_objectness_top_k: int = 0,
        grasp_to_tcp: np.ndarray | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.score_thresh = score_thresh
        self.force_objectness_top_k = max(0, int(force_objectness_top_k))
        self.grasp_to_tcp = IDENTITY_GRASP_TO_TCP.copy() if grasp_to_tcp is None else np.asarray(grasp_to_tcp, dtype=np.float32)
        validate_homogeneous_matrix(self.grasp_to_tcp, name="graspnet_grasp_to_tcp")
        self._net = None
        self.last_debug_info: dict[str, object] = {}

    def _add_paths(self) -> None:
        for rel in ("models", "dataset", "utils", "pointnet2"):
            path = str(self.repo_root / rel)
            if path not in sys.path:
                sys.path.insert(0, path)
        api_path = str(self.repo_root.parent / "graspnetAPI")
        if api_path not in sys.path:
            sys.path.insert(0, api_path)

    def load(self):
        if self._net is not None:
            return self._net
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"GraspNet checkpoint not found: {self.checkpoint_path}")
        self._add_paths()
        import torch
        from graspnet import GraspNet

        net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self._net = net
        return net

    def detect(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        workspace_mask: np.ndarray,
        *,
        debug_dir: Path | None = None,
    ) -> GraspCandidates:
        """Run GraspNet and return candidates in the camera frame."""

        self._add_paths()
        import torch
        from collision_detector import ModelFreeCollisionDetector
        from data_utils import CameraInfo, create_point_cloud_from_depth_image
        from graspnet import pred_decode
        from graspnetAPI import GraspGroup

        net = self.load()
        height, width = depth_m.shape[:2]
        camera = CameraInfo(width, height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 1.0)
        cloud = create_point_cloud_from_depth_image(depth_m.astype(np.float32), camera, organized=True)
        valid = workspace_mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0.0)
        cloud_masked = cloud[valid]
        finite_mask = np.isfinite(cloud_masked).all(axis=1)
        nonfinite_count = int(len(cloud_masked) - np.count_nonzero(finite_mask))
        cloud_masked = cloud_masked[finite_mask].astype(np.float32, copy=False)
        debug_info = self._point_cloud_debug_info(cloud_masked, nonfinite_count=nonfinite_count)
        debug_info.update(
            {
                "frame": "camera",
                "depth_units": "meters",
                "num_point_requested": int(self.num_point),
                "collision_thresh": float(self.collision_thresh),
                "voxel_size": float(self.voxel_size),
                "score_thresh": float(self.score_thresh),
                "force_objectness_top_k": int(self.force_objectness_top_k),
            }
        )
        print(f"[DEBUG] GraspNet Input Point Cloud Size: {cloud_masked.shape}", flush=True)
        print(
            "[DEBUG] GraspNet Input Point Cloud Extent "
            f"xyz_m={debug_info.get('extent_xyz_m')} z_range_m={debug_info.get('z_range_m')}",
            flush=True,
        )
        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            self._write_ply(debug_dir / "debug_graspnet_input.ply", cloud_masked)
            (debug_dir / "debug_graspnet_input.json").write_text(json.dumps(debug_info, indent=2, ensure_ascii=False))
        if len(cloud_masked) == 0:
            self.last_debug_info = debug_info
            return GraspCandidates(
                np.empty((0, 4, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        if len(cloud_masked) >= self.num_point:
            indices = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            indices = np.concatenate(
                [np.arange(len(cloud_masked)), np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)]
            )
        debug_info["sampled_point_count"] = int(len(indices))
        debug_info["sampled_with_replacement"] = bool(len(cloud_masked) < self.num_point)
        device = next(net.parameters()).device
        end_points = {"point_clouds": torch.from_numpy(cloud_masked[indices][None].astype(np.float32)).to(device)}
        with torch.no_grad():
            end_points = net(end_points)
            if "objectness_score" in end_points:
                objectness_score = end_points["objectness_score"][0].float()
                objectness_prob = torch.softmax(objectness_score, dim=0)[1]
                objectness_pred = torch.argmax(objectness_score, dim=0)
                objectness_positive_count = int(torch.count_nonzero(objectness_pred == 1).item())
                debug_info["objectness_seed_count"] = int(objectness_pred.numel())
                debug_info["objectness_positive_count"] = objectness_positive_count
                debug_info["objectness_prob_min"] = float(torch.min(objectness_prob).item())
                debug_info["objectness_prob_max"] = float(torch.max(objectness_prob).item())
                debug_info["objectness_prob_mean"] = float(torch.mean(objectness_prob).item())
                if objectness_positive_count == 0 and self.force_objectness_top_k > 0 and objectness_pred.numel() > 0:
                    force_count = min(int(self.force_objectness_top_k), int(objectness_pred.numel()))
                    top_prob, top_idx = torch.topk(objectness_prob, k=force_count, largest=True, sorted=False)
                    forced_score = torch.maximum(
                        objectness_score[1, top_idx],
                        objectness_score[0, top_idx] + torch.ones_like(top_prob),
                    )
                    end_points["objectness_score"][0, 1, top_idx] = forced_score
                    objectness_score = end_points["objectness_score"][0].float()
                    objectness_pred = torch.argmax(objectness_score, dim=0)
                    debug_info["objectness_forced"] = True
                    debug_info["objectness_forced_count"] = int(force_count)
                    debug_info["objectness_forced_prob_min"] = float(torch.min(top_prob).item())
                    debug_info["objectness_forced_prob_max"] = float(torch.max(top_prob).item())
                    debug_info["objectness_positive_count_after_force"] = int(torch.count_nonzero(objectness_pred == 1).item())
                else:
                    debug_info["objectness_forced"] = False
                    debug_info["objectness_forced_count"] = 0
            grasp_preds = pred_decode(end_points)
        grasp_group = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        debug_info["raw_decoded_count"] = int(len(grasp_group))
        if len(grasp_group) > 0:
            scores = np.asarray(grasp_group.scores, dtype=np.float32)
            debug_info["raw_score_min"] = float(np.nanmin(scores))
            debug_info["raw_score_max"] = float(np.nanmax(scores))
            debug_info["raw_score_mean"] = float(np.nanmean(scores))
        if self.collision_thresh > 0.0:
            detector = ModelFreeCollisionDetector(cloud_masked, voxel_size=self.voxel_size)
            collision_mask = detector.detect(grasp_group, approach_dist=0.05, collision_thresh=self.collision_thresh)
            debug_info["collision_checked"] = True
            debug_info["collision_rejected_count"] = int(np.count_nonzero(collision_mask))
            grasp_group = grasp_group[~collision_mask]
        else:
            debug_info["collision_checked"] = False
            debug_info["collision_rejected_count"] = 0
        debug_info["post_collision_count"] = int(len(grasp_group))
        grasp_group.nms()
        debug_info["post_nms_count"] = int(len(grasp_group))
        grasp_group.sort_by_score()
        if self.score_thresh > 0.0 and len(grasp_group) > 0:
            score_mask = np.asarray(grasp_group.scores, dtype=np.float32) >= float(self.score_thresh)
            debug_info["score_rejected_count"] = int(len(grasp_group) - np.count_nonzero(score_mask))
            grasp_group = grasp_group[score_mask]
        else:
            debug_info["score_rejected_count"] = 0
        debug_info["final_count"] = int(len(grasp_group))
        if debug_dir is not None:
            (debug_dir / "debug_graspnet_input.json").write_text(json.dumps(debug_info, indent=2, ensure_ascii=False))
        self.last_debug_info = debug_info
        translations = np.asarray(grasp_group.translations, dtype=np.float32)
        rotations = np.asarray(grasp_group.rotation_matrices, dtype=np.float32)
        depths = np.asarray(grasp_group.depths, dtype=np.float32)
        depth_offsets = np.zeros((len(grasp_group), 3), dtype=np.float32)
        depth_offsets[:, 0] = depths
        tcp_points = translations + np.einsum("nij,nj->ni", rotations, depth_offsets)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(grasp_group), axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = tcp_points
        poses = poses @ self.grasp_to_tcp[None, :, :]
        centers_2d = _project_points(poses[:, :3, 3], intrinsics)
        return GraspCandidates(
            poses=poses.astype(np.float32),
            scores=np.asarray(grasp_group.scores, dtype=np.float32),
            widths=np.asarray(grasp_group.widths, dtype=np.float32),
            centers_2d=centers_2d,
            depths=depths,
        )

    @staticmethod
    def _point_cloud_debug_info(points: np.ndarray, *, nonfinite_count: int) -> dict[str, object]:
        info: dict[str, object] = {
            "point_count": int(points.shape[0]),
            "shape": list(points.shape),
            "nonfinite_removed_count": int(nonfinite_count),
        }
        if points.shape[0] == 0:
            return info
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        extent = maxs - mins
        info.update(
            {
                "min_xyz_m": mins.astype(float).tolist(),
                "max_xyz_m": maxs.astype(float).tolist(),
                "extent_xyz_m": extent.astype(float).tolist(),
                "z_range_m": float(extent[2]),
            }
        )
        return info

    @staticmethod
    def _write_ply(path: Path, points: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        with path.open("w", encoding="ascii") as stream:
            stream.write("ply\n")
            stream.write("format ascii 1.0\n")
            stream.write(f"element vertex {len(pts)}\n")
            stream.write("property float x\n")
            stream.write("property float y\n")
            stream.write("property float z\n")
            stream.write("end_header\n")
            for x, y, z in pts:
                stream.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def _project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.maximum(points_camera[:, 2], 1e-6)
    u = intrinsics[0, 0] * points_camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)
