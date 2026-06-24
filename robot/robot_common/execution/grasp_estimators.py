"""
Adapters for mature RGB-D grasp estimators.

The task pipelines already provide RGB-D frames, masks, and camera intrinsics.
This module keeps the estimator boundary explicit: GraspNet / AnyGrasp are the
success path, while missing backends fail loudly instead of using analytic scene
truth shortcuts.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robot_common.infra.config import MODELS_DIR, PROJECT_ROOT
from robot_common.infra.logging import logger
from robot_common.execution.mature_backends import (
    ANYGRASP_BACKEND,
    GRASPNET_BACKEND,
    MissingBackendError,
    check_backend,
)
from robot_common.execution.wsl_bridge import (
    bash_quote,
    process_detail,
    robot_stack_env_prefix,
    run_wsl,
    run_wsl_with_retries,
    windows_path_to_wsl,
    wsl_bridge_enabled,
)


@dataclass
class GraspPose:
    """6-DoF grasp pose returned by a mature estimator.

    ``position`` is in the same camera frame as the input point cloud unless the
    caller overwrites it with a task-level world transform. ``metadata`` stores
    backend-specific fields such as width, raw grasp arrays, and point counts.
    """

    position: np.ndarray
    rotation: np.ndarray
    score: float
    source: str
    width: float = 0.06
    approach_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    jaw_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    metadata: dict = field(default_factory=dict)

    @property
    def transform(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T


def _ensure_third_party_on_path():
    """Expose local third-party checkouts without requiring editable installs."""
    candidates = [
        PROJECT_ROOT / "third_party" / "graspnet-baseline",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "models",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "dataset",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "utils",
        PROJECT_ROOT / "third_party" / "graspnetAPI",
        PROJECT_ROOT / "third_party" / "anygrasp_sdk",
    ]
    for path in candidates:
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def _make_point_cloud(rgb: np.ndarray, depth: np.ndarray,
                      intrinsics: dict, mask: Optional[np.ndarray] = None,
                      max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Convert RGB-D to an unorganized point cloud in camera coordinates."""
    if depth is None:
        raise ValueError("depth image is required for RGB-D grasp estimation")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    rgb = np.asarray(rgb)
    h, w = depth.shape
    fx = float(intrinsics.get("fx", 500.0))
    fy = float(intrinsics.get("fy", fx))
    cx = float(intrinsics.get("cx", w / 2.0))
    cy = float(intrinsics.get("cy", h / 2.0))

    valid = np.isfinite(depth) & (depth > 1e-4)
    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.shape != depth.shape:
            mask_arr = mask_arr[:h, :w]
        valid &= mask_arr
    if not np.any(valid):
        raise ValueError("RGB-D frame contains no valid target depth points")

    vv, uu = np.nonzero(valid)
    z = depth[vv, uu]
    x = (uu.astype(np.float32) - cx) * z / fx
    y = (vv.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    if rgb.ndim == 3 and rgb.shape[:2] == depth.shape:
        colors = rgb[vv, uu, :3].astype(np.float32) / 255.0
    else:
        colors = np.ones_like(points, dtype=np.float32)

    if len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def _load_graspnet_grasp_group():
    """Load GraspNetAPI.GraspGroup without importing the heavy eval package."""
    try:
        from graspnetAPI.grasp import GraspGroup

        return GraspGroup
    except Exception:
        package_dir = PROJECT_ROOT / "third_party" / "graspnetAPI" / "graspnetAPI"
        grasp_py = package_dir / "grasp.py"
        if not grasp_py.exists():
            raise
        package_name = "_graspnet_api_light"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_dir)]
            sys.modules[package_name] = package
        module_name = f"{package_name}.grasp"
        if module_name in sys.modules:
            return sys.modules[module_name].GraspGroup
        spec = importlib.util.spec_from_file_location(module_name, grasp_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load GraspNetAPI grasp.py from {grasp_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.GraspGroup


def _pose_from_candidate(candidate, source: str, metadata: Optional[dict] = None) -> GraspPose:
    """Normalize common GraspNet/AnyGrasp candidate representations."""
    metadata = dict(metadata or {})
    if hasattr(candidate, "translation"):
        position = np.asarray(candidate.translation, dtype=float)
    elif hasattr(candidate, "position"):
        position = np.asarray(candidate.position, dtype=float)
    else:
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        position = arr[:3]

    if hasattr(candidate, "rotation_matrix"):
        rotation = np.asarray(candidate.rotation_matrix, dtype=float).reshape(3, 3)
    elif hasattr(candidate, "rotation"):
        rotation = np.asarray(candidate.rotation, dtype=float).reshape(3, 3)
    else:
        arr = np.asarray(candidate, dtype=float).reshape(-1)
        rotation = arr[3:12].reshape(3, 3) if arr.size >= 12 else np.eye(3)

    score = 0.0
    for attr in ("score", "objectness", "confidence"):
        if hasattr(candidate, attr):
            score = float(getattr(candidate, attr))
            break
    width = float(getattr(candidate, "width", metadata.get("width", 0.06)))
    approach_axis = rotation[:, 0] if rotation.shape == (3, 3) else np.array([1.0, 0.0, 0.0])
    jaw_axis = rotation[:, 1] if rotation.shape == (3, 3) else np.array([0.0, 1.0, 0.0])
    return GraspPose(
        position=position,
        rotation=rotation,
        score=score,
        width=width,
        source=source,
        approach_axis=np.asarray(approach_axis, dtype=float),
        jaw_axis=np.asarray(jaw_axis, dtype=float),
        metadata=metadata,
    )


class AnyGraspEstimator:
    """AnyGrasp SDK adapter.

    AnyGrasp is kept as a first-class backend for future license-enabled runs.
    Until the SDK/license is installed, availability remains false and callers
    should select GraspNet for the作业链路.
    """

    def __init__(self, required: bool = True, checkpoint: str = ""):
        _ensure_third_party_on_path()
        self.required = required
        self.available = check_backend(ANYGRASP_BACKEND, required=required)
        self.checkpoint = checkpoint
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if not self.available:
            raise MissingBackendError("AnyGrasp backend is not available")
        try:
            sdk = importlib.import_module("anygrasp_sdk")
            if hasattr(sdk, "AnyGrasp"):
                self._model = sdk.AnyGrasp()
            else:
                raise AttributeError("anygrasp_sdk.AnyGrasp not found")
        except Exception as exc:
            raise MissingBackendError(
                "AnyGrasp SDK is installed but could not be initialized; "
                "check license, CUDA, and checkpoint configuration."
            ) from exc

    def estimate(self, rgb: np.ndarray, depth: np.ndarray,
                 intrinsics: dict, mask: Optional[np.ndarray] = None) -> GraspPose:
        self._load_model()
        points, colors = _make_point_cloud(rgb, depth, intrinsics, mask)
        try:
            if hasattr(self._model, "get_grasp"):
                grasps = self._model.get_grasp(points, colors)
            elif hasattr(self._model, "predict"):
                grasps = self._model.predict(points, colors)
            else:
                raise AttributeError("AnyGrasp model exposes neither get_grasp nor predict")
        except Exception as exc:
            raise RuntimeError("AnyGrasp inference failed") from exc
        if grasps is None or len(grasps) == 0:
            raise RuntimeError("AnyGrasp returned no grasps")
        best = max(grasps, key=lambda g: float(getattr(g, "score", 0.0)))
        pose = _pose_from_candidate(best, "anygrasp", {"num_points": len(points)})
        logger.info(f"AnyGrasp grasp score={pose.score:.3f}, width={pose.width:.3f}")
        return pose


class GraspNetEstimator:
    """GraspNet baseline RGB-D adapter."""

    def __init__(self, required: bool = True, checkpoint: str = "",
                 allow_wsl_bridge: bool = True):
        _ensure_third_party_on_path()
        self.required = required
        self.checkpoint = Path(checkpoint) if checkpoint else MODELS_DIR / "graspnet-rs" / "checkpoint-rs.tar"
        self.allow_wsl_bridge = allow_wsl_bridge
        self.wsl_available = False
        self.wsl_error = ""
        self.available = False
        self.refresh_availability()
        if required and not self.available:
            raise MissingBackendError(self.availability_detail())
        self._model = None
        self._backend = ""
        self._torch = None
        self._pred_decode = None
        self._GraspGroup = None

    def refresh_availability(self) -> bool:
        """Refresh local/WSL availability instead of trusting construction time."""
        self.available = check_backend(GRASPNET_BACKEND, required=False)
        self.wsl_available = False
        if (
            not self.available
            and self.allow_wsl_bridge
            and wsl_bridge_enabled()
        ):
            self.wsl_available = self._check_wsl_bridge()
            self.available = self.wsl_available
        return self.available

    def availability_detail(self) -> str:
        if self.available:
            if self.wsl_available:
                return "GraspNet WSL bridge is ready"
            return "Local GraspNet baseline is ready"
        if self.wsl_error:
            return f"GraspNet WSL bridge not ready: {self.wsl_error}"
        return (
            "GraspNet baseline is not available locally, and the WSL bridge "
            "is not ready"
        )

    def _check_wsl_bridge(self) -> bool:
        try:
            checkpoint = windows_path_to_wsl(self.checkpoint)
            cmd = (
                robot_stack_env_prefix(include_graspnet=True)
                + f" && test -f {bash_quote(checkpoint)}"
                + " && python -c \"import pathlib; import graspnet; "
                "import pointnet2._ext; import knn_pytorch.knn_pytorch; "
                "from robot_common.execution.grasp_estimators import _load_graspnet_grasp_group; "
                "_load_graspnet_grasp_group(); "
                "assert pathlib.Path('models/graspnet-rs/checkpoint-rs.tar').exists(); "
                "print('graspnet_bridge_ready')\""
            )
            result = run_wsl_with_retries(
                cmd,
                timeout=180,
                attempts=3,
                delay_s=3.0,
            )
            if result.returncode != 0:
                self.wsl_error = process_detail(result)
                logger.warning(f"GraspNet WSL bridge not ready: {self.wsl_error[:300]}")
            else:
                self.wsl_error = ""
            return result.returncode == 0
        except Exception as exc:
            self.wsl_error = str(exc)
            return False

    def _win_path_to_wsl(self, path: Path) -> str:
        return windows_path_to_wsl(path)

    def _estimate_via_wsl(self, rgb: np.ndarray, depth: np.ndarray,
                          intrinsics: dict, mask: Optional[np.ndarray] = None) -> GraspPose:
        with tempfile.TemporaryDirectory(prefix="graspnet_bridge_") as tmp:
            tmp_path = Path(tmp)
            payload = tmp_path / "payload.npz"
            output = tmp_path / "pose.json"
            arrays = {
                "rgb": np.asarray(rgb),
                "depth": np.asarray(depth, dtype=np.float32),
                "intrinsics_json": np.asarray(json.dumps(intrinsics)),
            }
            if mask is not None:
                arrays["mask"] = np.asarray(mask).astype(np.uint8)
            np.savez_compressed(payload, **arrays)

            payload_wsl = self._win_path_to_wsl(payload)
            output_wsl = self._win_path_to_wsl(output)
            cmd = (
                robot_stack_env_prefix(include_graspnet=True)
                + f" && python scripts/graspnet_infer_once.py "
                f"{bash_quote(payload_wsl)} {bash_quote(output_wsl)}"
            )
            result = run_wsl(
                cmd,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "WSL GraspNet inference failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            data = json.loads(output.read_text(encoding="utf-8"))
            return GraspPose(
                position=np.asarray(data["position"], dtype=float),
                rotation=np.asarray(data["rotation"], dtype=float),
                score=float(data["score"]),
                width=float(data["width"]),
                source="graspnet_wsl_bridge",
                approach_axis=np.asarray(data.get("approach_axis", [1.0, 0.0, 0.0]), dtype=float),
                jaw_axis=np.asarray(data.get("jaw_axis", [0.0, 1.0, 0.0]), dtype=float),
                metadata=data.get("metadata", {}),
            )

    def _load_model(self):
        if self._model is not None:
            return
        if self.wsl_available and self._model is None:
            raise MissingBackendError("Local GraspNet is not available; WSL bridge is inference-only")
        if not self.available:
            raise MissingBackendError("GraspNet backend is not available")

        _ensure_third_party_on_path()
        try:
            import torch
            from graspnet import GraspNet, pred_decode

            GraspGroup = _load_graspnet_grasp_group()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = GraspNet(
                input_feature_dim=0,
                num_view=300,
                num_angle=12,
                num_depth=4,
                cylinder_radius=0.05,
                hmin=-0.02,
                hmax_list=[0.01, 0.02, 0.03, 0.04],
                is_training=False,
            )
            checkpoint = torch.load(str(self.checkpoint), map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"])
            net.to(device)
            net.eval()
            self._model = net
            self._torch = torch
            self._pred_decode = pred_decode
            self._GraspGroup = GraspGroup
            self._backend = "graspnet-baseline"
            logger.info(f"GraspNet baseline loaded: {self.checkpoint}")
            return
        except Exception as exc:
            raise MissingBackendError(
                "GraspNet baseline runtime/checkpoint could not be initialized. "
                "Install pointnet2/knn CUDA ops and place checkpoint-rs.tar under "
                "models/graspnet-rs/."
            ) from exc

    def _infer(self, points: np.ndarray, colors: np.ndarray,
               intrinsics: dict) -> list:
        torch = self._torch
        device = next(self._model.parameters()).device
        if len(points) >= 20000:
            idx = np.random.default_rng(42).choice(len(points), 20000, replace=False)
        else:
            idx1 = np.arange(len(points))
            idx2 = np.random.default_rng(42).choice(len(points), 20000 - len(points), replace=True)
            idx = np.concatenate([idx1, idx2], axis=0)
        sampled = points[idx].astype(np.float32)
        end_points = {
            "point_clouds": torch.from_numpy(sampled[np.newaxis]).to(device),
            "cloud_colors": colors[idx],
        }
        with torch.no_grad():
            end_points = self._model(end_points)
            preds = self._pred_decode(end_points)
        gg_array = preds[0].detach().cpu().numpy()
        return self._GraspGroup(gg_array)

    def estimate(self, rgb: np.ndarray, depth: np.ndarray,
                 intrinsics: dict, mask: Optional[np.ndarray] = None) -> GraspPose:
        if self.wsl_available and self._model is None:
            return self._estimate_via_wsl(rgb, depth, intrinsics, mask)
        if not self.available:
            raise MissingBackendError("GraspNet backend is not available")
        points, colors = _make_point_cloud(rgb, depth, intrinsics, mask)
        self._load_model()
        try:
            grasps = self._infer(points, colors, intrinsics)
        except Exception as exc:
            raise RuntimeError(f"GraspNet inference failed: {exc}") from exc

        if grasps is None or len(grasps) == 0:
            raise RuntimeError("GraspNet returned no grasps")

        # Upstream GraspGroup exposes sort_by_score / nms in some versions.
        try:
            if hasattr(grasps, "nms"):
                grasps = grasps.nms()
            if hasattr(grasps, "sort_by_score"):
                grasps = grasps.sort_by_score()
                best = grasps[0]
            else:
                best = max(grasps, key=lambda g: float(getattr(g, "score", 0.0)))
        except Exception:
            best = grasps[0]

        pose = _pose_from_candidate(
            best,
            "graspnet",
            {"num_points": len(points), "runtime": self._backend},
        )
        pose.metadata["approach_axis"] = pose.approach_axis.tolist()
        pose.metadata["jaw_axis"] = pose.jaw_axis.tolist()
        logger.info(f"GraspNet grasp score={pose.score:.3f}, width={pose.width:.3f}")
        return pose
