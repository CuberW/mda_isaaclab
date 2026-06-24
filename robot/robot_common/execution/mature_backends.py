"""
Runtime checks for mature planning and grasp backends.

These helpers keep the task pipelines honest: a configured mature backend must
be importable and have its expected model artifact available. Demo code should
not silently replace AnyGrasp, GraspNet, or WBC with local shortcuts.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from robot_common.infra.config import MODELS_DIR, PROJECT_ROOT


def _ensure_third_party_on_path():
    candidates = [
        PROJECT_ROOT / "third_party" / "graspnetAPI",
        PROJECT_ROOT / "third_party" / "graspnet-baseline",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "models",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "pointnet2",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "knn",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "utils",
        PROJECT_ROOT / "third_party" / "anygrasp_sdk",
    ]
    ros_names = []
    env_ros = os.environ.get("ROS_DISTRO")
    if env_ros:
        ros_names.append(env_ros)
    ros_names.extend(["jazzy", "humble", "iron"])
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for name in dict.fromkeys(ros_names):
        candidates.extend([
            Path("/opt") / "ros" / name / "lib" / py_tag / "site-packages",
            Path("/opt") / "ros" / name / "lib" / py_tag / "dist-packages",
        ])

    for path in candidates:
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


@dataclass(frozen=True)
class BackendRequirement:
    name: str
    modules: tuple[str, ...] = ()
    files: tuple[Path, ...] = ()


class MissingBackendError(RuntimeError):
    """Raised when a configured mature backend is not installed or configured."""


def _missing_modules(modules: Iterable[str]) -> list[str]:
    _ensure_third_party_on_path()
    missing = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _missing_files(files: Iterable[Path]) -> list[str]:
    return [str(path) for path in files if not path.exists()]


def check_backend(req: BackendRequirement, required: bool = True) -> bool:
    """Check a backend and optionally raise with actionable detail."""
    missing_mods = _missing_modules(req.modules)
    missing_paths = _missing_files(req.files)
    ok = not missing_mods and not missing_paths
    if ok or not required:
        return ok
    details = []
    if missing_mods:
        details.append(f"missing modules: {', '.join(missing_mods)}")
    if missing_paths:
        details.append(f"missing files: {', '.join(missing_paths)}")
    raise MissingBackendError(f"{req.name} backend is not available ({'; '.join(details)})")


def describe_backend(req: BackendRequirement) -> str:
    """Return a concise missing-parts description for health output."""
    missing_mods = _missing_modules(req.modules)
    missing_paths = _missing_files(req.files)
    details = []
    if missing_mods:
        details.append(f"missing modules: {', '.join(missing_mods)}")
    if missing_paths:
        details.append(f"missing files: {', '.join(missing_paths)}")
    return "; ".join(details)


MINK_BACKEND = BackendRequirement(
    name="Mink + quadprog",
    modules=("mink", "quadprog"),
)

GROUNDING_DINO_BACKEND = BackendRequirement(
    name="GroundingDINO",
    modules=("transformers", "torch"),
    files=(MODELS_DIR / "grounding-dino-base" / "config.json",),
)

SAM_BACKEND = BackendRequirement(
    name="SAM",
    modules=("segment_anything", "torch"),
    files=(MODELS_DIR / "sam-vit-b" / "sam_vit_b_01ec64.pth",),
)

ANYGRASP_BACKEND = BackendRequirement(
    name="AnyGrasp",
    modules=("anygrasp_sdk",),
)

GRASPNET_BACKEND = BackendRequirement(
    name="GraspNet baseline",
    modules=("graspnet", "graspnetAPI", "pointnet2._ext", "knn_pytorch.knn_pytorch"),
    files=(
        MODELS_DIR / "graspnet-rs" / "checkpoint-rs.tar",
        PROJECT_ROOT / "third_party" / "graspnet-baseline" / "models" / "graspnet.py",
    ),
)

ROBOSUITE_WBC_BACKEND = BackendRequirement(
    name="robosuite WBC",
    modules=("robosuite",),
)
