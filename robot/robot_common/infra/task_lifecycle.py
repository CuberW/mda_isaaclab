"""Shared task-pipeline lifecycle helpers.

These helpers keep orchestration code focused on the task flow. Loading raw
YAML, opening debug folders, prewarming lazy perception models, and persisting
motion traces are cross-cutting concerns shared by all three tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_common.infra.debug_artifacts import EpisodeDebugWriter
from robot_common.infra.logging import logger


def load_raw_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load a task YAML file as a plain dictionary."""
    with Path(config_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def debug_enabled(raw_config: dict[str, Any]) -> bool:
    """Return whether debug artifacts should be written for a task config."""
    return bool((raw_config or {}).get("task", {}).get("debug_artifacts", True))


def create_episode_debug_writer(
    task_id: str,
    episode_id: int,
    raw_config: dict[str, Any],
) -> EpisodeDebugWriter:
    """Create the per-episode debug writer using the common config flag."""
    return EpisodeDebugWriter(
        task_id,
        episode_id,
        enabled=debug_enabled(raw_config),
    )


def prewarm_perception_models(perception: Any, *, segmentor: bool = True) -> None:
    """Load lazy detector/segmentor models before a viewer opens."""
    if perception is not None and getattr(perception, "detector", None) is not None:
        try:
            perception.detector._load_model()
        except AttributeError:
            pass
    if segmentor and perception is not None and getattr(perception, "segmentor", None) is not None:
        try:
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            perception.segmentor.segment(rgb, (8, 8, 32, 32))
        except Exception:
            pass


def save_motion_trace(motion_trace: Any, task_id: str, episode_id: int) -> dict[str, Any]:
    """Save a motion trace and return its summary."""
    if motion_trace is None:
        return {}
    summary = motion_trace.summary()
    path = Path("results") / "traces" / f"task_{task_id}_ep{episode_id}.json"
    motion_trace.save(path)
    logger.info(f"Motion trace saved: {path} smooth={summary['smooth']}")
    return summary
