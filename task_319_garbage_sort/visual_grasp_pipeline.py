"""Task 319 visual grasp pipeline command entrypoint.

This entrypoint coordinates model preparation, target selection, waste
classification, and grasp-candidate filtering/selection.  The full IsaacLab
execution path uses the same modules but should be launched with cameras
enabled from an IsaacLab script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT = ROOT / "models/graspnet-rs/checkpoint-rs.tar"

from task_319_garbage_sort.grasp_pipeline.grasping.grasp_selector import select_best_grasp
from task_319_garbage_sort.grasp_pipeline.grasping.mask_filter import filter_grasps_by_mask
from task_319_garbage_sort.grasp_pipeline.perception.scene_observer import TASK319_OBJECT_BY_NAME, ordered_targets
from task_319_garbage_sort.grasp_pipeline.reasoning.waste_classifier import classify_waste
from task_319_garbage_sort.grasp_pipeline.types import GraspCandidates, PerceivedObject




def maybe_prepare_deps(download_missing_models: bool) -> None:
    if not download_missing_models:
        return
    script = ROOT / "task_319_garbage_sort/scripts/prepare_grasp_deps.py"
    subprocess.run([sys.executable, str(script), "--download_missing_models"], check=True)


def choose_target(target_object: str | None, target_category: str | None):
    if target_object:
        if target_object not in TASK319_OBJECT_BY_NAME:
            raise KeyError(f"Unknown target object {target_object!r}.")
        return TASK319_OBJECT_BY_NAME[target_object]
    candidates = ordered_targets(target_category)
    if not candidates:
        raise RuntimeError(f"No target candidates for category {target_category!r}.")
    return candidates[0]


def synthetic_grasp_selection(target_spec) -> dict:
    """Run a deterministic mask-filter/selector smoke test without IsaacLab."""

    mask = np.zeros((80, 100), dtype=bool)
    mask[25:58, 35:70] = True
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    poses[:, :3, 3] = np.array(
        [
            [0.0, 0.0, 0.86],
            [0.02, 0.0, 0.88],
            [0.20, 0.0, 0.88],
            [0.0, 0.0, 0.79],
        ],
        dtype=np.float32,
    )
    poses[:, 2, 2] = -1.0
    candidates = GraspCandidates(
        poses=poses,
        scores=np.array([0.65, 0.80, 0.95, 0.99], dtype=np.float32),
        widths=np.array([0.05, 0.09, 0.05, 0.05], dtype=np.float32),
        centers_2d=np.array([[50, 42], [54, 42], [90, 42], [52, 42]], dtype=np.float32),
    )
    target = PerceivedObject(
        object_id=0,
        object_name=target_spec.object_name,
        class_name=target_spec.class_name,
        confidence=1.0,
        bbox_xyxy=(35.0, 25.0, 70.0, 58.0),
        mask=mask,
        center_2d=(52.0, 42.0),
        center_3d=np.array([0.0, 0.0, 0.86], dtype=np.float32),
        waste_category=target_spec.waste_category,
    )
    filtered = filter_grasps_by_mask(candidates, target.mask, margin=5)
    selected = select_best_grasp(filtered, target, table_surface_z=0.81)
    if selected is None:
        raise RuntimeError("Synthetic selector failed to choose a grasp.")
    return {
        "filtered_count": int(len(filtered.scores)),
        "selected_score": selected.score,
        "selected_width_m": selected.width,
        "selected_center_2d": selected.center_2d,
        "selected_pose_world": selected.pose_world.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 319 visual grasp pipeline coordinator.")
    parser.add_argument("--target_object", default="")
    parser.add_argument("--target_category", default="")
    parser.add_argument("--perception_backend", choices=["gt", "yolo", "grounded_sam"], default="gt")
    parser.add_argument("--classifier_backend", choices=["table", "ollama"], default="table")
    parser.add_argument("--ollama_model", default="qwen2.5")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--download_missing_models", action="store_true")
    parser.add_argument("--dry_run_graspnet", action="store_true", help="Run selector smoke test without camera data.")
    args = parser.parse_args()

    maybe_prepare_deps(args.download_missing_models)
    target = choose_target(args.target_object or None, args.target_category or None)
    waste = classify_waste(target.object_name, target.class_name, backend=args.classifier_backend, ollama_model=args.ollama_model)
    result = {
        "target_object": target.object_name,
        "class_name": target.class_name,
        "waste_category": waste.category,
        "classifier_backend": waste.backend,
        "perception_backend": args.perception_backend,
        "checkpoint": str(Path(args.checkpoint)),
    }
    if args.dry_run_graspnet:
        result["grasp_selection"] = synthetic_grasp_selection(target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
