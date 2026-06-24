#!/usr/bin/env python
"""
Unified runner for active robot tasks.

Usage:
  python run_all.py --task 319             # Run task 3.19 only
  python run_all.py --task all --eval      # Run all active tasks evaluation
  python run_all.py --task 22 --instruction "用双手把长杆放到指定区域"
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from task_319_garbage_sort.constants import DEFAULT_TASK_319_CONFIG
from robot_common.infra.logging import setup_logger
from robot_common.infra.metrics import MetricsTracker

logger = setup_logger("run_all")


def _parse_csv_names(value: str) -> list[str]:
    """Parse comma/semicolon separated object names from CLI."""
    if not value:
        return []
    return [
        item.strip()
        for chunk in str(value).split(";")
        for item in chunk.split(",")
        if item.strip()
    ]


def _override_319_demo_objects(pipeline, demo_objects: int | None):
    """Override task_319 demo object count without editing YAML files."""
    if demo_objects is None:
        return
    count = max(0, int(demo_objects))
    task_cfg = pipeline._raw_config.setdefault("task", {})
    task_cfg["demo_objects"] = count
    task_cfg["viewer_demo_objects"] = count


def run_task_319(config: str = DEFAULT_TASK_319_CONFIG,
                 eval_mode: bool = False, episodes: int = 3,
                 show_viewer: bool = True, full_stack: bool = False,
                 trash_objects: list[str] | None = None,
                 demo_objects: int | None = None,
                 live_debug: bool = False):
    """Run Task 3.19: Garbage Sorting."""
    from task_319_garbage_sort import GarbageSortingPipeline
    logger.info("=" * 60)
    logger.info("Task 3.19: Kuavo Garbage Sorting")
    logger.info("=" * 60)

    pipeline = GarbageSortingPipeline(config)
    try:
        _override_319_demo_objects(pipeline, demo_objects)
        if live_debug and hasattr(pipeline, "enable_live_debug"):
            pipeline.enable_live_debug(True)
        if full_stack and hasattr(pipeline, "validate_mature_backends"):
            pipeline.validate_mature_backends()
        if hasattr(pipeline, "prewarm"):
            pipeline.prewarm()
        if show_viewer:
            _run_with_viewer(
                pipeline,
                eval_mode,
                episodes,
                None,
                run_kwargs={"trash_objects": trash_objects} if trash_objects else None,
            )
            return pipeline.metrics.summary(), _pipeline_succeeded(pipeline)
        if eval_mode:
            pipeline.run_evaluation(episodes)
        else:
            if trash_objects:
                pipeline.run_single_episode(trash_objects=trash_objects)
            else:
                pipeline.run_single_episode()
            pipeline.metrics.print_summary()
        return pipeline.metrics.summary(), _pipeline_succeeded(pipeline)
    finally:
        pipeline.cleanup()


def _run_with_viewer(
    pipeline,
    eval_mode: bool,
    episodes: int,
    instruction: str | None,
    run_kwargs: dict | None = None,
):
    """Wrapper to run pipeline with MuJoCo viewer. Viewer stays open after pipeline finishes."""
    import mujoco
    with mujoco.viewer.launch_passive(pipeline.env.model, pipeline.env.data) as v:
        pipeline.env.set_viewer(v)
        pipeline.env.render_mode = "viewer"
        setattr(pipeline, "visual_mode", True)
        if eval_mode:
            try:
                pipeline.run_evaluation(episodes)
            except TypeError:
                pipeline.run_evaluation()
        else:
            if instruction is None:
                pipeline.run_single_episode(**(run_kwargs or {}))
            else:
                pipeline.run_single_episode(instruction)
        if hasattr(pipeline, 'metrics'):
            pipeline.metrics.print_summary()
        # Keep viewer alive until user closes it
        print("\nPipeline complete. Viewer stays open - close the window to exit.")
        while v.is_running():
            v.sync()
            time.sleep(0.033)  # ~30fps
    pipeline.cleanup()


def run_task_22(config: str = "configs/task_22_dual_arm.yaml",
                instruction: str = "", eval_mode: bool = False,
                show_viewer: bool = True, full_stack: bool = False):
    """Run Task 2.2: Dual-Arm VLA Transport."""
    from task_22_dual_arm import DualArmVLAPipeline
    logger.info("=" * 60)
    logger.info("Task 2.2: Dual-Arm VLA Collaborative Transport")
    logger.info("=" * 60)

    pipeline = DualArmVLAPipeline(config)
    try:
        if full_stack and hasattr(pipeline, "validate_mature_backends"):
            pipeline.validate_mature_backends()
        if hasattr(pipeline, "prewarm"):
            pipeline.prewarm()
        if show_viewer:
            _run_with_viewer(pipeline, eval_mode, 0, instruction)
            return pipeline.metrics.summary(), _pipeline_succeeded(pipeline)
        if eval_mode:
            pipeline.run_evaluation()
        else:
            pipeline.run_single_episode(instruction)
            pipeline.metrics.print_summary()
        return pipeline.metrics.summary(), _pipeline_succeeded(pipeline)
    finally:
        pipeline.cleanup()


def _summary_succeeded(summary: dict) -> bool:
    """Return True only when aggregate task metrics are acceptable."""
    if not summary:
        return False
    if float(summary.get("success_rate", 0.0)) <= 0.0:
        return False
    return True


def _pipeline_succeeded(pipeline) -> bool:
    """Validate per-episode success and motion trace quality."""
    if not _summary_succeeded(pipeline.metrics.summary()):
        return False
    for ep in pipeline.metrics.episodes:
        trace = ep.custom.get("motion_trace", {}) if isinstance(ep.custom, dict) else {}
        custom = ep.custom if isinstance(ep.custom, dict) else {}
        if trace and not trace.get("smooth", False):
            return False
        if ep.task_type == "garbage_sort" and ep.success:
            nav = custom.get("navigation_records", [])
            delivery = custom.get("bin_delivery_results", [])
            if not (
                ep.object_detected
                and ep.grasp_success
                and len(nav) >= 2
                and any(bool(item.get("success", False)) for item in delivery)
            ):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified Robot System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --task 319                    # Single episode garbage sorting
  python run_all.py --task 22 --instruction "用双手把长杆放到指定区域"
  python run_all.py --task all --eval             # Evaluate all active tasks
        """
    )
    parser.add_argument("--task", type=str, default="319",
                       choices=["319", "22", "all"],
                       help="Task to run (319=garbage, 22=dual-arm, all)")
    parser.add_argument("--config", type=str, default="",
                       help="Override config file path")
    parser.add_argument("--instruction", type=str, default="",
                       help="Natural language instruction")
    parser.add_argument("--eval", action="store_true",
                       help="Run full evaluation (multiple episodes)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of evaluation episodes (for task 319)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without MuJoCo viewer (offscreen mode)")
    parser.add_argument("--full", action="store_true",
                       help="Require mature external backends before running")
    parser.add_argument("--demo-objects", type=int, default=None,
                       help="Task 3.19 only: number of scene trash objects to process without editing YAML")
    parser.add_argument("--trash-objects", type=str, default="",
                       help="Task 3.19 only: comma-separated scene body names, e.g. trash_01,trash_04")
    parser.add_argument("--live-debug", action="store_true",
                       help="Task 3.19 only: show live perception/grasp/control debug panels")

    args = parser.parse_args()
    show_viewer = not args.headless
    trash_objects = _parse_csv_names(args.trash_objects)

    if args.task == "319":
        config = args.config or DEFAULT_TASK_319_CONFIG
        summary, ok = run_task_319(
            config,
            args.eval,
            args.episodes,
            show_viewer,
            args.full,
            trash_objects=trash_objects,
            demo_objects=args.demo_objects,
            live_debug=args.live_debug,
        )
        if not show_viewer and not ok:
            return 1

    elif args.task == "22":
        config = args.config or "configs/task_22_dual_arm.yaml"
        summary, ok = run_task_22(config, args.instruction, args.eval, show_viewer, args.full)
        if not show_viewer and not ok:
            return 1

    elif args.task == "all":
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING ACTIVE TASKS")
        logger.info("=" * 80)

        results = {}

        # Task 3.19
        config_319 = args.config or DEFAULT_TASK_319_CONFIG
        try:
            summary, ok = run_task_319(
                config_319,
                args.eval,
                args.episodes,
                show_viewer,
                args.full,
                trash_objects=trash_objects,
                demo_objects=args.demo_objects,
                live_debug=args.live_debug,
            )
            results["3.19"] = "OK" if ok else "FAIL: metrics/trace"
        except Exception as e:
            logger.error(f"Task 3.19 failed: {e}")
            results["3.19"] = f"FAIL: {e}"

        # Task 2.2
        config_22 = args.config or "configs/task_22_dual_arm.yaml"
        try:
            summary, ok = run_task_22(config_22, args.instruction, args.eval, show_viewer, args.full)
            results["2.2"] = "OK" if ok else "FAIL: metrics/trace"
        except Exception as e:
            logger.error(f"Task 2.2 failed: {e}")
            results["2.2"] = f"FAIL: {e}"

        logger.info("\n" + "=" * 80)
        logger.info("ALL TASKS COMPLETE")
        for task_id, status in results.items():
            logger.info(f"  Task {task_id}: {status}")
        logger.info("=" * 80)
        if not show_viewer and any(not str(status).startswith("OK") for status in results.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
