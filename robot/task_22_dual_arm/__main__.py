"""Main entry point for Task 2.2 — python -m task_22_dual_arm."""

from .pipeline import DualArmVLAPipeline


def main():
    """Main entry point for Task 2.2."""
    import argparse
    parser = argparse.ArgumentParser(description="Task 2.2: Dual-Arm VLA Transport")
    parser.add_argument("--config", default="configs/task_22_dual_arm.yaml")
    parser.add_argument("--instruction", default="用双手把长杆放到指定区域")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    pipeline = DualArmVLAPipeline(args.config)
    try:
        if args.eval:
            pipeline.run_evaluation()
        else:
            pipeline.run_single_episode(args.instruction)
            pipeline.metrics.print_summary()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
