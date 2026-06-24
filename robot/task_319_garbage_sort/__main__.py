"""Main entry point for Task 3.19 — python -m task_319_garbage_sort."""

from .constants import DEFAULT_TASK_319_CONFIG
from .pipeline import GarbageSortingPipeline


def main():
    """Main entry point for Task 3.19."""
    import argparse
    parser = argparse.ArgumentParser(description="Task 3.19: Garbage Sorting")
    parser.add_argument("--config", default=DEFAULT_TASK_319_CONFIG)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    pipeline = GarbageSortingPipeline(args.config)
    try:
        if args.eval:
            pipeline.run_evaluation(args.episodes)
        else:
            pipeline.run_single_episode()
            pipeline.metrics.print_summary()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
