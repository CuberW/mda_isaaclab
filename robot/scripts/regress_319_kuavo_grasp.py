"""Fast local regression for the Task 3.19 Kuavo grasp loop.

This script intentionally bypasses perception and bin delivery. It exists to
keep the manipulation bug small: can the current Kuavo model navigate to a
reachable base pose, close the official claw on a real scene object, attach
only after verified finger contact, and lift?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_319_garbage_sort import GarbageSortingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Regress Kuavo 3.19 grasp only")
    parser.add_argument("--config", default="configs/task_319_kuavo_wheel.yaml")
    parser.add_argument("--target", default="trash_01")
    parser.add_argument("--out", default="results/debug/kuavo_grasp_regression.json")
    args = parser.parse_args()

    pipeline = GarbageSortingPipeline(args.config)
    try:
        result = pipeline.run_kuavo_grasp_regression(args.target)
        out_path = PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok") and result.get("attached") and result.get("lift_delta", 0.0) > 0.05 else 1
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
