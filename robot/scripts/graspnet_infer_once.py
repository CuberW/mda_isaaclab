#!/usr/bin/env python
"""Run one GraspNet RGB-D inference from an NPZ payload.

This script is intentionally small so it can be called from Windows through
WSL. The viewer stays native on Windows while the CUDA-only GraspNet backend can
run in the prepared Linux stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robot_common.execution.grasp_estimators import GraspNetEstimator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = np.load(args.payload, allow_pickle=True)
    rgb = data["rgb"]
    depth = data["depth"]
    intrinsics = json.loads(str(data["intrinsics_json"]))
    mask = data["mask"] if "mask" in data.files else None

    pose = GraspNetEstimator(required=True).estimate(rgb, depth, intrinsics, mask=mask)
    payload = {
        "position": pose.position.astype(float).tolist(),
        "rotation": pose.rotation.astype(float).tolist(),
        "approach_axis": pose.approach_axis.astype(float).tolist(),
        "jaw_axis": pose.jaw_axis.astype(float).tolist(),
        "score": float(pose.score),
        "width": float(pose.width),
        "source": pose.source,
        "metadata": pose.metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
