"""Task-319 grasp-and-sort state-machine entrypoint.

This wrapper keeps the public command focused on the real task chain while the
shared scene/perception/control implementation stays in visual_grasp_record_demo.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("visual_grasp_record_demo.py")), run_name="__main__")
