"""Task-319 grasp-and-sort state-machine entrypoint.

This wrapper keeps the public command focused on the real task chain while the
shared scene/perception/control implementation stays in visual_grasp_record_demo.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if "--debug_cube_simple_topdown" in sys.argv[1:]:
        sys.argv = [str(script_dir / "topdown_cube_grasp_test.py")] + [
            arg for arg in sys.argv[1:] if arg != "--debug_cube_simple_topdown"
        ]
        runpy.run_path(str(script_dir / "topdown_cube_grasp_test.py"), run_name="__main__")
    else:
        runpy.run_path(str(script_dir / "visual_grasp_record_demo.py"), run_name="__main__")
