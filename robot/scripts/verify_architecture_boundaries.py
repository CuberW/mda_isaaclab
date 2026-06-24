#!/usr/bin/env python
"""Verify perception-planning-control architecture import boundaries.

Task pipelines are orchestration modules. They may use infra/decision helpers,
but robot capability imports should go through the public seams:
``perception``, ``planning``, and ``control``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIRS = [
    PROJECT_ROOT / "task_319_garbage_sort",
    PROJECT_ROOT / "task_22_dual_arm",
]


def _task_py_files():
    """Yield all .py files in task directories."""
    for d in TASK_DIRS:
        yield from sorted(d.glob("*.py"))

DISALLOWED_PREFIXES = (
    "robot_common.env",
    "robot_common.perception",
    "robot_common.execution",
)

ALLOWED_DIRECT_IMPORTS = {
    # Cross-cutting support stays in robot_common until a later migration.
    "robot_common.infra",
    "robot_common.decision",
}


def _import_module_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def _is_disallowed(module: str) -> bool:
    if any(module == allowed or module.startswith(f"{allowed}.") for allowed in ALLOWED_DIRECT_IMPORTS):
        return False
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in DISALLOWED_PREFIXES)


def check_task_import_boundaries() -> tuple[bool, list[str]]:
    violations: list[str] = []
    for path in _task_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for node in ast.walk(tree):
            for module in _import_module_names(node):
                if _is_disallowed(module):
                    violations.append(f"{rel}:{getattr(node, 'lineno', '?')} imports {module}")
    return not violations, violations


def main() -> int:
    ok, violations = check_task_import_boundaries()
    print("\n" + "=" * 60)
    print("Architecture Boundary Verification")
    print("=" * 60)
    if ok:
        print("[OK] task pipelines use perception/planning/control seams")
    else:
        print("[FAIL] task pipelines bypass architecture seams")
        for item in violations:
            print(f"  {item}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
