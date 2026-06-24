"""Shared configuration seam for cross-task tunables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).resolve().parent
SHARED_PARAMETERS = CONFIG_ROOT / "shared_parameters.yaml"


@lru_cache(maxsize=1)
def load_shared_parameters() -> dict[str, Any]:
    if not SHARED_PARAMETERS.exists():
        return {}
    return yaml.safe_load(SHARED_PARAMETERS.read_text(encoding="utf-8")) or {}


def shared_config(path: str, default: Any = None) -> Any:
    current: Any = load_shared_parameters()
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


__all__ = ["CONFIG_ROOT", "SHARED_PARAMETERS", "load_shared_parameters", "shared_config"]

