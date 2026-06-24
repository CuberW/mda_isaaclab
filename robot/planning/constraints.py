"""Constraint-planning seam.

ReKep-style constraint generation should live behind this seam when added.
Keeping the seam explicit prevents task pipelines from embedding constraint
construction directly in episode-control code.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConstraintSpec:
    """A planner-facing task constraint description."""

    name: str
    frame: str
    target: list[float]
    tolerance: float
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["ConstraintSpec"]

