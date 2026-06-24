"""
Metrics tracking for robot task evaluation.

Tracks: success rate, timing, errors, grasp quality, etc.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int = 0
    success: bool = False
    task_type: str = ""
    instruction: str = ""

    # Timing
    total_time: float = 0.0
    perception_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0

    # Task-specific
    num_steps: int = 0
    grasp_attempts: int = 0
    grasp_success: bool = False
    object_detected: bool = False
    object_class: str = ""
    detected_class: str = ""
    classification_correct: bool = False

    # Errors
    position_error: float = 0.0         # Final position error (m)
    orientation_error: float = 0.0      # Final orientation error (rad)
    sync_error: float = 0.0             # Dual-arm sync error (m)

    # Custom
    custom: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "total_time": round(self.total_time, 3),
            "perception_time": round(self.perception_time, 3),
            "planning_time": round(self.planning_time, 3),
            "execution_time": round(self.execution_time, 3),
            "num_steps": self.num_steps,
            "grasp_attempts": self.grasp_attempts,
            "grasp_success": self.grasp_success,
            "object_detected": self.object_detected,
            "object_class": self.object_class,
            "detected_class": self.detected_class,
            "classification_correct": self.classification_correct,
            "position_error": round(self.position_error, 4),
            "orientation_error": round(self.orientation_error, 4),
            "sync_error": round(self.sync_error, 4),
            "custom": self.custom,
        }


class MetricsTracker:
    """Tracks metrics across multiple episodes and generates statistics."""

    def __init__(self, task_name: str = ""):
        self.task_name = task_name
        self.episodes: list[EpisodeMetrics] = []
        self.start_time = time.time()

    def new_episode(self, episode_id: int = 0, instruction: str = "",
                    task_type: str = "") -> EpisodeMetrics:
        """Start tracking a new episode."""
        ep = EpisodeMetrics(
            episode_id=episode_id if episode_id else len(self.episodes),
            instruction=instruction,
            task_type=task_type or self.task_name,
        )
        self.episodes.append(ep)
        return ep

    def summary(self) -> dict:
        """Compute summary statistics across all episodes."""
        if not self.episodes:
            return {"error": "No episodes recorded"}

        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.success)
        grasp_successes = sum(1 for e in self.episodes if e.grasp_success)
        class_correct = sum(1 for e in self.episodes if e.classification_correct)
        detected = sum(1 for e in self.episodes if e.object_detected)

        times = [e.total_time for e in self.episodes]
        pos_errors = [e.position_error for e in self.episodes if e.success]

        return {
            "task_name": self.task_name,
            "total_episodes": n,
            "success_rate": round(successes / n, 3) if n else 0,
            "success_count": successes,
            "grasp_success_rate": round(grasp_successes / n, 3) if n else 0,
            "classification_accuracy": round(class_correct / n, 3) if n and any(e.object_class for e in self.episodes) else 0,
            "detection_rate": round(detected / n, 3) if n else 0,
            "avg_time": round(sum(times) / len(times), 3),
            "min_time": round(min(times), 3) if times else 0,
            "max_time": round(max(times), 3) if times else 0,
            "avg_position_error_mm": round(sum(pos_errors) / len(pos_errors) * 1000, 1) if pos_errors else 0,
            "total_elapsed": round(time.time() - self.start_time, 1),
        }

    def confusion_matrix(self) -> dict:
        """Build confusion matrix for classification tasks."""
        matrix = defaultdict(lambda: defaultdict(int))
        classes = set()
        for e in self.episodes:
            if e.object_class and e.detected_class:
                matrix[e.object_class][e.detected_class] += 1
                classes.add(e.object_class)
                classes.add(e.detected_class)
        return {
            "classes": sorted(classes),
            "matrix": {c: {d: matrix[c][d] for d in sorted(classes)} for c in sorted(classes)}
        }

    def save(self, path: str):
        """Save metrics to JSON."""
        data = {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        if any(e.object_class for e in self.episodes):
            data["confusion_matrix"] = self.confusion_matrix()

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Pretty print summary."""
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  Task: {s['task_name']}")
        print(f"  Episodes: {s['total_episodes']}")
        print(f"  Success Rate: {s['success_rate']*100:.1f}% ({s['success_count']}/{s['total_episodes']})")
        if s.get('grasp_success_rate', 0) > 0:
            print(f"  Grasp Success: {s['grasp_success_rate']*100:.1f}%")
        if s.get('classification_accuracy', 0) > 0:
            print(f"  Classification Accuracy: {s['classification_accuracy']*100:.1f}%")
        if s.get('detection_rate', 0) > 0:
            print(f"  Detection Rate: {s['detection_rate']*100:.1f}%")
        print(f"  Avg Time: {s['avg_time']:.2f}s")
        if s.get('avg_position_error_mm', 0) > 0:
            print(f"  Avg Position Error: {s['avg_position_error_mm']:.1f}mm")
        print(f"  Total Elapsed: {s['total_elapsed']:.1f}s")
        print(f"{'='*60}\n")
