"""
Visualization utilities for robot simulation results.

Supports:
  - Detection visualization (bounding boxes, masks)
  - Trajectory plotting
  - Metrics dashboard generation
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from robot_common.infra.logging import logger

# Conditional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def draw_detections(image: np.ndarray, detections: list,
                    color_map: dict = None) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: RGB image (H, W, 3), uint8
        detections: List of DetectionResult objects
        color_map: Optional dict mapping class_name → (R, G, B)

    Returns:
        Annotated image
    """
    if not HAS_PIL:
        logger.warning("PIL not available, returning original image")
        return image

    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    draw = ImageDraw.Draw(pil_img)

    # Default colors
    default_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0),
    ]

    for i, det in enumerate(detections):
        color = default_colors[i % len(default_colors)]
        if color_map and det.class_name in color_map:
            color = color_map[det.class_name]
            # Convert 0-1 to 0-255 if needed
            if all(0 <= c <= 1 for c in color):
                color = tuple(int(c * 255) for c in color)

        if det.bbox:
            x1, y1, x2, y2 = det.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{det.class_name} {det.confidence:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill=color)

        # Draw mask if available
        if det.mask is not None:
            mask_overlay = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
            mask_color = tuple(list(color) + [80])  # Semi-transparent
            mask_overlay[det.mask > 0] = mask_color
            mask_img = Image.fromarray(mask_overlay, 'RGBA')
            pil_img = pil_img.convert('RGBA')
            pil_img = Image.alpha_composite(pil_img, mask_img)
            pil_img = pil_img.convert('RGB')

    return np.array(pil_img)


def plot_metrics_summary(metrics_tracker, save_path: str = ""):
    """Generate metrics summary visualization.

    Creates:
      - Success rate bar chart
      - Timing breakdown pie chart
      - Per-episode metrics table
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available for plotting")
        return

    summary = metrics_tracker.summary()
    if "error" in summary:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Success rate gauge
    ax1 = axes[0]
    success_rate = summary.get("success_rate", 0)
    ax1.barh(["Success Rate"], [success_rate * 100], color="green" if success_rate >= 0.7 else "orange")
    ax1.barh(["Success Rate"], [100 - success_rate * 100],
             left=[success_rate * 100], color="lightgray")
    ax1.set_xlim(0, 100)
    ax1.set_title(f"Success Rate: {success_rate*100:.1f}%")
    ax1.text(success_rate * 100 / 2, 0, f"{success_rate*100:.1f}%",
             ha="center", va="center", fontsize=14, fontweight="bold")

    # 2. Sub-metrics
    ax2 = axes[1]
    sub_metrics = {
        "Grasp": summary.get("grasp_success_rate", 0),
        "Detection": summary.get("detection_rate", 0),
        "Classification": summary.get("classification_accuracy", 0),
    }
    labels = list(sub_metrics.keys())
    values = [v * 100 for v in sub_metrics.values()]
    colors = ["steelblue", "darkorange", "forestgreen"]
    bars = ax2.bar(labels, values, color=colors)
    ax2.set_ylim(0, 105)
    ax2.set_title("Sub-Metrics")
    for bar, val in zip(bars, values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # 3. Timing overview
    ax3 = axes[2]
    times = [e.total_time for e in metrics_tracker.episodes]
    if times:
        ax3.plot(range(1, len(times) + 1), times, 'o-', color="steelblue", markersize=8)
        avg = np.mean(times)
        ax3.axhline(y=avg, color="red", linestyle="--", label=f"Avg: {avg:.2f}s")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Time (s)")
        ax3.set_title("Episode Duration")
        ax3.legend()

    fig.suptitle(f"Task: {summary.get('task_name', 'Unknown')} | "
                 f"Episodes: {summary.get('total_episodes', 0)}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Metrics plot saved: {save_path}")
    else:
        # Save to default location
        path = Path(f"metrics_{summary.get('task_name', 'unknown')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Metrics plot saved: {path}")

    plt.close()


def plot_confusion_matrix(metrics_tracker, save_path: str = ""):
    """Plot confusion matrix for classification tasks."""
    if not HAS_MATPLOTLIB:
        return

    cm = metrics_tracker.confusion_matrix()
    if not cm or not cm.get("classes"):
        return

    classes = cm["classes"]
    matrix = cm["matrix"]
    n = len(classes)

    fig, ax = plt.subplots(figsize=(n * 1.2, n * 1.0))
    data = np.zeros((n, n))
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            data[i, j] = matrix.get(true_cls, {}).get(pred_cls, 0)

    im = ax.imshow(data, cmap="Blues")

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # Values in cells
    for i in range(n):
        for j in range(n):
            text_color = "white" if data[i, j] > data.max() / 2 else "black"
            ax.text(j, i, int(data[i, j]), ha="center", va="center", color=text_color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    path = save_path or f"confusion_matrix_{metrics_tracker.task_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def create_comparison_grid(images: List[np.ndarray], titles: List[str],
                           cols: int = 3, save_path: str = "") -> np.ndarray:
    """Create a grid of images for side-by-side comparison.

    Useful for: head camera vs wrist camera, before vs after grasp,
    clean vs dirty object classification.
    """
    if not HAS_MATPLOTLIB:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else axes

    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img if img.dtype == np.uint8 else img)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Also return as numpy array
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return result
