"""Waste classification for task-319 objects."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from task_319_garbage_sort.grasp_pipeline.perception.scene_observer import TASK319_OBJECT_BY_NAME

WASTE_CATEGORIES = ("可回收物", "厨余垃圾", "有害垃圾", "其他垃圾")

CLASS_NAME_TABLE = {
    "cracker box": "可回收物",
    "sugar box": "可回收物",
    "metal can": "可回收物",
    "plastic bottle": "可回收物",
    "banana": "厨余垃圾",
    "food can": "厨余垃圾",
    "bleach cleanser": "有害垃圾",
    "marker": "有害垃圾",
    "foam brick": "其他垃圾",
    "ceramic mug": "其他垃圾",
}


@dataclass(slots=True)
class WasteClass:
    category: str
    category_id: int
    reason: str
    backend: str = "table"


def _normalize_category(category: str) -> str | None:
    for valid in WASTE_CATEGORIES:
        if category == valid or valid in category:
            return valid
    return None


def classify_by_table(object_name: str, class_name: str | None = None) -> WasteClass:
    spec = TASK319_OBJECT_BY_NAME.get(object_name)
    if spec is not None:
        category = spec.waste_category
        return WasteClass(category, WASTE_CATEGORIES.index(category), f"{spec.class_name} maps to {category}.")
    if class_name:
        category = CLASS_NAME_TABLE.get(class_name.lower())
        if category is not None:
            return WasteClass(category, WASTE_CATEGORIES.index(category), f"{class_name} maps to {category}.")
    return WasteClass("其他垃圾", WASTE_CATEGORIES.index("其他垃圾"), "Unknown object; conservative fallback.")


def classify_with_ollama(object_name: str, class_name: str | None = None, model: str = "qwen2.5") -> WasteClass:
    """Classify via a local Ollama model; fall back to the deterministic table."""

    prompt = (
        "你是垃圾分类专家。中国大陆四大垃圾分类标准是：可回收物、厨余垃圾、有害垃圾、其他垃圾。"
        f"请判断物体 {class_name or object_name} 属于哪一类。只回复 JSON："
        "{\"category\":\"类别名\",\"reason\":\"判断依据\"}"
    )
    try:
        completed = subprocess.run(
            ["ollama", "run", model, prompt],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        category = _normalize_category(str(payload.get("category", "")))
        if category is not None:
            return WasteClass(category, WASTE_CATEGORIES.index(category), str(payload.get("reason", "")), "ollama")
    except Exception:
        pass
    return classify_by_table(object_name, class_name)


def classify_waste(
    object_name: str,
    class_name: str | None = None,
    *,
    backend: str = "table",
    ollama_model: str = "qwen2.5",
) -> WasteClass:
    if backend == "ollama":
        return classify_with_ollama(object_name, class_name, ollama_model)
    return classify_by_table(object_name, class_name)
