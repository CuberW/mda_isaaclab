"""Scene-object metadata for the task-319 YCB table scene."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SceneObjectSpec:
    object_name: str
    ycb_name: str
    class_name: str
    waste_category: str
    semantic_class: str


TASK319_OBJECTS: tuple[SceneObjectSpec, ...] = (
    SceneObjectSpec("trash_cracker_box_0", "003_cracker_box", "cracker box", "可回收物", "recyclable_paper"),
    SceneObjectSpec("trash_sugar_box_0", "004_sugar_box", "sugar box", "可回收物", "recyclable_paper"),
    SceneObjectSpec("trash_tomato_soup_can_0", "005_tomato_soup_can", "metal can", "可回收物", "recyclable_metal"),
    SceneObjectSpec("trash_mustard_bottle_0", "006_mustard_bottle", "plastic bottle", "可回收物", "recyclable_plastic"),
    SceneObjectSpec("trash_banana_0", "011_banana", "banana", "厨余垃圾", "kitchen_food"),
    SceneObjectSpec("trash_potted_meat_can_0", "010_potted_meat_can", "food can", "厨余垃圾", "kitchen_food_residue"),
    SceneObjectSpec("trash_bleach_cleanser_0", "021_bleach_cleanser", "bleach cleanser", "有害垃圾", "hazard_chemical"),
    SceneObjectSpec("trash_battery_0", "battery_block", "battery", "有害垃圾", "hazard_battery"),
    SceneObjectSpec("trash_foam_brick_0", "061_foam_brick", "foam brick", "其他垃圾", "other_waste"),
    SceneObjectSpec("trash_mug_0", "025_mug", "ceramic mug", "其他垃圾", "other_waste"),
)

TASK319_OBJECT_BY_NAME = {spec.object_name: spec for spec in TASK319_OBJECTS}


def ordered_targets(target_category: str | None = None) -> list[SceneObjectSpec]:
    """Return scene objects in the default grasp priority order."""

    priority = {"有害垃圾": 0, "厨余垃圾": 1, "可回收物": 2, "其他垃圾": 3}
    objects = list(TASK319_OBJECTS)
    if target_category:
        objects = [obj for obj in objects if obj.waste_category == target_category]
    objects.sort(key=lambda obj: (priority.get(obj.waste_category, 99), obj.object_name))
    return objects
