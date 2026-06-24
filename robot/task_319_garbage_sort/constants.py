"""Garbage sorting task constants — category mappings, bin positions, detection prompts."""

import numpy as np


DEFAULT_TASK_319_CONFIG = "configs/task_319_kuavo_wheel.yaml"


# Garbage categories (ASCII internal keys; display labels are handled in docs/UI).
CATEGORY_RECYCLABLE = "recyclable"
CATEGORY_KITCHEN = "kitchen_waste"
CATEGORY_HAZARDOUS = "hazardous"
CATEGORY_OTHER = "other"
GARBAGE_CATEGORIES = [
    CATEGORY_RECYCLABLE,
    CATEGORY_KITCHEN,
    CATEGORY_HAZARDOUS,
    CATEGORY_OTHER,
]

OBJECT_TO_CATEGORY = {
    "plastic_bottle": CATEGORY_RECYCLABLE,
    "newspaper": CATEGORY_RECYCLABLE,
    "aluminum_can": CATEGORY_RECYCLABLE,
    "glass_jar": CATEGORY_RECYCLABLE,
    "cardboard": CATEGORY_RECYCLABLE,
    "metal_can": CATEGORY_RECYCLABLE,
    "plastic_bag": CATEGORY_RECYCLABLE,
    "paper": CATEGORY_RECYCLABLE,
    "apple_core": CATEGORY_KITCHEN,
    "banana_peel": CATEGORY_KITCHEN,
    "food_scrap": CATEGORY_KITCHEN,
    "vegetable": CATEGORY_KITCHEN,
    "bread": CATEGORY_KITCHEN,
    "egg_shell": CATEGORY_KITCHEN,
    "battery": CATEGORY_HAZARDOUS,
    "medicine": CATEGORY_HAZARDOUS,
    "light_bulb": CATEGORY_HAZARDOUS,
    "paint_can": CATEGORY_HAZARDOUS,
    "tissue": CATEGORY_OTHER,
    "ceramic": CATEGORY_OTHER,
    "cigarette_butt": CATEGORY_OTHER,
}

SCENE_TRASH_TO_CATEGORY = {
    "trash_01": CATEGORY_RECYCLABLE,
    "trash_02": CATEGORY_RECYCLABLE,
    "trash_03": CATEGORY_RECYCLABLE,
    "trash_04": CATEGORY_KITCHEN,
    "trash_05": CATEGORY_KITCHEN,
    "trash_06": CATEGORY_KITCHEN,
    "trash_07": CATEGORY_HAZARDOUS,
    "trash_08": CATEGORY_HAZARDOUS,
    "trash_09": CATEGORY_HAZARDOUS,
    "trash_10": CATEGORY_OTHER,
    "trash_11": CATEGORY_OTHER,
    "trash_12": CATEGORY_OTHER,
}

COCO_CLASS_TO_CATEGORY = {
    "bottle": CATEGORY_RECYCLABLE,
    "cup": CATEGORY_RECYCLABLE,
    "book": CATEGORY_RECYCLABLE,
    "banana": CATEGORY_KITCHEN,
    "apple": CATEGORY_KITCHEN,
    "orange": CATEGORY_KITCHEN,
    "cell phone": CATEGORY_HAZARDOUS,
    "remote": CATEGORY_HAZARDOUS,
    "mouse": CATEGORY_HAZARDOUS,
    "sports ball": CATEGORY_OTHER,
    "teddy bear": CATEGORY_OTHER,
}

BIN_POSITIONS = {
    CATEGORY_RECYCLABLE: np.array([0.75, 0.45, 0.3]),
    CATEGORY_KITCHEN: np.array([0.95, 0.20, 0.3]),
    CATEGORY_HAZARDOUS: np.array([0.95, -0.45, 0.3]),
    CATEGORY_OTHER: np.array([0.75, -0.75, 0.3]),
}

BIN_BODY_BY_CATEGORY = {
    CATEGORY_RECYCLABLE: "bin_recyclable",
    CATEGORY_KITCHEN: "bin_kitchen",
    CATEGORY_HAZARDOUS: "bin_hazardous",
    CATEGORY_OTHER: "bin_other",
}

GARBAGE_DETECTION_PROMPTS = [
    "trash object",
    "bottle",
    "can",
    "cup",
    "box",
    "paper",
    "apple",
    "banana",
    "ball",
    "marker",
]
