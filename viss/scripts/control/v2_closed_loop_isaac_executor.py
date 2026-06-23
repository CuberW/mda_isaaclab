import asyncio
import builtins
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 0. 停止旧任务
# ============================================================

OLD_TASK_NAME = "TRASHBOT_V2_CLOSED_LOOP_TASK"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old V2 closed-loop task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old task: {repr(e)}")


# ============================================================
# 1. 文件路径
# ============================================================

PLAN_JSON = Path(r"D:\isaac_projects\v2_visual_task_plan.json")
RESULT_JSON = Path(r"D:\isaac_projects\v2_closed_loop_isaac_result.json")
CALIB_JSON = Path(r"D:\isaac_projects\v2_pixel_table_calibration.json")


# ============================================================
# 2. 场景路径
# ============================================================

ROBOT_PATH = "/World/TrashBotRobot"
TABLE_PATH = "/World/TrashSortingScene/table"

BIN_PATHS = {
    "bin_recyclable_blue": "/World/TrashSortingScene/bin_recyclable_blue",
    "bin_kitchen_green": "/World/TrashSortingScene/bin_kitchen_green",
    "bin_hazardous_red": "/World/TrashSortingScene/bin_hazardous_red",
    "bin_other_gray": "/World/TrashSortingScene/bin_other_gray",
}


# ============================================================
# 3. 物体映射
# ============================================================

HIGH_TRUST_RAW_TO_PRIM = {
    "can": ["/World/TrashSortingScene/trash_can"],
    "battery": ["/World/TrashSortingScene/trash_battery"],
    "battery1": ["/World/TrashSortingScene/trash_battery"],
    "battery5": ["/World/TrashSortingScene/trash_battery"],
    "drugbox": ["/World/TrashSortingScene/trash_medicine_box"],
    "drug": ["/World/TrashSortingScene/trash_medicine_box"],
    "drugbag": ["/World/TrashSortingScene/trash_medicine_box"],
    "papercup": ["/World/TrashSortingScene/trash_broken_cup"],
}

MEDIUM_TRUST_RAW_TO_PRIM = {
    "paper": [
        "/World/TrashSortingScene/trash_tissue",
        "/World/TrashSortingScene/trash_paper_box",
    ],
    "bottle": [
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_plastic_bottle",
    ],
    "bottle2": [
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_plastic_bottle",
    ],
}

LOW_TRUST_RAW_TO_PRIM = {
    "potato": ["/World/TrashSortingScene/trash_apple_core"],
    "potatocut": ["/World/TrashSortingScene/trash_banana_peel"],
    "rabbitcut": ["/World/TrashSortingScene/trash_banana_peel"],
    "mooli": ["/World/TrashSortingScene/trash_apple_core"],
    "brick": ["/World/TrashSortingScene/trash_paper_box"],
    "china": ["/World/TrashSortingScene/trash_broken_cup"],
    "stone": ["/World/TrashSortingScene/trash_broken_cup"],
    "capsule": ["/World/TrashSortingScene/trash_medicine_box"],
}

RAW_CLASS_TO_PRIMS = {}
RAW_CLASS_TO_PRIMS.update(HIGH_TRUST_RAW_TO_PRIM)
RAW_CLASS_TO_PRIMS.update(MEDIUM_TRUST_RAW_TO_PRIM)
RAW_CLASS_TO_PRIMS.update(LOW_TRUST_RAW_TO_PRIM)

CATEGORY_TO_PRIMS = {
    "recyclable": [
        "/World/TrashSortingScene/trash_plastic_bottle",
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_can",
        "/World/TrashSortingScene/trash_tissue",
        "/World/TrashSortingScene/trash_paper_box",
        "/World/TrashSortingScene/trash_broken_cup",
    ],
    "kitchen": [
        "/World/TrashSortingScene/trash_apple_core",
        "/World/TrashSortingScene/trash_banana_peel",
    ],
    "hazardous": [
        "/World/TrashSortingScene/trash_battery",
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "other": [
        "/World/TrashSortingScene/trash_paper_box",
    ],
}


CATEGORY_CN = {
    "recyclable": "可回收垃圾",
    "kitchen": "厨余垃圾",
    "hazardous": "有害垃圾",
    "other": "其他垃圾",
    "unknown": "未知类别",
}

BIN_CN = {
    "bin_recyclable_blue": "蓝色可回收垃圾桶",
    "bin_kitchen_green": "绿色厨余垃圾桶",
    "bin_hazardous_red": "红色有害垃圾桶",
    "bin_other_gray": "灰色其他垃圾桶",
    "unknown": "未知垃圾桶",
}


# ============================================================
# 4. 参数
# ============================================================

ROBOT_HOME = Gf.Vec3d(1.20, -1.20, 0.00)
APPROACH_OFFSET = Gf.Vec3d(0.00, -0.55, 0.00)
GRIP_OFFSET = Gf.Vec3d(0.00, 0.35, 0.95)

DROP_HEIGHT_ABOVE_BIN = 0.85
DROP_INSIDE_HEIGHT = 0.20

MOVE_FRAMES = 55
PICK_FRAMES = 25
DROP_FRAMES = 25
RETURN_FRAMES = 45
WAIT_FRAMES_WHEN_IDLE = 20

HIDE_OBJECT_AFTER_DROP = True

MAX_ATTACH_DISTANCE = 0.25
ALLOW_CATEGORY_FALLBACK = False

INLIER_THRESHOLD_M = 0.18
GOOD_INLIER_RMSE_M = 0.08
WARN_INLIER_RMSE_M = 0.15
MIN_INLIERS = 4

PROCESSED_PLAN_IDS = set()
LAST_GOOD_CALIB = None


# ============================================================
# 5. USD 工具
# ============================================================

def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。")
    return stage


def make_sdf_path(path):
    if path is None:
        return None
    path_str = str(path).strip()
    if not path_str or not path_str.startswith("/"):
        return None
    return Sdf.Path(path_str)


def get_prim(stage, path):
    sdf_path = make_sdf_path(path)
    if sdf_path is None:
        return None
    try:
        prim = stage.GetPrimAtPath(sdf_path)
    except Exception as e:
        print(f"[WARN] GetPrimAtPath failed: {path}, error={repr(e)}")
        return None
    if not prim or not prim.IsValid():
        return None
    return prim


def get_translate(prim):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            if value is not None:
                return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))

    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))


def set_translate(prim, vec):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))
            return

    op = xform.AddTranslateOp()
    op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))


def set_visibility(prim, visible):
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def is_prim_visible(prim):
    try:
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.ComputeVisibility()
        return visibility != UsdGeom.Tokens.invisible
    except Exception:
        return True


def compute_world_bbox(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.ComputeAlignedBox()


def get_world_bbox_center_xy(stage, prim):
    try:
        box = compute_world_bbox(stage, prim)
        min_pt = box.GetMin()
        max_pt = box.GetMax()
        x = (float(min_pt[0]) + float(max_pt[0])) / 2.0
        y = (float(min_pt[1]) + float(max_pt[1])) / 2.0
        return x, y, "bbox_center"
    except Exception:
        pos = get_translate(prim)
        return float(pos[0]), float(pos[1]), "translate_fallback"


def get_table_top_z(stage):
    table_prim = get_prim(stage, TABLE_PATH)
    if table_prim is None:
        return 0.29
    try:
        box = compute_world_bbox(stage, table_prim)
        return float(box.GetMax()[2])
    except Exception:
        return 0.29


def vec_distance_xy(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def lerp_vec(a, b, t):
    return Gf.Vec3d(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


async def next_frame():
    await omni.kit.app.get_app().next_update_async()


async def pause_frames(num_frames):
    for _ in range(num_frames):
        await next_frame()


async def animate_prim(prim, start, end, frames):
    for i in range(frames + 1):
        t = ease(i / max(frames, 1))
        set_translate(prim, lerp_vec(start, end, t))
        await next_frame()


async def animate_robot_with_object(robot_prim, object_prim, robot_start, robot_end, grip_offset, frames):
    for i in range(frames + 1):
        t = ease(i / max(frames, 1))
        robot_pos = lerp_vec(robot_start, robot_end, t)
        object_pos = robot_pos + grip_offset
        set_translate(robot_prim, robot_pos)
        set_translate(object_prim, object_pos)
        await next_frame()


# ============================================================
# 6. JSON
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_plan():
    if not PLAN_JSON.exists():
        return None

    try:
        data = load_json(PLAN_JSON)
    except Exception as e:
        print(f"[WAIT] plan 暂时不可读：{repr(e)}")
        return None

    plan_id = data.get("plan_id")
    if not plan_id:
        return None

    if plan_id in PROCESSED_PLAN_IDS:
        return None

    if data.get("selected_task") is None:
        return None

    return data


# ============================================================
# 7. Homography 标定
# ============================================================

def get_task_centroid(task):
    centroid = task.get("centroid_px")
    if centroid is not None and len(centroid) == 2:
        return [float(centroid[0]), float(centroid[1])]

    bbox = task.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def get_mapping_by_level(level):
    if level == "high":
        return dict(HIGH_TRUST_RAW_TO_PRIM)

    if level == "medium":
        merged = {}
        merged.update(HIGH_TRUST_RAW_TO_PRIM)
        merged.update(MEDIUM_TRUST_RAW_TO_PRIM)
        return merged

    if level == "low":
        merged = {}
        merged.update(HIGH_TRUST_RAW_TO_PRIM)
        merged.update(MEDIUM_TRUST_RAW_TO_PRIM)
        merged.update(LOW_TRUST_RAW_TO_PRIM)
        return merged

    raise ValueError(f"Unknown trust level: {level}")


def build_candidate_pairs(stage, plan, trust_level):
    tasks = plan.get("tasks", [])
    mapping = get_mapping_by_level(trust_level)

    # Group tasks by raw_class
    tasks_by_class = {}
    for task in tasks:
        raw_class = task.get("raw_class_name", task.get("class_name", "unknown"))
        centroid = get_task_centroid(task)
        if centroid is not None:
            tasks_by_class.setdefault(raw_class, []).append((task, centroid))

    pairs = []
    
    for raw_class, task_list in tasks_by_class.items():
        candidate_paths = mapping.get(raw_class, [])
        if not candidate_paths:
            continue
            
        # Get all visible prims for this raw_class
        visible_prims = []
        for path in candidate_paths:
            prim = get_prim(stage, path)
            if prim is not None and is_prim_visible(prim):
                x, y, world_source = get_world_bbox_center_xy(stage, prim)
                visible_prims.append((path, prim, x, y, world_source))
                
        if not visible_prims:
            continue
            
        # Sort tasks by pixel u coordinate (ascending)
        task_list_sorted = sorted(task_list, key=lambda item: item[1][0])
        
        # Sort visible prims by world x coordinate (ascending)
        visible_prims_sorted = sorted(visible_prims, key=lambda item: item[2])
        
        # Match them one-to-one in sorted order
        for (task, centroid), prim_info in zip(task_list_sorted, visible_prims_sorted):
            path, prim, x, y, world_source = prim_info
            u, v = centroid
            pairs.append({
                "raw_class_name": raw_class,
                "task_id": task.get("task_id"),
                "confidence": float(task.get("confidence", 0.0)),
                "prim_path": path,
                "pixel_uv": [round(float(u), 3), round(float(v), 3)],
                "world_xy": [round(float(x), 4), round(float(y), 4)],
                "world_xy_source": world_source,
                "pair_trust": trust_level,
            })
            
    return pairs


def build_best_candidate_pairs(stage, plan):
    for level in ["high", "medium", "low"]:
        pairs = build_candidate_pairs(stage, plan, level)
        if len(pairs) >= 4:
            return pairs, level
    return pairs, "low"


def solve_homography(pixel_points, world_points):
    if len(pixel_points) < 4:
        raise RuntimeError(f"Need at least 4 pairs, got {len(pixel_points)}")

    rows = []
    for (u, v), (x, y) in zip(pixel_points, world_points):
        u = float(u)
        v = float(v)
        x = float(x)
        y = float(y)

        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])

    A = np.array(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1, :]
    H = h.reshape(3, 3)

    if abs(H[2, 2]) < 1e-12:
        raise RuntimeError("Invalid homography")

    return H / H[2, 2]


def apply_homography(H, u, v):
    p = np.array([float(u), float(v), 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-12:
        raise RuntimeError("Invalid homography projection")
    return float(q[0] / q[2]), float(q[1] / q[2])


def evaluate_homography(H, pairs):
    checked = []
    all_errors = []
    inlier_errors = []

    for idx, pair in enumerate(pairs):
        u, v = pair["pixel_uv"]
        x_gt, y_gt = pair["world_xy"]
        x_pred, y_pred = apply_homography(H, u, v)
        err = math.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
        is_inlier = err <= INLIER_THRESHOLD_M

        item = dict(pair)
        item["pair_index"] = idx
        item["pred_world_xy"] = [round(float(x_pred), 4), round(float(y_pred), 4)]
        item["error_xy_m"] = round(float(err), 4)
        item["is_inlier"] = bool(is_inlier)

        checked.append(item)
        all_errors.append(err)
        if is_inlier:
            inlier_errors.append(err)

    all_rmse = math.sqrt(sum(e * e for e in all_errors) / len(all_errors)) if all_errors else 999.0
    inlier_rmse = math.sqrt(sum(e * e for e in inlier_errors) / len(inlier_errors)) if inlier_errors else 999.0

    return {
        "checked_pairs": checked,
        "all_rmse": all_rmse,
        "all_max_error": max(all_errors) if all_errors else 999.0,
        "inlier_rmse": inlier_rmse,
        "inlier_max_error": max(inlier_errors) if inlier_errors else 999.0,
        "inlier_count": len(inlier_errors),
        "outlier_count": len(all_errors) - len(inlier_errors),
    }


def robust_fit_homography(pairs):
    if len(pairs) < 4:
        raise RuntimeError(f"Need at least 4 candidate pairs, got {len(pairs)}")

    best = None
    pair_indices = list(range(len(pairs)))

    for combo in itertools.combinations(pair_indices, 4):
        subset = [pairs[i] for i in combo]
        pixel_points = [p["pixel_uv"] for p in subset]
        world_points = [p["world_xy"] for p in subset]

        try:
            H = solve_homography(pixel_points, world_points)
            eval_result = evaluate_homography(H, pairs)
        except Exception:
            continue

        score = (
            eval_result["inlier_count"],
            -eval_result["inlier_rmse"],
            -eval_result["all_rmse"],
        )

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "combo": combo,
                "H": H,
                "eval": eval_result,
            }

    if best is None:
        raise RuntimeError("Failed to fit homography")

    initial_eval = best["eval"]
    inlier_pairs = [
        pair for pair, checked in zip(pairs, initial_eval["checked_pairs"])
        if checked["is_inlier"]
    ]

    if len(inlier_pairs) >= MIN_INLIERS:
        pixel_points = [p["pixel_uv"] for p in inlier_pairs]
        world_points = [p["world_xy"] for p in inlier_pairs]
        H_refit = solve_homography(pixel_points, world_points)
        final_eval = evaluate_homography(H_refit, pairs)
        return H_refit, final_eval, list(best["combo"]), len(inlier_pairs)

    return best["H"], initial_eval, list(best["combo"]), initial_eval["inlier_count"]


def calibrate_or_use_cache(stage, plan):
    global LAST_GOOD_CALIB

    pairs, trust_level_used = build_best_candidate_pairs(stage, plan)

    if len(pairs) >= 4:
        try:
            H, eval_result, initial_combo, used_inlier_count = robust_fit_homography(pairs)
            status = "ok"

            if eval_result["inlier_count"] < MIN_INLIERS:
                status = "bad_not_enough_inliers"
            elif eval_result["inlier_rmse"] <= GOOD_INLIER_RMSE_M:
                status = "ok"
            elif eval_result["inlier_rmse"] <= WARN_INLIER_RMSE_M:
                status = "warn_medium_inlier_error"
            else:
                status = "bad_large_inlier_error"

            calib = {
                "mode": "V2_4_CLOSED_LOOP_HOMOGRAPHY",
                "status": status,
                "trust_level_used": trust_level_used,
                "table_z": round(float(get_table_top_z(stage)), 4),
                "homography_pixel_to_world_xy": H.tolist(),
                "inlier_count": int(eval_result["inlier_count"]),
                "outlier_count": int(eval_result["outlier_count"]),
                "inlier_rmse_xy_m": round(float(eval_result["inlier_rmse"]), 4),
                "inlier_max_error_xy_m": round(float(eval_result["inlier_max_error"]), 4),
                "all_rmse_xy_m": round(float(eval_result["all_rmse"]), 4),
                "all_max_error_xy_m": round(float(eval_result["all_max_error"]), 4),
                "rmse_xy_m": round(float(eval_result["inlier_rmse"]), 4),
                "max_error_xy_m": round(float(eval_result["inlier_max_error"]), 4),
                "initial_combo": initial_combo,
                "used_inlier_count_for_refit": int(used_inlier_count),
                "pairs": eval_result["checked_pairs"],
                "calibration_source": "current_plan",
            }

            save_json_atomic(CALIB_JSON, calib)

            if status in ["ok", "warn_medium_inlier_error"]:
                LAST_GOOD_CALIB = calib
                return calib

            print(f"[WARN] current calibration bad: {status}")

        except Exception as e:
            print(f"[WARN] calibration failed: {repr(e)}")

    if LAST_GOOD_CALIB is not None:
        cached = dict(LAST_GOOD_CALIB)
        cached["calibration_source"] = "memory_cache"
        return cached

    if CALIB_JSON.exists():
        try:
            cached = load_json(CALIB_JSON)
            if cached.get("status") in ["ok", "warn_medium_inlier_error"]:
                cached["calibration_source"] = "file_cache"
                LAST_GOOD_CALIB = cached
                return cached
        except Exception:
            pass

    raise RuntimeError("No valid homography calibration available.")


def pixel_to_world_by_homography(calib, u, v):
    H = np.array(calib["homography_pixel_to_world_xy"], dtype=np.float64)
    x, y = apply_homography(H, u, v)
    z = float(calib.get("table_z", 0.29))
    return Gf.Vec3d(x, y, z)


# ============================================================
# 8. 任务选择与执行
# ============================================================

def get_task_raw_class(task):
    return task.get("raw_class_name", task.get("class_name", "unknown"))


def get_task_category(task):
    return task.get("garbage_category", task.get("category", "unknown"))


def get_task_target_bin(task):
    return task.get("target_bin", "unknown")


def unique_list(items):
    out = []
    seen = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def choose_object_for_visual_target(stage, task, visual_world):
    raw_cls = get_task_raw_class(task)
    category = get_task_category(task)

    raw_candidates = unique_list(RAW_CLASS_TO_PRIMS.get(raw_cls, []))
    scored_raw = []

    for path in raw_candidates:
        prim = get_prim(stage, path)
        if prim is None:
            continue
        if not is_prim_visible(prim):
            continue

        x, y, _ = get_world_bbox_center_xy(stage, prim)
        pos = Gf.Vec3d(x, y, float(visual_world[2]))
        dist = vec_distance_xy(pos, visual_world)
        scored_raw.append((dist, path, prim, "raw_class_candidate"))

    if scored_raw:
        scored_raw.sort(key=lambda x: x[0])
        dist, path, prim, reason = scored_raw[0]
        return {
            "prim": prim,
            "path": path,
            "distance_xy": dist,
            "selection_reason": reason,
            "strict_match": dist <= MAX_ATTACH_DISTANCE,
        }

    if not ALLOW_CATEGORY_FALLBACK:
        return {
            "prim": None,
            "path": None,
            "distance_xy": None,
            "selection_reason": "no_raw_class_candidate_and_category_fallback_disabled",
            "strict_match": False,
        }

    category_candidates = unique_list(CATEGORY_TO_PRIMS.get(category, []))
    scored_category = []

    for path in category_candidates:
        prim = get_prim(stage, path)
        if prim is None:
            continue
        if not is_prim_visible(prim):
            continue

        x, y, _ = get_world_bbox_center_xy(stage, prim)
        pos = Gf.Vec3d(x, y, float(visual_world[2]))
        dist = vec_distance_xy(pos, visual_world)
        scored_category.append((dist, path, prim, "category_fallback_candidate"))

    if not scored_category:
        return {
            "prim": None,
            "path": None,
            "distance_xy": None,
            "selection_reason": "no_candidate",
            "strict_match": False,
        }

    scored_category.sort(key=lambda x: x[0])
    dist, path, prim, reason = scored_category[0]
    return {
        "prim": prim,
        "path": path,
        "distance_xy": dist,
        "selection_reason": reason,
        "strict_match": dist <= MAX_ATTACH_DISTANCE,
    }


async def execute_plan(stage, robot_prim, plan):
    plan_id = plan["plan_id"]
    task = plan["selected_task"]

    raw_cls = get_task_raw_class(task)
    category = get_task_category(task)
    target_bin = get_task_target_bin(task)

    centroid = task.get("centroid_px")
    if centroid is None or len(centroid) != 2:
        raise RuntimeError("selected_task has no centroid_px")

    u = float(centroid[0])
    v = float(centroid[1])

    calib = calibrate_or_use_cache(stage, plan)
    visual_world = pixel_to_world_by_homography(calib, u, v)

    bin_path = BIN_PATHS.get(target_bin)
    bin_prim = get_prim(stage, bin_path) if bin_path else None

    obj_info = choose_object_for_visual_target(stage, task, visual_world)
    obj_prim = obj_info["prim"]
    obj_path = obj_info["path"]

    print("\n" + "=" * 80)
    print(f"[PLAN] {plan_id}")
    print(f"[SELECTED] raw={raw_cls}, category={category}, target_bin={target_bin}")
    print(f"[PIXEL] u={u:.2f}, v={v:.2f}")
    print(f"[VISUAL WORLD] {visual_world}")
    print(f"[CALIB] source={calib.get('calibration_source')}, status={calib.get('status')}, rmse={calib.get('rmse_xy_m')}")
    print(f"[ATTACH] {obj_path}, dist={obj_info['distance_xy']}, strict={obj_info['strict_match']}")

    start_time = time.time()

    record = {
        "plan_id": plan_id,
        "cycle_id": plan.get("cycle_id"),
        "mode": "V2_4_VISUAL_CLOSED_LOOP",
        "source_image": plan.get("source_image"),
        "task_id": task.get("task_id"),
        "raw_class_name": raw_cls,
        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),
        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),
        "confidence": float(task.get("confidence", 0.0)),
        "centroid_px": [round(u, 3), round(v, 3)],
        "visual_world_xyz": [
            round(float(visual_world[0]), 4),
            round(float(visual_world[1]), 4),
            round(float(visual_world[2]), 4),
        ],
        "homography_used": True,
        "calibration_status": calib.get("status"),
        "calibration_source": calib.get("calibration_source"),
        "calibration_rmse_xy_m": calib.get("rmse_xy_m"),
        "calibration_inlier_count": calib.get("inlier_count"),
        "attached_object_prim": obj_path,
        "attach_distance_xy": round(float(obj_info["distance_xy"]), 4) if obj_info["distance_xy"] is not None else None,
        "attach_selection_reason": obj_info["selection_reason"],
        "attach_strict_match": bool(obj_info["strict_match"]),
        "bin_prim": bin_path,
        "status": "pending",
        "action_completed": False,
        "visual_grasp_used": True,
        "model_coordinate_used_for_planning": False,
        "model_coordinate_used_for_mesh_animation": True,
        "duration_sec": 0.0,
    }

    if bin_prim is None:
        record["status"] = "failed_no_bin"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        return record

    if obj_prim is None:
        record["status"] = "failed_no_matching_object_prim"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        return record

    if not obj_info["strict_match"]:
        record["status"] = "failed_visual_object_mismatch"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        return record

    try:
        set_translate(robot_prim, ROBOT_HOME)
        await next_frame()

        robot_start = get_translate(robot_prim)
        obj_start = get_translate(obj_prim)
        bin_pos = get_translate(bin_prim)

        approach_pos = Gf.Vec3d(
            float(visual_world[0] + APPROACH_OFFSET[0]),
            float(visual_world[1] + APPROACH_OFFSET[1]),
            float(ROBOT_HOME[2]),
        )

        print(f"[APPROACH - VISUAL] {robot_start} -> {approach_pos}")
        await animate_prim(robot_prim, robot_start, approach_pos, MOVE_FRAMES)

        grip_pos = approach_pos + GRIP_OFFSET

        print(f"[GRASP] {obj_start} -> {grip_pos}")
        await animate_prim(obj_prim, obj_start, grip_pos, PICK_FRAMES)

        bin_approach_pos = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1] - 0.55),
            float(ROBOT_HOME[2]),
        )

        print(f"[TRANSPORT] -> {bin_approach_pos}")
        await animate_robot_with_object(
            robot_prim,
            obj_prim,
            approach_pos,
            bin_approach_pos,
            GRIP_OFFSET,
            MOVE_FRAMES,
        )

        object_current = bin_approach_pos + GRIP_OFFSET

        drop_above = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1]),
            float(bin_pos[2] + DROP_HEIGHT_ABOVE_BIN),
        )

        drop_inside = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1]),
            float(bin_pos[2] + DROP_INSIDE_HEIGHT),
        )

        print(f"[DROP] {object_current} -> {drop_above} -> {drop_inside}")
        await animate_prim(obj_prim, object_current, drop_above, DROP_FRAMES)
        await animate_prim(obj_prim, drop_above, drop_inside, DROP_FRAMES)

        if HIDE_OBJECT_AFTER_DROP:
            set_visibility(obj_prim, False)
            print(f"[HIDE] {obj_path}")

        robot_now = get_translate(robot_prim)
        print(f"[RETURN] {robot_now} -> {ROBOT_HOME}")
        await animate_prim(robot_prim, robot_now, ROBOT_HOME, RETURN_FRAMES)

        record["status"] = "success"
        record["action_completed"] = True

    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise

    except Exception as e:
        record["status"] = "failed_exception"
        record["error"] = repr(e)
        print(f"[FAILED_EXCEPTION] {repr(e)}")

    record["duration_sec"] = round(time.time() - start_time, 3)
    save_json_atomic(RESULT_JSON, record)

    print("[RESULT]")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"[SAVED] {RESULT_JSON}")

    return record


# ============================================================
# 9. 主循环
# ============================================================

async def main_loop():
    print("[START] v2_closed_loop_isaac_executor.py")
    print(f"[WAIT PLAN] {PLAN_JSON}")

    stage = get_stage()
    robot_prim = get_prim(stage, ROBOT_PATH)

    if robot_prim is None:
        raise RuntimeError(f"找不到机器人 Prim：{ROBOT_PATH}")

    set_translate(robot_prim, ROBOT_HOME)
    await next_frame()

    while True:
        plan = read_plan()

        if plan is None:
            await pause_frames(WAIT_FRAMES_WHEN_IDLE)
            continue

        plan_id = plan["plan_id"]
        PROCESSED_PLAN_IDS.add(plan_id)

        await execute_plan(stage, robot_prim, plan)
        await pause_frames(20)


task = asyncio.ensure_future(main_loop())
setattr(builtins, OLD_TASK_NAME, task)