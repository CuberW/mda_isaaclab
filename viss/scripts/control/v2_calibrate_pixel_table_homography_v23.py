import itertools
import json
import math
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 1. 文件路径
# ============================================================

PLAN_JSON = Path(r"D:\isaac_projects\v2_visual_task_plan.json")
CALIB_JSON = Path(r"D:\isaac_projects\v2_pixel_table_calibration.json")


# ============================================================
# 2. 场景路径
# ============================================================

TABLE_PATH = "/World/TrashSortingScene/table"


# ============================================================
# 3. 标定映射
# ============================================================

# 高可信类别：物体唯一、YOLO 框稳定、与训练类匹配较好
HIGH_TRUST_RAW_TO_PRIM = {
    "can": [
        "/World/TrashSortingScene/trash_can",
    ],
    "battery": [
        "/World/TrashSortingScene/trash_battery",
    ],
    "battery1": [
        "/World/TrashSortingScene/trash_battery",
    ],
    "battery5": [
        "/World/TrashSortingScene/trash_battery",
    ],
    "drugbox": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "drug": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "drugbag": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "papercup": [
        "/World/TrashSortingScene/trash_broken_cup",
    ],
}

# 中等可信类别：可作为候选，但可能存在一类多实例或形状导致中心点不稳定
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

# 低可信类别：仿真物体和 YOLO 训练类语义不完全一致，默认不参与标定
LOW_TRUST_RAW_TO_PRIM = {
    "potato": [
        "/World/TrashSortingScene/trash_apple_core",
    ],
    "potatocut": [
        "/World/TrashSortingScene/trash_banana_peel",
    ],
    "rabbitcut": [
        "/World/TrashSortingScene/trash_banana_peel",
    ],
    "mooli": [
        "/World/TrashSortingScene/trash_apple_core",
    ],
    "brick": [
        "/World/TrashSortingScene/trash_paper_box",
    ],
    "china": [
        "/World/TrashSortingScene/trash_broken_cup",
    ],
    "stone": [
        "/World/TrashSortingScene/trash_broken_cup",
    ],
    "capsule": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
}


# 标定参数
INLIER_THRESHOLD_M = 0.18
GOOD_INLIER_RMSE_M = 0.08
WARN_INLIER_RMSE_M = 0.15
MIN_INLIERS = 4


# ============================================================
# 4. USD 工具函数
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

    if not path_str:
        return None

    if not path_str.startswith("/"):
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


def compute_world_bbox(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )

    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.ComputeAlignedBox()


def get_world_bbox_center_xy(stage, prim):
    """
    使用世界包围盒中心作为视觉中心。
    GLB/USD 的 Translate 原点不一定在物体视觉中心，所以不能直接用 Translate。
    """
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
        print("[WARN] table not found, use fallback z=0.29")
        return 0.29

    try:
        box = compute_world_bbox(stage, table_prim)
        return float(box.GetMax()[2])
    except Exception:
        return 0.29


# ============================================================
# 5. JSON 工具
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


# ============================================================
# 6. 标定点构造
# ============================================================

def get_task_centroid(task):
    centroid = task.get("centroid_px")

    if centroid is not None and len(centroid) == 2:
        return [float(centroid[0]), float(centroid[1])]

    bbox = task.get("bbox_xyxy")

    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]

    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    ]


def get_mapping_by_level(level):
    if level == "high":
        return HIGH_TRUST_RAW_TO_PRIM

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
            
        # Get all prims for this raw_class
        visible_prims = []
        for path in candidate_paths:
            prim = get_prim(stage, path)
            if prim is not None:
                x, y, world_source = get_world_bbox_center_xy(stage, prim)
                visible_prims.append((path, prim, x, y, world_source))
                
        if not visible_prims:
            continue
            
        # Sort tasks by pixel u coordinate (ascending)
        task_list_sorted = sorted(task_list, key=lambda item: item[1][0])
        
        # Sort prims by world x coordinate (ascending)
        visible_prims_sorted = sorted(visible_prims, key=lambda item: item[2])
        
        # Match them one-to-one in sorted order
        for (task, centroid), prim_info in zip(task_list_sorted, visible_prims_sorted):
            path, prim, x, y, world_source = prim_info
            u, v = centroid
            
            if raw_class in HIGH_TRUST_RAW_TO_PRIM:
                pair_trust = "high"
            elif raw_class in MEDIUM_TRUST_RAW_TO_PRIM:
                pair_trust = "medium"
            else:
                pair_trust = "low"
                
            pairs.append({
                "raw_class_name": raw_class,
                "task_id": task.get("task_id"),
                "confidence": float(task.get("confidence", 0.0)),
                "prim_path": path,
                "pixel_uv": [round(float(u), 3), round(float(v), 3)],
                "world_xy": [round(float(x), 4), round(float(y), 4)],
                "world_xy_source": world_source,
                "pair_trust": pair_trust,
            })
            
    return pairs


def build_best_candidate_pairs(stage, plan):
    """
    优先使用高可信点。
    如果不足 4 个，再引入 medium；
    仍不足，再引入 low。
    """
    for level in ["high", "medium", "low"]:
        pairs = build_candidate_pairs(stage, plan, level)
        print(f"[PAIRS] trust_level={level}, count={len(pairs)}")

        if len(pairs) >= 4:
            return pairs, level

    return pairs, "low"


# ============================================================
# 7. 单应矩阵求解与评估
# ============================================================

def solve_homography(pixel_points, world_points):
    """
    求 H，使得：
      [x, y, 1]^T ~ H [u, v, 1]^T
    """
    if len(pixel_points) < 4:
        raise RuntimeError(f"Need at least 4 pairs, got {len(pixel_points)}")

    rows = []

    for (u, v), (x, y) in zip(pixel_points, world_points):
        u = float(u)
        v = float(v)
        x = float(x)
        y = float(y)

        rows.append([
            -u, -v, -1.0,
            0.0, 0.0, 0.0,
            u * x, v * x, x,
        ])

        rows.append([
            0.0, 0.0, 0.0,
            -u, -v, -1.0,
            u * y, v * y, y,
        ])

    A = np.array(rows, dtype=np.float64)

    _, _, vt = np.linalg.svd(A)
    h = vt[-1, :]
    H = h.reshape(3, 3)

    if abs(H[2, 2]) < 1e-12:
        raise RuntimeError("Invalid homography: H[2,2] too small")

    H = H / H[2, 2]

    return H


def apply_homography(H, u, v):
    p = np.array([float(u), float(v), 1.0], dtype=np.float64)
    q = H @ p

    if abs(q[2]) < 1e-12:
        raise RuntimeError("Invalid homography projection")

    x = float(q[0] / q[2])
    y = float(q[1] / q[2])

    return x, y


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
    all_max_error = max(all_errors) if all_errors else 999.0

    inlier_rmse = (
        math.sqrt(sum(e * e for e in inlier_errors) / len(inlier_errors))
        if inlier_errors else 999.0
    )
    inlier_max_error = max(inlier_errors) if inlier_errors else 999.0

    return {
        "checked_pairs": checked,
        "all_errors": all_errors,
        "inlier_errors": inlier_errors,
        "all_rmse": all_rmse,
        "all_max_error": all_max_error,
        "inlier_rmse": inlier_rmse,
        "inlier_max_error": inlier_max_error,
        "inlier_count": len(inlier_errors),
        "outlier_count": len(all_errors) - len(inlier_errors),
    }


def robust_fit_homography(pairs):
    """
    枚举 4 点组合，选择：
      1. 内点最多；
      2. 内点 RMSE 最小；
      3. 所有点 RMSE 最小。
    最后用内点重新拟合一次。
    """
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
        raise RuntimeError("Failed to fit any homography model.")

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

        return {
            "H": H_refit,
            "initial_combo": list(best["combo"]),
            "used_inlier_count_for_refit": len(inlier_pairs),
            "eval": final_eval,
        }

    return {
        "H": best["H"],
        "initial_combo": list(best["combo"]),
        "used_inlier_count_for_refit": initial_eval["inlier_count"],
        "eval": initial_eval,
    }


def determine_status(eval_result):
    inlier_count = eval_result["inlier_count"]
    inlier_rmse = eval_result["inlier_rmse"]

    if inlier_count < MIN_INLIERS:
        return "bad_not_enough_inliers"

    if inlier_rmse <= GOOD_INLIER_RMSE_M:
        return "ok"

    if inlier_rmse <= WARN_INLIER_RMSE_M:
        return "warn_medium_inlier_error"

    return "bad_large_inlier_error"


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("[START] V2.3 robust inlier homography calibration")
    print("=" * 80)

    stage = get_stage()
    plan = load_json(PLAN_JSON)

    image_width = int(plan.get("image_width", 0))
    image_height = int(plan.get("image_height", 0))
    table_z = get_table_top_z(stage)

    pairs, trust_level_used = build_best_candidate_pairs(stage, plan)

    print(f"[PAIRS USED] trust_level_used={trust_level_used}, count={len(pairs)}")
    print(json.dumps(pairs, ensure_ascii=False, indent=2))

    if len(pairs) < 4:
        result = {
            "mode": "V2_3_ROBUST_INLIER_HOMOGRAPHY",
            "status": "failed_not_enough_pairs",
            "num_pairs": len(pairs),
            "pairs": pairs,
            "message": "Need at least 4 calibration pairs.",
        }

        save_json_atomic(CALIB_JSON, result)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[SAVED] {CALIB_JSON}")
        raise RuntimeError("Not enough calibration pairs.")

    fit = robust_fit_homography(pairs)

    H = fit["H"]
    eval_result = fit["eval"]
    status = determine_status(eval_result)

    result = {
        "mode": "V2_3_ROBUST_INLIER_HOMOGRAPHY",
        "status": status,
        "source_plan": str(PLAN_JSON),
        "image_width": image_width,
        "image_height": image_height,
        "table_z": round(float(table_z), 4),

        "trust_level_used": trust_level_used,
        "num_pairs": len(pairs),
        "inlier_threshold_m": INLIER_THRESHOLD_M,
        "min_inliers": MIN_INLIERS,

        "inlier_count": int(eval_result["inlier_count"]),
        "outlier_count": int(eval_result["outlier_count"]),
        "used_inlier_count_for_refit": int(fit["used_inlier_count_for_refit"]),
        "initial_combo": fit["initial_combo"],

        "homography_pixel_to_world_xy": H.tolist(),

        # 注意：执行是否允许，主要看 inlier_rmse_xy_m
        "inlier_rmse_xy_m": round(float(eval_result["inlier_rmse"]), 4),
        "inlier_max_error_xy_m": round(float(eval_result["inlier_max_error"]), 4),

        # all_rmse 仅用于调试，包含 outlier，不作为是否可用的主判断
        "all_rmse_xy_m": round(float(eval_result["all_rmse"]), 4),
        "all_max_error_xy_m": round(float(eval_result["all_max_error"]), 4),

        # 为了兼容旧执行器，保留 rmse_xy_m 字段，但这里写 inlier rmse
        "rmse_xy_m": round(float(eval_result["inlier_rmse"]), 4),
        "max_error_xy_m": round(float(eval_result["inlier_max_error"]), 4),

        "pairs": eval_result["checked_pairs"],

        "note": (
            "V2.3 uses inlier-only RMSE as calibration quality. "
            "Outliers are kept in pairs for debugging but are not used to reject a valid calibration."
        ),
    }

    save_json_atomic(CALIB_JSON, result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[SAVED] {CALIB_JSON}")

    if status == "ok":
        print("[OK] 标定可用，可以执行 V2.1/V2.3 视觉抓取。")
    elif status == "warn_medium_inlier_error":
        print("[WARN] 标定可试运行，但执行后必须检查 attach_distance_xy。")
    else:
        print("[BAD] 标定不可用。请重新恢复布局、采图、YOLO、生成 plan。")

    print("=" * 80)


main()