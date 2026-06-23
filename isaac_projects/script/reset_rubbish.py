import builtins
import json
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 0. 停止旧任务，防止恢复场景后又被自动执行
# ============================================================

OLD_TASK_NAMES = [
    "TRASHBOT_CLOSED_LOOP_TASK",
    "TRASHBOT_V27_HEAD_CAMERA_ACTION_SERVER",
    "TRASHBOT_V25_HEAD_CAMERA_SINGLE_TASK_V2",
    "TRASHBOT_V25_PREPARE_SEARCH_POSE_TASK",
    "TRASHBOT_V25_GOTO_TABLE_TASK",
]

for name in OLD_TASK_NAMES:
    old_task = getattr(builtins, name, None)
    if old_task is not None:
        try:
            old_task.cancel()
            print(f"[CANCEL] old task cancelled: {name}")
        except Exception as e:
            print(f"[WARN] failed to cancel old task {name}: {repr(e)}")
        try:
            delattr(builtins, name)
        except Exception:
            pass


# ============================================================
# 1. 文件清理
# ============================================================

CLEAR_HANDSHAKE_FILES = True

HANDSHAKE_FILES = [
    Path(r"D:\isaac_projects\closed_loop_task_plan.json"),
    Path(r"D:\isaac_projects\closed_loop_isaac_result.json"),

    Path(r"D:\isaac_projects\v2_visual_task_plan.json"),
    Path(r"D:\isaac_projects\v2_visual_execution_result.json"),
    Path(r"D:\isaac_projects\v2_closed_loop_isaac_result.json"),

    Path(r"D:\isaac_projects\v26_view_command.json"),
    Path(r"D:\isaac_projects\v26_view_result.json"),

    Path(r"D:\isaac_projects\v27_head_camera_action_result.json"),
]


# ============================================================
# 2. 场景路径
# ============================================================

ROBOT_PATH = "/World/TrashBotRobot"
CAMERA_PATH = "/World/TrashCamera"
TABLE_PATH = "/World/TrashSortingScene/table"

BIN_PATHS = [
    "/World/TrashSortingScene/bin_recyclable_blue",
    "/World/TrashSortingScene/bin_kitchen_green",
    "/World/TrashSortingScene/bin_hazardous_red",
    "/World/TrashSortingScene/bin_other_gray",
]


# ============================================================
# 3. 目标布局
#
# 重要：
# 这里的 XY 不再表示 USD Translate 原点位置，
# 而是表示“物体世界包围盒 bbox 的中心 XY”。
#
# 这样可以避免 GLB 模型原点不在几何中心导致的左右偏移。
# ============================================================

TRASH_LAYOUT_XY = {
    # 上方区域：纸盒、两个瓶子、罐子
    "/World/TrashSortingScene/trash_paper_box": (-1.00, 0.40),
    "/World/TrashSortingScene/trash_plastic_bottle": (-1.40, 0.20),
    "/World/TrashSortingScene/trash_dirty_bottle": (-1.00, 0.10),
    "/World/TrashSortingScene/trash_can": (-0.52, 0.30),

    # 中间区域：苹果、香蕉、电池、药盒
    "/World/TrashSortingScene/trash_apple_core": (-1.24, -0.12),
    "/World/TrashSortingScene/trash_banana_peel": (-0.86, -0.14),
    "/World/TrashSortingScene/trash_battery": (-0.50, -0.12),
    "/World/TrashSortingScene/trash_medicine_box": (-0.50, -0.34),

    # 下方区域：纸杯、纸团
    "/World/TrashSortingScene/trash_broken_cup": (-1.24, -0.40),
    "/World/TrashSortingScene/trash_tissue": (-0.78, -0.46),
}

ROBOT_HOME = Gf.Vec3d(1.20, -1.20, 0.00)

FALLBACK_TABLE_TOP_Z = 0.76

# 贴桌面时留一点间隙，避免视觉上闪烁或 z-fighting。
SURFACE_CLEARANCE = 0.002


# ============================================================
# 4. 相机参数
# ============================================================

APPLY_CAMERA_TUNING = False

CAMERA_FOCAL_LENGTH = 16.0
CAMERA_HORIZONTAL_APERTURE = 36.0
CAMERA_VERTICAL_APERTURE = 24.0


# ============================================================
# 5. USD 工具函数
# ============================================================

def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。请先打开 trashbot_scene.usd。")
    return stage


def make_sdf_path(path):
    if path is None:
        return None

    path_str = str(path).strip()

    if not path_str:
        return None

    if not path_str.startswith("/"):
        print(f"[WARN] invalid prim path: {path_str}")
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


def set_visible(prim, visible=True):
    imageable = UsdGeom.Imageable(prim)

    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


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
    """
    只设置 Translate，不 ClearXformOpOrder，不修改 Scale。
    """
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))
            return

    op = xform.AddTranslateOp()
    op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))


def compute_world_bbox(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.ComputeAlignedBox()


def bbox_info(stage, prim):
    box = compute_world_bbox(stage, prim)
    min_pt = box.GetMin()
    max_pt = box.GetMax()

    center = Gf.Vec3d(
        (float(min_pt[0]) + float(max_pt[0])) / 2.0,
        (float(min_pt[1]) + float(max_pt[1])) / 2.0,
        (float(min_pt[2]) + float(max_pt[2])) / 2.0,
    )

    size = Gf.Vec3d(
        float(max_pt[0]) - float(min_pt[0]),
        float(max_pt[1]) - float(min_pt[1]),
        float(max_pt[2]) - float(min_pt[2]),
    )

    return {
        "min": Gf.Vec3d(float(min_pt[0]), float(min_pt[1]), float(min_pt[2])),
        "max": Gf.Vec3d(float(max_pt[0]), float(max_pt[1]), float(max_pt[2])),
        "center": center,
        "size": size,
    }


def get_table_top_z(stage):
    table_prim = get_prim(stage, TABLE_PATH)

    if table_prim is None:
        print(f"[WARN] table not found: {TABLE_PATH}, use fallback z={FALLBACK_TABLE_TOP_Z}")
        return FALLBACK_TABLE_TOP_Z

    try:
        info = bbox_info(stage, table_prim)
        top_z = float(info["max"][2])
        print(f"[TABLE] top_z={top_z:.4f}")
        return top_z
    except Exception as e:
        print(f"[WARN] failed to compute table top z: {repr(e)}")
        return FALLBACK_TABLE_TOP_Z


def vec3_to_list(v):
    return [
        round(float(v[0]), 4),
        round(float(v[1]), 4),
        round(float(v[2]), 4),
    ]


# ============================================================
# 6. 清理闭环文件
# ============================================================

def clear_handshake_files():
    if not CLEAR_HANDSHAKE_FILES:
        return

    for path in HANDSHAKE_FILES:
        try:
            if path.exists():
                path.unlink()
                print(f"[DELETE] {path}")
        except Exception as e:
            print(f"[WARN] failed to delete {path}: {repr(e)}")


# ============================================================
# 7. 恢复垃圾桶、机器人、相机、垃圾
# ============================================================

def restore_bins(stage):
    for path in BIN_PATHS:
        prim = get_prim(stage, path)

        if prim is None:
            print(f"[MISS BIN] {path}")
            continue

        set_visible(prim, True)
        print(f"[VISIBLE BIN] {path}")


def restore_robot(stage):
    robot_prim = get_prim(stage, ROBOT_PATH)

    if robot_prim is None:
        print(f"[MISS ROBOT] {ROBOT_PATH}")
        return

    set_visible(robot_prim, True)
    set_translate(robot_prim, ROBOT_HOME)

    print(f"[RESTORE ROBOT] {ROBOT_PATH} -> {ROBOT_HOME}")


def tune_camera(stage):
    if not APPLY_CAMERA_TUNING:
        return

    camera_prim = get_prim(stage, CAMERA_PATH)

    if camera_prim is None:
        print(f"[MISS CAMERA] {CAMERA_PATH}")
        return

    camera = UsdGeom.Camera(camera_prim)

    camera.GetFocalLengthAttr().Set(float(CAMERA_FOCAL_LENGTH))
    camera.GetHorizontalApertureAttr().Set(float(CAMERA_HORIZONTAL_APERTURE))
    camera.GetVerticalApertureAttr().Set(float(CAMERA_VERTICAL_APERTURE))

    print(f"[CAMERA] focal_length={CAMERA_FOCAL_LENGTH}")
    print(f"[CAMERA] horizontal_aperture={CAMERA_HORIZONTAL_APERTURE}")
    print(f"[CAMERA] vertical_aperture={CAMERA_VERTICAL_APERTURE}")


def align_object_bbox_to_target(stage, prim, target_x, target_y, table_top_z):
    """
    用 bbox 对齐物体，而不是用模型原点或 half height。

    目标：
    1. bbox center x/y = target_x/target_y
    2. bbox min_z = table_top_z + SURFACE_CLEARANCE

    这样可以解决：
    - GLB 原点不在几何中心；
    - 物体中心不在桌面上；
    - half height 假设失效；
    - 物体悬浮或嵌入桌面。
    """
    before_translate = get_translate(prim)
    before_bbox = bbox_info(stage, prim)

    before_center = before_bbox["center"]
    before_min = before_bbox["min"]

    target_min_z = float(table_top_z + SURFACE_CLEARANCE)

    delta = Gf.Vec3d(
        float(target_x - before_center[0]),
        float(target_y - before_center[1]),
        float(target_min_z - before_min[2]),
    )

    target_translate = Gf.Vec3d(
        float(before_translate[0] + delta[0]),
        float(before_translate[1] + delta[1]),
        float(before_translate[2] + delta[2]),
    )

    set_translate(prim, target_translate)

    after_translate = get_translate(prim)
    after_bbox = bbox_info(stage, prim)

    return {
        "before_translate": vec3_to_list(before_translate),
        "after_translate": vec3_to_list(after_translate),

        "before_bbox_min": vec3_to_list(before_bbox["min"]),
        "before_bbox_max": vec3_to_list(before_bbox["max"]),
        "before_bbox_center": vec3_to_list(before_bbox["center"]),
        "before_bbox_size": vec3_to_list(before_bbox["size"]),

        "after_bbox_min": vec3_to_list(after_bbox["min"]),
        "after_bbox_max": vec3_to_list(after_bbox["max"]),
        "after_bbox_center": vec3_to_list(after_bbox["center"]),
        "after_bbox_size": vec3_to_list(after_bbox["size"]),

        "target_bbox_center_xy": [
            round(float(target_x), 4),
            round(float(target_y), 4),
        ],
        "target_bbox_min_z": round(float(target_min_z), 4),

        "delta_translate": vec3_to_list(delta),

        "final_bottom_error_z": round(float(after_bbox["min"][2] - target_min_z), 6),
        "final_center_error_xy": [
            round(float(after_bbox["center"][0] - target_x), 6),
            round(float(after_bbox["center"][1] - target_y), 6),
        ],
    }


def restore_trash_objects(stage):
    table_top_z = get_table_top_z(stage)

    restored = []
    missing = []

    for path, xy in TRASH_LAYOUT_XY.items():
        target_x, target_y = xy

        prim = get_prim(stage, path)

        if prim is None:
            print(f"[MISS TRASH] {path}")
            missing.append(path)
            continue

        set_visible(prim, True)

        try:
            info = align_object_bbox_to_target(
                stage=stage,
                prim=prim,
                target_x=float(target_x),
                target_y=float(target_y),
                table_top_z=float(table_top_z),
            )

            restored_item = {
                "path": path,
                **info,
            }

            restored.append(restored_item)

            after_center = info["after_bbox_center"]
            after_min = info["after_bbox_min"]

            print(
                f"[RESTORE TRASH] {path}\n"
                f"  bbox_center_xy -> ({after_center[0]:.4f}, {after_center[1]:.4f})\n"
                f"  bbox_min_z     -> {after_min[2]:.4f}\n"
                f"  bottom_err_z   -> {info['final_bottom_error_z']}\n"
                f"  center_err_xy  -> {info['final_center_error_xy']}"
            )

        except Exception as e:
            print(f"[WARN] failed to restore trash {path}: {repr(e)}")
            missing.append(path)

    return restored, missing


def save_restore_log(restored, missing):
    log_path = Path(r"D:\isaac_projects\target_layout_restore_log.json")
    gt_path = Path(r"D:\isaac_projects\v27_object_ground_truth_layout.json")

    data = {
        "message": "Trash objects restored by bbox center XY and bbox bottom Z.",
        "placement_rule": "bbox_center_xy_to_layout_and_bbox_min_z_to_table_top",
        "num_restored": len(restored),
        "num_missing": len(missing),
        "surface_clearance": SURFACE_CLEARANCE,
        "camera_focal_length": CAMERA_FOCAL_LENGTH,
        "restored": restored,
        "missing": missing,
    }

    gt = {
        "message": "Ground-truth layout after restoration. XY uses bbox center, not USD translate origin.",
        "objects": [
            {
                "path": item["path"],
                "bbox_center_xyz": item["after_bbox_center"],
                "bbox_min_xyz": item["after_bbox_min"],
                "bbox_max_xyz": item["after_bbox_max"],
                "bbox_size_xyz": item["after_bbox_size"],
                "translate_xyz": item["after_translate"],
            }
            for item in restored
        ],
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {log_path}")
    except Exception as e:
        print(f"[WARN] failed to save restore log: {repr(e)}")

    try:
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {gt_path}")
    except Exception as e:
        print(f"[WARN] failed to save gt layout: {repr(e)}")


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("[START] Restore Target Trash Layout")
    print("=" * 80)

    stage = get_stage()

    clear_handshake_files()
    restore_bins(stage)
    restore_robot(stage)
    tune_camera(stage)

    restored, missing = restore_trash_objects(stage)

    save_restore_log(restored, missing)

    print("=" * 80)
    print(f"[DONE] restored={len(restored)}, missing={len(missing)}")
    print("[NOTE] 建议另存场景：D:/isaac_projects/trashbot_scene_v27_bbox_aligned.usd")
    print("=" * 80)


main()