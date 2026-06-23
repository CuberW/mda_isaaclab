import builtins
import json
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 0. 停止旧闭环任务，防止恢复后又被自动执行
# ============================================================

OLD_TASK_NAME = "TRASHBOT_CLOSED_LOOP_TASK"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old closed-loop Isaac task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old task: {repr(e)}")


# ============================================================
# 1. 文件清理
# ============================================================

CLEAR_HANDSHAKE_FILES = True

HANDSHAKE_FILES = [
    Path(r"D:\isaac_projects\closed_loop_task_plan.json"),
    Path(r"D:\isaac_projects\closed_loop_isaac_result.json"),
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
#    目标：接近你发的图一
#    只改 Translate，不改 Scale
# ============================================================

TRASH_LAYOUT_XY = {
    # 上方区域：纸盒、两个瓶子、罐子
    "/World/TrashSortingScene/trash_paper_box": (-1, 0.4),
    "/World/TrashSortingScene/trash_plastic_bottle": (-1.4, 0.2),
    "/World/TrashSortingScene/trash_dirty_bottle": (-1, 0.1),
    "/World/TrashSortingScene/trash_can": (-0.52, 0.30),

    # 中间区域：苹果、香蕉、电池、药盒
    "/World/TrashSortingScene/trash_apple_core": (-1.24, -0.12),
    "/World/TrashSortingScene/trash_banana_peel": (-0.86, -0.14),
    "/World/TrashSortingScene/trash_battery": (-0.5, -0.12),
    "/World/TrashSortingScene/trash_medicine_box": (-0.5, -0.34),

    # 下方区域：纸杯、纸团
    "/World/TrashSortingScene/trash_broken_cup": (-1.24, -0.4),
    "/World/TrashSortingScene/trash_tissue": (-0.78, -0.46),
}

ROBOT_HOME = Gf.Vec3d(1.20, -1.20, 0.00)

FALLBACK_TABLE_TOP_Z = 0.76
EXTRA_Z_LIFT = 0.02


# ============================================================
# 4. 相机参数
# ============================================================

APPLY_CAMERA_TUNING = True

# 图二太近，先把焦距调小。
# 如果还太近，改成 14.0；
# 如果太远，改成 20.0。
CAMERA_FOCAL_LENGTH = 16.0

# 保持当前相机姿态，只调视野，不强制改旋转。
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


def get_table_top_z(stage):
    table_prim = get_prim(stage, TABLE_PATH)

    if table_prim is None:
        print(f"[WARN] table not found: {TABLE_PATH}, use fallback z={FALLBACK_TABLE_TOP_Z}")
        return FALLBACK_TABLE_TOP_Z

    try:
        box = compute_world_bbox(stage, table_prim)
        top_z = float(box.GetMax()[2])
        print(f"[TABLE] top_z={top_z:.4f}")
        return top_z
    except Exception as e:
        print(f"[WARN] failed to compute table top z: {repr(e)}")
        return FALLBACK_TABLE_TOP_Z


def compute_object_half_height(stage, prim):
    try:
        box = compute_world_bbox(stage, prim)
        min_z = float(box.GetMin()[2])
        max_z = float(box.GetMax()[2])
        height = max(max_z - min_z, 0.02)
        return height / 2.0
    except Exception:
        return 0.05


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


def restore_trash_objects(stage):
    table_top_z = get_table_top_z(stage)

    restored = []
    missing = []

    for path, xy in TRASH_LAYOUT_XY.items():
        x, y = xy

        prim = get_prim(stage, path)

        if prim is None:
            print(f"[MISS TRASH] {path}")
            missing.append(path)
            continue

        set_visible(prim, True)

        half_h = compute_object_half_height(stage, prim)
        z = table_top_z + half_h + EXTRA_Z_LIFT

        target_pos = Gf.Vec3d(float(x), float(y), float(z))
        set_translate(prim, target_pos)

        restored.append({
            "path": path,
            "position": [
                round(float(x), 4),
                round(float(y), 4),
                round(float(z), 4),
            ],
            "half_height": round(float(half_h), 4),
        })

        print(f"[RESTORE TRASH] {path} -> {target_pos}")

    return restored, missing


def save_restore_log(restored, missing):
    log_path = Path(r"D:\isaac_projects\target_layout_restore_log.json")

    data = {
        "message": "Trash objects restored to target YOLO layout.",
        "num_restored": len(restored),
        "num_missing": len(missing),
        "camera_focal_length": CAMERA_FOCAL_LENGTH,
        "restored": restored,
        "missing": missing,
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SAVED] {log_path}")

    except Exception as e:
        print(f"[WARN] failed to save log: {repr(e)}")


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

    restored, missing = restore_trash_objects(stage)

    save_restore_log(restored, missing)

    print("=" * 80)
    print(f"[DONE] restored={len(restored)}, missing={len(missing)}")
    print("=" * 80)


main()