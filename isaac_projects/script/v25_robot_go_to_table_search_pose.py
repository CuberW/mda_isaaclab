import asyncio
import builtins
import math
import time
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


OLD_TASK_NAME = "TRASHBOT_V25_GOTO_TABLE_TASK"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old goto-table task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old goto-table task: {repr(e)}")


ROBOT_PATH = "/World/TrashBotRobot"
HEAD_CAMERA_PATH = "/World/TrashBotHeadCamera"

# 桌面垃圾主要集中在 x=-1.2~-0.3, y=-0.5~0.4
# 当前机器人 home 一般在 (1.2, -1.2, 0)，太远。
# 这个搜索位让机器人来到桌面前方，头部相机视角会明显放大垃圾。
TABLE_SEARCH_POSE = Gf.Vec3d(-0.15, -0.95, 0.00)

# 稍微近一点的候选位。如果第一个位置还太远，可以改用这个。
TABLE_SEARCH_POSE_NEAR = Gf.Vec3d(-0.35, -0.88, 0.00)

# 默认用较稳的位置，避免相机过近导致视野丢失。
USE_NEAR_POSE = False

# 头部相机看向桌面中心偏前位置。移动到桌边后，目标会变大。
LOOK_AT_POINT = Gf.Vec3d(-0.65, -0.05, 0.33)

HEAD_OFFSET = Gf.Vec3d(0.00, 0.00, 1.55)

MOVE_FRAMES = 120
SETTLE_FRAMES = 60

RESULT_JSON = Path(r"D:\isaac_projects\v25_robot_go_to_table_result.json")


def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。")
    return stage


def get_prim(stage, path):
    prim = stage.GetPrimAtPath(Sdf.Path(str(path)))
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


def clear_xform_ops(prim):
    # 只用于相机，不用于垃圾模型
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()


def set_world_matrix(prim, matrix):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(matrix)
            return

    clear_xform_ops(prim)
    op = xform.AddTransformOp()
    op.Set(matrix)


def make_camera_look_at_matrix(eye, target, up=Gf.Vec3d(0.0, 0.0, 1.0)):
    # USD Camera 默认沿本地 -Z 看；SetLookAt 返回 view matrix，要取 inverse。
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye, target, up)
    return view.GetInverse()


def update_head_camera(stage, robot_pos):
    cam_prim = get_prim(stage, HEAD_CAMERA_PATH)
    if cam_prim is None:
        print(f"[WARN] head camera not found: {HEAD_CAMERA_PATH}")
        return None

    eye = robot_pos + HEAD_OFFSET
    mat = make_camera_look_at_matrix(eye, LOOK_AT_POINT)
    set_world_matrix(cam_prim, mat)

    return eye


def lerp_vec(a, b, t):
    return Gf.Vec3d(
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
    )


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


async def next_frame():
    await omni.kit.app.get_app().next_update_async()


async def pause_frames(n):
    for _ in range(n):
        await next_frame()


def save_json(path, data):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


async def goto_table():
    print("=" * 80)
    print("[START] V2.5 robot go to table search pose")
    print("=" * 80)

    stage = get_stage()

    robot_prim = get_prim(stage, ROBOT_PATH)
    if robot_prim is None:
        raise RuntimeError(f"找不到机器人 Prim：{ROBOT_PATH}")

    target = TABLE_SEARCH_POSE_NEAR if USE_NEAR_POSE else TABLE_SEARCH_POSE

    start = get_translate(robot_prim)

    print(f"[ROBOT START] {start}")
    print(f"[ROBOT TARGET] {target}")
    print(f"[HEAD CAMERA LOOK_AT] {LOOK_AT_POINT}")

    t0 = time.time()

    for i in range(MOVE_FRAMES + 1):
        ratio = ease(i / max(MOVE_FRAMES, 1))
        pos = lerp_vec(start, target, ratio)

        set_translate(robot_prim, pos)
        eye = update_head_camera(stage, pos)

        if i % 30 == 0:
            print(
                f"[MOVE] frame={i:03d}, robot=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                f"head_eye=({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})" if eye else ""
            )

        await next_frame()

    final_pos = get_translate(robot_prim)
    final_eye = update_head_camera(stage, final_pos)

    await pause_frames(SETTLE_FRAMES)

    result = {
        "mode": "V2_5_ROBOT_GOTO_TABLE_SEARCH_POSE",
        "robot_path": ROBOT_PATH,
        "head_camera_path": HEAD_CAMERA_PATH,
        "start_xyz": [
            round(float(start[0]), 4),
            round(float(start[1]), 4),
            round(float(start[2]), 4),
        ],
        "target_xyz": [
            round(float(target[0]), 4),
            round(float(target[1]), 4),
            round(float(target[2]), 4),
        ],
        "final_xyz": [
            round(float(final_pos[0]), 4),
            round(float(final_pos[1]), 4),
            round(float(final_pos[2]), 4),
        ],
        "head_camera_eye_xyz": [
            round(float(final_eye[0]), 4),
            round(float(final_eye[1]), 4),
            round(float(final_eye[2]), 4),
        ] if final_eye else None,
        "look_at_xyz": [
            round(float(LOOK_AT_POINT[0]), 4),
            round(float(LOOK_AT_POINT[1]), 4),
            round(float(LOOK_AT_POINT[2]), 4),
        ],
        "status": "success",
        "duration_sec": round(time.time() - t0, 3),
    }

    save_json(RESULT_JSON, result)

    print("[RESULT]")
    print(result)
    print(f"[SAVED] {RESULT_JSON}")
    print("=" * 80)


task = asyncio.ensure_future(goto_table())
setattr(builtins, OLD_TASK_NAME, task)