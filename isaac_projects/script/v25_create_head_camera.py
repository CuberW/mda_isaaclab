import asyncio
import builtins
import math

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 0. 停掉旧头部相机跟随任务
# ============================================================

OLD_TASK_NAME = "TRASHBOT_V25_HEAD_CAMERA_FOLLOW_TASK"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old head camera follow task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old task: {repr(e)}")


# ============================================================
# 1. 配置
# ============================================================

ROBOT_PATH = "/World/TrashBotRobot"
HEAD_CAMERA_PATH = "/World/TrashBotHeadCamera"

# 头部相机相对机器人底盘的世界偏移。
# 当前机器人是逻辑模型，没有真实头部 TF，所以先用 robot translate + 这个 offset 模拟头部相机。
HEAD_OFFSET = Gf.Vec3d(0.00, 0.00, 1.55)

# 相机默认看向桌面中心。后续可以改成看向选中的目标或机器人前方。
LOOK_AT_POINT = Gf.Vec3d(-0.65, 0.00, 0.33)

# 视野参数。头部相机比固定俯视相机要广一点。
FOCAL_LENGTH = 18.0
HORIZONTAL_APERTURE = 36.0
VERTICAL_APERTURE = 24.0

# 每几帧更新一次相机姿态
UPDATE_INTERVAL_FRAMES = 1


# ============================================================
# 2. USD 工具
# ============================================================

def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。")
    return stage


def make_sdf_path(path):
    return Sdf.Path(str(path))


def get_prim(stage, path):
    prim = stage.GetPrimAtPath(make_sdf_path(path))
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


def clear_xform_ops(prim):
    """
    这里只用于新建的相机，不用于垃圾模型。
    垃圾模型不要 ClearXformOpOrder，避免清掉 Scale。
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()


def set_world_matrix(prim, matrix):
    xform = UsdGeom.Xformable(prim)

    ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(matrix)
            return

    clear_xform_ops(prim)
    op = xform.AddTransformOp()
    op.Set(matrix)


def make_camera_look_at_matrix(eye, target, up=Gf.Vec3d(0.0, 0.0, 1.0)):
    """
    USD Camera 默认沿本地 -Z 方向看。
    Gf.Matrix4d.SetLookAt 得到 view matrix，因此要取逆作为 camera-to-world。
    """
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye, target, up)

    cam_to_world = view.GetInverse()
    return cam_to_world


def create_or_update_head_camera(stage):
    cam_prim = get_prim(stage, HEAD_CAMERA_PATH)

    if cam_prim is None:
        camera = UsdGeom.Camera.Define(stage, make_sdf_path(HEAD_CAMERA_PATH))
        cam_prim = camera.GetPrim()
        print(f"[CREATE] {HEAD_CAMERA_PATH}")
    else:
        camera = UsdGeom.Camera(cam_prim)
        print(f"[FOUND] {HEAD_CAMERA_PATH}")

    camera.GetFocalLengthAttr().Set(float(FOCAL_LENGTH))
    camera.GetHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE))
    camera.GetVerticalApertureAttr().Set(float(VERTICAL_APERTURE))

    # 近裁剪不要太大，否则近处物体消失
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

    return cam_prim


async def next_frame():
    await omni.kit.app.get_app().next_update_async()


async def follow_loop():
    print("=" * 80)
    print("[START] V2.5 head camera follow")
    print(f"[CAMERA] {HEAD_CAMERA_PATH}")
    print("=" * 80)

    stage = get_stage()

    robot_prim = get_prim(stage, ROBOT_PATH)
    if robot_prim is None:
        raise RuntimeError(f"找不到机器人 Prim：{ROBOT_PATH}")

    cam_prim = create_or_update_head_camera(stage)

    frame_idx = 0

    while True:
        robot_pos = get_translate(robot_prim)
        eye = robot_pos + HEAD_OFFSET

        # 这里让头部相机始终看向桌面中心，相当于机器人头部云台凝视桌面。
        target = LOOK_AT_POINT

        mat = make_camera_look_at_matrix(eye, target)
        set_world_matrix(cam_prim, mat)

        if frame_idx % 60 == 0:
            print(
                f"[HEAD CAMERA] eye=({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}) "
                f"look_at=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
            )

        frame_idx += 1

        for _ in range(UPDATE_INTERVAL_FRAMES):
            await next_frame()


task = asyncio.ensure_future(follow_loop())
setattr(builtins, OLD_TASK_NAME, task)