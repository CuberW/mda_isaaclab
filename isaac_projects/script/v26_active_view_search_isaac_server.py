import asyncio
import builtins
import json
import math
import time
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom


TASK_NAME = "TRASHBOT_V26_ACTIVE_VIEW_SEARCH_SERVER"

COMMAND_JSON = Path(r"D:\isaac_projects\v26_view_command.json")
RESULT_JSON = Path(r"D:\isaac_projects\v26_view_result.json")

ROBOT_PATH = "/World/TrashBotRobot"
HEAD_CAMERA_PATH = "/World/TrashBotHeadCamera"

HEAD_OFFSET = Gf.Vec3d(0.00, 0.00, 1.55)

DEFAULT_MOVE_FRAMES = 35
DEFAULT_SETTLE_FRAMES = 12

CAMERA_FOCAL_LENGTH = 18.0
CAMERA_HORIZONTAL_APERTURE = 36.0
CAMERA_VERTICAL_APERTURE = 24.0

VIEW_CANDIDATES = {
    "front_mid": {
        "view_id": "front_mid",
        "robot_xyz": [-0.35, -0.88, 0.0],
        "look_at_xyz": [-0.65, -0.12, 0.30],
    },
    "front_left": {
        "view_id": "front_left",
        "robot_xyz": [-0.70, -0.92, 0.0],
        "look_at_xyz": [-0.75, -0.10, 0.30],
    },
    "front_right": {
        "view_id": "front_right",
        "robot_xyz": [0.00, -0.92, 0.0],
        "look_at_xyz": [-0.45, -0.10, 0.30],
    },
    "near_mid": {
        "view_id": "near_mid",
        "robot_xyz": [-0.45, -0.78, 0.0],
        "look_at_xyz": [-0.65, -0.08, 0.30],
    },
    "far_mid": {
        "view_id": "far_mid",
        "robot_xyz": [-0.25, -1.08, 0.0],
        "look_at_xyz": [-0.65, -0.12, 0.30],
    },
}


old_task = getattr(builtins, TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old", TASK_NAME)
    except Exception as e:
        print("[WARN] failed to cancel old task:", repr(e))
    try:
        delattr(builtins, TASK_NAME)
    except Exception:
        pass


def v26_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage opened.")
    return stage


def v26_get_prim(stage, path):
    prim = stage.GetPrimAtPath(Sdf.Path(str(path)))
    if not prim or not prim.IsValid():
        return None
    return prim


def v26_get_translate(prim):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            if value is not None:
                return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))

    mat = xform.ComputeLocalToWorldTransform(0)
    t = mat.ExtractTranslation()
    return Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))


def v26_set_translate(prim, vec):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))
            return

    op = xform.AddTranslateOp()
    op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))


def v26_set_world_matrix(prim, matrix):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp()
    op.Set(matrix)


def v26_make_look_at_matrix(eye, target, up=Gf.Vec3d(0.0, 0.0, 1.0)):
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye, target, up)
    return view.GetInverse()


def v26_vec3(xyz):
    return Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def v26_vec3_list(v):
    return [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]


def v26_dist_xy(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def v26_lerp(a, b, t):
    return Gf.Vec3d(
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
    )


def v26_ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def v26_load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def v26_save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def v26_define_head_camera(stage):
    prim = v26_get_prim(stage, HEAD_CAMERA_PATH)

    if prim is None:
        cam = UsdGeom.Camera.Define(stage, Sdf.Path(HEAD_CAMERA_PATH))
        prim = cam.GetPrim()
        print("[CREATE CAMERA]", HEAD_CAMERA_PATH)
    else:
        cam = UsdGeom.Camera(prim)

    cam.GetFocalLengthAttr().Set(float(CAMERA_FOCAL_LENGTH))
    cam.GetHorizontalApertureAttr().Set(float(CAMERA_HORIZONTAL_APERTURE))
    cam.GetVerticalApertureAttr().Set(float(CAMERA_VERTICAL_APERTURE))

    return prim


def v26_update_head_camera(stage, robot_pos, look_at):
    cam_prim = v26_define_head_camera(stage)

    eye = robot_pos + HEAD_OFFSET
    mat = v26_make_look_at_matrix(eye, look_at)
    v26_set_world_matrix(cam_prim, mat)

    return eye


async def v26_next_frame():
    await omni.kit.app.get_app().next_update_async()


async def v26_pause_frames(n):
    for _ in range(max(0, int(n))):
        await v26_next_frame()


async def v26_move_robot(stage, robot_prim, start, target, look_at, frames):
    distance = v26_dist_xy(start, target)

    # 关键修复：如果当前位置和目标位置一样，不做动画循环，直接写相机并返回。
    if distance < 1e-6:
        eye = v26_update_head_camera(stage, target, look_at)
        v26_set_translate(robot_prim, target)
        await v26_next_frame()
        print(f"[MOVE SKIP] already at target, robot={v26_vec3_list(target)}, eye={v26_vec3_list(eye)}")
        return eye

    last_eye = None

    for i in range(int(frames) + 1):
        t = v26_ease(i / max(int(frames), 1))
        pos = v26_lerp(start, target, t)

        v26_set_translate(robot_prim, pos)
        last_eye = v26_update_head_camera(stage, pos, look_at)

        if i == 0 or i == int(frames) or i % 15 == 0:
            print(
                f"[MOVE] frame={i:03d}, "
                f"robot=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}), "
                f"eye=({last_eye[0]:.3f},{last_eye[1]:.3f},{last_eye[2]:.3f})"
            )

        await v26_next_frame()

    return last_eye


def v26_resolve_command(command):
    view_id = command.get("view_id", "front_mid")

    base = dict(VIEW_CANDIDATES.get(view_id, VIEW_CANDIDATES["front_mid"]))

    if command.get("robot_xyz") is not None:
        base["robot_xyz"] = command["robot_xyz"]

    if command.get("look_at_xyz") is not None:
        base["look_at_xyz"] = command["look_at_xyz"]

    base["view_id"] = view_id
    return base


async def v26_execute_command(stage, robot_prim, command):
    t0 = time.time()

    command_id = command.get("command_id")
    view = v26_resolve_command(command)

    target_robot = v26_vec3(view["robot_xyz"])
    look_at = v26_vec3(view["look_at_xyz"])

    move_frames = int(command.get("move_frames", DEFAULT_MOVE_FRAMES))
    settle_frames = int(command.get("settle_frames", DEFAULT_SETTLE_FRAMES))

    robot_start = v26_get_translate(robot_prim)

    print("\n" + "=" * 80)
    print(f"[V2.6 COMMAND] {command_id}")
    print(f"[VIEW] {view['view_id']}")
    print(f"[ROBOT] {robot_start} -> {target_robot}")
    print(f"[LOOK_AT] {look_at}")
    print("=" * 80)

    try:
        eye = await v26_move_robot(
            stage=stage,
            robot_prim=robot_prim,
            start=robot_start,
            target=target_robot,
            look_at=look_at,
            frames=move_frames,
        )

        await v26_pause_frames(settle_frames)

        final_robot = v26_get_translate(robot_prim)
        eye = v26_update_head_camera(stage, final_robot, look_at)
        await v26_next_frame()

        result = {
            "status": "success",
            "mode": "V2_6_ACTIVE_VIEW_SEARCH_SERVER_REWRITE",
            "command_id": command_id,
            "view_id": view["view_id"],
            "robot_xyz": v26_vec3_list(final_robot),
            "look_at_xyz": v26_vec3_list(look_at),
            "camera_path": HEAD_CAMERA_PATH,
            "camera_eye_xyz": v26_vec3_list(eye),
            "move_frames": move_frames,
            "settle_frames": settle_frames,
            "duration_sec": round(time.time() - t0, 4),
            "timestamp": time.time(),
        }

    except Exception as e:
        result = {
            "status": "failed_exception",
            "mode": "V2_6_ACTIVE_VIEW_SEARCH_SERVER_REWRITE",
            "command_id": command_id,
            "view_id": view.get("view_id"),
            "error": repr(e),
            "duration_sec": round(time.time() - t0, 4),
            "timestamp": time.time(),
        }

    v26_save_json_atomic(RESULT_JSON, result)

    print("[VIEW RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[SAVED]", RESULT_JSON)
    print("=" * 80)

    return result


async def v26_main_loop():
    print("=" * 80)
    print("[START] V2.6 active view search server - rewritten")
    print("[WAIT COMMAND]", COMMAND_JSON)
    print("[WRITE RESULT]", RESULT_JSON)
    print("=" * 80)

    stage = v26_stage()

    robot_prim = v26_get_prim(stage, ROBOT_PATH)
    if robot_prim is None:
        raise RuntimeError(f"Cannot find robot: {ROBOT_PATH}")

    v26_define_head_camera(stage)

    processed = set()

    while True:
        command = v26_load_json(COMMAND_JSON)

        if not command:
            await v26_pause_frames(5)
            continue

        command_id = command.get("command_id")

        if not command_id:
            await v26_pause_frames(5)
            continue

        if command_id in processed:
            await v26_pause_frames(5)
            continue

        processed.add(command_id)

        await v26_execute_command(stage, robot_prim, command)
        await v26_pause_frames(3)


task = asyncio.ensure_future(v26_main_loop())
setattr(builtins, TASK_NAME, task)