import asyncio
import builtins
import json
import math
import time
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# =============================================================================
# V2.7 Head-Camera Action Server
# Rewritten version:
# - no generic function names, all helper functions are prefixed with v27_
# - dynamic approach direction instead of fixed global y offset
# - grip point directly aligns above visual_world
# =============================================================================


TASK_NAME = "TRASHBOT_V27_HEAD_CAMERA_ACTION_SERVER"

# 只取消同类动作服务器和可能干扰头部相机姿态的旧 V2.5 任务。
# 不取消 v26 active view search server。
CONFLICT_TASK_NAMES = [
    TASK_NAME,
    "TRASHBOT_V25_HEAD_CAMERA_SINGLE_TASK_V2",
    "TRASHBOT_V25_PREPARE_SEARCH_POSE_TASK",
    "TRASHBOT_V25_GOTO_TABLE_TASK",
    "TRASHBOT_V25_HEAD_CAMERA_FOLLOW_TASK",
    "TRASHBOT_V25_HEAD_CAMERA_TASK",
]

for name in CONFLICT_TASK_NAMES:
    old_task = getattr(builtins, name, None)
    if old_task is not None:
        try:
            old_task.cancel()
            print(f"[CANCEL] old task: {name}")
        except Exception as e:
            print(f"[WARN] failed to cancel {name}: {repr(e)}")
        try:
            delattr(builtins, name)
        except Exception:
            pass


PLAN_JSON = Path(r"D:\isaac_projects\v2_visual_task_plan.json")
ACTION_RESULT_JSON = Path(r"D:\isaac_projects\v27_head_camera_action_result.json")

ROBOT_PATH = "/World/TrashBotRobot"
HEAD_CAMERA_PATH = "/World/TrashBotHeadCamera"
TABLE_PATH = "/World/TrashSortingScene/table"

BIN_PATHS = {
    "bin_recyclable_blue": "/World/TrashSortingScene/bin_recyclable_blue",
    "bin_kitchen_green": "/World/TrashSortingScene/bin_kitchen_green",
    "bin_hazardous_red": "/World/TrashSortingScene/bin_hazardous_red",
    "bin_other_gray": "/World/TrashSortingScene/bin_other_gray",
}

RAW_CLASS_TO_PRIMS = {
    "battery": ["/World/TrashSortingScene/trash_battery"],
    "battery1": ["/World/TrashSortingScene/trash_battery"],
    "battery5": ["/World/TrashSortingScene/trash_battery"],

    "drugbox": ["/World/TrashSortingScene/trash_medicine_box"],
    "drug": ["/World/TrashSortingScene/trash_medicine_box"],
    "drugbag": ["/World/TrashSortingScene/trash_medicine_box"],
    "capsule": ["/World/TrashSortingScene/trash_medicine_box"],

    "can": ["/World/TrashSortingScene/trash_can"],

    "bottle": [
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_plastic_bottle",
    ],
    "bottle2": [
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_plastic_bottle",
    ],

    "paper": [
        "/World/TrashSortingScene/trash_paper_box",
        "/World/TrashSortingScene/trash_tissue",
    ],

    "papercup": ["/World/TrashSortingScene/trash_broken_cup"],

    "potato": ["/World/TrashSortingScene/trash_apple_core"],
    "mooli": ["/World/TrashSortingScene/trash_apple_core"],
    "potatocut": ["/World/TrashSortingScene/trash_banana_peel"],
    "rabbitcut": ["/World/TrashSortingScene/trash_banana_peel"],

    "brick": [],
    "china": [],
    "stone": [],
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


HEAD_OFFSET = Gf.Vec3d(0.00, 0.00, 1.55)
DEFAULT_LOOK_AT = Gf.Vec3d(-0.65, -0.12, 0.30)

# 动态靠近参数
STAND_OFF_DISTANCE = 0.45
MIN_STAND_OFF_DISTANCE = 0.22
GRASP_HEIGHT = 0.95

# 投放参数
DROP_HEIGHT_ABOVE_BIN = 0.85
DROP_INSIDE_HEIGHT = 0.20

# 动画速度
MOVE_FRAMES = 35
PICK_FRAMES = 15
DROP_FRAMES = 15
RETURN_FRAMES = 25
WAIT_FRAMES = 20

# 视觉匹配阈值
MAX_ATTACH_DISTANCE = 0.30

# 投放后隐藏物体，方便后续继续闭环处理
HIDE_OBJECT_AFTER_DROP = True

# 启动时是否忽略已经存在的旧 plan，防止吃旧任务
IGNORE_EXISTING_PLAN_ON_START = True

PROCESSED_PLAN_IDS = set()


def v27_get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。")
    return stage


def v27_get_prim(stage, path):
    prim = stage.GetPrimAtPath(Sdf.Path(str(path)))
    if not prim or not prim.IsValid():
        return None
    return prim


def v27_get_translate(prim):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            if value is not None:
                return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))

    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))


def v27_set_translate(prim, vec):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))
            return

    op = xform.AddTranslateOp()
    op.Set(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))


def v27_set_visibility(prim, visible):
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def v27_is_visible(prim):
    try:
        return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
    except Exception:
        return True


def v27_clear_xform_ops(prim):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()


def v27_set_world_matrix(prim, matrix):
    xform = UsdGeom.Xformable(prim)

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(matrix)
            return

    v27_clear_xform_ops(prim)
    op = xform.AddTransformOp()
    op.Set(matrix)


def v27_make_camera_look_at_matrix(eye, target, up=Gf.Vec3d(0.0, 0.0, 1.0)):
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye, target, up)
    return view.GetInverse()


def v27_update_head_camera(stage, robot_pos, look_at):
    cam_prim = v27_get_prim(stage, HEAD_CAMERA_PATH)
    if cam_prim is None:
        raise RuntimeError(f"找不到头部相机：{HEAD_CAMERA_PATH}")

    eye = robot_pos + HEAD_OFFSET
    mat = v27_make_camera_look_at_matrix(eye, look_at)
    v27_set_world_matrix(cam_prim, mat)
    return eye


def v27_compute_world_bbox(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.ComputeAlignedBox()


def v27_get_world_bbox_center(stage, prim):
    try:
        box = v27_compute_world_bbox(stage, prim)
        min_pt = box.GetMin()
        max_pt = box.GetMax()

        return Gf.Vec3d(
            (float(min_pt[0]) + float(max_pt[0])) / 2.0,
            (float(min_pt[1]) + float(max_pt[1])) / 2.0,
            (float(min_pt[2]) + float(max_pt[2])) / 2.0,
        )
    except Exception:
        return v27_get_translate(prim)


def v27_get_table_top_z(stage):
    table_prim = v27_get_prim(stage, TABLE_PATH)
    if table_prim is None:
        return 0.29

    try:
        box = v27_compute_world_bbox(stage, table_prim)
        return float(box.GetMax()[2])
    except Exception:
        return 0.29


def v27_vec_distance_xy(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def v27_lerp_vec(a, b, t):
    return Gf.Vec3d(
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
    )


def v27_ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


async def v27_next_frame():
    await omni.kit.app.get_app().next_update_async()


async def v27_pause_frames(n):
    for _ in range(n):
        await v27_next_frame()


async def v27_animate_robot(stage, robot_prim, start, end, frames, look_at):
    for i in range(frames + 1):
        t = v27_ease(i / max(frames, 1))
        pos = v27_lerp_vec(start, end, t)
        v27_set_translate(robot_prim, pos)
        v27_update_head_camera(stage, pos, look_at)
        await v27_next_frame()


async def v27_animate_object(object_prim, start, end, frames):
    for i in range(frames + 1):
        t = v27_ease(i / max(frames, 1))
        pos = v27_lerp_vec(start, end, t)
        v27_set_translate(object_prim, pos)
        await v27_next_frame()


async def v27_animate_robot_with_object(stage, robot_prim, object_prim, robot_start, robot_end, carry_offset, frames, look_at):
    for i in range(frames + 1):
        t = v27_ease(i / max(frames, 1))
        robot_pos = v27_lerp_vec(robot_start, robot_end, t)
        object_pos = robot_pos + carry_offset

        v27_set_translate(robot_prim, robot_pos)
        v27_set_translate(object_prim, object_pos)
        v27_update_head_camera(stage, robot_pos, look_at)

        await v27_next_frame()


def v27_load_json_safe(path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def v27_save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def v27_parse_vec3(value, default):
    if value is None or len(value) != 3:
        return default

    return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))


def v27_get_task_raw_class(task):
    return task.get("raw_class_name", task.get("class_name", "unknown"))


def v27_get_task_category(task):
    return task.get("garbage_category", task.get("category", "unknown"))


def v27_get_task_target_bin(task):
    return task.get("target_bin", "unknown")


def v27_get_task_centroid(task):
    centroid = task.get("centroid_px")
    if centroid is not None and len(centroid) == 2:
        return [float(centroid[0]), float(centroid[1])]

    bbox = task.get("bbox_xyxy")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    return None


def v27_get_plan_look_at(plan):
    view = plan.get("v26_selected_view") or plan.get("selected_view") or {}
    return v27_parse_vec3(view.get("look_at_xyz"), DEFAULT_LOOK_AT)


def v27_pixel_to_world_on_table(stage, camera_path, u, v, image_width, image_height):
    camera_prim = v27_get_prim(stage, camera_path)
    if camera_prim is None:
        raise RuntimeError(f"找不到相机 Prim：{camera_path}")

    camera = UsdGeom.Camera(camera_prim)

    focal_length = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())

    x_ndc = (float(u) + 0.5) / float(image_width) - 0.5
    y_ndc = 0.5 - (float(v) + 0.5) / float(image_height)

    x_camera = x_ndc * horizontal_aperture
    y_camera = y_ndc * vertical_aperture
    z_camera = -focal_length

    ray_cam = Gf.Vec3d(x_camera, y_camera, z_camera)
    ray_cam.Normalize()

    xform = UsdGeom.Xformable(camera_prim)
    cam_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    cam_pos = cam_to_world.ExtractTranslation()
    ray_world = cam_to_world.TransformDir(ray_cam)
    ray_world.Normalize()

    table_z = v27_get_table_top_z(stage)

    denom = float(ray_world[2])
    if abs(denom) < 1e-8:
        raise RuntimeError("Camera ray is parallel to table plane.")

    t = (table_z - float(cam_pos[2])) / denom

    if t <= 0:
        raise RuntimeError(f"Ray intersection behind camera: t={t}")

    hit = Gf.Vec3d(
        float(cam_pos[0] + ray_world[0] * t),
        float(cam_pos[1] + ray_world[1] * t),
        float(table_z),
    )

    debug = {
        "camera_path": camera_path,
        "pixel_u": round(float(u), 3),
        "pixel_v": round(float(v), 3),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_focal_length": round(focal_length, 4),
        "camera_horizontal_aperture": round(horizontal_aperture, 4),
        "camera_vertical_aperture": round(vertical_aperture, 4),
        "camera_position": [
            round(float(cam_pos[0]), 4),
            round(float(cam_pos[1]), 4),
            round(float(cam_pos[2]), 4),
        ],
        "ray_world": [
            round(float(ray_world[0]), 6),
            round(float(ray_world[1]), 6),
            round(float(ray_world[2]), 6),
        ],
        "table_z": round(float(table_z), 4),
        "intersection_t": round(float(t), 4),
        "world_hit": [
            round(float(hit[0]), 4),
            round(float(hit[1]), 4),
            round(float(hit[2]), 4),
        ],
    }

    return hit, debug


def v27_choose_object(stage, raw_class, visual_world):
    candidates = RAW_CLASS_TO_PRIMS.get(raw_class, [])
    scored = []

    for path in candidates:
        prim = v27_get_prim(stage, path)
        if prim is None:
            continue

        if not v27_is_visible(prim):
            continue

        center = v27_get_world_bbox_center(stage, prim)
        dist = v27_vec_distance_xy(center, visual_world)

        scored.append((dist, path, prim, center))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    dist, path, prim, center = scored[0]

    return {
        "distance_xy": float(dist),
        "path": path,
        "prim": prim,
        "center": center,
        "strict_match": bool(dist <= MAX_ATTACH_DISTANCE),
    }


def v27_compute_dynamic_approach_and_grip(robot_start, visual_world):
    """
    动态靠近：
    - 方向：robot_start -> visual_world
    - 停靠点：目标点反方向 STAND_OFF_DISTANCE
    - 夹爪点：visual_world 正上方
    """
    dx = float(visual_world[0] - robot_start[0])
    dy = float(visual_world[1] - robot_start[1])
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-6:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    if dist < STAND_OFF_DISTANCE:
        stand_off = max(MIN_STAND_OFF_DISTANCE, dist * 0.65)
    else:
        stand_off = STAND_OFF_DISTANCE

    approach_pos = Gf.Vec3d(
        float(visual_world[0] - ux * stand_off),
        float(visual_world[1] - uy * stand_off),
        float(robot_start[2]),
    )

    grip_pos = Gf.Vec3d(
        float(visual_world[0]),
        float(visual_world[1]),
        float(GRASP_HEIGHT),
    )

    carry_offset = Gf.Vec3d(
        float(grip_pos[0] - approach_pos[0]),
        float(grip_pos[1] - approach_pos[1]),
        float(grip_pos[2] - approach_pos[2]),
    )

    return approach_pos, grip_pos, carry_offset, stand_off


def v27_make_base_record(plan, task, raw_class, category, target_bin, u, v, image_width, image_height):
    return {
        "mode": "V2_7_HEAD_CAMERA_ALL_TRASH_ACTION_REWRITE",
        "plan_id": plan.get("plan_id"),
        "cycle_id": plan.get("cycle_id"),
        "source_image": plan.get("source_image"),
        "raw_class_name": raw_class,
        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),
        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),
        "confidence": float(task.get("confidence", 0.0)),
        "centroid_px": [round(float(u), 3), round(float(v), 3)],
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_path": HEAD_CAMERA_PATH,
        "head_camera_used": True,
        "relative_position_computed": True,
        "visual_grasp_used": True,
        "model_coordinate_used_for_planning": False,
        "model_coordinate_used_for_mesh_animation": True,
        "action_completed": False,
        "object_hidden_after_drop": False,
        "status": "pending",
        "duration_sec": 0.0,
    }


async def v27_execute_plan(stage, robot_prim, plan):
    t0 = time.time()

    plan_id = plan.get("plan_id")
    task = plan.get("selected_task")

    if task is None:
        raise RuntimeError("plan 中没有 selected_task")

    raw_class = v27_get_task_raw_class(task)
    category = v27_get_task_category(task)
    target_bin = v27_get_task_target_bin(task)

    centroid = v27_get_task_centroid(task)
    if centroid is None:
        raise RuntimeError("selected_task 中没有 centroid_px 或 bbox_xyxy")

    image_width = int(plan.get("image_width", 480))
    image_height = int(plan.get("image_height", 320))

    u, v = centroid

    look_at = v27_get_plan_look_at(plan)

    robot_start_pose = v27_get_translate(robot_prim)

    # 重新按照 plan 对应视角设置头部相机，保证投影与采图视角一致
    v27_update_head_camera(stage, robot_start_pose, look_at)
    await v27_next_frame()

    visual_world, projection_debug = v27_pixel_to_world_on_table(
        stage=stage,
        camera_path=HEAD_CAMERA_PATH,
        u=u,
        v=v,
        image_width=image_width,
        image_height=image_height,
    )

    target_rel_robot = Gf.Vec3d(
        float(visual_world[0] - robot_start_pose[0]),
        float(visual_world[1] - robot_start_pose[1]),
        float(visual_world[2] - robot_start_pose[2]),
    )

    obj_info = v27_choose_object(stage, raw_class, visual_world)

    bin_path = BIN_PATHS.get(target_bin)
    bin_prim = v27_get_prim(stage, bin_path) if bin_path else None

    record = v27_make_base_record(
        plan=plan,
        task=task,
        raw_class=raw_class,
        category=category,
        target_bin=target_bin,
        u=u,
        v=v,
        image_width=image_width,
        image_height=image_height,
    )

    record.update({
        "robot_pose_at_projection_xyz": [
            round(float(robot_start_pose[0]), 4),
            round(float(robot_start_pose[1]), 4),
            round(float(robot_start_pose[2]), 4),
        ],
        "look_at_xyz": [
            round(float(look_at[0]), 4),
            round(float(look_at[1]), 4),
            round(float(look_at[2]), 4),
        ],
        "visual_world_xyz": [
            round(float(visual_world[0]), 4),
            round(float(visual_world[1]), 4),
            round(float(visual_world[2]), 4),
        ],
        "target_relative_to_robot_xyz": [
            round(float(target_rel_robot[0]), 4),
            round(float(target_rel_robot[1]), 4),
            round(float(target_rel_robot[2]), 4),
        ],
        "projection_debug": projection_debug,
        "attached_object_prim": obj_info["path"] if obj_info else None,
        "attach_distance_xy": round(float(obj_info["distance_xy"]), 4) if obj_info else None,
        "attach_strict_match": bool(obj_info["strict_match"]) if obj_info else False,
        "bin_prim": bin_path,
    })

    print("\n" + "=" * 80)
    print(f"[V2.7 ACTION REWRITE] plan_id={plan_id}, raw={raw_class}, category={category}, bin={target_bin}")
    print(f"[ROBOT START] {robot_start_pose}")
    print(f"[PIXEL] u={u:.2f}, v={v:.2f}, image={image_width}x{image_height}")
    print(f"[WORLD] {visual_world}")
    print(f"[REL ROBOT] {target_rel_robot}")
    print(f"[ATTACH] {record['attached_object_prim']}, dist={record['attach_distance_xy']}")
    print("=" * 80)

    if bin_prim is None:
        record["status"] = "failed_no_bin"
        record["duration_sec"] = round(time.time() - t0, 3)
        v27_save_json_atomic(ACTION_RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return record

    if obj_info is None:
        record["status"] = "failed_no_matching_object"
        record["duration_sec"] = round(time.time() - t0, 3)
        v27_save_json_atomic(ACTION_RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return record

    if not obj_info["strict_match"]:
        record["status"] = "failed_visual_object_mismatch"
        record["duration_sec"] = round(time.time() - t0, 3)
        v27_save_json_atomic(ACTION_RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return record

    try:
        obj_prim = obj_info["prim"]

        robot_start = v27_get_translate(robot_prim)
        obj_start = v27_get_translate(obj_prim)
        bin_pos = v27_get_translate(bin_prim)

        approach_pos, grip_pos, carry_offset, stand_off = v27_compute_dynamic_approach_and_grip(
            robot_start=robot_start,
            visual_world=visual_world,
        )

        record["dynamic_approach"] = {
            "stand_off_distance": round(float(stand_off), 4),
            "approach_xyz": [
                round(float(approach_pos[0]), 4),
                round(float(approach_pos[1]), 4),
                round(float(approach_pos[2]), 4),
            ],
            "grip_xyz": [
                round(float(grip_pos[0]), 4),
                round(float(grip_pos[1]), 4),
                round(float(grip_pos[2]), 4),
            ],
            "carry_offset_xyz": [
                round(float(carry_offset[0]), 4),
                round(float(carry_offset[1]), 4),
                round(float(carry_offset[2]), 4),
            ],
        }

        print(f"[DYNAMIC APPROACH] {robot_start} -> {approach_pos}, stand_off={stand_off:.3f}")
        print(f"[DYNAMIC GRIP] obj={obj_start} -> grip={grip_pos}, carry_offset={carry_offset}")

        # 1. 机器人动态靠近目标
        await v27_animate_robot(stage, robot_prim, robot_start, approach_pos, MOVE_FRAMES, look_at)

        # 2. 逻辑抓取：物体移动到 visual_world 正上方
        await v27_animate_object(obj_prim, obj_start, grip_pos, PICK_FRAMES)

        # 3. 运输到目标垃圾桶前方
        bin_approach_pos = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1] - 0.55),
            float(robot_start[2]),
        )

        print(f"[TRANSPORT] approach={approach_pos} -> bin_approach={bin_approach_pos}")

        await v27_animate_robot_with_object(
            stage=stage,
            robot_prim=robot_prim,
            object_prim=obj_prim,
            robot_start=approach_pos,
            robot_end=bin_approach_pos,
            carry_offset=carry_offset,
            frames=MOVE_FRAMES,
            look_at=look_at,
        )

        object_current = bin_approach_pos + carry_offset

        # 4. 投放
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

        await v27_animate_object(obj_prim, object_current, drop_above, DROP_FRAMES)
        await v27_animate_object(obj_prim, drop_above, drop_inside, DROP_FRAMES)

        if HIDE_OBJECT_AFTER_DROP:
            v27_set_visibility(obj_prim, False)
            record["object_hidden_after_drop"] = True

        # 5. 回到抓取前搜索位，方便下一轮继续主动搜索
        robot_now = v27_get_translate(robot_prim)
        print(f"[RETURN] {robot_now} -> {robot_start_pose}")

        await v27_animate_robot(stage, robot_prim, robot_now, robot_start_pose, RETURN_FRAMES, look_at)

        record["status"] = "success"
        record["action_completed"] = True

    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["duration_sec"] = round(time.time() - t0, 3)
        v27_save_json_atomic(ACTION_RESULT_JSON, record)
        raise

    except Exception as e:
        record["status"] = "failed_exception"
        record["error"] = repr(e)
        print(f"[FAILED_EXCEPTION] {repr(e)}")

    record["duration_sec"] = round(time.time() - t0, 3)
    v27_save_json_atomic(ACTION_RESULT_JSON, record)

    print("=" * 80)
    print("[V2.7 ACTION RESULT]")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"[SAVED] {ACTION_RESULT_JSON}")
    print("=" * 80)

    return record


async def v27_main_loop():
    print("=" * 80)
    print("[START] V2.7 head camera action server - rewritten")
    print(f"[WAIT PLAN] {PLAN_JSON}")
    print("=" * 80)

    stage = v27_get_stage()

    robot_prim = v27_get_prim(stage, ROBOT_PATH)
    if robot_prim is None:
        raise RuntimeError(f"找不到机器人：{ROBOT_PATH}")

    cam_prim = v27_get_prim(stage, HEAD_CAMERA_PATH)
    if cam_prim is None:
        raise RuntimeError(f"找不到头部相机：{HEAD_CAMERA_PATH}")

    if IGNORE_EXISTING_PLAN_ON_START:
        existing_plan = v27_load_json_safe(PLAN_JSON)
        existing_plan_id = existing_plan.get("plan_id") if existing_plan else None
        if existing_plan_id:
            PROCESSED_PLAN_IDS.add(existing_plan_id)
            print(f"[SKIP EXISTING PLAN ON START] {existing_plan_id}")

    while True:
        plan = v27_load_json_safe(PLAN_JSON)

        if plan is None:
            await v27_pause_frames(WAIT_FRAMES)
            continue

        plan_id = plan.get("plan_id")

        if not plan_id:
            await v27_pause_frames(WAIT_FRAMES)
            continue

        if plan_id in PROCESSED_PLAN_IDS:
            await v27_pause_frames(WAIT_FRAMES)
            continue

        if plan.get("selected_task") is None:
            await v27_pause_frames(WAIT_FRAMES)
            continue

        PROCESSED_PLAN_IDS.add(plan_id)

        await v27_execute_plan(stage, robot_prim, plan)
        await v27_pause_frames(10)


task = asyncio.ensure_future(v27_main_loop())
setattr(builtins, TASK_NAME, task)