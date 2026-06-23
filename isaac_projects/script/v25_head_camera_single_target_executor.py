import asyncio
import builtins
import json
import math
import time
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


OLD_TASK_NAME = "TRASHBOT_V25_HEAD_CAMERA_SINGLE_TASK_V2"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old V2.5 head camera single task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old task: {repr(e)}")


PLAN_JSON = Path(r"D:\isaac_projects\v2_visual_task_plan.json")
RESULT_JSON = Path(r"D:\isaac_projects\v25_head_camera_single_result.json")

CAMERA_PATH = "/World/TrashBotHeadCamera"
ROBOT_PATH = "/World/TrashBotRobot"
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


APPROACH_OFFSET = Gf.Vec3d(0.00, -0.45, 0.00)
GRIP_OFFSET = Gf.Vec3d(0.00, 0.35, 0.95)

DROP_HEIGHT_ABOVE_BIN = 0.85
DROP_INSIDE_HEIGHT = 0.20

MOVE_FRAMES = 55
PICK_FRAMES = 25
DROP_FRAMES = 25
RETURN_FRAMES = 45

# 头部相机是扩展示范，允许略大误差，但超过 0.60 仍判失败
MAX_ATTACH_DISTANCE = 0.60
HIDE_OBJECT_AFTER_DROP = False


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

    mat = xform.ComputeLocalToWorldTransform(0)
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


def compute_world_bbox(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.ComputeAlignedBox()


def get_world_bbox_center(stage, prim):
    try:
        box = compute_world_bbox(stage, prim)
        min_pt = box.GetMin()
        max_pt = box.GetMax()
        return Gf.Vec3d(
            (float(min_pt[0]) + float(max_pt[0])) / 2.0,
            (float(min_pt[1]) + float(max_pt[1])) / 2.0,
            (float(min_pt[2]) + float(max_pt[2])) / 2.0,
        )
    except Exception:
        return get_translate(prim)


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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def get_task_raw_class(task):
    return task.get("raw_class_name", task.get("class_name", "unknown"))


def get_task_category(task):
    return task.get("garbage_category", task.get("category", "unknown"))


def get_task_target_bin(task):
    return task.get("target_bin", "unknown")


def pixel_to_world_on_table_by_camera(stage, camera_path, pixel_u, pixel_v, image_width, image_height):
    camera_prim = get_prim(stage, camera_path)
    if camera_prim is None:
        raise RuntimeError(f"找不到相机 Prim：{camera_path}")

    camera = UsdGeom.Camera(camera_prim)

    focal_length = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())

    x_ndc = (float(pixel_u) + 0.5) / float(image_width) - 0.5
    y_ndc = 0.5 - (float(pixel_v) + 0.5) / float(image_height)

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

    table_z = get_table_top_z(stage)

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
        "pixel_u": round(float(pixel_u), 3),
        "pixel_v": round(float(pixel_v), 3),
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


def world_to_robot_relative(robot_prim, world_point):
    robot_pos = get_translate(robot_prim)
    return Gf.Vec3d(
        float(world_point[0] - robot_pos[0]),
        float(world_point[1] - robot_pos[1]),
        float(world_point[2] - robot_pos[2]),
    )


def choose_object(stage, raw_class, visual_world):
    candidates = RAW_CLASS_TO_PRIMS.get(raw_class, [])
    scored = []

    for path in candidates:
        prim = get_prim(stage, path)
        if prim is None:
            continue

        center = get_world_bbox_center(stage, prim)
        dist = vec_distance_xy(center, visual_world)
        scored.append((dist, path, prim, center))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    dist, path, prim, center = scored[0]

    return {
        "distance_xy": dist,
        "path": path,
        "prim": prim,
        "center": center,
        "strict_match": dist <= MAX_ATTACH_DISTANCE,
    }


async def execute_once():
    print("=" * 80)
    print("[START] V2.5 head camera single target executor V2")
    print("=" * 80)

    stage = get_stage()
    plan = load_json(PLAN_JSON)

    robot_prim = get_prim(stage, ROBOT_PATH)
    if robot_prim is None:
        raise RuntimeError(f"找不到机器人：{ROBOT_PATH}")

    camera_prim = get_prim(stage, CAMERA_PATH)
    if camera_prim is None:
        raise RuntimeError(f"找不到头部相机：{CAMERA_PATH}")

    task = plan.get("selected_task")
    if task is None:
        raise RuntimeError("plan 中没有 selected_task")

    raw_class = get_task_raw_class(task)
    category = get_task_category(task)
    target_bin = get_task_target_bin(task)

    centroid = task.get("centroid_px")
    if centroid is None or len(centroid) != 2:
        raise RuntimeError("selected_task 没有 centroid_px")

    image_width = int(plan["image_width"])
    image_height = int(plan["image_height"])

    u = float(centroid[0])
    v = float(centroid[1])

    # 注意：这里不再重置机器人到 home。相机位姿必须和刚才采图时保持一致。
    robot_start_pose = get_translate(robot_prim)

    visual_world, projection_debug = pixel_to_world_on_table_by_camera(
        stage=stage,
        camera_path=CAMERA_PATH,
        pixel_u=u,
        pixel_v=v,
        image_width=image_width,
        image_height=image_height,
    )

    target_rel_robot = world_to_robot_relative(robot_prim, visual_world)
    obj_info = choose_object(stage, raw_class, visual_world)

    bin_path = BIN_PATHS.get(target_bin)
    bin_prim = get_prim(stage, bin_path) if bin_path else None

    start_time = time.time()

    record = {
        "mode": "V2_5_HEAD_CAMERA_SINGLE_TARGET_V2",
        "plan_id": plan.get("plan_id"),
        "source_image": plan.get("source_image"),
        "raw_class_name": raw_class,
        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),
        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),
        "confidence": float(task.get("confidence", 0.0)),
        "centroid_px": [round(u, 3), round(v, 3)],
        "image_width": image_width,
        "image_height": image_height,
        "camera_path": CAMERA_PATH,
        "robot_pose_at_projection_xyz": [
            round(float(robot_start_pose[0]), 4),
            round(float(robot_start_pose[1]), 4),
            round(float(robot_start_pose[2]), 4),
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
        "visual_grasp_used": True,
        "head_camera_used": True,
        "relative_position_computed": True,
        "model_coordinate_used_for_planning": False,
        "model_coordinate_used_for_mesh_animation": True,
        "status": "pending",
        "duration_sec": 0.0,
    }

    print(f"[SELECTED] raw={raw_class}, category={category}, bin={target_bin}")
    print(f"[ROBOT POSE] {robot_start_pose}")
    print(f"[PIXEL] u={u:.2f}, v={v:.2f}, image={image_width}x{image_height}")
    print(f"[WORLD] {visual_world}")
    print(f"[REL ROBOT] {target_rel_robot}")
    print(f"[ATTACH] {record['attached_object_prim']}, dist={record['attach_distance_xy']}")

    if obj_info is None:
        record["status"] = "failed_no_matching_object"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    if not obj_info["strict_match"]:
        record["status"] = "failed_visual_object_mismatch"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    if bin_prim is None:
        record["status"] = "failed_no_bin"
        record["duration_sec"] = round(time.time() - start_time, 3)
        save_json_atomic(RESULT_JSON, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    try:
        obj_prim = obj_info["prim"]

        robot_start = get_translate(robot_prim)
        obj_start = get_translate(obj_prim)
        bin_pos = get_translate(bin_prim)

        approach_pos = Gf.Vec3d(
            float(visual_world[0] + APPROACH_OFFSET[0]),
            float(visual_world[1] + APPROACH_OFFSET[1]),
            float(robot_start[2]),
        )

        print(f"[APPROACH - HEAD VISUAL] {robot_start} -> {approach_pos}")
        await animate_prim(robot_prim, robot_start, approach_pos, MOVE_FRAMES)

        grip_pos = approach_pos + GRIP_OFFSET

        print(f"[GRASP] {obj_start} -> {grip_pos}")
        await animate_prim(obj_prim, obj_start, grip_pos, PICK_FRAMES)

        bin_approach_pos = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1] - 0.55),
            float(robot_start[2]),
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

        print(f"[RETURN] robot -> search pose {robot_start_pose}")
        robot_now = get_translate(robot_prim)
        await animate_prim(robot_prim, robot_now, robot_start_pose, RETURN_FRAMES)

        record["status"] = "success"

    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise

    except Exception as e:
        record["status"] = "failed_exception"
        record["error"] = repr(e)
        print(f"[FAILED_EXCEPTION] {repr(e)}")

    record["duration_sec"] = round(time.time() - start_time, 3)
    save_json_atomic(RESULT_JSON, record)

    print("=" * 80)
    print("[RESULT]")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"[SAVED] {RESULT_JSON}")
    print("=" * 80)


task = asyncio.ensure_future(execute_once())
setattr(builtins, OLD_TASK_NAME, task)