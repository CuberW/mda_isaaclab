import asyncio
import builtins
import json
import math
import time
from pathlib import Path

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom


# ============================================================
# 0. 停掉旧 V2 任务
# ============================================================

OLD_TASK_NAME = "TRASHBOT_V2_VISUAL_TASK"

old_task = getattr(builtins, OLD_TASK_NAME, None)
if old_task is not None:
    try:
        old_task.cancel()
        print("[CANCEL] old V2 visual task cancelled.")
    except Exception as e:
        print(f"[WARN] failed to cancel old V2 task: {repr(e)}")


# ============================================================
# 1. 文件路径
# ============================================================

PLAN_JSON = Path(r"D:\isaac_projects\v2_visual_task_plan.json")
RESULT_JSON = Path(r"D:\isaac_projects\v2_visual_execution_result.json")


# ============================================================
# 2. 场景路径
# ============================================================

CAMERA_PATH = "/World/TrashCamera"
ROBOT_PATH = "/World/TrashBotRobot"
TABLE_PATH = "/World/TrashSortingScene/table"

BIN_PATHS = {
    "bin_recyclable_blue": "/World/TrashSortingScene/bin_recyclable_blue",
    "bin_kitchen_green": "/World/TrashSortingScene/bin_kitchen_green",
    "bin_hazardous_red": "/World/TrashSortingScene/bin_hazardous_red",
    "bin_other_gray": "/World/TrashSortingScene/bin_other_gray",
}


TRASH_PRIMS = [
    "/World/TrashSortingScene/trash_plastic_bottle",
    "/World/TrashSortingScene/trash_dirty_bottle",
    "/World/TrashSortingScene/trash_can",
    "/World/TrashSortingScene/trash_paper_box",
    "/World/TrashSortingScene/trash_apple_core",
    "/World/TrashSortingScene/trash_banana_peel",
    "/World/TrashSortingScene/trash_battery",
    "/World/TrashSortingScene/trash_medicine_box",
    "/World/TrashSortingScene/trash_broken_cup",
    "/World/TrashSortingScene/trash_tissue",
]


RAW_CLASS_TO_PRIMS = {
    "bottle": [
        "/World/TrashSortingScene/trash_plastic_bottle",
        "/World/TrashSortingScene/trash_dirty_bottle",
    ],
    "bottle2": [
        "/World/TrashSortingScene/trash_dirty_bottle",
        "/World/TrashSortingScene/trash_plastic_bottle",
    ],
    "can": [
        "/World/TrashSortingScene/trash_can",
    ],
    "paper": [
        "/World/TrashSortingScene/trash_tissue",
        "/World/TrashSortingScene/trash_paper_box",
    ],
    "papercup": [
        "/World/TrashSortingScene/trash_broken_cup",
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
    "drug": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "drugbag": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "drugbox": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
    "capsule": [
        "/World/TrashSortingScene/trash_medicine_box",
    ],
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
}


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
# 3. 运动参数
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

# 如果视觉估计点和候选 Prim 坐标距离超过这个值，仍可 fallback，但会在日志中标出
MAX_STRICT_ATTACH_DISTANCE = 0.45

# 抓完是否隐藏物体。V2 单次验证阶段建议 False，方便观察；闭环验证时可改 True。
HIDE_OBJECT_AFTER_DROP = False


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


def get_table_top_z(stage):
    table_prim = get_prim(stage, TABLE_PATH)

    if table_prim is None:
        print("[WARN] table prim not found, use fallback table z = 0.76")
        return 0.76

    try:
        box = compute_world_bbox(stage, table_prim)
        top_z = float(box.GetMax()[2])
        print(f"[TABLE] top_z={top_z:.4f}")
        return top_z
    except Exception as e:
        print(f"[WARN] failed to compute table top z: {repr(e)}, use fallback 0.76")
        return 0.76


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


# ============================================================
# 5. V2 核心：像素点反算桌面世界坐标
# ============================================================

def pixel_to_world_on_table(stage, pixel_u, pixel_v, image_width, image_height):
    camera_prim = get_prim(stage, CAMERA_PATH)

    if camera_prim is None:
        raise RuntimeError(f"找不到相机 Prim：{CAMERA_PATH}")

    camera = UsdGeom.Camera(camera_prim)

    focal_length = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera.GetVerticalApertureAttr().Get())

    if focal_length <= 0:
        raise RuntimeError(f"Invalid camera focal length: {focal_length}")

    # 像素坐标转相机成像平面坐标，USD 相机默认看向本地 -Z
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
        raise RuntimeError(
            f"Ray intersection is behind camera. t={t}, cam_pos={cam_pos}, ray={ray_world}"
        )

    hit = Gf.Vec3d(
        float(cam_pos[0] + ray_world[0] * t),
        float(cam_pos[1] + ray_world[1] * t),
        float(table_z),
    )

    debug = {
        "pixel_u": round(float(pixel_u), 3),
        "pixel_v": round(float(pixel_v), 3),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_focal_length": focal_length,
        "camera_horizontal_aperture": horizontal_aperture,
        "camera_vertical_aperture": vertical_aperture,
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


# ============================================================
# 6. 根据视觉点选择要移动的垃圾 Prim
# ============================================================

def unique_list(items):
    out = []
    seen = set()

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def get_task_raw_class(task):
    return task.get("raw_class_name", task.get("class_name", "unknown"))


def get_task_category(task):
    return task.get("garbage_category", task.get("category", "unknown"))


def get_task_target_bin(task):
    return task.get("target_bin", "unknown")


def choose_nearest_object_prim(stage, task, visual_world):
    raw_cls = get_task_raw_class(task)
    category = get_task_category(task)

    candidate_paths = []
    candidate_paths.extend(RAW_CLASS_TO_PRIMS.get(raw_cls, []))
    candidate_paths.extend(CATEGORY_TO_PRIMS.get(category, []))
    candidate_paths = unique_list(candidate_paths)

    scored = []

    for path in candidate_paths:
        prim = get_prim(stage, path)

        if prim is None:
            continue

        if not is_prim_visible(prim):
            continue

        pos = get_translate(prim)
        dist = vec_distance_xy(pos, visual_world)

        scored.append((dist, path, prim, "class_or_category_candidate"))

    if scored:
        scored.sort(key=lambda x: x[0])
        dist, path, prim, reason = scored[0]

        strict = dist <= MAX_STRICT_ATTACH_DISTANCE

        return {
            "path": path,
            "prim": prim,
            "distance_xy": dist,
            "selection_reason": reason,
            "strict_match": strict,
        }

    # fallback：在所有垃圾中找视觉点最近的可见 Prim
    fallback = []

    for path in TRASH_PRIMS:
        prim = get_prim(stage, path)

        if prim is None:
            continue

        if not is_prim_visible(prim):
            continue

        pos = get_translate(prim)
        dist = vec_distance_xy(pos, visual_world)
        fallback.append((dist, path, prim, "nearest_any_visible_trash"))

    if not fallback:
        return {
            "path": None,
            "prim": None,
            "distance_xy": None,
            "selection_reason": "no_visible_trash",
            "strict_match": False,
        }

    fallback.sort(key=lambda x: x[0])
    dist, path, prim, reason = fallback[0]

    return {
        "path": path,
        "prim": prim,
        "distance_xy": dist,
        "selection_reason": reason,
        "strict_match": dist <= MAX_STRICT_ATTACH_DISTANCE,
    }


# ============================================================
# 7. 文件读取
# ============================================================

def load_plan():
    if not PLAN_JSON.exists():
        raise FileNotFoundError(f"V2 plan not found: {PLAN_JSON}")

    with open(PLAN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "selected_task" not in data or data["selected_task"] is None:
        raise ValueError("Plan has no selected_task.")

    if "image_width" not in data or "image_height" not in data:
        raise ValueError("Plan must contain image_width and image_height.")

    return data


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


# ============================================================
# 8. 单次 V2 视觉抓取执行
# ============================================================

async def execute_v2_once():
    print("=" * 80)
    print("[START] V2 visual grasp executor")
    print("=" * 80)

    stage = get_stage()
    plan = load_plan()

    task = plan["selected_task"]

    image_width = int(plan["image_width"])
    image_height = int(plan["image_height"])

    centroid = task.get("centroid_px")
    if centroid is None or len(centroid) != 2:
        raise ValueError("selected_task has no valid centroid_px.")

    pixel_u = float(centroid[0])
    pixel_v = float(centroid[1])

    raw_cls = get_task_raw_class(task)
    category = get_task_category(task)
    target_bin = get_task_target_bin(task)

    print(f"[PLAN] {plan.get('plan_id')}")
    print(f"[SOURCE IMAGE] {plan.get('source_image')}")
    print(f"[SELECTED] raw={raw_cls}, category={category}, target_bin={target_bin}")
    print(f"[PIXEL] u={pixel_u:.2f}, v={pixel_v:.2f}, image={image_width}x{image_height}")

    robot_prim = get_prim(stage, ROBOT_PATH)

    if robot_prim is None:
        raise RuntimeError(f"找不到机器人 Prim：{ROBOT_PATH}")

    bin_path = BIN_PATHS.get(target_bin)
    bin_prim = get_prim(stage, bin_path) if bin_path else None

    if bin_prim is None:
        raise RuntimeError(f"找不到目标垃圾桶：{target_bin}, path={bin_path}")

    visual_world, projection_debug = pixel_to_world_on_table(
        stage=stage,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        image_width=image_width,
        image_height=image_height,
    )

    print(f"[V2 WORLD HIT] {visual_world}")

    obj_info = choose_nearest_object_prim(
        stage=stage,
        task=task,
        visual_world=visual_world,
    )

    obj_prim = obj_info["prim"]
    obj_path = obj_info["path"]

    if obj_prim is None:
        raise RuntimeError("没有找到可移动垃圾 Prim。")

    print(f"[ATTACH PRIM] {obj_path}")
    print(f"[ATTACH DIST XY] {obj_info['distance_xy']:.4f}")
    print(f"[ATTACH REASON] {obj_info['selection_reason']}")
    print(f"[STRICT MATCH] {obj_info['strict_match']}")

    start_time = time.time()

    record = {
        "plan_id": plan.get("plan_id"),
        "mode": "V2_VISUAL_PIXEL_TO_WORLD",
        "source_image": plan.get("source_image"),

        "task_id": task.get("task_id"),
        "raw_class_name": raw_cls,
        "garbage_category": category,
        "garbage_category_cn": CATEGORY_CN.get(category, category),
        "target_bin": target_bin,
        "target_bin_cn": BIN_CN.get(target_bin, target_bin),
        "confidence": float(task.get("confidence", 0.0)),

        "centroid_px": [round(pixel_u, 3), round(pixel_v, 3)],
        "image_width": image_width,
        "image_height": image_height,

        "visual_world_xyz": [
            round(float(visual_world[0]), 4),
            round(float(visual_world[1]), 4),
            round(float(visual_world[2]), 4),
        ],
        "projection_debug": projection_debug,

        "attached_object_prim": obj_path,
        "attach_distance_xy": round(float(obj_info["distance_xy"]), 4),
        "attach_selection_reason": obj_info["selection_reason"],
        "attach_strict_match": bool(obj_info["strict_match"]),

        "bin_prim": bin_path,

        "status": "pending",
        "visual_grasp_used": True,
        "model_coordinate_used_for_planning": False,
        "model_coordinate_used_for_mesh_animation": True,

        "duration_sec": 0.0,
    }

    try:
        set_translate(robot_prim, ROBOT_HOME)
        await next_frame()

        obj_actual_start = get_translate(obj_prim)
        bin_pos = get_translate(bin_prim)
        robot_start = get_translate(robot_prim)

        # 关键变化：approach_pos 来自 YOLO 像素反算的 visual_world，而不是 obj_actual_start
        approach_pos = Gf.Vec3d(
            float(visual_world[0] + APPROACH_OFFSET[0]),
            float(visual_world[1] + APPROACH_OFFSET[1]),
            float(ROBOT_HOME[2]),
        )

        print(f"[APPROACH - VISUAL] robot {robot_start} -> {approach_pos}")
        await animate_prim(robot_prim, robot_start, approach_pos, MOVE_FRAMES)

        grip_pos = approach_pos + GRIP_OFFSET

        print(f"[GRASP] attached object {obj_actual_start} -> {grip_pos}")
        await animate_prim(obj_prim, obj_actual_start, grip_pos, PICK_FRAMES)

        bin_approach_pos = Gf.Vec3d(
            float(bin_pos[0]),
            float(bin_pos[1] - 0.55),
            float(ROBOT_HOME[2]),
        )

        print(f"[TRANSPORT] robot+object -> {bin_approach_pos}")
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

        print(f"[DROP] object {object_current} -> {drop_above} -> {drop_inside}")
        await animate_prim(obj_prim, object_current, drop_above, DROP_FRAMES)
        await animate_prim(obj_prim, drop_above, drop_inside, DROP_FRAMES)

        if HIDE_OBJECT_AFTER_DROP:
            set_visibility(obj_prim, False)
            print(f"[HIDE] {obj_path}")

        robot_now = get_translate(robot_prim)

        print(f"[RETURN] robot {robot_now} -> {ROBOT_HOME}")
        await animate_prim(robot_prim, robot_now, ROBOT_HOME, RETURN_FRAMES)

        record["status"] = "success"

    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise

    except Exception as e:
        record["status"] = "failed_exception"
        record["error"] = repr(e)
        print(f"[FAILED_EXCEPTION] {repr(e)}")

    record["duration_sec"] = round(time.time() - start_time, 3)

    atomic_write_json(RESULT_JSON, record)

    print("=" * 80)
    print("[RESULT]")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"[SAVED] {RESULT_JSON}")
    print("=" * 80)


task = asyncio.ensure_future(execute_v2_once())
setattr(builtins, OLD_TASK_NAME, task)