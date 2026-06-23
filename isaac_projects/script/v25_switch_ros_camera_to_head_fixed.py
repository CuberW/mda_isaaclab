import omni.usd
import omni.graph.core as og
from pxr import Sdf


HEAD_CAMERA_PATH = "/World/TrashBotHeadCamera"
RENDER_PRODUCT_NODE = "/Graph/ROS_Camera/RenderProduct"
RGB_PUBLISH_NODE = "/Graph/ROS_Camera/RGBPublish"
CAMERA_INFO_NODE = "/Graph/ROS_Camera/CameraInfoPublish"


def set_og_attr(attr_path, value):
    try:
        attr = og.Controller.attribute(attr_path)
        og.Controller.set(attr, value)
        print(f"[OG SET] {attr_path} = {value}")
        return True
    except Exception as e:
        print(f"[OG WARN] failed set {attr_path}: {repr(e)}")
        return False


def set_usd_attr(prim_path, attr_name, value):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))

    if not prim or not prim.IsValid():
        print(f"[USD WARN] prim not found: {prim_path}")
        return False

    attr = prim.GetAttribute(attr_name)

    if not attr:
        print(f"[USD WARN] attr not found: {prim_path}.{attr_name}")
        return False

    try:
        attr.Set(value)
        print(f"[USD SET] {prim_path}.{attr_name} = {value}")
        return True
    except Exception as e:
        print(f"[USD WARN] failed set {prim_path}.{attr_name}: {repr(e)}")
        return False


def print_node_attrs(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))

    if not prim or not prim.IsValid():
        print(f"[WARN] prim not found: {prim_path}")
        return

    print("\n" + "=" * 80)
    print(f"[ATTRS] {prim_path}")
    print("=" * 80)

    for attr in prim.GetAttributes():
        try:
            value = attr.Get()
        except Exception:
            value = None

        print(f"{attr.GetName()} | type={attr.GetTypeName()} | value={value}")


def main():
    print("=" * 80)
    print("[START] switch ROS camera to head camera")
    print("=" * 80)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD Stage。")

    head_cam = stage.GetPrimAtPath(Sdf.Path(HEAD_CAMERA_PATH))
    if not head_cam or not head_cam.IsValid():
        raise RuntimeError(f"找不到头部相机：{HEAD_CAMERA_PATH}。请先运行 v25_create_head_camera.py")

    render_node = stage.GetPrimAtPath(Sdf.Path(RENDER_PRODUCT_NODE))
    if not render_node or not render_node.IsValid():
        raise RuntimeError(f"找不到 RenderProduct 节点：{RENDER_PRODUCT_NODE}")

    # 1. 优先用 OmniGraph API 设置 cameraPrim
    ok = False

    candidate_attr_paths = [
        f"{RENDER_PRODUCT_NODE}.inputs:cameraPrim",
        f"{RENDER_PRODUCT_NODE}.inputs:camera",
        f"{RENDER_PRODUCT_NODE}.inputs:cameraPath",
    ]

    for attr_path in candidate_attr_paths:
        if set_og_attr(attr_path, HEAD_CAMERA_PATH):
            ok = True

    # 2. 兜底：直接用 USD Attribute 设置
    candidate_usd_attrs = [
        "inputs:cameraPrim",
        "inputs:camera",
        "inputs:cameraPath",
    ]

    for attr_name in candidate_usd_attrs:
        if set_usd_attr(RENDER_PRODUCT_NODE, attr_name, Sdf.Path(HEAD_CAMERA_PATH)):
            ok = True
        if set_usd_attr(RENDER_PRODUCT_NODE, attr_name, HEAD_CAMERA_PATH):
            ok = True

    # 3. frameId 改成头部相机，topic 先保持不变，避免 WSL 命令要改
    set_og_attr(f"{RGB_PUBLISH_NODE}.inputs:frameId", "trash_head_camera")
    set_og_attr(f"{CAMERA_INFO_NODE}.inputs:frameId", "trash_head_camera")

    # topic 保持：
    # RGBPublish.inputs:topicName = camera/rgb
    # CameraInfoPublish.inputs:topicName = camera_info

    print_node_attrs(RENDER_PRODUCT_NODE)
    print_node_attrs(RGB_PUBLISH_NODE)
    print_node_attrs(CAMERA_INFO_NODE)

    print("=" * 80)

    if ok:
        print("[OK] 已尝试把 ROS RenderProduct 相机切换到 /World/TrashBotHeadCamera")
        print("[NEXT] 播放仿真后，在 WSL 重新采图检查是否变成头部相机视角。")
    else:
        print("[WARN] 未能确认写入 cameraPrim。请手动打开 Action Graph：")
        print("       /Graph/ROS_Camera/RenderProduct")
        print("       把 inputs:cameraPrim 改成 /World/TrashBotHeadCamera")

    print("=" * 80)


main()