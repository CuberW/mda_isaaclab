"""Build a Kuavo S62 URDF with a physically attached right two-finger gripper.

The original Kuavo S62 asset has no actuated gripper joints.  Previous task-319
prototypes spawned the gripper as a separate articulation and teleported it to
the wrist every physics step, which makes contact forces non-physical.  This
module keeps the upstream robot URDF untouched and generates a temporary merged
URDF where the gripper base is fixed to ``zarm_r7_end_effector``.  The default
mount rotates the gripper so its local +X finger direction is inline with the
right-wrist local -Z forearm/end-effector axis.
"""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


RIGHT_WRIST_LINK = "zarm_r7_end_effector"
GRIPPER_BASE_LINK = "gripper_base"
MOUNT_JOINT_NAME = "right_gripper_mount_joint"
GENERATOR_VERSION = "v4_inline_gripper_axis"
INLINE_GRIPPER_MOUNT_RPY = (0.0, 1.5707963267948966, 0.0)
RIGHT_WRIST_PLACEHOLDER_LINKS = {
    "zarm_r7_end_effector",
    "zarm_r7_end_effector_1",
    "zarm_r7_end_effector_2",
}


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()[:12]


def ensure_kuavo_with_gripper_urdf(
    kuavo_urdf: Path,
    gripper_urdf: Path,
    *,
    output_dir: Path | None = None,
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_rpy: tuple[float, float, float] = INLINE_GRIPPER_MOUNT_RPY,
    strip_right_wrist_collision: bool = True,
) -> Path:
    """Return a generated URDF path for Kuavo with an attached right gripper.

    The generated file is cached in ``/tmp`` by content digest.  It is safe to
    call this function at module import time before IsaacLab converts the URDF.
    """

    kuavo_urdf = kuavo_urdf.resolve()
    gripper_urdf = gripper_urdf.resolve()
    if not kuavo_urdf.is_file():
        raise FileNotFoundError(f"Kuavo URDF not found: {kuavo_urdf}")
    if not gripper_urdf.is_file():
        raise FileNotFoundError(f"Gripper URDF not found: {gripper_urdf}")

    output_dir = output_dir or Path(tempfile.gettempdir()) / "task319_urdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = f"{GENERATOR_VERSION}_{_file_digest(kuavo_urdf)}_{_file_digest(gripper_urdf)}"
    output_path = output_dir / f"kuavo_s62_with_right_gripper_{digest}.urdf"
    if output_path.is_file():
        return output_path

    robot_tree = ET.parse(kuavo_urdf)
    robot_root = robot_tree.getroot()
    gripper_root = ET.parse(gripper_urdf).getroot()

    kuavo_package_root = next((parent for parent in kuavo_urdf.parents if parent.name == "kuavo_assets"), None)
    if kuavo_package_root is not None:
        for mesh in robot_root.findall(".//mesh"):
            filename = mesh.attrib.get("filename", "")
            prefix = "package://kuavo_assets/"
            if filename.startswith(prefix):
                mesh.set("filename", str(kuavo_package_root / filename[len(prefix):]))

    link_names = {elem.attrib.get("name") for elem in robot_root.findall("link")}
    joint_names = {elem.attrib.get("name") for elem in robot_root.findall("joint")}
    if RIGHT_WRIST_LINK not in link_names:
        raise RuntimeError(f"Cannot attach gripper: missing wrist link {RIGHT_WRIST_LINK!r}.")
    if GRIPPER_BASE_LINK in link_names:
        raise RuntimeError(f"Cannot attach gripper: link name already exists: {GRIPPER_BASE_LINK!r}.")
    if MOUNT_JOINT_NAME in joint_names:
        raise RuntimeError(f"Cannot attach gripper: joint name already exists: {MOUNT_JOINT_NAME!r}.")

    if strip_right_wrist_collision:
        for link in robot_root.findall("link"):
            if link.attrib.get("name") in RIGHT_WRIST_PLACEHOLDER_LINKS:
                for child in list(link):
                    if child.tag in {"visual", "collision"}:
                        link.remove(child)

    for elem in list(gripper_root):
        if elem.tag in {"link", "joint", "material", "gazebo"}:
            robot_root.append(elem)

    mount_joint = ET.Element("joint", {"name": MOUNT_JOINT_NAME, "type": "fixed"})
    ET.SubElement(
        mount_joint,
        "origin",
        {
            "xyz": f"{mount_xyz[0]} {mount_xyz[1]} {mount_xyz[2]}",
            "rpy": f"{mount_rpy[0]} {mount_rpy[1]} {mount_rpy[2]}",
        },
    )
    ET.SubElement(mount_joint, "parent", {"link": RIGHT_WRIST_LINK})
    ET.SubElement(mount_joint, "child", {"link": GRIPPER_BASE_LINK})
    robot_root.append(mount_joint)

    robot_root.set("name", f"{robot_root.attrib.get('name', 'kuavo_s62')}_with_right_gripper")
    ET.indent(robot_tree, space="  ")
    robot_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
