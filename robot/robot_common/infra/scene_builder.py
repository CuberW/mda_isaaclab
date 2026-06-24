"""
MuJoCo scene builder - programmatic construction of simulation scenes.

Useful for creating custom test scenes without manually editing XML.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from robot_common.infra.logging import logger


class SceneBuilder:
    """Programmatic MuJoCo scene builder.

    Creates MJCF XML files with robots, objects, cameras, and lighting.
    """

    def __init__(self, model_name: str = "custom_scene"):
        self.model_name = model_name
        self.root = ET.Element("mujoco", model=model_name)
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true")

        # Options
        ET.SubElement(self.root, "option",
                      timestep="0.002",
                      gravity="0 0 -9.81",
                      integrator="implicitfast")

        # Visual
        ET.SubElement(self.root, "visual")
        self.root.find("visual").append(ET.Element("global",
                                                    offwidth="640", offheight="480"))

        # Defaults
        self.default = ET.SubElement(self.root, "default")

        # Assets
        self.asset = ET.SubElement(self.root, "asset")

        # World body
        self.worldbody = ET.SubElement(self.root, "worldbody")

        # Contact excludes
        self.contact = ET.SubElement(self.root, "contact")

        # Actuators
        self.actuator = ET.SubElement(self.root, "actuator")

    def add_light(self, pos: Tuple[float, float, float] = (2, 2, 3),
                  directional: bool = True):
        """Add a light source."""
        attrs = {"pos": f"{pos[0]} {pos[1]} {pos[2]}"}
        if directional:
            attrs["directional"] = "true"
            attrs["dir"] = f"{-pos[0]/3} {-pos[1]/3} {-pos[2]/3}"
        ET.SubElement(self.worldbody, "light", **attrs)

    def add_floor(self, size: Tuple[float, float] = (3.0, 3.0),
                  rgba: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)):
        """Add a floor plane."""
        ET.SubElement(self.worldbody, "geom",
                      name="floor", type="plane",
                      size=f"{size[0]} {size[1]} 0.01",
                      rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

    def add_box(self, name: str, size: Tuple[float, float, float],
                pos: Tuple[float, float, float],
                rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
                has_freejoint: bool = True,
                mass: float = 0.1):
        """Add a box object."""
        body = ET.SubElement(self.worldbody, "body", name=name,
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        if has_freejoint:
            ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom",
                      type="box",
                      size=f"{size[0]} {size[1]} {size[2]}",
                      rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
                      mass=str(mass))

    def add_cylinder(self, name: str, radius: float, height: float,
                     pos: Tuple[float, float, float],
                     rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
                     has_freejoint: bool = True,
                     mass: float = 0.1):
        """Add a cylinder object."""
        body = ET.SubElement(self.worldbody, "body", name=name,
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        if has_freejoint:
            ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom",
                      type="cylinder",
                      size=f"{radius} {height/2}",
                      rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
                      mass=str(mass))

    def add_sphere(self, name: str, radius: float,
                   pos: Tuple[float, float, float],
                   rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
                   has_freejoint: bool = True,
                   mass: float = 0.05):
        """Add a sphere object."""
        body = ET.SubElement(self.worldbody, "body", name=name,
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        if has_freejoint:
            ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom",
                      type="sphere",
                      size=str(radius),
                      rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
                      mass=str(mass))

    def add_camera(self, name: str,
                   pos: Tuple[float, float, float] = (1.5, 0, 1.5),
                   quat: Tuple[float, float, float, float] = (0.7, 0.7, 0, 0),
                   fovy: float = 60.0,
                   resolution: Tuple[int, int] = (640, 480)):
        """Add a camera to the scene."""
        # Camera is added as a body with a camera element
        body = ET.SubElement(self.worldbody, "body", name=f"{name}_body",
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        ET.SubElement(body, "camera", name=name,
                     fovy=str(fovy),
                     resolution=f"{resolution[0]} {resolution[1]}")

    def add_table(self, name: str = "table",
                  size: Tuple[float, float, float] = (0.6, 0.4, 0.02),
                  pos: Tuple[float, float, float] = (0.5, 0, 0.4),
                  leg_height: float = 0.38,
                  rgba: Tuple[float, float, float, float] = (0.6, 0.4, 0.2, 1.0)):
        """Add a table with legs."""
        # Tabletop
        body = ET.SubElement(self.worldbody, "body", name=name,
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        ET.SubElement(body, "geom", name=f"{name}_top",
                     type="box",
                     size=f"{size[0]} {size[1]} {size[2]}",
                     rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

        # Legs
        for i, (lx, ly) in enumerate([
            (size[0] - 0.05, size[1] - 0.05),
            (-size[0] + 0.05, size[1] - 0.05),
            (size[0] - 0.05, -size[1] + 0.05),
            (-size[0] + 0.05, -size[1] + 0.05),
        ]):
            ET.SubElement(body, "geom", name=f"{name}_leg_{i}",
                         type="cylinder",
                         size=f"0.02 {leg_height/2}",
                         pos=f"{lx} {ly} {-leg_height/2 - size[2]}",
                         rgba="0.5 0.3 0.1 1")

    def add_trash_bin(self, name: str,
                      pos: Tuple[float, float, float],
                      rgba: Tuple[float, float, float, float],
                      radius: float = 0.12, height: float = 0.36):
        """Add a trash bin (cylinder container)."""
        body = ET.SubElement(self.worldbody, "body", name=name,
                            pos=f"{pos[0]} {pos[1]} {pos[2]}")
        # Outer cylinder (hollow visual)
        ET.SubElement(body, "geom", name=f"{name}_wall",
                     type="cylinder",
                     size=f"{radius} {height/2}",
                     rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")

    def include_xml(self, path: str):
        """Include another MJCF XML file (e.g., robot model)."""
        # MuJoCo include is a special processing instruction, but for XML we use:
        attrs = {"file": path}
        ET.SubElement(self.root, "include", **attrs)

    def add_actuator_control(self, joint_name: str, ctrl_range: str = "-1 1",
                             actuator_type: str = "motor"):
        """Add an actuator for a joint."""
        if actuator_type == "motor":
            ET.SubElement(self.actuator, "motor",
                         name=joint_name, joint=joint_name,
                         ctrlrange=ctrl_range)
        elif actuator_type == "position":
            ET.SubElement(self.actuator, "position",
                         name=joint_name, joint=joint_name,
                         ctrlrange=ctrl_range, kp="10")

    def save(self, path: str):
        """Save the scene to an MJCF XML file."""
        # Pretty-print XML
        self._indent(self.root)
        tree = ET.ElementTree(self.root)
        with open(path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        logger.info(f"Scene saved: {path}")

    def _indent(self, elem, level=0):
        """Pretty-print XML indentation."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def create_garbage_sorting_scene(output_path: str = ""):
    """Create a complete garbage sorting scene programmatically.

    Scene: Stretch robot + 4 trash bins + 10+ garbage items + table
    """
    builder = SceneBuilder("garbage_sorting_scene")

    # Include Stretch robot
    builder.include_xml("simulation/menagerie/hello_robot_stretch/stretch.xml")

    builder.add_light(pos=(2, 2, 3))
    builder.add_floor(size=(3.0, 3.0))

    # Table
    builder.add_table(pos=(1.5, 0, 0.375), size=(0.5, 0.3, 0.375))

    # Trash bins (4 categories, 4 colors)
    bins = [
        ("bin_recyclable", (3.2, 1.2, 0.12), (0.2, 0.3, 0.9, 1.0)),
        ("bin_kitchen", (3.2, 0.4, 0.12), (0.2, 0.8, 0.3, 1.0)),
        ("bin_hazardous", (3.2, -0.4, 0.12), (0.9, 0.2, 0.2, 1.0)),
        ("bin_other", (3.2, -1.2, 0.12), (0.5, 0.5, 0.5, 1.0)),
    ]
    for name, pos, rgba in bins:
        builder.add_trash_bin(name, pos, rgba)

    # Garbage items (10 items scattered on table)
    items = [
        ("plastic_bottle", (1.4, 0.08, 0.76), "cylinder", 0.03, 0.10, (0.2, 0.5, 1.0, 1.0)),
        ("aluminum_can", (1.6, -0.05, 0.76), "cylinder", 0.025, 0.06, (0.7, 0.7, 0.7, 1.0)),
        ("apple_core", (1.45, -0.10, 0.76), "sphere", 0.025, 0.0, (0.8, 0.2, 0.2, 1.0)),
        ("banana_peel", (1.55, 0.12, 0.76), "capsule", 0.015, 0.04, (0.9, 0.9, 0.1, 1.0)),
        ("battery", (1.5, 0.15, 0.76), "cylinder", 0.01, 0.04, (0.3, 0.3, 0.3, 1.0)),
        ("tissue_box", (1.35, -0.12, 0.76), "box", 0.03, 0.005, (1.0, 1.0, 1.0, 1.0)),
        ("newspaper", (1.6, 0.05, 0.76), "box", 0.04, 0.003, (1.0, 1.0, 0.8, 1.0)),
        ("glass_jar", (1.35, 0.02, 0.76), "cylinder", 0.02, 0.07, (0.6, 0.8, 1.0, 1.0)),
        ("ceramic_cup", (1.42, 0.0, 0.76), "cylinder", 0.025, 0.05, (0.8, 0.8, 0.7, 1.0)),
        ("medicine_bottle", (1.58, -0.08, 0.76), "cylinder", 0.015, 0.05, (0.9, 0.5, 0.2, 1.0)),
    ]

    for name, pos, shape, s1, s2, rgba in items:
        if shape == "cylinder":
            builder.add_cylinder(name, s1, s2, pos, rgba, mass=0.1)
        elif shape == "sphere":
            builder.add_sphere(name, s1, pos, rgba, mass=0.05)
        elif shape == "box":
            builder.add_box(name, (s1, abs(s2), s1), pos, rgba, mass=0.1)
        elif shape == "capsule":
            builder.add_cylinder(name, s1, s2 * 2, pos, rgba, mass=0.05)

    # Camera
    builder.add_camera("overhead_camera", pos=(1.5, 0, 2.0), quat=(0.7, 0.7, 0, 0))

    save_path = output_path or "simulation/custom/garbage_sorting_scene.xml"
    builder.save(save_path)
    return save_path


if __name__ == "__main__":
    create_garbage_sorting_scene()
