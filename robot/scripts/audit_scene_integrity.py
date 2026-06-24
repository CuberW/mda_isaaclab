#!/usr/bin/env python
"""Audit MuJoCo task scenes for camera frame and physical layout issues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robot_common.env import MuJoCoEnv


OUT_DIR = PROJECT_ROOT / "results" / "scene_audit"

SCENES = {
    "319": {
        "path": "simulation/menagerie/hello_robot_stretch/trash_sorting.xml",
        "cameras": ["camera_rgb", "camera_depth", "overview_cam"],
        "bodies": [
            "base_link", "table", "bin_recyclable", "bin_kitchen",
            "bin_hazardous", "bin_other",
            *[f"trash_{i:02d}" for i in range(1, 13)],
        ],
        "table": "table",
        "robot_base": "base_link",
        "support_bodies": {"base_link"},
    },
    "22": {
        "path": "simulation/robots/dual_panda_scene.xml",
        "cameras": ["scene_camera"],
        "bodies": [
            "table", "l_base", "r_base", "long_rod", "rod_support",
            "box_obj", "target_region",
        ],
        "table": "table",
        "robot_base": "l_base",
        "support_bodies": {"table", "l_base", "r_base", "rod_support", "target_region"},
    },
}


def _name(model, obj_type, idx: int) -> str:
    return mujoco.mj_id2name(model, obj_type, idx) or f"{obj_type.name}_{idx}"


def _body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _is_descendant(model, child: int, parent: int) -> bool:
    cur = child
    while cur > 0:
        if cur == parent:
            return True
        cur = int(model.body_parentid[cur])
    return child == parent


def _geom_aabb(model, data, gid: int):
    geom_type = int(model.geom_type[gid])
    size = model.geom_size[gid].copy()
    pos = data.geom_xpos[gid].copy()
    mat = data.geom_xmat[gid].reshape(3, 3).copy()
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.array([size[0], size[0], size[0]])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.array([size[0], size[0], size[1]])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([size[0], size[0], size[1] + size[0]])
    else:
        half = np.maximum(size[:3], 0.001)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(pos + mat @ (np.array([sx, sy, sz]) * half))
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0)


def _body_aabb(model, data, body_name: str):
    bid = _body_id(model, body_name)
    if bid < 0:
        return None
    aabbs = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        if _is_descendant(model, body_id, bid):
            aabb = _geom_aabb(model, data, gid)
            if aabb is not None:
                aabbs.append(aabb)
    if not aabbs:
        return None
    mins = np.min([item[0] for item in aabbs], axis=0)
    maxs = np.max([item[1] for item in aabbs], axis=0)
    return mins, maxs


def _camera_axes(env: MuJoCoEnv, cam: str) -> dict:
    pos, _ = env.get_camera_pose(cam)
    center = env.camera_point_to_world(np.array([0.0, 0.0, 1.0]), cam)
    forward = env.camera_point_to_world(np.array([0.0, 0.0, 1.2]), cam) - center
    right = env.camera_point_to_world(np.array([0.2, 0.0, 1.0]), cam) - center
    down = env.camera_point_to_world(np.array([0.0, 0.2, 1.0]), cam) - center
    forward /= np.linalg.norm(forward)
    right /= np.linalg.norm(right)
    down /= np.linalg.norm(down)
    return {
        "position": np.round(pos, 5).tolist(),
        "forward_world": np.round(forward, 5).tolist(),
        "right_world": np.round(right, 5).tolist(),
        "down_world": np.round(down, 5).tolist(),
        "down_dot_gravity": float(np.dot(down, np.array([0.0, 0.0, -1.0]))),
        "right_abs_vertical": float(abs(right[2])),
    }


def _contacts(env: MuJoCoEnv):
    contacts = []
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        g1 = _name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = _name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        contacts.append({
            "geom1": g1,
            "geom2": g2,
            "dist": float(contact.dist),
            "position": np.round(contact.pos, 5).tolist(),
        })
    return sorted(contacts, key=lambda row: row["dist"])


def _render_contact_sheet(images: list[tuple[str, np.ndarray]], output: Path):
    thumbs = []
    for title, img in images:
        pil = Image.fromarray(img).resize((320, 240))
        canvas = Image.new("RGB", (320, 278), (245, 245, 245))
        canvas.paste(pil, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 240, 320, 278), fill=(20, 20, 20))
        draw.text((8, 248), title[:42], fill=(255, 255, 255))
        thumbs.append(canvas)
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 278), (230, 230, 230))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 320, (idx // cols) * 278))
    sheet.save(output)


def audit_scene(task_id: str, spec: dict, settle_steps: int) -> dict:
    env = MuJoCoEnv(spec["path"], width=640, height=480, camera_name=spec["cameras"][0])
    try:
        env.reset()
        for _ in range(settle_steps):
            env.step()
        report: dict = {
            "task": task_id,
            "xml_path": spec["path"],
            "cameras": {},
            "bodies": {},
            "contacts": _contacts(env),
            "violations": [],
        }
        images = []
        for cam in spec["cameras"]:
            axes = _camera_axes(env, cam)
            report["cameras"][cam] = axes
            img = env.render(cam)
            images.append((f"{task_id}:{cam}", img))
            if axes["right_abs_vertical"] > 0.35:
                report["violations"].append(
                    f"{cam}: image-right axis is too vertical ({axes['right_abs_vertical']:.3f})"
                )
        for body in spec["bodies"]:
            aabb = _body_aabb(env.model, env.data, body)
            if aabb is None:
                report["bodies"][body] = {"missing": True}
                report["violations"].append(f"{body}: missing or no geoms")
                continue
            mn, mx = aabb
            report["bodies"][body] = {
                "min": np.round(mn, 5).tolist(),
                "max": np.round(mx, 5).tolist(),
                "center": np.round((mn + mx) * 0.5, 5).tolist(),
            }
            support_bodies = set(spec.get("support_bodies", set()))
            if body not in support_bodies and mn[2] < -0.003:
                report["violations"].append(f"{body}: below floor by {-mn[2]:.4f}m")
        table_name = spec.get("table")
        if table_name and table_name in report["bodies"] and "max" in report["bodies"][table_name]:
            table = report["bodies"][table_name]
            table_top = float(table["max"][2])
            for body, body_report in report["bodies"].items():
                support_bodies = set(spec.get("support_bodies", set()))
                if body in support_bodies or body in {table_name, spec.get("robot_base", "")} or "missing" in body_report:
                    continue
                center = np.asarray(body_report["center"], dtype=float)
                table_min = np.asarray(table["min"], dtype=float)
                table_max = np.asarray(table["max"], dtype=float)
                over_table = (
                    table_min[0] - 0.02 <= center[0] <= table_max[0] + 0.02
                    and table_min[1] - 0.02 <= center[1] <= table_max[1] + 0.02
                )
                if over_table and float(body_report["min"][2]) < table_top - 0.003:
                    report["violations"].append(
                        f"{body}: penetrates table top by {table_top - float(body_report['min'][2]):.4f}m"
                    )
        max_pen = min([c["dist"] for c in report["contacts"]], default=0.0)
        report["max_penetration"] = float(max(0.0, -max_pen))
        severe = [
            c for c in report["contacts"]
            if c["dist"] < -0.002
            and "grasp_pad" not in c["geom1"]
            and "grasp_pad" not in c["geom2"]
            and not (
                c["geom1"].startswith("mjOBJ_GEOM_")
                and c["geom2"].startswith("mjOBJ_GEOM_")
            )
        ]
        for contact in severe:
            report["violations"].append(
                f"contact {contact['geom1']} <-> {contact['geom2']} penetration {-contact['dist']:.4f}m"
            )
        sheet_path = OUT_DIR / f"task_{task_id}_contact_sheet.png"
        _render_contact_sheet(images, sheet_path)
        report["contact_sheet"] = str(sheet_path.relative_to(PROJECT_ROOT))
        report["ok"] = not report["violations"]
        return report
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(SCENES), action="append")
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = args.task or sorted(SCENES)
    reports = [audit_scene(task, SCENES[task], args.settle_steps) for task in tasks]
    output = OUT_DIR / "scene_integrity_report.json"
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[{status}] task {report['task']} -> {report['contact_sheet']}")
        for violation in report["violations"][:20]:
            print(f"  - {violation}")
        if len(report["violations"]) > 20:
            print(f"  ... {len(report['violations']) - 20} more")
    print(f"Report: {output}")
    ok = all(item["ok"] for item in reports)
    return 0 if ok or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
