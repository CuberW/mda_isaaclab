"""Pinch-grasp quality checks shared by task pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from control.end_effectors import EndEffectorSpec


@dataclass(frozen=True)
class PinchQuality:
    ok: bool
    reason: str
    contact_bodies: tuple[str, ...]
    pinch_center: np.ndarray
    object_center: np.ndarray
    pinch_distance: float
    lateral_distance: float
    vertical_distance: float
    object_radius: float
    center_limit: float
    bilateral_contact: bool
    object_between_fingers: bool
    finger_span: float

    def detail(self) -> str:
        return (
            f"{self.reason} bodies={list(self.contact_bodies)} "
            f"pinch_dist={self.pinch_distance:.3f}m "
            f"lat={self.lateral_distance:.3f}m "
            f"z={self.vertical_distance:.3f}m "
            f"limit={self.center_limit:.3f}m "
            f"radius={self.object_radius:.3f}m "
            f"between={self.object_between_fingers}"
        )


def body_grasp_extent(model, body_name: str) -> tuple[float, float]:
    """Return conservative object radius and half-height from MuJoCo geoms."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return 0.03, 0.05
    radius = 0.03
    half_height = 0.05
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    for gid in range(start, start + count):
        geom_type = int(model.geom_type[gid])
        size = np.asarray(model.geom_size[gid], dtype=float)
        if geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            radius = max(radius, float(size[0]))
            if len(size) > 1:
                half_height = max(half_height, float(size[1]))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = max(radius, float(size[0]))
            half_height = max(half_height, float(size[0]))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            radius = max(radius, float(max(size[0], size[1])))
            half_height = max(half_height, float(size[2]))
    return radius, half_height


def existing_bodies(env, names: Iterable[str]) -> list[str]:
    body_names = set(getattr(env, "_body_names", []))
    return [name for name in names if name in body_names]


def geom_positions(env, names: Iterable[str]) -> list[np.ndarray]:
    """Return world positions for named MuJoCo geoms that exist."""
    positions: list[np.ndarray] = []
    if not hasattr(env, "geom_index"):
        return positions
    for name in names:
        gid = env.geom_index(name)
        if gid is not None and int(gid) >= 0:
            positions.append(env.data.geom_xpos[int(gid)].copy())
    return positions


def pinch_center(env, spec: EndEffectorSpec) -> np.ndarray:
    """Return the current world-space center between official finger frames."""
    pads = geom_positions(env, getattr(spec, "finger_pad_geoms", ()))
    if len(pads) >= 2:
        return np.mean(pads[:2], axis=0)
    finger_tips = existing_bodies(env, spec.finger_tip_bodies)
    if len(finger_tips) >= 2:
        return np.mean([env.get_body_position(name) for name in finger_tips[:2]], axis=0)
    fingers = existing_bodies(env, spec.finger_bodies)
    if len(fingers) >= 2:
        return np.mean([env.get_body_position(name) for name in fingers[:2]], axis=0)
    if spec.pinch_body in getattr(env, "_body_names", []):
        return env.get_body_position(spec.pinch_body)
    return env.get_body_position(spec.primary_gripper_body)


def finger_span(env, spec: EndEffectorSpec) -> float:
    pads = geom_positions(env, getattr(spec, "finger_pad_geoms", ()))
    if len(pads) >= 2:
        return float(np.linalg.norm(pads[0] - pads[1]))
    fingers = existing_bodies(env, spec.finger_tip_bodies) or existing_bodies(env, spec.finger_bodies)
    if len(fingers) < 2:
        return 0.0
    return float(np.linalg.norm(env.get_body_position(fingers[0]) - env.get_body_position(fingers[1])))


def object_between_fingers(env, spec: EndEffectorSpec, obj_body: str,
                           object_radius: float, margin: float = 0.018) -> bool:
    pads = geom_positions(env, getattr(spec, "finger_pad_geoms", ()))
    if len(pads) >= 2:
        p0, p1 = pads[:2]
    else:
        fingers = existing_bodies(env, spec.finger_tip_bodies) or existing_bodies(env, spec.finger_bodies)
        if len(fingers) < 2:
            return False
        p0 = env.get_body_position(fingers[0])
        p1 = env.get_body_position(fingers[1])
    if obj_body not in getattr(env, "_body_names", []):
        return False
    obj = env.get_body_position(obj_body)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return False
    unit = axis / length
    proj = float(np.dot(obj - p0, unit))
    closest = p0 + np.clip(proj, 0.0, length) * unit
    perpendicular = float(np.linalg.norm(obj - closest))
    inside_segment = -margin <= proj <= length + margin
    return bool(inside_segment and perpendicular <= object_radius + margin)


def evaluate_pinch_grasp(
    env,
    grasp_manager,
    spec: EndEffectorSpec,
    obj_body: str,
    *,
    center_margin: float = 0.026,
    min_center_limit: float = 0.040,
    max_center_limit: float = 0.080,
    require_bilateral_contact: bool = False,
) -> PinchQuality:
    """Check whether an object is genuinely captured near the pinch center."""
    mujoco.mj_forward(env.model, env.data)
    obj = env.get_body_position(obj_body) if obj_body in getattr(env, "_body_names", []) else np.zeros(3)
    center = pinch_center(env, spec)
    radius, half_height = body_grasp_extent(env.model, obj_body)
    center_limit = float(np.clip(radius + center_margin, min_center_limit, max_center_limit))
    bodies = tuple(grasp_manager.contacting_gripper_bodies(spec.weld_name, obj_body))
    bilateral = grasp_manager.detect_contact(spec.weld_name, obj_body, min_gripper_bodies=2)
    between = object_between_fingers(env, spec, obj_body, radius)
    dist = float(np.linalg.norm(center - obj))
    lateral_dist = float(np.linalg.norm(center[:2] - obj[:2]))
    vertical_dist = float(abs(center[2] - obj[2]))
    rim_or_side_centered = bool(
        lateral_dist <= center_limit
        and vertical_dist <= half_height + center_margin + 0.035
    )
    span = finger_span(env, spec)
    real_finger_bodies = set(spec.finger_bodies or spec.gripper_bodies)
    real_finger_bodies.update(spec.finger_tip_bodies)
    has_real_contact = bool(set(bodies).intersection(real_finger_bodies))
    center_ok = bool(dist <= center_limit or rim_or_side_centered)
    if require_bilateral_contact:
        capture_ok = bool(bilateral and between)
    else:
        capture_ok = bool(bilateral or between or rim_or_side_centered)
    ok = bool(has_real_contact and center_ok and capture_ok)
    if ok:
        reason = "verified_pinch_capture"
    elif not has_real_contact:
        reason = "no_real_finger_contact"
    elif not center_ok:
        reason = "pinch_center_too_far"
    elif not capture_ok:
        reason = "object_not_between_fingers"
    else:
        reason = "pinch_capture_failed"
    return PinchQuality(
        ok=ok,
        reason=reason,
        contact_bodies=bodies,
        pinch_center=center,
        object_center=obj,
        pinch_distance=dist,
        lateral_distance=lateral_dist,
        vertical_distance=vertical_dist,
        object_radius=radius,
        center_limit=center_limit,
        bilateral_contact=bool(bilateral),
        object_between_fingers=bool(between),
        finger_span=span,
    )
