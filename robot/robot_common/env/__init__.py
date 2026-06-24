"""
MuJoCo environment wrapper - unified simulation interface.

Supports: Mujoco XML scene loading, step simulation, camera rendering,
         joint control, sensor reading, collision detection.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjModel, MjData, Renderer

from robot_common.infra.logging import logger

# Try to import mediapy-friendly rendering
try:
    import mediapy as media
    HAS_MEDIA = True
except ImportError:
    HAS_MEDIA = False


class MuJoCoEnv:
    """Unified MuJoCo environment wrapper for all three tasks."""

    def __init__(
        self,
        xml_path: str,
        render_mode: str = "offscreen",
        camera_name: str = "head_camera",
        width: int = 640,
        height: int = 480,
        control_freq: float = 50.0,
    ):
        """
        Initialize MuJoCo environment.

        Args:
            xml_path: Path to MJCF scene XML file
            render_mode: "offscreen" or "window"
            camera_name: Default camera for rendering
            width: Render width
            height: Render height
            control_freq: Control frequency in Hz
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {xml_path}")

        logger.info(f"Loading MuJoCo scene: {xml_path}")
        self.model = MjModel.from_xml_path(str(xml_path))
        self.data = MjData(self.model)

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.control_freq = control_freq
        self.dt = self.model.opt.timestep
        self._steps_per_control = max(1, int(1.0 / (control_freq * self.dt)))

        # Renderer: lazy-init to avoid OpenGL context conflict with viewer
        self._viewer = None
        self.renderer = None
        self._lazy_renderer = (render_mode != "viewer")
        self._motion_trace = None

        # Compute initial forward kinematics (fixes black screen on viewer launch)
        mujoco.mj_forward(self.model, self.data)

        # State
        self._step_count = 0
        self._episode_time = 0.0
        self._done = False

        # Joint indices cache
        self._actuator_names = self._get_actuator_names()
        self._joint_names = self._get_joint_names()
        self._body_names = self._get_body_names()
        self._camera_names = self._get_camera_names()

        logger.info(f"  Actuators: {len(self._actuator_names)}")
        logger.info(f"  Joints: {len(self._joint_names)}")
        logger.info(f"  Bodies: {len(self._body_names)}")
        logger.info(f"  Cameras: {self._camera_names}")

    # ── Properties ──────────────────────────────────────────
    @property
    def nu(self) -> int:
        """Number of actuators (control dimensions)."""
        return self.model.nu

    @property
    def nq(self) -> int:
        """Number of joint positions."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Number of joint velocities."""
        return self.model.nv

    @property
    def timestep(self) -> float:
        return self.dt

    @property
    def time(self) -> float:
        return self.data.time

    @property
    def step_count(self) -> int:
        return self._step_count

    # ── Name lookups ────────────────────────────────────────
    def _get_actuator_names(self) -> list:
        return [self.model.actuator(i).name for i in range(self.model.nu)]

    def _get_joint_names(self) -> list:
        return [self.model.joint(i).name for i in range(self.model.njnt)]

    def _get_body_names(self) -> list:
        names = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name and name not in names:
                names.append(name)
        return names

    def _get_camera_names(self) -> list:
        names = []
        for i in range(self.model.ncam):
            names.append(self.model.camera(i).name)
        return names

    def actuator_index(self, name: str) -> Optional[int]:
        """Get actuator index by name."""
        try:
            return self._actuator_names.index(name)
        except ValueError:
            return None

    def joint_index(self, name: str) -> Optional[int]:
        """Get joint qpos address by name."""
        try:
            return self.model.joint(name).qposadr[0]
        except Exception:
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            except Exception:
                return None

    def body_index(self, name: str) -> Optional[int]:
        """Get body ID by name."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            return None

    def site_index(self, name: str) -> Optional[int]:
        """Get site ID by name."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        except Exception:
            return None

    def geom_index(self, name: str) -> Optional[int]:
        """Get geom ID by name."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        except Exception:
            return None

    # ── State access ────────────────────────────────────────
    def get_body_position(self, name: str) -> np.ndarray:
        """Get body position in world frame."""
        bid = self.body_index(name)
        if bid is None:
            return np.zeros(3)
        return self.data.xpos[bid].copy()

    def get_body_rotation(self, name: str) -> np.ndarray:
        """Get body rotation matrix (3x3)."""
        bid = self.body_index(name)
        if bid is None:
            return np.eye(3)
        return self.data.xmat[bid].reshape(3, 3).copy()

    def get_body_quat(self, name: str) -> np.ndarray:
        """Get body orientation as quaternion (w,x,y,z)."""
        bid = self.body_index(name)
        if bid is None:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return self.data.xquat[bid].copy()

    def get_site_position(self, name: str) -> np.ndarray:
        """Get site position in world frame."""
        sid = self.site_index(name)
        if sid is None:
            return np.zeros(3)
        return self.data.site_xpos[sid].copy()

    def get_joint_position(self, name: str) -> float:
        """Get scalar joint position by name."""
        jid = self.joint_index(name)
        if jid is None:
            return 0.0
        return float(self.data.qpos[jid])

    def get_joint_velocity(self, name: str) -> float:
        """Get scalar joint velocity by name."""
        try:
            dof_id = self.model.joint(name).dofadr[0]
            return float(self.data.qvel[dof_id])
        except Exception:
            return 0.0

    def get_qpos(self) -> np.ndarray:
        """Get full joint position array."""
        return self.data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        """Get full joint velocity array."""
        return self.data.qvel.copy()

    def get_actuator_positions(self) -> np.ndarray:
        """Get current actuator (control) positions."""
        return self.data.ctrl.copy()

    # ── Control ─────────────────────────────────────────────
    def set_control(self, ctrl: np.ndarray):
        """Set actuator control signals."""
        self.data.ctrl[:] = ctrl[:self.model.nu]

    def set_joint_positions(self, positions: dict):
        """Set joint positions by name."""
        for name, value in positions.items():
            jid = self.joint_index(name)
            if jid is not None:
                self.data.qpos[jid] = value

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._episode_time = 0.0
        self._done = False
        if self._motion_trace is not None:
            self._motion_trace.reset()

    def set_viewer(self, viewer):
        """Attach a MuJoCo viewer for live rendering."""
        self._viewer = viewer

    def set_motion_trace(self, trace):
        """Attach a motion trace collector for per-step diagnostics."""
        self._motion_trace = trace

    def step(self, ctrl: Optional[np.ndarray] = None) -> bool:
        """
        Advance simulation by one control step.

        Args:
            ctrl: Control signal (optional)

        Returns:
            True if simulation is still running
        """
        if ctrl is not None:
            self.set_control(ctrl)

        # Step physics multiple times per control step
        for _ in range(self._steps_per_control):
            mujoco.mj_step(self.model, self.data)
        # Sync viewer once per control step (not every sub-step)
        if self._viewer is not None:
            self._viewer.sync()

        self._step_count += 1
        self._episode_time = self.data.time
        if self._motion_trace is not None:
            self._motion_trace.record(self)
        return not self._done

    # ── Rendering ───────────────────────────────────────────
    def _ensure_renderer(self):
        """Lazy-init renderer to avoid OpenGL conflict with viewer."""
        if self.renderer is None and self._lazy_renderer:
            self.renderer = Renderer(self.model, self.height, self.width)

    def render(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render RGB image from specified camera."""
        cam = camera_name or self.camera_name
        self._ensure_renderer()

        if self.renderer is not None:
            # Try named camera first
            cam_id = -1
            for i in range(self.model.ncam):
                if self.model.camera(i).name == cam:
                    cam_id = i
                    break

            try:
                if cam_id >= 0:
                    self.renderer.update_scene(self.data, camera=cam_id)
                elif self.model.ncam > 0:
                    self.renderer.update_scene(self.data, camera=0)
                else:
                    # No cameras in scene - render from default view
                    self.renderer.update_scene(self.data)
            except Exception:
                try:
                    self.renderer.update_scene(self.data)
                except Exception:
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)

            return self.renderer.render()
        else:
            # Window mode or no renderer
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def render_depth(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render depth image from specified camera."""
        cam = camera_name or self.camera_name
        self._ensure_renderer()
        if self.renderer is not None:
            cam_id = -1
            for i in range(self.model.ncam):
                if self.model.camera(i).name == cam:
                    cam_id = i
                    break
            try:
                if cam_id >= 0:
                    self.renderer.update_scene(self.data, camera=cam_id)
                elif self.model.ncam > 0:
                    self.renderer.update_scene(self.data, camera=0)
                else:
                    self.renderer.update_scene(self.data)
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render()
                self.renderer.disable_depth_rendering()
                return depth
            except Exception:
                return np.zeros((self.height, self.width), dtype=np.float32)
        return np.zeros((self.height, self.width), dtype=np.float32)

    def render_rgbd(self, camera_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB + Depth from specified camera."""
        return self.render(camera_name), self.render_depth(camera_name)

    # ── Camera intrinsics ───────────────────────────────────
    def get_camera_intrinsics(self, camera_name: Optional[str] = None) -> dict:
        """Get camera intrinsic parameters."""
        cam_name = camera_name or self.camera_name
        cam_id = -1
        for i in range(self.model.ncam):
            if self.model.camera(i).name == cam_name:
                cam_id = i
                break

        if cam_id < 0:
            return {"fx": 500, "fy": 500, "cx": self.width / 2, "cy": self.height / 2}

        cam = self.model.camera(cam_id)
        # Fovy to focal length approximation
        fovy = cam.fovy[0] if hasattr(cam, 'fovy') and len(cam.fovy) > 0 else 60.0
        f = (self.height / 2) / np.tan(np.deg2rad(fovy) / 2)
        return {
            "fx": f,
            "fy": f,
            "cx": self.width / 2,
            "cy": self.height / 2,
            "fovy": fovy,
            "width": self.width,
            "height": self.height,
        }

    def get_camera_pose(self, camera_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get a MuJoCo camera pose in world coordinates.

        Returns:
            (position, rotation_matrix), where rotation_matrix maps MuJoCo's
            OpenGL camera frame to world coordinates.
        """
        cam_name = camera_name or self.camera_name
        cam_id = -1
        for i in range(self.model.ncam):
            if self.model.camera(i).name == cam_name:
                cam_id = i
                break
        if cam_id < 0:
            return np.zeros(3), np.eye(3)
        return (
            self.data.cam_xpos[cam_id].copy(),
            self.data.cam_xmat[cam_id].reshape(3, 3).copy(),
        )

    def camera_point_to_world(self, point_camera: np.ndarray,
                              camera_name: Optional[str] = None) -> np.ndarray:
        """Transform a point from the project RGB-D camera convention to world.

        ``depth_to_pointcloud`` and ``pixel_to_3d`` use an image-friendly camera
        convention: +x right, +y down, +z forward. MuJoCo camera poses use the
        OpenGL convention: +x right, +y up, -z forward. Convert before applying
        the camera pose.
        """
        pos, rot = self.get_camera_pose(camera_name)
        p = np.asarray(point_camera[:3], dtype=float)
        mujoco_cam = np.array([p[0], -p[1], -p[2]], dtype=float)
        return pos + rot @ mujoco_cam

    def world_point_to_camera(self, point_world: np.ndarray,
                              camera_name: Optional[str] = None) -> np.ndarray:
        """Transform a world point into the project RGB-D camera convention.

        The returned convention matches ``depth_to_pointcloud``:
        +x right, +y down, +z forward.
        """
        pos, rot = self.get_camera_pose(camera_name)
        world_delta = np.asarray(point_world[:3], dtype=float) - pos
        mujoco_cam = rot.T @ world_delta
        return np.array([mujoco_cam[0], -mujoco_cam[1], -mujoco_cam[2]], dtype=float)

    def project_world_point(self, point_world: np.ndarray,
                            camera_name: Optional[str] = None) -> dict:
        """Project a world point into camera pixel coordinates.

        Returns a dict with camera-frame coordinates, pixel coordinates, and a
        visibility flag. It intentionally does not perform occlusion testing.
        """
        p_cam = self.world_point_to_camera(point_world, camera_name)
        intr = self.get_camera_intrinsics(camera_name)
        z = float(p_cam[2])
        if z <= 1e-6:
            return {
                "camera": p_cam,
                "pixel": np.array([np.nan, np.nan], dtype=float),
                "visible": False,
            }
        u = float(intr["fx"] * p_cam[0] / z + intr["cx"])
        v = float(intr["fy"] * p_cam[1] / z + intr["cy"])
        visible = 0.0 <= u < float(intr.get("width", self.width)) and 0.0 <= v < float(intr.get("height", self.height))
        return {
            "camera": p_cam,
            "pixel": np.array([u, v], dtype=float),
            "visible": bool(visible),
        }

    def pointcloud_to_world(self, points_camera: np.ndarray,
                            camera_name: Optional[str] = None) -> np.ndarray:
        """Vectorized camera-point to world transform for point clouds."""
        pos, rot = self.get_camera_pose(camera_name)
        pts = np.asarray(points_camera, dtype=float).reshape(-1, 3)
        mujoco_pts = pts.copy()
        mujoco_pts[:, 1] *= -1.0
        mujoco_pts[:, 2] *= -1.0
        return (pos + mujoco_pts @ rot.T).reshape(np.asarray(points_camera).shape)

    def depth_to_pointcloud(self, depth: np.ndarray,
                            camera_name: Optional[str] = None) -> np.ndarray:
        """Convert depth image to organized point cloud."""
        intrinsics = self.get_camera_intrinsics(camera_name)
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        h, w = depth.shape[:2]

        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        z = depth
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        return np.stack([x, y, z], axis=-1)  # (H, W, 3)

    def pixel_to_3d(self, pixel: Tuple[int, int], depth: np.ndarray,
                    camera_name: Optional[str] = None) -> np.ndarray:
        """Convert a single pixel to 3D world coordinate."""
        intrinsics = self.get_camera_intrinsics(camera_name)
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        u, v = pixel
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.array([x, y, z])

    # ── Contact and collision ───────────────────────────────
    def get_contacts(self, body1: str, body2: str = "") -> list:
        """Get contact points involving named bodies."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name if contact.geom1 >= 0 else ""
            geom2_name = self.model.geom(contact.geom2).name if contact.geom2 >= 0 else ""

            if body1 and body1 not in geom1_name and body1 not in geom2_name:
                continue
            if body2 and body2 not in geom1_name and body2 not in geom2_name:
                continue

            contacts.append({
                "geom1": geom1_name,
                "geom2": geom2_name,
                "position": contact.pos.copy(),
                "frame": contact.frame.copy(),
                "distance": contact.dist,
            })
        return contacts

    def check_self_collision(self) -> bool:
        """Check if robot is in self-collision."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            # Check if both geoms are robot parts (not environment)
            # This is a heuristic - specific robot parts should be tagged
            if contact.dist < 0 and "floor" not in geom1_name and "floor" not in geom2_name:
                return True
        return False

    # ── Info ────────────────────────────────────────────────
    def close(self):
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
