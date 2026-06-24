"""
Grasp Manager — dynamic weld-constraint grasping for MuJoCo.

Implements the OpenAI MAE GrabObjWrapper pattern:
1. Weld equator_weak constraints are pre-defined in scene XML (active="false")
2. On grasp, the constraint is activated with the correct relative pose
3. On release, the constraint is deactivated

This gives 100% reliable grasping compared to pure friction-based grasping,
while still allowing physics-based contact detection before activation.

Reference: openai/multi-agent-emergence-environments (GrabObjWrapper)
"""

from typing import Optional, List, Tuple
import numpy as np
import mujoco

from robot_common.infra.logging import logger


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w,x,y,z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_invert(q: np.ndarray) -> np.ndarray:
    """Invert a quaternion (w,x,y,z format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.sum(q**2)


def mat2quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    # Use mujoco's built-in
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


class GraspManager:
    """Manage dynamic weld constraints for grasping objects.

    Usage pattern:
        # In scene XML, pre-define weld constraints:
        #   <equality>
        #     <weld name="grasp_N" body1="gripper_body" body2="obj_body"
        #           active="false" torquescale="5"/>
        #   </equality>

        # In Python:
        gm = GraspManager(model, data)
        gm.register_weld("grasp_0", gripper_body="l_finger_l",
                         default_obj_body="obj_cup")
        gm.register_weld("grasp_1", gripper_body="r_finger_l",
                         default_obj_body="obj_can")

        # On grasp:
        if gm.detect_contact("grasp_0"):
            gm.attach("grasp_0", "box_obj")

        # Wait...

        # On release:
        gm.release("grasp_0")
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._welds: dict = {}  # name -> {eq_id, gripper_body, gripper_geoms}

    def register_weld(
        self,
        weld_name: str,
        gripper_body: str,
        default_obj_body: str = "",
        gripper_geoms: Optional[List[str]] = None,
        extra_gripper_bodies: Optional[List[str]] = None,
    ):
        """Register a pre-defined weld constraint for management.

        Args:
            weld_name: Name of the weld equality constraint in the XML
            gripper_body: Name of the gripper body (body1 in the weld)
            default_obj_body: Default object body (body2 in the weld — can be
                             changed dynamically via eq_obj2id)
            gripper_geoms: List of geom names belonging to the gripper fingers,
                          used for contact detection. If None, searches for
                          geoms under gripper_body.
            extra_gripper_bodies: Additional gripper bodies whose geoms should
                          count for contact detection. The weld body can stay
                          on one finger, while contact is verified across both
                          fingers.
        """
        try:
            eq_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name
            )
        except Exception:
            logger.error(f"Weld '{weld_name}' not found in model equalities")
            return False

        if eq_id < 0:
            logger.error(f"Weld '{weld_name}' not found in model equalities")
            return False

        # Find gripper geom IDs for contact detection
        if gripper_geoms is None:
            gripper_geom_ids = self._find_body_geom_ids(gripper_body)
        else:
            gripper_geom_ids = []
            for gname in gripper_geoms:
                try:
                    gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                    gripper_geom_ids.append(gid)
                except Exception:
                    pass
        for body_name in extra_gripper_bodies or []:
            gripper_geom_ids.extend(self._find_body_geom_ids(body_name))
        gripper_geom_ids = sorted(set(gripper_geom_ids))

        self._welds[weld_name] = {
            "eq_id": eq_id,
            "gripper_body": gripper_body,
            "gripper_body_names": [gripper_body] + list(extra_gripper_bodies or []),
            "gripper_geom_ids": gripper_geom_ids,
            "default_obj_body": default_obj_body,
            "active": False,
        }
        logger.info(
            f"Registered weld '{weld_name}': gripper={gripper_body}, "
            f"geom_ids={gripper_geom_ids}"
        )
        return True

    def _find_body_geom_ids(self, body_name: str) -> List[int]:
        """Find all geom IDs belonging to a body."""
        geom_ids = []
        try:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            start = self.model.body_geomadr[body_id]
            count = self.model.body_geomnum[body_id]
            geom_ids = list(range(start, start + count))
        except Exception:
            pass
        return geom_ids

    def detect_contact(
        self,
        weld_name: str,
        obj_body: Optional[str] = None,
        min_contact_force: float = 0.0,
        min_gripper_bodies: int = 1,
    ) -> bool:
        """Check if the gripper is in contact with an object.

        Args:
            weld_name: Registered weld name
            obj_body: Object body to check contact with. If None, checks
                     any contact involving gripper geoms.
            min_contact_force: Minimum contact force to count as contact
            min_gripper_bodies: Number of distinct registered gripper bodies
                     that must touch the object. Use 2 for real two-finger
                     pinch checks.

        Returns:
            True if gripper geoms are in contact with the object
        """
        if weld_name not in self._welds:
            return False

        info = self._welds[weld_name]
        gripper_geom_ids = set(info["gripper_geom_ids"])
        if not gripper_geom_ids:
            return False

        obj_geom_ids = set()
        if obj_body:
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, obj_body
                )
                start = self.model.body_geomadr[body_id]
                count = self.model.body_geomnum[body_id]
                obj_geom_ids = set(range(start, start + count))
            except Exception:
                pass

        contacting_bodies: set[str] = set()
        geom_to_body = {
            gid: self.model.body(int(self.model.geom_bodyid[gid])).name
            for gid in gripper_geom_ids
        }

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)

            # Check force threshold
            if min_contact_force > 0:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                if np.linalg.norm(force[:3]) < min_contact_force:
                    continue

            # Check if this contact is between gripper and object
            g_in_gripper = g1 in gripper_geom_ids or g2 in gripper_geom_ids
            if not g_in_gripper:
                continue

            if obj_geom_ids:
                g_in_obj = g1 in obj_geom_ids or g2 in obj_geom_ids
                if not g_in_obj:
                    continue

            gripper_geom = g1 if g1 in gripper_geom_ids else g2
            body_name = geom_to_body.get(gripper_geom, "")
            if body_name:
                contacting_bodies.add(body_name)
            if len(contacting_bodies) >= max(1, int(min_gripper_bodies)):
                return True

        return False

    def contacting_gripper_bodies(
        self,
        weld_name: str,
        obj_body: Optional[str] = None,
    ) -> List[str]:
        """Return registered gripper bodies currently touching an object."""
        if weld_name not in self._welds:
            return []

        info = self._welds[weld_name]
        gripper_geom_ids = set(info["gripper_geom_ids"])
        if not gripper_geom_ids:
            return []

        obj_geom_ids = set()
        if obj_body:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obj_body
            )
            if body_id >= 0:
                start = self.model.body_geomadr[body_id]
                count = self.model.body_geomnum[body_id]
                obj_geom_ids = set(range(start, start + count))

        geom_to_body = {
            gid: self.model.body(int(self.model.geom_bodyid[gid])).name
            for gid in gripper_geom_ids
        }
        bodies = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 not in gripper_geom_ids and g2 not in gripper_geom_ids:
                continue
            if obj_geom_ids and g1 not in obj_geom_ids and g2 not in obj_geom_ids:
                continue
            gripper_geom = g1 if g1 in gripper_geom_ids else g2
            body_name = geom_to_body.get(gripper_geom, "")
            if body_name:
                bodies.add(body_name)
        return sorted(bodies)

    def contacted_body(self, weld_name: str) -> Optional[str]:
        """Return the non-gripper body currently touching a registered gripper."""
        if weld_name not in self._welds:
            return None

        info = self._welds[weld_name]
        gripper_geom_ids = set(info["gripper_geom_ids"])
        if not gripper_geom_ids:
            return None

        gripper_body_ids = set()
        for body_name in info.get("gripper_body_names", [info["gripper_body"]]):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                gripper_body_ids.add(int(body_id))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if g1 not in gripper_geom_ids and g2 not in gripper_geom_ids:
                continue
            other_geom = g2 if g1 in gripper_geom_ids else g1
            body_id = int(self.model.geom_bodyid[other_geom])
            if body_id in gripper_body_ids or body_id == 0:
                continue
            body_name = self.model.body(body_id).name
            if body_name:
                return body_name
        return None

    def attach(
        self,
        weld_name: str,
        obj_body: str,
        anchor_world: Optional[np.ndarray] = None,
    ) -> bool:
        """Attach (weld) the gripper to an object.

        Computes the relative pose between gripper and object,
        sets the weld constraint data, and activates it.

        Args:
            weld_name: Registered weld name
            obj_body: Name of the object body to weld to
            anchor_world: Optional world-space point where the gripper and
                object should be welded. Use this for pinch-center attachment
                so objects do not hang from one finger body.

        Returns:
            True if successful
        """
        if weld_name not in self._welds:
            logger.error(f"Weld '{weld_name}' not registered")
            return False

        info = self._welds[weld_name]
        eq_id = info["eq_id"]
        gripper_body = info["gripper_body"]

        try:
            obj_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obj_body
            )
        except Exception:
            logger.error(f"Object body '{obj_body}' not found")
            return False

        # Ensure forward kinematics is up to date
        mujoco.mj_forward(self.model, self.data)

        # Get gripper and object poses
        g_pos = self.data.body(gripper_body).xpos.copy()
        g_rot = self.data.body(gripper_body).xmat.copy().reshape(3, 3)
        o_pos = self.data.body(obj_body).xpos.copy()
        o_rot = self.data.body(obj_body).xmat.copy().reshape(3, 3)

        # MuJoCo weld data layout is:
        #   [anchor2_in_body2(3), anchor1_in_body1(3), relquat_body2_in_body1(4), torquescale]
        # The constraint code evaluates:
        #   pos1 = body1 + R1 @ anchor1
        #   pos2 = body2 + R2 @ anchor2
        # By default, preserve the historical behavior: keep the object origin
        # fixed. When a pinch-center anchor is supplied, weld the object at that
        # physical capture point instead of hanging it from a single finger.
        if anchor_world is None:
            anchor_world = o_pos
        else:
            anchor_world = np.asarray(anchor_world[:3], dtype=float)
        anchor_obj = o_rot.T @ (anchor_world - o_pos)
        anchor_gripper = g_rot.T @ (anchor_world - g_pos)
        rel_rot = g_rot.T @ o_rot
        rel_quat = mat2quat(rel_rot)

        self.model.eq_data[eq_id, 0:3] = anchor_obj
        self.model.eq_data[eq_id, 3:6] = anchor_gripper
        self.model.eq_data[eq_id, 6:10] = rel_quat

        # Set object body reference
        self.model.eq_obj2id[eq_id] = obj_body_id

        # Activate the constraint for the current MjData. ``eq_active0`` is the
        # model reset default, so leave it unchanged to avoid sticky grabs
        # across episodes.
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[eq_id] = 1
        else:
            self.model.eq_active0[eq_id] = 1
        info["active"] = True
        mujoco.mj_forward(self.model, self.data)

        logger.debug(
            f"Weld '{weld_name}' attached to '{obj_body}': "
            f"anchor_gripper={anchor_gripper.round(3)}, rel_quat={rel_quat.round(3)}"
        )
        return True

    def release(self, weld_name: str) -> bool:
        """Release (unweld) a previously attached object.

        Args:
            weld_name: Registered weld name

        Returns:
            True if successful
        """
        if weld_name not in self._welds:
            logger.error(f"Weld '{weld_name}' not registered")
            return False

        info = self._welds[weld_name]
        eq_id = info["eq_id"]

        # Deactivate the constraint
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[eq_id] = 0
        self.model.eq_active0[eq_id] = 0
        info["active"] = False
        mujoco.mj_forward(self.model, self.data)

        logger.debug(f"Weld '{weld_name}' released")
        return True

    def is_attached(self, weld_name: str) -> bool:
        """Check if a weld constraint is currently active."""
        if weld_name not in self._welds:
            return False
        eq_id = self._welds[weld_name]["eq_id"]
        if hasattr(self.data, "eq_active"):
            return bool(self.data.eq_active[eq_id])
        return self._welds[weld_name]["active"]

    def release_all(self):
        """Release all active weld constraints."""
        for name in self._welds:
            if self._welds[name]["active"]:
                self.release(name)
