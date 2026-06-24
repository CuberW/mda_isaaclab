"""ReKep 到 Isaac Sim 5.1 + Kuavo 轮式双臂机器人的适配层。

这个文件只做“移植封装”，不把 ReKep 官方优化器重新发明一遍：

1. 输入层接收外部全局相机/Grounded-SAM2 已经给出的世界系 3D 关键点。
2. 约束层硬编码防碰撞的 Hover -> Grasp 两阶段抓取逻辑。
3. 输出层把 ReKep/约束优化得到的 SE(3) 末端航点交给 Isaac Sim 5.1 的 IK。
4. 执行层把离散关节航点用 CubicSpline 上采样，再用 ArticulationController 下发 PD joint position。

TODO(Isaac 工作站接入时必须补齐)：
- Kuavo articulation prim path，例如 "/World/Kuavo"。
- 左/右臂 joint_names 的真实顺序，必须与 Isaac articulation 中 DOF 顺序一致。
- 左/右末端 link/frame 名称，例如 "zarm_r7_end_effector" 或真实夹爪 pinch frame。
- Lula 或 Pinocchio IK solver 的实例化方式。不同项目的 robot descriptor / URDF / XRDF 路径不同，
  本文件通过 ``KuavoIKSolverProtocol`` 抽象出来，避免把路径写死。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


ArrayLike = Sequence[float] | np.ndarray


class ArmSide(str, Enum):
    """Kuavo 左/右臂选择。"""

    LEFT = "left"
    RIGHT = "right"


class ReKepPhase(str, Enum):
    """ReKep 执行阶段。"""

    HOVER = "hover"
    GRASP = "grasp"
    LIFT = "lift"
    RELEASE = "release"


@dataclass(frozen=True)
class SE3Pose:
    """世界坐标系下的末端位姿。

    position: xyz，单位 m。
    quat_xyzw: 四元数，Isaac 常用 xyzw；若底层控制器需要 wxyz，请在发送前转换。
    """

    position: np.ndarray
    quat_xyzw: np.ndarray

    @staticmethod
    def from_position_z_down(position: ArrayLike, yaw_rad: float = 0.0) -> "SE3Pose":
        """构造末端 Z 轴竖直向下的抓取姿态。

        约定：工具坐标系局部 +Z 指向夹爪接近方向。这里让局部 +Z 对准世界 -Z，
        从而保证 Hover 和 Grasp 只能从物体上方竖直下探，避免侧向击飞物体。
        """

        pos = np.asarray(position, dtype=float).reshape(3)
        # 先绕 X 轴旋转 180 度，使局部 +Z 指向世界 -Z；再叠加绕世界 Z 的 yaw。
        rot = Rotation.from_euler("z", yaw_rad) * Rotation.from_euler("x", np.pi)
        return SE3Pose(position=pos, quat_xyzw=rot.as_quat())

    def as_matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(self.quat_xyzw).as_matrix()
        transform[:3, 3] = self.position
        return transform


@dataclass
class ReKepKeypointInput:
    """外部感知模块传入的干净世界系关键点。

    这些点应由“全局相机 + Grounded SAM2 + 深度/点云反投影 + 外参标定”得到。
    本适配层不再做多相机融合，也不关心像素坐标。
    """

    target_obj_keypoints: np.ndarray
    container_keypoints: np.ndarray | None = None
    obstacle_keypoints: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def target_center(self) -> np.ndarray:
        pts = np.asarray(self.target_obj_keypoints, dtype=float).reshape(-1, 3)
        if pts.size == 0:
            raise ValueError("target_obj_keypoints 不能为空")
        return np.mean(pts, axis=0)

    def container_center(self) -> np.ndarray | None:
        if self.container_keypoints is None:
            return None
        pts = np.asarray(self.container_keypoints, dtype=float).reshape(-1, 3)
        return np.mean(pts, axis=0) if pts.size else None


@dataclass(frozen=True)
class TwoStageGraspConstraints:
    """防碰撞两阶段抓取约束。

    Hover State:
    - 末端先到目标中心正上方 hover_height_m。
    - 末端局部 Z 轴严格朝下。

    Grasp State:
    - 末端只能沿局部 Z 轴，也就是世界 -Z 方向直线下降。
    - xy 不允许明显偏移，避免从侧面扫到物体。
    """

    hover_height_m: float = 0.15
    grasp_depth_m: float = 0.0
    xy_tolerance_m: float = 0.01
    z_axis_alignment_weight: float = 100.0
    straight_down_weight: float = 100.0

    def build_hover_pose(self, target_center_world: np.ndarray, yaw_rad: float = 0.0) -> SE3Pose:
        hover = np.asarray(target_center_world, dtype=float).reshape(3).copy()
        hover[2] += self.hover_height_m
        return SE3Pose.from_position_z_down(hover, yaw_rad=yaw_rad)

    def build_grasp_pose(self, target_center_world: np.ndarray, yaw_rad: float = 0.0) -> SE3Pose:
        grasp = np.asarray(target_center_world, dtype=float).reshape(3).copy()
        grasp[2] += self.grasp_depth_m
        return SE3Pose.from_position_z_down(grasp, yaw_rad=yaw_rad)

    def hover_cost(self, ee_pose: SE3Pose, target_center_world: np.ndarray) -> float:
        desired = self.build_hover_pose(target_center_world)
        pos_err = np.linalg.norm(ee_pose.position - desired.position)
        z_align_err = self._z_down_alignment_error(ee_pose)
        return float(pos_err**2 + self.z_axis_alignment_weight * z_align_err**2)

    def grasp_cost(self, ee_pose: SE3Pose, hover_pose: SE3Pose, target_center_world: np.ndarray) -> float:
        desired = self.build_grasp_pose(target_center_world)
        xy_err = np.linalg.norm(ee_pose.position[:2] - hover_pose.position[:2])
        z_err = abs(float(ee_pose.position[2] - desired.position[2]))
        z_align_err = self._z_down_alignment_error(ee_pose)
        # xy 超过容差后重罚，鼓励从 hover 点竖直下探。
        xy_violation = max(0.0, xy_err - self.xy_tolerance_m)
        return float(
            z_err**2
            + self.straight_down_weight * xy_violation**2
            + self.z_axis_alignment_weight * z_align_err**2
        )

    @staticmethod
    def _z_down_alignment_error(pose: SE3Pose) -> float:
        rot = Rotation.from_quat(pose.quat_xyzw).as_matrix()
        local_z_in_world = rot[:, 2]
        desired = np.array([0.0, 0.0, -1.0])
        return float(1.0 - np.clip(np.dot(local_z_in_world, desired), -1.0, 1.0))


@dataclass
class ReKepPlan:
    """ReKep/约束层输出的末端航点计划。"""

    side: ArmSide
    ee_waypoints: list[SE3Pose]
    phases: list[ReKepPhase]
    source: str = "rekep_two_stage_grasp_constraints"

    def validate(self) -> None:
        if not self.ee_waypoints:
            raise ValueError("ReKepPlan.ee_waypoints 不能为空")
        if len(self.ee_waypoints) != len(self.phases):
            raise ValueError("ee_waypoints 和 phases 长度必须一致")


class ReKepOptimizerProtocol(Protocol):
    """官方 ReKep 优化器的最小协议。

    如果你已经接入 huangwl18/ReKep 官方 optimizer，可以写一个薄 wrapper，
    把它的输出转成 list[SE3Pose] 即可。
    """

    def optimize(
        self,
        initial_waypoints: Sequence[SE3Pose],
        keypoints: ReKepKeypointInput,
        constraints: TwoStageGraspConstraints,
    ) -> Sequence[SE3Pose]:
        ...


class KuavoIKSolverProtocol(Protocol):
    """Kuavo IK 求解器协议，兼容 Lula / Pinocchio / 自研 IK。

    TODO(服务器接入):
    - Lula: 在 Isaac Sim 中用 ArticulationKinematicsSolver / LulaKinematicsSolver 封装此接口。
    - Pinocchio: 用 URDF + frame_id 求解并返回 joint_positions。
    """

    def solve(
        self,
        side: ArmSide,
        target_pose: SE3Pose,
        seed_joint_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


@dataclass
class KuavoIsaacConfig:
    """Isaac Sim 中 Kuavo articulation 和关节配置。"""

    articulation_prim_path: str
    right_arm_joint_names: list[str]
    left_arm_joint_names: list[str]
    right_ee_frame_name: str
    left_ee_frame_name: str
    control_hz: float = 60.0
    position_kp: float = 80.0
    position_kd: float = 8.0

    def arm_joint_names(self, side: ArmSide) -> list[str]:
        return self.left_arm_joint_names if side == ArmSide.LEFT else self.right_arm_joint_names

    @staticmethod
    def default_s62(articulation_prim_path: str = "/World/Kuavo") -> "KuavoIsaacConfig":
        """Kuavo 5-W v62 / biped_s62 的默认关节命名。

        这组名字与当前仓库里 biped_s62 MJCF/URDF 的 zarm 命名一致。Isaac 导入
        USD 后如果 DOF 名发生变化，脚本会再从 articulation.dof_names 自动校正。
        """

        return KuavoIsaacConfig(
            articulation_prim_path=articulation_prim_path,
            right_arm_joint_names=[f"zarm_r{i}_joint" for i in range(1, 8)],
            left_arm_joint_names=[f"zarm_l{i}_joint" for i in range(1, 8)],
            right_ee_frame_name="zarm_r7_end_effector",
            left_ee_frame_name="zarm_l7_end_effector",
        )

    def with_detected_joint_names(self, dof_names: Sequence[str]) -> "KuavoIsaacConfig":
        """从 Isaac articulation 的 DOF 名里自动筛出左右 7 轴。"""

        names = list(dof_names)

        def _ordered(prefix: str) -> list[str]:
            found = [name for name in names if name.startswith(prefix) and name.endswith("_joint")]
            return sorted(found, key=lambda n: int(n.split(prefix, 1)[1].split("_joint", 1)[0]))

        left = _ordered("zarm_l")
        right = _ordered("zarm_r")
        return KuavoIsaacConfig(
            articulation_prim_path=self.articulation_prim_path,
            right_arm_joint_names=right if len(right) == 7 else self.right_arm_joint_names,
            left_arm_joint_names=left if len(left) == 7 else self.left_arm_joint_names,
            right_ee_frame_name=self.right_ee_frame_name,
            left_ee_frame_name=self.left_ee_frame_name,
            control_hz=self.control_hz,
            position_kp=self.position_kp,
            position_kd=self.position_kd,
        )


class CubicSplineTrajectorySmoother:
    """把 ReKep 的稀疏 joint waypoints 上采样为平滑密集轨迹。"""

    def __init__(self, min_points: int = 120):
        self.min_points = int(min_points)

    def smooth_joint_waypoints(
        self,
        joint_waypoints: np.ndarray,
        total_time_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(joint_waypoints, dtype=float)
        if q.ndim != 2:
            raise ValueError("joint_waypoints 必须是 [num_waypoints, num_joints]")
        if q.shape[0] < 2:
            raise ValueError("至少需要两个关节航点才能插值")

        n_src = q.shape[0]
        n_dst = max(self.min_points, n_src * 20)
        src_t = np.linspace(0.0, 1.0, n_src)
        dst_t = np.linspace(0.0, 1.0, n_dst)
        spline = CubicSpline(src_t, q, axis=0, bc_type="clamped")
        dense_q = spline(dst_t)

        if total_time_s is None:
            total_time_s = max(1.0, n_dst / 60.0)
        dense_time = np.linspace(0.0, float(total_time_s), n_dst)
        return dense_time, dense_q


class IsaacArticulationAdapter:
    """Isaac Sim 5.1 ArticulationController 的薄封装。

    为了让本文件能在非 Isaac 环境下 import，所有 Isaac 依赖都延迟导入。
    """

    def __init__(self, world: Any, config: KuavoIsaacConfig):
        self.world = world
        self.config = config
        self.robot = None
        self.controller = None

    def initialize(self) -> None:
        """从 stage 获取 Kuavo articulation 并初始化 controller。

        TODO(Isaac 工作站接入):
        如果你使用的是 Isaac Sim 5.1 的新版 Articulation API，可能需要根据项目模板把
        Articulation import 路径改成实际可用路径。
        """

        try:
            from isaacsim.core.prims import Articulation
            from isaacsim.core.utils.types import ArticulationAction
        except Exception:  # pragma: no cover - 本机没有 Isaac 时不运行
            try:
                from omni.isaac.core.articulations import Articulation
                from omni.isaac.core.utils.types import ArticulationAction
            except Exception as exc:
                raise ImportError(
                    "无法导入 Isaac Sim Articulation API。请在 Isaac Sim 5.1 Python 环境中运行。"
                ) from exc

        self._articulation_action_cls = ArticulationAction
        self.robot = Articulation(prim_paths_expr=self.config.articulation_prim_path)
        self.robot.initialize()
        self.controller = self.robot.get_articulation_controller()
        if hasattr(self.robot, "dof_names"):
            self.config = self.config.with_detected_joint_names(list(self.robot.dof_names))

    def get_joint_positions(self, joint_names: Sequence[str]) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("IsaacArticulationAdapter.initialize() 尚未调用")
        all_names = list(self.robot.dof_names)
        all_q = np.asarray(self.robot.get_joint_positions(), dtype=float)
        indices = [all_names.index(name) for name in joint_names]
        return all_q[indices]

    def apply_joint_positions(self, joint_names: Sequence[str], joint_positions: np.ndarray) -> None:
        if self.robot is None or self.controller is None:
            raise RuntimeError("IsaacArticulationAdapter.initialize() 尚未调用")
        all_names = list(self.robot.dof_names)
        indices = np.asarray([all_names.index(name) for name in joint_names], dtype=np.int32)
        q = np.asarray(joint_positions, dtype=float)
        action = self._articulation_action_cls(joint_positions=q, joint_indices=indices)
        self.controller.apply_action(action)


class ReKepIntegration:
    """ReKep 官方核心逻辑到 Isaac Sim + Kuavo 的集成入口。"""

    def __init__(
        self,
        isaac_adapter: IsaacArticulationAdapter,
        ik_solver: KuavoIKSolverProtocol,
        kuavo_config: KuavoIsaacConfig,
        rekep_optimizer: ReKepOptimizerProtocol | None = None,
        constraints: TwoStageGraspConstraints | None = None,
        smoother: CubicSplineTrajectorySmoother | None = None,
    ):
        self.isaac = isaac_adapter
        self.ik_solver = ik_solver
        self.config = kuavo_config
        self.rekep_optimizer = rekep_optimizer
        self.constraints = constraints or TwoStageGraspConstraints()
        self.smoother = smoother or CubicSplineTrajectorySmoother(min_points=120)

    def build_two_stage_plan(
        self,
        keypoints: ReKepKeypointInput,
        side: ArmSide = ArmSide.RIGHT,
        yaw_rad: float = 0.0,
    ) -> ReKepPlan:
        """由世界系关键点生成 Hover -> Grasp -> Lift 的 ReKep 航点计划。"""

        target_center = keypoints.target_center()
        hover = self.constraints.build_hover_pose(target_center, yaw_rad=yaw_rad)
        grasp = self.constraints.build_grasp_pose(target_center, yaw_rad=yaw_rad)
        lift_pos = grasp.position.copy()
        lift_pos[2] += self.constraints.hover_height_m
        lift = SE3Pose.from_position_z_down(lift_pos, yaw_rad=yaw_rad)

        initial_waypoints = [hover, grasp, lift]
        if self.rekep_optimizer is not None:
            optimized = list(self.rekep_optimizer.optimize(initial_waypoints, keypoints, self.constraints))
            # 保守处理：官方优化器输出仍必须保留“两阶段竖直下探”的语义。
            if len(optimized) >= 2:
                optimized[0] = hover
                optimized[1] = grasp
            waypoints = optimized
        else:
            waypoints = initial_waypoints

        phases = [ReKepPhase.HOVER, ReKepPhase.GRASP, ReKepPhase.LIFT][: len(waypoints)]
        plan = ReKepPlan(side=side, ee_waypoints=waypoints, phases=phases)
        plan.validate()
        return plan

    def solve_joint_waypoints(self, plan: ReKepPlan) -> np.ndarray:
        """把 SE(3) 末端航点转换成 Kuavo 关节航点。"""

        plan.validate()
        joint_names = self.config.arm_joint_names(plan.side)
        seed = self.isaac.get_joint_positions(joint_names)
        q_waypoints: list[np.ndarray] = []

        for pose, phase in zip(plan.ee_waypoints, plan.phases):
            q = self.ik_solver.solve(plan.side, pose, seed_joint_positions=seed)
            q = np.asarray(q, dtype=float).reshape(-1)
            if q.size != len(joint_names):
                raise ValueError(
                    f"IK 输出维度 {q.size} 与 {plan.side.value} arm joint 数 {len(joint_names)} 不一致"
                )
            q_waypoints.append(q)
            seed = q

            # 这里保留阶段信息，方便后续接入夹爪控制：
            # - HOVER: gripper open
            # - GRASP: 下降后 close gripper
            # - LIFT: verify contact 后抬升
            _ = phase

        return np.vstack(q_waypoints)

    def execute_plan(
        self,
        plan: ReKepPlan,
        total_time_s: float | None = None,
        on_phase_boundary: Callable[[ReKepPhase], None] | None = None,
    ) -> None:
        """求 IK、平滑轨迹，并在 Isaac world.step() 循环中执行。"""

        joint_names = self.config.arm_joint_names(plan.side)
        q_sparse = self.solve_joint_waypoints(plan)
        _, q_dense = self.smoother.smooth_joint_waypoints(q_sparse, total_time_s=total_time_s)

        # 简单阶段边界：把稀疏航点映射到密集轨迹索引，用于开/闭夹爪等事件。
        boundary_indices = self._phase_boundary_indices(len(plan.phases), len(q_dense))
        boundary_map = dict(zip(boundary_indices, plan.phases))

        for i, q in enumerate(q_dense):
            if on_phase_boundary is not None and i in boundary_map:
                on_phase_boundary(boundary_map[i])
            self.isaac.apply_joint_positions(joint_names, q)
            self.world_step()

    def world_step(self) -> None:
        """固定频率推进 Isaac Sim。

        Isaac 的 world.step(render=True/False) 签名在模板里可能略有差异，所以这里做兼容调用。
        """

        try:
            self.isaac.world.step(render=True)
        except TypeError:
            self.isaac.world.step()

    @staticmethod
    def _phase_boundary_indices(num_phases: int, num_dense: int) -> list[int]:
        if num_phases <= 1:
            return [0]
        return [int(round(i * (num_dense - 1) / (num_phases - 1))) for i in range(num_phases)]


class DummyReKepOptimizer:
    """无官方 ReKep 优化器时的占位优化器。

    仅用于接口联调：直接返回输入航点。正式作业要替换成 huangwl18/ReKep 官方优化器 wrapper。
    """

    def optimize(
        self,
        initial_waypoints: Sequence[SE3Pose],
        keypoints: ReKepKeypointInput,
        constraints: TwoStageGraspConstraints,
    ) -> Sequence[SE3Pose]:
        return list(initial_waypoints)


class DummyIKSolver:
    """IK 接口占位类。

    TODO(必须替换)：服务器 Isaac 环境中请用 Lula 或 Pinocchio 实现 KuavoIKSolverProtocol。
    """

    def __init__(self, num_joints: int = 7):
        self.num_joints = int(num_joints)

    def solve(
        self,
        side: ArmSide,
        target_pose: SE3Pose,
        seed_joint_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        if seed_joint_positions is not None:
            return np.asarray(seed_joint_positions, dtype=float).copy()
        return np.zeros(self.num_joints, dtype=float)


class LulaKuavoIKSolver:
    """Isaac Sim Lula IK 的 Kuavo 包装器。

    由于不同 Isaac Sim 5.1 安装和 USD/URDF 导入流程会导致 Lula 类路径略有差别，
    本类把所有 Isaac 依赖延迟到运行时导入，并在失败时输出可操作的错误。
    """

    def __init__(
        self,
        config: KuavoIsaacConfig,
        robot_description_path: str,
        urdf_path: str,
    ):
        self.config = config
        self.robot_description_path = robot_description_path
        self.urdf_path = urdf_path
        self._solver_by_side: dict[ArmSide, Any] = {}

    def initialize(self, articulation: Any) -> None:
        try:
            from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
        except Exception as exc:  # pragma: no cover - 仅 Isaac 环境运行
            raise ImportError(
                "无法导入 Isaac Lula IK。请确认在 Isaac Sim 5.1 Python 中运行，并启用 "
                "omni.isaac.motion_generation 扩展。"
            ) from exc

        for side in (ArmSide.LEFT, ArmSide.RIGHT):
            ee_frame = (
                self.config.left_ee_frame_name
                if side == ArmSide.LEFT
                else self.config.right_ee_frame_name
            )
            lula = LulaKinematicsSolver(
                robot_description_path=self.robot_description_path,
                urdf_path=self.urdf_path,
            )
            self._solver_by_side[side] = ArticulationKinematicsSolver(
                articulation,
                lula,
                ee_frame,
            )

    def solve(
        self,
        side: ArmSide,
        target_pose: SE3Pose,
        seed_joint_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        solver = self._solver_by_side.get(side)
        if solver is None:
            raise RuntimeError("LulaKuavoIKSolver.initialize(articulation) 尚未调用")
        # Isaac 常用 scalar-first quaternion；SE3Pose 保存 xyzw，所以这里转换。
        xyzw = target_pose.quat_xyzw
        quat_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
        result, success = solver.compute_inverse_kinematics(
            target_position=target_pose.position,
            target_orientation=quat_wxyz,
        )
        if not success:
            raise RuntimeError(f"Lula IK 求解失败: side={side.value}, target={target_pose.position.tolist()}")
        if hasattr(result, "joint_positions"):
            return np.asarray(result.joint_positions, dtype=float)
        return np.asarray(result, dtype=float)


__all__ = [
    "ArmSide",
    "CubicSplineTrajectorySmoother",
    "DummyIKSolver",
    "DummyReKepOptimizer",
    "IsaacArticulationAdapter",
    "KuavoIKSolverProtocol",
    "KuavoIsaacConfig",
    "LulaKuavoIKSolver",
    "ReKepIntegration",
    "ReKepKeypointInput",
    "ReKepOptimizerProtocol",
    "ReKepPhase",
    "ReKepPlan",
    "SE3Pose",
    "TwoStageGraspConstraints",
]
