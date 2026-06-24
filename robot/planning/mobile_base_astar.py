"""Scene-derived 2D A* planner for mobile-base waypoint generation.

The planner builds a simple occupancy grid from MuJoCo geometry positions and
sizes. It is deliberately small: it produces executable base waypoints from
scene geometry instead of hard-coded lanes for a particular tabletop layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import mujoco
import numpy as np


@dataclass(frozen=True)
class BasePathPlan:
    waypoints: list[np.ndarray]
    source: str
    success: bool
    reason: str = ""


class MobileBaseAStarPlanner:
    """Plan base XY paths on a scene-derived 2D occupancy grid."""

    def __init__(
        self,
        env,
        resolution: float = 0.08,
        bounds: tuple[float, float, float, float] = (-0.75, 1.75, -1.20, 1.05),
    ):
        self.env = env
        self.resolution = float(resolution)
        self.bounds = tuple(float(v) for v in bounds)

    def plan(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        obstacle_inflation: float = 0.28,
        goal_clearance: float = 0.18,
        max_waypoints: int = 16,
    ) -> BasePathPlan:
        start = np.asarray(start_pose[:3], dtype=float)
        goal = np.asarray(goal_pose[:3], dtype=float)
        grid = self._build_grid(obstacle_inflation=obstacle_inflation)
        start_cell = self._world_to_cell(start[:2])
        goal_cell = self._world_to_cell(goal[:2])
        if not self._in_grid(start_cell) or not self._in_grid(goal_cell):
            return BasePathPlan([], "astar_scene_geometry", False, "start_or_goal_out_of_bounds")
        self._clear_disc(grid, start_cell, radius_m=goal_clearance)
        if grid[goal_cell]:
            return BasePathPlan([], "astar_scene_geometry", False, "goal_occupied")
        cells = self._astar(grid, start_cell, goal_cell)
        if not cells:
            if self._line_is_clear(grid, start_cell, goal_cell):
                return BasePathPlan(
                    [np.array([goal[0], goal[1], goal[2]], dtype=float)],
                    "straight_scene_geometry",
                    True,
                )
            return BasePathPlan([], "astar_scene_geometry", False, "no_grid_path")
        points = [self._cell_to_world(cell) for cell in cells]
        points = self._simplify_polyline(points)
        points = self._downsample(points, max_waypoints=max_waypoints)
        waypoints = []
        for idx, xy in enumerate(points[1:], start=1):
            yaw = float(goal[2]) if idx == len(points) - 1 else float(start[2])
            waypoints.append(np.array([xy[0], xy[1], yaw], dtype=float))
        if waypoints:
            waypoints[-1][2] = float(goal[2])
        return BasePathPlan(waypoints, "astar_scene_geometry", True)

    def _build_grid(self, obstacle_inflation: float) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.bounds
        nx = int(math.ceil((xmax - xmin) / self.resolution)) + 1
        ny = int(math.ceil((ymax - ymin) / self.resolution)) + 1
        grid = np.zeros((nx, ny), dtype=bool)
        for center, radius in self._obstacle_discs():
            inflated = float(radius + obstacle_inflation)
            min_cell = self._world_to_cell(center - inflated)
            max_cell = self._world_to_cell(center + inflated)
            for ix in range(max(0, min_cell[0]), min(nx, max_cell[0] + 1)):
                for iy in range(max(0, min_cell[1]), min(ny, max_cell[1] + 1)):
                    xy = self._cell_to_world((ix, iy))
                    if float(np.linalg.norm(xy - center)) <= inflated:
                        grid[ix, iy] = True
        return grid

    def _obstacle_discs(self) -> Iterable[tuple[np.ndarray, float]]:
        model = self.env.model
        data = self.env.data
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body_id = int(model.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if self._ignore_geom(name, body_name):
                continue
            pos = np.asarray(data.geom_xpos[gid, :2], dtype=float)
            size = np.asarray(model.geom_size[gid], dtype=float)
            geom_type = int(model.geom_type[gid])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                radius = float(np.linalg.norm(size[:2]))
            elif geom_type in (
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            ):
                radius = float(size[0])
            else:
                radius = float(max(size[:2]) if size.size >= 2 else max(size))
            if radius <= 0.015:
                continue
            yield pos, radius

    @staticmethod
    def _ignore_geom(name: str, body_name: str = "") -> bool:
        lowered = f"{name} {body_name}".lower()
        ignored_tokens = (
            "floor", "ground", "trash_", "r_f_fingers", "r_b_fingers",
            "r_pinch", "l_fingers", "base_", "wheel", "zarm", "zhead",
            "camera", "claw", "torso", "waist", "knee", "leg_", "legjoint",
            "arm_", "forearm", "hand_", "fingers", "finger", "head_",
        )
        return any(token in lowered for token in ignored_tokens)

    def _world_to_cell(self, xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        xmin, _, ymin, _ = self.bounds
        arr = np.asarray(xy, dtype=float)
        return (
            int(round((float(arr[0]) - xmin) / self.resolution)),
            int(round((float(arr[1]) - ymin) / self.resolution)),
        )

    def _cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        xmin, _, ymin, _ = self.bounds
        return np.array([
            xmin + float(cell[0]) * self.resolution,
            ymin + float(cell[1]) * self.resolution,
        ], dtype=float)

    @staticmethod
    def _in_grid_static(grid: np.ndarray, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]

    def _in_grid(self, cell: tuple[int, int]) -> bool:
        xmin, xmax, ymin, ymax = self.bounds
        xy = self._cell_to_world(cell)
        return xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax

    def _clear_disc(self, grid: np.ndarray, center: tuple[int, int], radius_m: float) -> None:
        radius = max(1, int(math.ceil(float(radius_m) / self.resolution)))
        for ix in range(center[0] - radius, center[0] + radius + 1):
            for iy in range(center[1] - radius, center[1] + radius + 1):
                cell = (ix, iy)
                if self._in_grid_static(grid, cell):
                    if math.hypot(ix - center[0], iy - center[1]) <= radius:
                        grid[ix, iy] = False

    def _astar(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        frontier: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(frontier, (0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._in_grid_static(grid, nxt) or grid[nxt]:
                    continue
                step_cost = math.hypot(dx, dy)
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        if goal not in came_from:
            return []
        path = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _line_is_clear(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            t = i / float(steps)
            cell = (
                int(round(start[0] + t * dx)),
                int(round(start[1] + t * dy)),
            )
            if not self._in_grid_static(grid, cell) or grid[cell]:
                return False
        return True

    @staticmethod
    def _simplify_polyline(points: list[np.ndarray]) -> list[np.ndarray]:
        if len(points) <= 2:
            return points
        out = [points[0]]
        prev_dir = None
        for idx in range(1, len(points)):
            direction = np.sign(points[idx] - points[idx - 1]).astype(int)
            direction_key = (int(direction[0]), int(direction[1]))
            if prev_dir is not None and direction_key != prev_dir:
                out.append(points[idx - 1])
            prev_dir = direction_key
        out.append(points[-1])
        return out

    @staticmethod
    def _downsample(points: list[np.ndarray], max_waypoints: int) -> list[np.ndarray]:
        if len(points) <= max_waypoints + 1:
            return points
        keep = np.linspace(0, len(points) - 1, max_waypoints + 1).round().astype(int)
        return [points[int(i)] for i in keep]
