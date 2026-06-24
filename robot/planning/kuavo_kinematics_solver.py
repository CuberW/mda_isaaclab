"""Standalone Kuavo IK bridge backed by the isolated C ABI wrapper."""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path

from robot_common.infra.config import PROJECT_ROOT


class KuavoKinematicsSolver:
    """Thin Python wrapper around the standalone Kuavo IK shared library."""

    _MODEL_DOF_PATTERN = re.compile(r"\bmodelDof\s+(\d+)")
    _MODEL_TYPE_PATTERN = re.compile(r"\bmanipulatorModelType\s+(\d+)")

    def __init__(
        self,
        lib_path: str | Path | None = None,
        urdf_path: str | Path | None = None,
        task_info_path: str | Path | None = None,
        arm_index: int = 1,
        is_whole_body: bool = True,
        linear_error_max: float = 0.01,
        angular_error_max: float = 0.02,
    ) -> None:
        self.lib_path = self._resolve_library_path(lib_path, required=False)
        self.urdf_path = self._resolve_urdf_path(urdf_path)
        self.task_info_path = self._resolve_task_info_path(task_info_path)
        self.arm_index = int(arm_index)
        self.is_whole_body = bool(is_whole_body)
        self.linear_error_max = float(linear_error_max)
        self.angular_error_max = float(angular_error_max)

        self.arm_dim, self.manipulator_model_type = self._load_task_info(self.task_info_path)
        self.state_dim = self._infer_state_dim(self.arm_dim, self.manipulator_model_type)
        self.safe_home_q = self._build_safe_home_q()
        self._docker_container = os.getenv("KUAVO_STANDALONE_IK_DOCKER_CONTAINER", "kuavo_official_ros")
        self._docker_lib_path = os.getenv(
            "KUAVO_STANDALONE_IK_DOCKER_LIB",
            "/root/standalone_ik/build/libstandalone_ik.so",
        )
        self._docker_urdf_path = os.getenv(
            "KUAVO_STANDALONE_IK_DOCKER_URDF",
            "/root/kuavo_ws_linux/src/kuavo_assets/models/biped_s62/urdf/biped_s62.urdf",
        )
        self._docker_task_info_path = os.getenv(
            "KUAVO_STANDALONE_IK_DOCKER_TASK_INFO",
            "/root/kuavo_ws_linux/src/humanoid-wheel-control/humanoid_wheel_interface/config/kuavo_s62/task.info",
        )
        self._docker_timeout_s = float(os.getenv("KUAVO_STANDALONE_IK_DOCKER_TIMEOUT_S", "20.0"))
        self._use_docker_backend = False
        self._lib = None
        self._handle = None

        local_init_error = self._try_init_local_backend()
        if local_init_error is not None:
            if not self._docker_backend_available():
                raise RuntimeError(local_init_error)
            self._use_docker_backend = True

        self.last_status = 0
        self.last_linear_error = math.inf
        self.last_angular_error = math.inf
        self.last_seed_source = "safe_home"
        self.last_q: list[float] = []
        self.dof_names = self._load_dof_names()

    def close(self) -> None:
        if self._use_docker_backend:
            self._handle = None
            return
        if getattr(self, "_handle", None):
            if os.getenv("KUAVO_STANDALONE_IK_DESTROY_ON_CLOSE", "").strip() != "1":
                self._handle = None
                return
            self._lib.ik_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def solve_ik(self, target_6d: list[float], current_q: list[float] = None) -> list[float]:
        """Solve arm IK and fall back to the validated safe home seed on failure."""
        if len(target_6d) < 6:
            raise ValueError("target_6d must contain [x, y, z, yaw, pitch, roll]")

        seed = self._normalize_seed(current_q)
        status, q_best, linear_error, angular_error = self._solve_once(target_6d, seed)
        used_seed = "current_q" if current_q is not None else "safe_home"

        if self._needs_retry(status, linear_error, angular_error) and current_q is not None:
            retry_status, retry_q, retry_linear_error, retry_angular_error = self._solve_once(
                target_6d,
                self.safe_home_q,
            )
            if self._is_better_result(
                retry_status,
                retry_linear_error,
                retry_angular_error,
                status,
                linear_error,
                angular_error,
            ):
                status = retry_status
                q_best = retry_q
                linear_error = retry_linear_error
                angular_error = retry_angular_error
                used_seed = "safe_home_retry"

        self.last_status = status
        self.last_linear_error = linear_error
        self.last_angular_error = angular_error
        self.last_seed_source = used_seed
        self.last_q = list(q_best)

        if self._needs_retry(status, linear_error, angular_error):
            raise RuntimeError(
                "standalone Kuavo IK failed: "
                f"status={status} linear_error={linear_error:.6f} "
                f"angular_error={angular_error:.6f} seed={used_seed}"
            )

        return list(q_best)

    def _configure_c_api(self) -> None:
        self._lib.ik_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self._lib.ik_create.restype = ctypes.c_void_p
        self._lib.ik_solve.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.ik_solve.restype = ctypes.c_int
        self._lib.ik_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ik_destroy.restype = None
        self._lib.ik_get_dof_name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._lib.ik_get_dof_name.restype = ctypes.c_int

    def _solve_once(self, target_6d: list[float], seed_q: list[float]) -> tuple[int, list[float], float, float]:
        if self._use_docker_backend:
            return self._solve_once_via_docker(target_6d, seed_q)
        target_array = (ctypes.c_double * 6)(*map(float, target_6d[:6]))
        seed_array = (ctypes.c_double * self.state_dim)(*map(float, seed_q[: self.state_dim]))
        out_q = (ctypes.c_double * self.arm_dim)()
        best_linear_error = ctypes.c_double(math.inf)
        best_angular_error = ctypes.c_double(math.inf)

        status = self._lib.ik_solve(
            self._handle,
            target_array,
            seed_array,
            self.state_dim,
            out_q,
            self.arm_dim,
            ctypes.byref(best_linear_error),
            ctypes.byref(best_angular_error),
        )
        q_best = [float(out_q[i]) for i in range(self.arm_dim)]
        return status, q_best, float(best_linear_error.value), float(best_angular_error.value)

    def _solve_once_via_docker(
        self,
        target_6d: list[float],
        seed_q: list[float],
    ) -> tuple[int, list[float], float, float]:
        payload = {
            "lib_path": self._docker_lib_path,
            "urdf_path": self._docker_urdf_path,
            "task_info_path": self._docker_task_info_path,
            "arm_index": int(self.arm_index),
            "is_whole_body": int(self.is_whole_body),
            "target_6d": [float(v) for v in target_6d[:6]],
            "seed_q": [float(v) for v in seed_q[: self.state_dim]],
            "state_dim": int(self.state_dim),
            "arm_dim": int(self.arm_dim),
        }
        script = """
import ctypes
import json
import math
import os
import sys

payload = json.loads(os.environ["KUAVO_STANDALONE_IK_PAYLOAD"])
lib = ctypes.CDLL(payload["lib_path"])
lib.ik_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.ik_create.restype = ctypes.c_void_p
lib.ik_solve.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
lib.ik_solve.restype = ctypes.c_int
lib.ik_destroy.argtypes = [ctypes.c_void_p]
lib.ik_destroy.restype = None

handle = lib.ik_create(
    payload["urdf_path"].encode("utf-8"),
    payload["task_info_path"].encode("utf-8"),
    int(payload["arm_index"]),
    int(payload["is_whole_body"]),
)
if not handle:
    raise RuntimeError("ik_create returned null")

target_pose = (ctypes.c_double * 6)(*payload["target_6d"])
seed_q = (ctypes.c_double * int(payload["state_dim"]))(*payload["seed_q"])
out_q = (ctypes.c_double * int(payload["arm_dim"]))()
best_linear_error = ctypes.c_double(math.inf)
best_angular_error = ctypes.c_double(math.inf)
try:
    status = lib.ik_solve(
        handle,
        target_pose,
        seed_q,
        int(payload["state_dim"]),
        out_q,
        int(payload["arm_dim"]),
        ctypes.byref(best_linear_error),
        ctypes.byref(best_angular_error),
    )
finally:
    lib.ik_destroy(handle)

print(json.dumps({
    "status": int(status),
    "q_best": [float(out_q[i]) for i in range(int(payload["arm_dim"]))],
    "linear_error": float(best_linear_error.value),
    "angular_error": float(best_angular_error.value),
}))
"""
        env = os.environ.copy()
        env["KUAVO_STANDALONE_IK_PAYLOAD"] = json.dumps(payload)
        cmd = [
            "docker",
            "exec",
            "-e",
            "KUAVO_STANDALONE_IK_PAYLOAD",
            self._docker_container,
            "bash",
            "-lc",
            (
                "source /opt/ros/noetic/setup.bash >/dev/null 2>&1; "
                "test -f /root/kuavo_ws_linux/installed/setup.bash "
                "&& source /root/kuavo_ws_linux/installed/setup.bash >/dev/null 2>&1; "
                "test -f /root/kuavo_ws_linux/devel/setup.bash "
                "&& source /root/kuavo_ws_linux/devel/setup.bash >/dev/null 2>&1; "
                "python3 -c "
                + shlex.quote(script)
            ),
        ]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=max(self._docker_timeout_s, 5.0),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"standalone IK docker solve timed out after {self._docker_timeout_s:.1f}s") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker standalone IK solve failed"
            raise RuntimeError(detail)
        try:
            data = json.loads(result.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise RuntimeError(f"invalid docker standalone IK response: {result.stdout!r}") from exc
        return (
            int(data.get("status", 1)),
            [float(v) for v in data.get("q_best", [])],
            float(data.get("linear_error", math.inf)),
            float(data.get("angular_error", math.inf)),
        )

    def _try_init_local_backend(self) -> str | None:
        try:
            dlopen_mode = ctypes.DEFAULT_MODE
            dlopen_mode |= getattr(os, "RTLD_GLOBAL", 0)
            dlopen_mode |= getattr(os, "RTLD_NODELETE", 0)
            self._lib = ctypes.CDLL(str(self.lib_path), mode=dlopen_mode)
            self._configure_c_api()
            self._handle = self._lib.ik_create(
                str(self.urdf_path).encode("utf-8"),
                str(self.task_info_path).encode("utf-8"),
                self.arm_index,
                int(self.is_whole_body),
            )
            if not self._handle:
                return (
                    "ik_create failed for "
                    f"lib={self.lib_path} urdf={self.urdf_path} task_info={self.task_info_path}"
                )
            return None
        except Exception as exc:
            self._lib = None
            self._handle = None
            return (
                "standalone IK local backend unavailable: "
                f"lib={self.lib_path} urdf={self.urdf_path} task_info={self.task_info_path} detail={exc}"
            )

    def _load_dof_names(self) -> list[str]:
        if self._use_docker_backend or self._lib is None or self._handle is None:
            return []
        names: list[str] = []
        for index in range(self.arm_dim):
            buf = ctypes.create_string_buffer(256)
            try:
                result = self._lib.ik_get_dof_name(self._handle, index, buf, len(buf))
            except Exception:
                return []
            if result < 0:
                return []
            names.append(buf.value.decode("utf-8", errors="replace"))
        return names

    def _docker_backend_available(self) -> bool:
        if os.getenv("KUAVO_STANDALONE_IK_DISABLE_DOCKER", "").strip() == "1":
            return False
        cmd = [
            "docker",
            "exec",
            self._docker_container,
            "sh",
            "-lc",
            "test -f "
            + shlex.quote(self._docker_lib_path)
            + " && test -f "
            + shlex.quote(self._docker_urdf_path)
            + " && test -f "
            + shlex.quote(self._docker_task_info_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=5.0,
            )
        except Exception:
            return False
        return result.returncode == 0

    def _normalize_seed(self, current_q: list[float] | None) -> list[float]:
        if current_q is None:
            return list(self.safe_home_q)

        values = [float(v) for v in current_q]
        if len(values) == self.state_dim:
            return values
        if len(values) == self.arm_dim:
            seed = list(self.safe_home_q)
            seed[-self.arm_dim:] = values
            return seed
        raise ValueError(
            f"current_q length must be state_dim={self.state_dim} or arm_dim={self.arm_dim}, "
            f"got {len(values)}"
        )

    def _build_safe_home_q(self) -> list[float]:
        seed = [0.0] * self.state_dim
        base_dim = self.state_dim - self.arm_dim
        if base_dim == 3:
            arm_biases = (
                [0.08, -0.12, 0.08, -0.12]
                + [0.05, -0.08, 0.10, 0.06, -0.06, 0.08, 0.04]
                + [-0.05, 0.08, -0.10, -0.06, 0.06, -0.08, -0.04]
            )
            if len(arm_biases) != self.arm_dim:
                raise RuntimeError(
                    f"validated safe home vector length {len(arm_biases)} does not match arm_dim={self.arm_dim}"
                )
            seed[base_dim:] = arm_biases
            return seed

        if base_dim >= 7:
            seed[2] = 0.85
            seed[6] = 1.0
        elif base_dim >= 3:
            seed[2] = 0.85

        for idx in range(base_dim, self.state_dim):
            seed[idx] = 0.1
        return seed

    def _needs_retry(self, status: int, linear_error: float, angular_error: float) -> bool:
        return (
            status != 0
            or not math.isfinite(linear_error)
            or not math.isfinite(angular_error)
            or linear_error > self.linear_error_max
            or angular_error > self.angular_error_max
        )

    @staticmethod
    def _is_better_result(
        new_status: int,
        new_linear_error: float,
        new_angular_error: float,
        old_status: int,
        old_linear_error: float,
        old_angular_error: float,
    ) -> bool:
        new_score = (
            0 if new_status == 0 else 1,
            float("inf") if not math.isfinite(new_linear_error) else new_linear_error,
            float("inf") if not math.isfinite(new_angular_error) else new_angular_error,
        )
        old_score = (
            0 if old_status == 0 else 1,
            float("inf") if not math.isfinite(old_linear_error) else old_linear_error,
            float("inf") if not math.isfinite(old_angular_error) else old_angular_error,
        )
        return new_score < old_score

    @classmethod
    def _load_task_info(cls, task_info_path: Path) -> tuple[int, int]:
        text = task_info_path.read_text(encoding="utf-8", errors="ignore")
        arm_dim_match = cls._MODEL_DOF_PATTERN.search(text)
        model_type_match = cls._MODEL_TYPE_PATTERN.search(text)
        if not arm_dim_match or not model_type_match:
            raise RuntimeError(f"failed to parse modelDof/manipulatorModelType from {task_info_path}")
        return int(arm_dim_match.group(1)), int(model_type_match.group(1))

    @staticmethod
    def _infer_state_dim(arm_dim: int, manipulator_model_type: int) -> int:
        if manipulator_model_type in (1, 4):
            return arm_dim + 3
        if manipulator_model_type in (2, 3):
            return arm_dim + 6
        if manipulator_model_type == 0:
            return arm_dim
        raise RuntimeError(f"unsupported manipulatorModelType={manipulator_model_type}")

    @classmethod
    def _resolve_library_path(cls, lib_path: str | Path | None, required: bool = True) -> Path:
        candidates: list[Path] = []
        if lib_path is not None:
            candidates.append(Path(lib_path))
        env_value = os.getenv("KUAVO_STANDALONE_IK_LIB")
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend(
            [
                PROJECT_ROOT / "standalone_ik" / "build" / "libstandalone_ik.so",
                PROJECT_ROOT / "standalone_ik" / "build" / "Release" / "standalone_ik.dll",
                PROJECT_ROOT / "standalone_ik" / "build" / "standalone_ik.dll",
                PROJECT_ROOT / "standalone_ik" / "build" / "libstandalone_ik.dylib",
            ]
        )
        if required:
            return cls._first_existing_path("standalone IK library", candidates)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                return path
        return Path(candidates[0]).expanduser()

    @classmethod
    def _resolve_urdf_path(cls, urdf_path: str | Path | None) -> Path:
        candidates: list[Path] = []
        if urdf_path is not None:
            candidates.append(Path(urdf_path))
        env_value = os.getenv("KUAVO_STANDALONE_IK_URDF")
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend(
            [
                PROJECT_ROOT
                / "third_party"
                / "kuavo-ros-opensource"
                / "src"
                / "kuavo_assets"
                / "models"
                / "biped_s62"
                / "urdf"
                / "biped_s62.urdf",
                PROJECT_ROOT / "simulation" / "robots" / "kuavo_wheel_s62" / "urdf" / "biped_s62.urdf",
            ]
        )
        return cls._first_existing_path("Kuavo URDF", candidates)

    @classmethod
    def _resolve_task_info_path(cls, task_info_path: str | Path | None) -> Path:
        candidates: list[Path] = []
        if task_info_path is not None:
            candidates.append(Path(task_info_path))
        env_value = os.getenv("KUAVO_STANDALONE_IK_TASK_INFO")
        if env_value:
            candidates.append(Path(env_value))
        candidates.extend(
            [
                PROJECT_ROOT
                / "third_party"
                / "kuavo-ros-opensource"
                / "src"
                / "humanoid-wheel-control"
                / "humanoid_wheel_interface"
                / "config"
                / "kuavo_s62"
                / "task.info",
            ]
        )
        return cls._first_existing_path("Kuavo task.info", candidates)

    @staticmethod
    def _first_existing_path(label: str, candidates: list[Path]) -> Path:
        checked: list[str] = []
        for candidate in candidates:
            path = Path(candidate).expanduser()
            checked.append(str(path))
            if path.exists():
                return path
        raise FileNotFoundError(f"could not find {label}; checked: {checked}")
