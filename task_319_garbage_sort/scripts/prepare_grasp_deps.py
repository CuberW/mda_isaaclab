#!/usr/bin/env python3
"""Prepare optional dependencies and model weights for task-319 grasping.

Default behavior is check-only.  Use explicit flags to install packages, build
CUDA extensions, or download weights.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRASPNET_REPO = ROOT / "third_party/graspnet-baseline"
GRASPNET_API = ROOT / "third_party/graspnetAPI"
GRASPNET_CKPT = ROOT / "models/graspnet-rs/checkpoint-rs.tar"
ANYGRASP_REPO = ROOT / "third_party/anygrasp_sdk"
ANYGRASP_CKPT = ROOT / "models/anygrasp/checkpoint_detection.tar"
ANYGRASP_DETECTION = ANYGRASP_REPO / "grasp_detection"
ANYGRASP_LICENSE = ANYGRASP_DETECTION / "license"
ANYGRASP_GSNET = ANYGRASP_DETECTION / "gsnet.so"
ANYGRASP_LIB_CXX = ANYGRASP_DETECTION / "lib_cxx.so"
YOLO_WEIGHTS = ROOT / "models/yolo/yolov8m-seg.pt"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GOOGLE_DRIVE_RS_ID = "1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def has_shared_library(name: str) -> bool:
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def pip_install(packages: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *packages])


def check(require_anygrasp: bool = False) -> bool:
    ok = True
    required_checks = {
        "torch": has_module("torch"),
        "open3d": has_module("open3d"),
        "transforms3d": has_module("transforms3d"),
        "autolab_core": has_module("autolab_core"),
        "cvxopt": has_module("cvxopt"),
        "grasp_nms": has_module("grasp_nms"),
        "graspnetAPI path": GRASPNET_API.exists(),
        "graspnet-baseline path": GRASPNET_REPO.exists(),
        "pointnet2 extension": has_module("pointnet2._ext"),
        "knn_pytorch extension": has_module("knn_pytorch.knn_pytorch"),
        "ultralytics": has_module("ultralytics"),
        "transformers": has_module("transformers"),
        "GraspNet checkpoint": GRASPNET_CKPT.is_file(),
        "YOLOv8m-seg weights": YOLO_WEIGHTS.is_file(),
    }
    optional_checks = {
        "AnyGrasp SDK path": ANYGRASP_REPO.exists(),
        "AnyGrasp checkpoint": ANYGRASP_CKPT.is_file(),
        "AnyGrasp gsnet.so": ANYGRASP_GSNET.is_file(),
        "AnyGrasp lib_cxx.so": ANYGRASP_LIB_CXX.is_file(),
        "AnyGrasp license": ANYGRASP_LICENSE.exists(),
        "MinkowskiEngine": has_module("MinkowskiEngine"),
        "OpenSSL libcrypto.so.1.1": has_shared_library("libcrypto.so.1.1"),
    }
    for label, passed in required_checks.items():
        print(f"{'[OK]' if passed else '[MISS]'} {label}")
        ok = ok and passed
    anygrasp_ok = True
    for label, passed in optional_checks.items():
        print(f"{'[OK]' if passed else '[MISS]'} optional {label}")
        anygrasp_ok = anygrasp_ok and passed
    return ok and (anygrasp_ok if require_anygrasp else True)

def download_graspnet_checkpoint() -> None:
    GRASPNET_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if GRASPNET_CKPT.is_file():
        return
    if not has_module("gdown"):
        pip_install(["gdown"])
    run(["gdown", "--id", GOOGLE_DRIVE_RS_ID, "-O", str(GRASPNET_CKPT)])


def cuda_build_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = Path(sys.prefix)
    if (cuda_home / "bin/nvcc").is_file():
        env.setdefault("CUDA_HOME", str(cuda_home))
        env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    return env


def build_pointnet2() -> None:
    pointnet_dir = GRASPNET_REPO / "pointnet2"
    if not pointnet_dir.is_dir():
        raise FileNotFoundError(pointnet_dir)
    run([sys.executable, "setup.py", "install"], cwd=pointnet_dir, env=cuda_build_env())


def build_knn() -> None:
    knn_dir = GRASPNET_REPO / "knn"
    if not knn_dir.is_dir():
        raise FileNotFoundError(knn_dir)
    run([sys.executable, "setup.py", "install"], cwd=knn_dir, env=cuda_build_env())


def download_yolo_weights() -> None:
    YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if YOLO_WEIGHTS.is_file():
        return
    if not has_module("ultralytics"):
        pip_install(["ultralytics"])
    # Ultralytics downloads public weights when the model name is constructed.
    code = (
        "from ultralytics import YOLO; "
        f"m=YOLO('yolov8m-seg.pt'); "
        f"m.export if False else None"
    )
    run([sys.executable, "-c", code], cwd=ROOT)
    downloaded = ROOT / "yolov8m-seg.pt"
    if downloaded.is_file():
        downloaded.replace(YOLO_WEIGHTS)
    elif not YOLO_WEIGHTS.is_file():
        raise FileNotFoundError("Ultralytics did not leave yolov8m-seg.pt in the current directory.")


def download_hf_llm() -> None:
    if not has_module("transformers"):
        pip_install(["transformers", "accelerate", "sentencepiece"])
    code = (
        "from transformers import AutoModelForCausalLM, AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{QWEN_MODEL}'); "
        f"AutoModelForCausalLM.from_pretrained('{QWEN_MODEL}')"
    )
    run([sys.executable, "-c", code])


def pull_ollama(model: str) -> None:
    run(["ollama", "pull", model])


def clone_anygrasp_sdk() -> None:
    if ANYGRASP_REPO.exists():
        print(f"[SKIP] AnyGrasp SDK already exists: {ANYGRASP_REPO}")
        return
    ANYGRASP_REPO.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "https://github.com/graspnet/anygrasp_sdk.git", str(ANYGRASP_REPO)])


def _find_versioned_binary(folder: Path, stem: str) -> Path:
    tag = sys.implementation.cache_tag
    matches = sorted(folder.glob(f"{stem}.{tag}*.so"))
    if matches:
        return matches[0]
    available = ", ".join(item.name for item in sorted(folder.glob(f"{stem}*.so"))) or "none"
    raise FileNotFoundError(f"No {stem} binary for Python tag {tag} in {folder}. Available: {available}")


def prepare_anygrasp_binaries() -> None:
    if not ANYGRASP_REPO.is_dir():
        raise FileNotFoundError(f"AnyGrasp SDK repo not found: {ANYGRASP_REPO}")
    gsnet_src = _find_versioned_binary(ANYGRASP_DETECTION / "gsnet_versions", "gsnet")
    lib_cxx_src = _find_versioned_binary(ANYGRASP_REPO / "license_registration/lib_cxx_versions", "lib_cxx")
    shutil.copy2(gsnet_src, ANYGRASP_GSNET)
    shutil.copy2(lib_cxx_src, ANYGRASP_LIB_CXX)
    print(f"[OK] copied {gsnet_src.name} -> {ANYGRASP_GSNET}")
    print(f"[OK] copied {lib_cxx_src.name} -> {ANYGRASP_LIB_CXX}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare task-319 grasping dependencies.")
    parser.add_argument("--install_python_deps", action="store_true", help="Install Python packages needed by optional backends.")
    parser.add_argument("--build_pointnet2", action="store_true", help="Build/install GraspNet pointnet2 CUDA extension.")
    parser.add_argument("--build_knn", action="store_true", help="Build/install GraspNet knn_pytorch extension.")
    parser.add_argument("--download_graspnet", action="store_true", help="Download checkpoint-rs.tar.")
    parser.add_argument("--download_yolo", action="store_true", help="Download YOLOv8m-seg weights.")
    parser.add_argument("--clone_anygrasp", action="store_true", help="Clone the public AnyGrasp SDK scaffold. Weights, license, and runtime dependencies must still be added manually from the SDK release.")
    parser.add_argument("--prepare_anygrasp_binaries", action="store_true", help="Copy Python-version-matched AnyGrasp gsnet/lib_cxx binaries into grasp_detection.")
    parser.add_argument("--require_anygrasp", action="store_true", help="Treat missing AnyGrasp SDK assets as a failed check.")
    parser.add_argument("--download_hf_llm", action="store_true", help="Download Qwen2.5-0.5B-Instruct via HuggingFace.")
    parser.add_argument("--pull_ollama", default="", help="Pull an Ollama model such as qwen2.5:0.5b.")
    parser.add_argument("--download_missing_models", action="store_true", help="Download GraspNet and YOLO weights if missing.")
    args = parser.parse_args()

    if args.install_python_deps:
        pip_install([
            "numpy==1.26.0",
            "scipy",
            "Pillow",
            "tqdm",
            "open3d",
            "gdown",
            "ultralytics",
            "opencv-python==4.10.0.84",
            "typing_extensions==4.12.2",
            "psutil==5.9.8",
            "uvicorn==0.29.0",
            "starlette==0.49.1",
            "transforms3d",
            "autolab-core",
            "cvxopt",
            "grasp_nms",
        ])
    if args.build_pointnet2:
        build_pointnet2()
    if args.build_knn:
        build_knn()
    if args.download_missing_models or args.download_graspnet:
        download_graspnet_checkpoint()
    if args.download_missing_models or args.download_yolo:
        download_yolo_weights()
    if args.clone_anygrasp:
        clone_anygrasp_sdk()
    if args.prepare_anygrasp_binaries:
        prepare_anygrasp_binaries()
    if args.download_hf_llm:
        download_hf_llm()
    if args.pull_ollama:
        pull_ollama(args.pull_ollama)
    ok = check(require_anygrasp=args.require_anygrasp)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
