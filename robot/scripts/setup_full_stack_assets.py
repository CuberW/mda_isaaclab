#!/usr/bin/env python
"""
Prepare model assets and third-party source locations for the full stack.

This script does not fabricate missing checkpoints. It downloads what can be
downloaded automatically and prints exact manual paths for license-gated or
network-blocked artifacts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"


def run(cmd: list[str], cwd: Path | None = None) -> bool:
    print("+", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  failed: {exc}")
        return False


def run_timeout(cmd: list[str], cwd: Path | None = None, timeout: int = 180) -> bool:
    print("+", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  failed: {exc}")
        return False


def ensure_git_repo(path: Path, url: str):
    if path.exists():
        print(f"[OK] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(path)])


def is_complete_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size <= 10 * 1024 * 1024:
        return False
    try:
        import torch

        checkpoint = torch.load(str(path), map_location="cpu")
        return isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    except Exception as exc:
        print(f"  invalid checkpoint at {path}: {exc}")
        return False


def curl_download(url: str, target: Path, timeout: int = 300) -> bool:
    curl = which("curl.exe") or which("curl")
    if curl is None:
        return False
    args = [
        curl,
        "-L",
        "--fail",
        "--connect-timeout",
        "20",
        "--max-time",
        str(timeout),
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "-o",
        str(target),
        url,
    ]
    if target.exists():
        args.insert(2, "-C")
        args.insert(3, "-")
    return run_timeout(args, timeout=timeout + 30)


def ensure_graspnet_checkpoint():
    target = MODELS_DIR / "graspnet-rs" / "checkpoint-rs.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_complete_checkpoint(target):
        print(f"[OK] GraspNet checkpoint: {target}")
        return

    print("[MISS] GraspNet checkpoint-rs.tar")

    # Direct URLs are faster and easier to resume than hub APIs on restricted
    # networks. Try the China-friendly mirror first, then the canonical host.
    direct_urls = [
        "https://hf-mirror.com/dgrachev/a2_pretrained/resolve/main/checkpoint-rs.tar?download=true",
        "https://huggingface.co/dgrachev/a2_pretrained/resolve/main/checkpoint-rs.tar?download=true",
    ]
    for url in direct_urls:
        if curl_download(url, target) and is_complete_checkpoint(target):
            print(f"[OK] GraspNet checkpoint: {target}")
            return

    try:
        import gdown
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "gdown"])
        try:
            import gdown
        except ImportError:
            gdown = None

    if gdown is not None:
        # Official graspnet-baseline README links the RealSense checkpoint via
        # this Google Drive id.
        file_id = "1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"
        try:
            gdown.download(id=file_id, output=str(target), quiet=False)
        except Exception as exc:
            print(f"  gdown failed: {exc}")

    if not is_complete_checkpoint(target):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                hf_hub_download = None
        if hf_hub_download is not None:
            try:
                downloaded = hf_hub_download(
                    repo_id="dgrachev/a2_pretrained",
                    filename="checkpoint-rs.tar",
                    local_dir=str(target.parent),
                    local_dir_use_symlinks=False,
                )
                downloaded_path = Path(downloaded)
                if downloaded_path != target and downloaded_path.exists():
                    downloaded_path.replace(target)
            except Exception as exc:
                print(f"  Hugging Face fallback failed: {exc}")

    if not is_complete_checkpoint(target):
        print(
            "\nManual action required:\n"
            f"  Place GraspNet RealSense checkpoint at:\n    {target}\n"
            "  Expected filename: checkpoint-rs.tar\n"
            "  Sources from graspnet/graspnet-baseline README:\n"
            "    Google Drive: https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing\n"
            "    Baidu Pan: https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A\n"
        )


def main() -> int:
    THIRD_PARTY_DIR.mkdir(exist_ok=True)
    ensure_git_repo(THIRD_PARTY_DIR / "graspnetAPI", "https://github.com/graspnet/graspnetAPI.git")
    ensure_git_repo(THIRD_PARTY_DIR / "graspnet-baseline", "https://github.com/graspnet/graspnet-baseline.git")
    ensure_graspnet_checkpoint()

    required = [
        MODELS_DIR / "grounding-dino-base" / "config.json",
        MODELS_DIR / "sam-vit-b" / "sam_vit_b_01ec64.pth",
        MODELS_DIR / "clip-vit-b32" / "config.json",
        MODELS_DIR / "graspnet-rs" / "checkpoint-rs.tar",
    ]
    ok = True
    for path in required:
        exists = path.exists()
        print(f"[{'OK' if exists else 'MISS'}] {path}")
        ok = ok and exists
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
