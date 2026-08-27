"""Technical environment and Git provenance capture."""

from __future__ import annotations

import importlib
from importlib import metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

from windblade import __version__
from windblade.utils import format_utc, utc_now


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_command(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def capture_git_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Capture Git state without requiring Git or a committed repository."""

    root = Path(repository_root).resolve()
    inside = _git_command(root, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        return {"git_commit": None, "git_dirty": None}
    commit = _git_command(root, "rev-parse", "HEAD")
    status = _git_command(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _capture_optional_torch() -> dict[str, Any]:
    details: dict[str, Any] = {
        "pytorch_available": False,
        "pytorch_version": _distribution_version("torch"),
        "torchvision_version": _distribution_version("torchvision"),
        "cuda_available": None,
        "cuda_device_count": None,
    }
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        details["pytorch_import_error"] = type(exc).__name__
        return details

    details["pytorch_available"] = True
    details["pytorch_version"] = getattr(torch, "__version__", details["pytorch_version"])
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        available = bool(cuda.is_available())
        details["cuda_available"] = available
        details["cuda_device_count"] = int(cuda.device_count()) if available else 0
    return details


def capture_environment(
    selected_device: str,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture non-sensitive technical metadata for an experiment manifest."""

    root = Path(repository_root).resolve() if repository_root is not None else Path.cwd()
    project_version = _distribution_version("windblade") or __version__
    environment: dict[str, Any] = {
        "captured_utc": format_utc(utc_now()),
        "operating_system": platform.system() or None,
        "operating_system_release": platform.release() or None,
        "platform": platform.platform() or None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "project_version": project_version,
        "hostname": socket.gethostname() or None,
        "cpu": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "selected_device": selected_device,
    }
    environment.update(_capture_optional_torch())
    environment.update(capture_git_provenance(root))
    return environment
