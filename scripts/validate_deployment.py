"""Validate the tracked artifacts and configuration required for cloud deployment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from windblade_demo.constants import (
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_RELATIVE_PATH,
    CHECKPOINT_STATE_FINGERPRINT,
)
from windblade_demo.inference import load_frozen_model
from windblade_demo.detector import (
    DETECTOR_CHECKPOINT, DETECTOR_CHECKPOINT_BYTES, DETECTOR_CHECKPOINT_SHA256,
)


EXPECTED_REQUIREMENTS = {
    "-e .",
    "streamlit==1.62.0",
    "streamlit-cropper==0.3.1",
    "opencv-python-headless==4.11.0.86",
    "./app/vendor/ultralytics-8.3.150-py3-none-any.whl",
    'torch==2.13.0+cpu; sys_platform != "darwin"',
    'torchvision==0.28.0+cpu; sys_platform != "darwin"',
}
VENDORED_ULTRALYTICS = Path("app/vendor/ultralytics-8.3.150-py3-none-any.whl")
VENDORED_ULTRALYTICS_SHA256 = "1827a8504ef9c70b3285069b70ce0bc92f434934a06d0c550601d0a5fc2455bb"
EXPECTED_OPENCV_REQUIREMENT = "Requires-Dist: opencv-python-headless==4.11.0.86"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.relative_to(root).as_posix()],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0


def validate_headless_wheel(path: Path) -> None:
    if sha256(path) != VENDORED_ULTRALYTICS_SHA256:
        raise RuntimeError("Vendored Ultralytics wheel SHA-256 does not match the deployment contract.")
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise RuntimeError("Vendored Ultralytics wheel has invalid metadata layout.")
            metadata = wheel.read(metadata_names[0]).decode("utf-8")
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Vendored Ultralytics wheel is unreadable.") from exc
    metadata_lines = set(metadata.splitlines())
    if "Name: ultralytics" not in metadata_lines or "Version: 8.3.150" not in metadata_lines:
        raise RuntimeError("Vendored Ultralytics wheel identity is invalid.")
    opencv_requirements = {
        line for line in metadata_lines
        if line.lower().startswith("requires-dist: opencv-")
    }
    if opencv_requirements != {EXPECTED_OPENCV_REQUIREMENT}:
        raise RuntimeError(
            "Vendored Ultralytics wheel must require only the pinned headless OpenCV distribution."
        )


def validate(root: Path, *, require_tracked: bool = True) -> dict:
    entrypoint = root / "app/app.py"
    requirements = root / "app/requirements.txt"
    config = root / ".streamlit/config.toml"
    system_packages = root / "packages.txt"
    ultralytics_wheel = root / VENDORED_ULTRALYTICS
    checkpoint = root / CHECKPOINT_RELATIVE_PATH
    metadata = checkpoint.with_suffix(".json")
    detector_checkpoint = root / DETECTOR_CHECKPOINT
    for path in (
        entrypoint, requirements, config, ultralytics_wheel,
        checkpoint, metadata, detector_checkpoint,
    ):
        if not path.is_file():
            raise RuntimeError(f"Required deployment file is missing: {path.relative_to(root)}")
    if system_packages.exists():
        raise RuntimeError("packages.txt must be absent so Streamlit skips the apt dependency stage.")

    requirement_lines = {
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = EXPECTED_REQUIREMENTS - requirement_lines
    if missing:
        raise RuntimeError(f"Deployment requirements are incomplete: {sorted(missing)}")
    if "ultralytics==8.3.150" in requirement_lines or any(
        line.lower().startswith("opencv-python")
        and not line.lower().startswith("opencv-python-headless==")
        for line in requirement_lines
    ):
        raise RuntimeError("Deployment requirements include a forbidden GUI OpenCV dependency path.")
    validate_headless_wheel(ultralytics_wheel)
    config_text = config.read_text(encoding="utf-8")
    if "gatherUsageStats = false" not in config_text or "headless = true" not in config_text:
        raise RuntimeError("Streamlit deployment configuration is incomplete.")
    if sha256(checkpoint) != CHECKPOINT_FILE_SHA256:
        raise RuntimeError("Deployment checkpoint SHA-256 does not match the frozen contract.")
    metadata_record = json.loads(metadata.read_text(encoding="utf-8"))
    if metadata_record.get("checkpoint_fingerprint") != CHECKPOINT_STATE_FINGERPRINT:
        raise RuntimeError("Deployment checkpoint metadata does not match the frozen contract.")
    if detector_checkpoint.stat().st_size != DETECTOR_CHECKPOINT_BYTES:
        raise RuntimeError("Deployment detector checkpoint size does not match the frozen contract.")
    if sha256(detector_checkpoint) != DETECTOR_CHECKPOINT_SHA256:
        raise RuntimeError("Deployment detector checkpoint SHA-256 does not match the frozen contract.")

    tracked_files = {
        path.relative_to(root).as_posix(): tracked(root, path)
        for path in (
            entrypoint, requirements, config, ultralytics_wheel,
            checkpoint, metadata, detector_checkpoint,
        )
    }
    if require_tracked and not all(tracked_files.values()):
        raise RuntimeError(f"Deployment files are not all tracked: {tracked_files}")

    loaded = load_frozen_model(root)
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "target": "Streamlit Community Cloud",
        "repository": "Stukhori/BadeScope",
        "branch": "main",
        "entrypoint": "app/app.py",
        "python_version": "3.11",
        "dependency_file": "app/requirements.txt",
        "configuration_file": ".streamlit/config.toml",
        "system_package_file": None,
        "system_packages": [],
        "opencv_distribution": "opencv-python-headless==4.11.0.86",
        "ultralytics_wheel": {
            "path": VENDORED_ULTRALYTICS.as_posix(),
            "sha256": VENDORED_ULTRALYTICS_SHA256,
        },
        "checkpoint": {
            "path": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "bytes": checkpoint.stat().st_size,
            "file_sha256": sha256(checkpoint),
            "state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
            "loaded_on_cpu": next(loaded.model.parameters()).device.type == "cpu",
            "evaluation_mode": not loaded.model.training,
        },
        "proposal_detector_checkpoint": {
            "path": DETECTOR_CHECKPOINT.as_posix(),
            "bytes": detector_checkpoint.stat().st_size,
            "file_sha256": sha256(detector_checkpoint),
            "device": "cpu",
            "package": "ultralytics==8.3.150 (vendored headless dependency metadata)",
        },
        "tracked_files": tracked_files,
        "external_downloads_at_runtime": 0,
        "secrets_required": False,
        "automatic_region_proposals_available": True,
        "automatic_region_proposals_experimental": True,
        "external_deployment_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = validate(root, require_tracked=not args.allow_untracked)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
