from __future__ import annotations

from pathlib import Path

from scripts.validate_deployment import (
    EXPECTED_OPENCV_REQUIREMENT,
    VENDORED_ULTRALYTICS,
    VENDORED_ULTRALYTICS_SHA256,
    sha256,
    validate,
    validate_headless_wheel,
)


ROOT = Path(__file__).resolve().parents[1]


def test_apt_stage_is_absent_and_headless_contract_is_deployable() -> None:
    assert not (ROOT / "packages.txt").exists()
    result = validate(ROOT)
    assert result["status"] == "PASS"
    assert result["system_package_file"] is None
    assert result["system_packages"] == []
    assert result["opencv_distribution"] == "opencv-python-headless==4.11.0.86"
    assert result["tracked_files"][VENDORED_ULTRALYTICS.as_posix()] is True


def test_vendored_ultralytics_requires_only_pinned_headless_opencv() -> None:
    wheel = ROOT / VENDORED_ULTRALYTICS
    assert sha256(wheel) == VENDORED_ULTRALYTICS_SHA256
    validate_headless_wheel(wheel)
    assert EXPECTED_OPENCV_REQUIREMENT == "Requires-Dist: opencv-python-headless==4.11.0.86"


def test_requirements_exclude_gui_opencv_distribution() -> None:
    for relative in ("requirements-app.txt", "app/requirements.txt"):
        lines = {
            line.strip() for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert "opencv-python-headless==4.11.0.86" in lines
        assert "./app/vendor/ultralytics-8.3.150-py3-none-any.whl" in lines
        assert not any(line.startswith("opencv-python==") for line in lines)
        assert "ultralytics==8.3.150" not in lines
