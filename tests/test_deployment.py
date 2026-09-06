from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_deployment import (
    EXPECTED_SYSTEM_PACKAGES,
    normalized_system_packages,
    validate,
    validate_system_packages,
)


ROOT = Path(__file__).resolve().parents[1]


def test_root_packages_file_is_exact_tracked_and_deployable() -> None:
    path = ROOT / "packages.txt"
    assert path.read_bytes() == b"libgl1\nlibglib2.0-0t64\n"
    assert normalized_system_packages(path) == ["libgl1", "libglib2.0-0t64"]
    result = validate(ROOT)
    assert result["status"] == "PASS"
    assert result["system_package_file"] == "packages.txt"
    assert result["system_packages"] == ["libgl1", "libglib2.0-0t64"]
    assert result["tracked_files"]["packages.txt"] is True


def test_comments_and_whitespace_do_not_change_normalized_contents(tmp_path: Path) -> None:
    path = tmp_path / "packages.txt"
    path.write_text(
        "# Streamlit system dependencies\n  libgl1  \n  libglib2.0-0t64  \n\n",
        encoding="utf-8",
    )
    assert validate_system_packages(path) == EXPECTED_SYSTEM_PACKAGES


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "# libgl1\n",
        "libgl1\n",
        "libglib2.0-0t64\n",
        "libgl1\nlibgl1\n",
        "libgl1\nlibglib2.0-0t64\nlibglib2.0-0t64\n",
        "libgl1\nlibglib2.0-0\n",
        "libgl1-mesa-glx\n",
        "libglib2.0-0t64\nlibgl1\n",
        "libgl1\nlibglib2.0-0t64\nlibx11-6\n",
    ],
)
def test_missing_duplicate_or_additional_system_packages_fail_closed(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "packages.txt"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly"):
        validate_system_packages(path)
