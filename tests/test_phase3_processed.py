from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from windblade.data.processed import (
    EXPECTED_CLASS_COUNTS,
    LABELS,
    LABEL_TO_ID,
    ProcessedDatasetError,
    _validate_generated,
)


def _row(instance_id: str = "1_0", source_id: str = "1", split: str = "train"):
    return {
        "instance_id": instance_id,
        "source_image_id": source_id,
        "split": split,
        "canonical_label": "craze",
        "class_id": 0,
        "crop_xmin": 0,
        "crop_ymin": 0,
        "crop_xmax": 64,
        "crop_ymax": 64,
        "crop_side": 64,
        "output_relative_path": f"images/{instance_id}.png",
    }


def _write_crop(root: Path, instance_id: str):
    path = root / "images" / f"{instance_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), (10, 20, 30)).save(path, format="PNG")


def test_multiple_instances_inherit_one_source_split(tmp_path: Path):
    rows = [_row("1_0"), _row("1_1")]
    for row in rows:
        _write_crop(tmp_path, row["instance_id"])
    _validate_generated(rows, tmp_path, {"1": "train"})


def test_unknown_or_changed_split_fails(tmp_path: Path):
    row = _row(split="unknown")
    _write_crop(tmp_path, row["instance_id"])
    with pytest.raises(ProcessedDatasetError, match="split inheritance"):
        _validate_generated([row], tmp_path, {"1": "train"})


def test_excluded_source_cannot_enter_processed_data(tmp_path: Path):
    row = _row(source_id="excluded")
    _write_crop(tmp_path, row["instance_id"])
    with pytest.raises(ProcessedDatasetError, match="excluded source"):
        _validate_generated([row], tmp_path, {"1": "train"})


def test_duplicate_instance_ids_fail(tmp_path: Path):
    row = _row()
    _write_crop(tmp_path, row["instance_id"])
    with pytest.raises(ProcessedDatasetError, match="duplicate instance"):
        _validate_generated([row, dict(row)], tmp_path, {"1": "train"})


def test_extra_output_image_fails(tmp_path: Path):
    row = _row()
    _write_crop(tmp_path, row["instance_id"])
    _write_crop(tmp_path, "extra_0")
    with pytest.raises(ProcessedDatasetError, match="missing or extra"):
        _validate_generated([row], tmp_path, {"1": "train"})


def test_frozen_label_order_and_expected_counts():
    assert LABELS == (
        "craze",
        "corrosion",
        "surface_injure",
        "thunderstrike",
        "crack",
        "hide_craze",
    )
    assert LABEL_TO_ID == {label: index for index, label in enumerate(LABELS)}
    assert [EXPECTED_CLASS_COUNTS[label] for label in LABELS] == [169, 178, 264, 60, 131, 263]
    assert sum(EXPECTED_CLASS_COUNTS.values()) == 1065
