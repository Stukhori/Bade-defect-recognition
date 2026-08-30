from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from windblade_demo.crops import (
    PixelBox,
    SelectionValidationError,
    contextual_crop,
    display_image,
    map_display_box,
    prepare_region,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/wtbd_crops_v1/manifest.csv"
PROCESSED = ROOT / "data/processed/wtbd_crops_v1"
RAW_IMAGES = ROOT / "data/raw/wtbd/WT blade defect dataset/JPEGImages"


def test_display_resize_and_coordinate_mapping_are_explicit():
    image = Image.new("RGB", (1400, 1000))
    displayed = display_image(image)
    assert displayed.size == (700, 500)
    selected = map_display_box(
        {"left": 100, "top": 50, "width": 200, "height": 100},
        display_size=displayed.size,
        original_size=image.size,
    )
    assert selected == PixelBox(200, 100, 600, 300)


@pytest.mark.parametrize(
    "rectangle",
    [
        {"left": 1, "top": 1, "width": 0, "height": 10},
        {"left": 1, "top": 1, "width": 10, "height": -1},
        {"left": float("nan"), "top": 1, "width": 10, "height": 10},
        {"left": 1, "top": 1, "width": "bad", "height": 10},
    ],
)
def test_invalid_manual_rectangles_are_rejected(rectangle):
    with pytest.raises(SelectionValidationError):
        map_display_box(rectangle, display_size=(100, 100), original_size=(200, 200))


@pytest.mark.parametrize(
    ("box", "expected_side"),
    [
        (PixelBox(0, 0, 5, 5), 64),
        (PixelBox(0, 400, 40, 500), 150),
        (PixelBox(400, 980, 500, 1024), 150),
        (PixelBox(300, 480, 700, 520), 600),
    ],
)
def test_contextual_crop_handles_minimum_edges_and_elongation(box, expected_side):
    result = contextual_crop(Image.new("RGB", (1024, 1024)), box)
    assert result.geometry.crop_side == expected_side
    assert result.model_input.mode == "RGB"
    assert result.model_input.size == (224, 224)
    assert result.geometry.crop_xmin <= box.left
    assert result.geometry.crop_ymin <= box.top
    assert result.geometry.crop_xmax >= box.right
    assert result.geometry.crop_ymax >= box.bottom


def _rows():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parity_candidates(rows):
    predicates = (
        lambda row: row["minimum_side_applied"] == "True",
        lambda row: row["boundary_shifted"] == "True" and row["max_side_clipped"] == "False",
        lambda row: row["max_side_clipped"] == "True",
        lambda row: row["minimum_side_applied"] == "False"
        and row["boundary_shifted"] == "False"
        and row["max_side_clipped"] == "False",
    )
    selected = []
    for predicate in predicates:
        match = next(row for row in rows if predicate(row))
        if match["instance_id"] not in {item["instance_id"] for item in selected}:
            selected.append(match)
    return selected


def test_manual_contextual_crop_is_pixel_identical_to_frozen_phase3_outputs():
    if not RAW_IMAGES.is_dir() or not (PROCESSED / "images").is_dir():
        pytest.skip("local frozen WTBD image payload is not available")
    for row in _parity_candidates(_rows()):
        with Image.open(RAW_IMAGES / row["source_filename"]) as opened:
            original = opened.convert("RGB").copy()
        selected = PixelBox(
            int(row["bbox_xmin"]) - 1,
            int(row["bbox_ymin"]) - 1,
            int(row["bbox_xmax"]),
            int(row["bbox_ymax"]),
        )
        observed = contextual_crop(original, selected)
        with Image.open(PROCESSED / row["output_relative_path"]) as opened:
            expected = opened.convert("RGB").copy()
        assert observed.geometry.crop_xmin == int(row["crop_xmin"])
        assert observed.geometry.crop_ymin == int(row["crop_ymin"])
        assert observed.geometry.crop_xmax == int(row["crop_xmax"])
        assert observed.geometry.crop_ymax == int(row["crop_ymax"])
        assert np.array_equal(np.asarray(observed.model_input), np.asarray(expected))


def test_prepared_224_rgb_crop_is_unchanged():
    source = Image.fromarray(np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3))
    assert np.array_equal(np.asarray(prepare_region(source)), np.asarray(source))
