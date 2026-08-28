from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
import pytest

from windblade.data.crops import CropValidationError, calculate_square_crop, generate_crop_png


@pytest.mark.parametrize(
    ("box", "size"),
    [
        ((401, 401, 500, 500), (1024, 1024)),  # centered square
        ((480, 300, 520, 700), (1024, 1024)),  # tall
        ((300, 480, 700, 520), (1024, 1024)),  # wide
        ((1, 400, 40, 500), (1024, 1024)),  # left
        ((980, 400, 1024, 500), (1024, 1024)),  # right
        ((400, 1, 500, 40), (1024, 1024)),  # top
        ((400, 980, 500, 1024), (1024, 1024)),  # bottom
        ((1, 1, 30, 30), (1024, 1024)),  # corner and minimum
        ((20, 20, 770, 770), (788, 788)),  # non-1024 and near full image
        ((200, 100, 700, 700), (800, 900)),  # non-square source
    ],
)
def test_square_crop_geometry_contains_box_and_stays_inside(box, size):
    xmin, ymin, xmax, ymax = box
    width, height = size
    crop = calculate_square_crop(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        image_width=width,
        image_height=height,
    )
    assert crop.crop_xmax - crop.crop_xmin == crop.crop_side
    assert crop.crop_ymax - crop.crop_ymin == crop.crop_side
    assert 0 <= crop.crop_xmin < crop.crop_xmax <= width
    assert 0 <= crop.crop_ymin < crop.crop_ymax <= height
    assert crop.crop_xmin <= xmin - 1
    assert crop.crop_ymin <= ymin - 1
    assert crop.crop_xmax >= xmax
    assert crop.crop_ymax >= ymax
    assert crop.crop_side >= xmax - xmin + 1
    assert crop.crop_side >= ymax - ymin + 1


def test_small_box_uses_64_pixel_minimum():
    crop = calculate_square_crop(
        xmin=500, ymin=500, xmax=505, ymax=505, image_width=1024, image_height=1024
    )
    assert crop.crop_side == 64
    assert crop.minimum_side_applied


def test_large_box_uses_context_threshold():
    crop = calculate_square_crop(
        xmin=400, ymin=400, xmax=499, ymax=449, image_width=1024, image_height=1024
    )
    assert crop.crop_side == 150
    assert not crop.minimum_side_applied


def test_full_image_clipping_is_recorded():
    crop = calculate_square_crop(
        xmin=1, ymin=1, xmax=788, ymax=788, image_width=788, image_height=788
    )
    assert crop.crop_side == 788
    assert crop.max_side_clipped
    assert (crop.crop_xmin, crop.crop_ymin, crop.crop_xmax, crop.crop_ymax) == (0, 0, 788, 788)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xmin": 0, "ymin": 1, "xmax": 10, "ymax": 10, "image_width": 20, "image_height": 20},
        {"xmin": 10, "ymin": 1, "xmax": 2, "ymax": 10, "image_width": 20, "image_height": 20},
        {"xmin": 1, "ymin": 1, "xmax": 21, "ymax": 10, "image_width": 20, "image_height": 20},
    ],
)
def test_invalid_box_fails(kwargs):
    with pytest.raises(CropValidationError):
        calculate_square_crop(**kwargs)


def test_png_generation_is_rgb_224_and_deterministic(tmp_path: Path):
    source = tmp_path / "source.png"
    pixels = bytes((index % 256 for index in range(100 * 80 * 3)))
    Image.frombytes("RGB", (100, 80), pixels).save(source)
    geometry = calculate_square_crop(
        xmin=1, ymin=1, xmax=30, ymax=20, image_width=100, image_height=80
    )
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    first_hash = generate_crop_png(source, first, geometry)
    second_hash = generate_crop_png(source, second, geometry)
    assert first_hash == second_hash == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (224, 224)


def test_generation_rejects_geometry_outside_decoded_source(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20)).save(source)
    geometry = calculate_square_crop(
        xmin=1, ymin=1, xmax=10, ymax=10, image_width=100, image_height=100
    )
    with pytest.raises(CropValidationError):
        generate_crop_png(source, tmp_path / "crop.png", geometry)
