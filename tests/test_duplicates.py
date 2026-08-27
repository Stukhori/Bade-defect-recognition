from __future__ import annotations

from pathlib import Path

from PIL import Image

from windblade.data.duplicates import (
    decoded_pixel_digest,
    dhash64,
    hamming_distance,
    near_duplicate_candidates,
    sha256_file,
)


def make_image(path: Path, changed: bool = False) -> None:
    image = Image.new("RGB", (16, 16), "black")
    for x in range(16):
        for y in range(16):
            value = x * 12 + y * 2
            image.putpixel((x, y), (value, value, value))
    if changed:
        image.putpixel((0, 0), (255, 255, 255))
    image.save(path, format="PNG")


def test_byte_and_pixel_digests_are_deterministic_and_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    same = tmp_path / "same.png"
    changed = tmp_path / "changed.png"
    make_image(first)
    make_image(same)
    make_image(changed, changed=True)

    assert sha256_file(first) == sha256_file(first)
    assert sha256_file(first) == sha256_file(same)
    assert sha256_file(first) != sha256_file(changed)
    assert decoded_pixel_digest(first) == decoded_pixel_digest(same)
    assert decoded_pixel_digest(first) != decoded_pixel_digest(changed)


def test_identical_images_have_zero_dhash_distance(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    make_image(first)
    make_image(second)

    assert hamming_distance(dhash64(first), dhash64(second)) == 0


def test_altered_fixture_hash_is_deterministic(tmp_path: Path) -> None:
    original = tmp_path / "original.png"
    changed = tmp_path / "changed.png"
    make_image(original)
    make_image(changed, changed=True)

    first_distance = hamming_distance(dhash64(original), dhash64(changed))
    second_distance = hamming_distance(dhash64(original), dhash64(changed))
    assert first_distance == second_distance
    assert 0 <= first_distance <= 64


def test_candidate_rows_include_split_and_exact_flags() -> None:
    records = [
        {
            "source_image_id": "0",
            "filename": "0.jpg",
            "sha256": "same",
            "pixel_sha256": "pixels",
            "dhash_int": 0,
            "official_split": "train",
        },
        {
            "source_image_id": "1",
            "filename": "1.jpg",
            "sha256": "same",
            "pixel_sha256": "pixels",
            "dhash_int": 0,
            "official_split": "test",
        },
    ]

    candidates = near_duplicate_candidates(records, maximum_distance=4)

    assert candidates == [
        {
            "image_a": "0.jpg",
            "image_b": "1.jpg",
            "exact_duplicate": True,
            "pixel_duplicate": True,
            "perceptual_distance": 0,
            "split_a": "train",
            "split_b": "test",
            "cross_split": True,
        }
    ]
