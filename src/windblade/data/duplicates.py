"""Exact and deterministic perceptual image duplicate screening."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_digest(path: str | Path) -> str:
    """Hash decoded RGB pixels plus dimensions, ignoring JPEG container metadata."""

    digest = hashlib.sha256()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def dhash64(path: str | Path) -> int:
    """Return a transparent 64-bit horizontal difference hash."""

    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = gray.tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def format_dhash(value: int) -> str:
    return f"{value:016x}"


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def duplicate_groups(records: Iterable[Mapping[str, Any]], field: str) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(str(record["source_image_id"]))
    groups = [sorted(ids, key=_image_id_key) for ids in grouped.values() if len(ids) > 1]
    return sorted(groups, key=lambda group: (_image_id_key(group[0]), len(group), group))


def _image_id_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def near_duplicate_candidates(
    records: list[Mapping[str, Any]],
    maximum_distance: int,
) -> list[dict[str, Any]]:
    """Screen all image pairs and return deterministic review candidates."""

    ordered = sorted(records, key=lambda row: _image_id_key(str(row["source_image_id"])))
    candidates: list[dict[str, Any]] = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            distance = hamming_distance(int(first["dhash_int"]), int(second["dhash_int"]))
            exact_file = first["sha256"] == second["sha256"]
            exact_pixel = first["pixel_sha256"] == second["pixel_sha256"]
            if not exact_file and not exact_pixel and distance > maximum_distance:
                continue
            split_a = first.get("official_split")
            split_b = second.get("official_split")
            candidates.append(
                {
                    "image_a": first["filename"],
                    "image_b": second["filename"],
                    "exact_duplicate": bool(exact_file),
                    "pixel_duplicate": bool(exact_pixel),
                    "perceptual_distance": distance,
                    "split_a": split_a,
                    "split_b": split_b,
                    "cross_split": bool(split_a and split_b and split_a != split_b),
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            row["perceptual_distance"],
            not row["exact_duplicate"],
            not row["pixel_duplicate"],
            not row["cross_split"],
            _image_id_key(Path(str(row["image_a"])).stem),
            _image_id_key(Path(str(row["image_b"])).stem),
        ),
    )
