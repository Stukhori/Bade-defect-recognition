"""Deterministic, padding-free contextual crops for WTBD instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


class CropValidationError(ValueError):
    """Raised when an annotation cannot produce a valid frozen crop."""


@dataclass(frozen=True)
class CropGeometry:
    """Zero-based, half-open pixel coordinates for a square Pillow crop."""

    crop_xmin: int
    crop_ymin: int
    crop_xmax: int
    crop_ymax: int
    crop_side: int
    minimum_side_applied: bool
    boundary_shifted: bool
    max_side_clipped: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_square_crop(
    *,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    image_width: int,
    image_height: int,
    context_multiplier: float = 1.5,
    minimum_side: int = 64,
) -> CropGeometry:
    """Return the frozen square crop around a 1-based inclusive VOC box.

    Input annotation coordinates retain the Phase 2 convention. Returned
    crop coordinates are zero-based and half-open, matching ``PIL.Image.crop``.
    """

    values = (xmin, ymin, xmax, ymax, image_width, image_height, minimum_side)
    if any(not isinstance(value, int) for value in values):
        raise CropValidationError("crop geometry inputs must be integers")
    if image_width <= 0 or image_height <= 0:
        raise CropValidationError("source dimensions must be positive")
    if not math.isfinite(context_multiplier) or context_multiplier <= 0:
        raise CropValidationError("context multiplier must be positive and finite")
    if minimum_side <= 0:
        raise CropValidationError("minimum crop side must be positive")
    if xmin < 1 or ymin < 1 or xmax > image_width or ymax > image_height:
        raise CropValidationError("VOC box lies outside the source image")
    if xmin > xmax or ymin > ymax:
        raise CropValidationError("VOC box has reversed coordinates")

    # Convert 1-based inclusive VOC bounds to zero-based half-open bounds.
    box_left, box_top, box_right, box_bottom = xmin - 1, ymin - 1, xmax, ymax
    box_width = box_right - box_left
    box_height = box_bottom - box_top
    base_side = max(box_width, box_height)
    context_side = math.ceil(context_multiplier * base_side)
    minimum_applied = context_side < minimum_side
    requested_side = max(minimum_side, context_side)
    maximum_side = min(image_width, image_height)
    crop_side = min(requested_side, maximum_side)
    max_side_clipped = requested_side > maximum_side
    if crop_side < box_width or crop_side < box_height:
        raise CropValidationError("largest in-image square cannot contain the annotation")

    center_x = (box_left + box_right) / 2.0
    center_y = (box_top + box_bottom) / 2.0
    centered_left = math.floor(center_x - crop_side / 2.0)
    centered_top = math.floor(center_y - crop_side / 2.0)
    crop_left = min(max(centered_left, 0), image_width - crop_side)
    crop_top = min(max(centered_top, 0), image_height - crop_side)
    crop_right = crop_left + crop_side
    crop_bottom = crop_top + crop_side
    boundary_shifted = crop_left != centered_left or crop_top != centered_top

    if not (
        0 <= crop_left < crop_right <= image_width
        and 0 <= crop_top < crop_bottom <= image_height
        and crop_right - crop_left == crop_bottom - crop_top == crop_side
    ):
        raise CropValidationError("computed crop is not a valid in-image square")
    if not (
        crop_left <= box_left
        and crop_top <= box_top
        and crop_right >= box_right
        and crop_bottom >= box_bottom
    ):
        raise CropValidationError("computed crop does not contain the complete annotation")

    return CropGeometry(
        crop_xmin=crop_left,
        crop_ymin=crop_top,
        crop_xmax=crop_right,
        crop_ymax=crop_bottom,
        crop_side=crop_side,
        minimum_side_applied=minimum_applied,
        boundary_shifted=boundary_shifted,
        max_side_clipped=max_side_clipped,
    )


def generate_crop_png(
    source_path: str | Path,
    output_path: str | Path,
    geometry: CropGeometry,
    *,
    output_size: tuple[int, int] = (224, 224),
) -> str:
    """Generate one deterministic RGB PNG and return its SHA-256 checksum."""

    destination = Path(output_path)
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise CropValidationError("output dimensions must be positive")
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
        if geometry.crop_xmax > source.width or geometry.crop_ymax > source.height:
            raise CropValidationError("crop geometry exceeds decoded source dimensions")
        crop = source.crop(
            (
                geometry.crop_xmin,
                geometry.crop_ymin,
                geometry.crop_xmax,
                geometry.crop_ymax,
            )
        )
        if crop.size != (geometry.crop_side, geometry.crop_side):
            raise CropValidationError("Pillow crop size disagrees with crop geometry")
        resized = crop.resize(output_size, resample=Image.Resampling.BILINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, format="PNG", optimize=False, compress_level=6)
    with Image.open(destination) as check:
        if check.format != "PNG" or check.mode != "RGB" or check.size != output_size:
            raise CropValidationError("generated image failed PNG/RGB/dimension validation")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, Image.Resampling.BILINEAR)


def _qc_sheet(
    rows: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    crop_root: Path,
    output_path: Path,
    title: str,
) -> None:
    tile_width, tile_height, columns = 620, 285, 2
    rows_count = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * tile_width, 36 + rows_count * tile_height), "white")
    draw_canvas = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw_canvas.text((8, 12), title, fill="black", font=font)
    for index, row in enumerate(rows):
        left = (index % columns) * tile_width
        top = 36 + (index // columns) * tile_height
        source_id = str(row["source_image_id"])
        with Image.open(source_paths[source_id]) as opened:
            original = opened.convert("RGB")
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        # PIL drawing uses zero-based display coordinates; convert VOC bounds.
        bbox = (
            int(row["bbox_xmin"]) - 1,
            int(row["bbox_ymin"]) - 1,
            int(row["bbox_xmax"]) - 1,
            int(row["bbox_ymax"]) - 1,
        )
        crop_box = (
            int(row["crop_xmin"]),
            int(row["crop_ymin"]),
            int(row["crop_xmax"]) - 1,
            int(row["crop_ymax"]) - 1,
        )
        draw.rectangle(crop_box, outline="#00a6ff", width=max(2, original.width // 300))
        draw.rectangle(bbox, outline="#ff3b30", width=max(2, original.width // 300))
        fitted = _fit(annotated, (360, 220))
        with Image.open(crop_root / str(row["output_relative_path"])) as opened_crop:
            crop = opened_crop.convert("RGB").copy()
        canvas.paste(fitted, (left + 5, top + 50))
        canvas.paste(crop, (left + 380, top + 50))
        heading = f"{row['canonical_label']} | source={source_id} | instance={row['instance_id']}"
        details = (
            f"bbox={row['bbox_width']}x{row['bbox_height']} | crop={row['crop_side']} | "
            "red=bbox blue=crop"
        )
        draw_canvas.text((left + 6, top + 8), heading, fill="black", font=font)
        draw_canvas.text((left + 6, top + 24), details, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False, compress_level=6)


def create_training_crop_qc(
    manifest_rows: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path],
    crop_root: str | Path,
    output_root: str | Path,
    *,
    labels: Sequence[str],
    seed: int,
    examples_per_class: int = 6,
    diagnostic_examples: int = 12,
) -> list[Path]:
    """Create deterministic, training-only crop QC sheets."""

    train = [row for row in manifest_rows if row["split"] == "train"]
    if len(train) != sum(1 for row in manifest_rows if row["split"] == "train"):
        raise CropValidationError("QC selection must contain training rows only")
    random_rows: list[Mapping[str, Any]] = []
    for class_index, label in enumerate(labels):
        candidates = sorted(
            (row for row in train if row["canonical_label"] == label),
            key=lambda row: str(row["instance_id"]),
        )
        rng = random.Random(seed + class_index)
        rng.shuffle(candidates)
        if len(candidates) < examples_per_class:
            raise CropValidationError(f"not enough training QC examples for class {label}")
        random_rows.extend(candidates[:examples_per_class])

    by_area = lambda row: int(row["bbox_width"]) * int(row["bbox_height"])
    by_elongation = lambda row: max(
        int(row["bbox_width"]) / int(row["bbox_height"]),
        int(row["bbox_height"]) / int(row["bbox_width"]),
    )
    by_boundary = lambda row: min(
        int(row["bbox_xmin"]) - 1,
        int(row["bbox_ymin"]) - 1,
        int(row["source_width"]) - int(row["bbox_xmax"]),
        int(row["source_height"]) - int(row["bbox_ymax"]),
    )
    selections = {
        "random_class_examples.png": (random_rows, "Random training examples: six per class"),
        "smallest_annotations.png": (sorted(train, key=by_area)[:diagnostic_examples], "Smallest training annotations"),
        "largest_annotations.png": (sorted(train, key=by_area, reverse=True)[:diagnostic_examples], "Largest training annotations"),
        "most_elongated_annotations.png": (
            sorted(train, key=by_elongation, reverse=True)[:diagnostic_examples],
            "Most elongated training annotations",
        ),
        "boundary_annotations.png": (
            sorted(train, key=by_boundary)[:diagnostic_examples],
            "Training annotations closest to an image boundary",
        ),
    }
    destination = Path(output_root)
    outputs: list[Path] = []
    for filename, (rows, title) in selections.items():
        output = destination / filename
        _qc_sheet(rows, source_paths, Path(crop_root), output, title)
        outputs.append(output)
    return outputs
