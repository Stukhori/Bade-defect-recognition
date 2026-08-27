"""Deterministic dataset-audit plots and human-review contact sheets."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BOX_COLORS = ["#ff3b30", "#00a6ff", "#34c759", "#ff9500", "#af52de", "#ff2d55"]


def _seed_for_label(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)


def _place_centered(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    x = left + max(0, (right - left - image.width) // 2)
    y = top + max(0, (bottom - top - image.height) // 2)
    canvas.paste(image, (x, y))


def create_annotation_contact_sheets(
    instances: Sequence[Mapping[str, Any]],
    image_paths: Mapping[str, Path],
    output_directory: str | Path,
    classes: Sequence[str],
    examples_per_class: int,
    seed: int,
) -> list[Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    font = ImageFont.load_default()
    tile_width, tile_height, header_height = 340, 370, 30
    columns = 3

    for class_index, label in enumerate(classes):
        candidates = [row for row in instances if row.get("canonical_label_if_unambiguous") == label]
        candidates.sort(key=lambda row: str(row["instance_id"]))
        rng = random.Random(_seed_for_label(seed, label))
        rng.shuffle(candidates)
        selected = candidates[:examples_per_class]
        rows = max(1, math.ceil(len(selected) / columns))
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        sheet_draw = ImageDraw.Draw(sheet)

        for index, instance in enumerate(selected):
            image_path = image_paths[str(instance["source_image_id"])]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            coordinates = (
                int(instance["xmin"]),
                int(instance["ymin"]),
                int(instance["xmax"]),
                int(instance["ymax"]),
            )
            color = BOX_COLORS[class_index % len(BOX_COLORS)]
            draw.rectangle(coordinates, outline=color, width=6)
            fitted = _fit_image(image, (tile_width - 10, tile_height - header_height - 10))
            column = index % columns
            row_index = index // columns
            tile_left = column * tile_width
            tile_top = row_index * tile_height
            title = f"{instance['source_image_id']} | {instance['raw_label']} | {instance['instance_id']}"
            sheet_draw.text((tile_left + 6, tile_top + 8), title, fill="black", font=font)
            _place_centered(
                sheet,
                fitted,
                (
                    tile_left + 5,
                    tile_top + header_height,
                    tile_left + tile_width - 5,
                    tile_top + tile_height - 5,
                ),
            )

        output = destination / f"{label}.png"
        sheet.save(output, format="PNG")
        output_paths.append(output)
    return output_paths


def create_near_duplicate_sheets(
    candidates: Sequence[Mapping[str, Any]],
    image_paths: Mapping[str, Path],
    output_directory: str | Path,
    pairs_per_sheet: int,
    maximum_pairs: int,
) -> list[Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    prioritized = [row for row in candidates if row.get("cross_split")]
    prioritized.extend(row for row in candidates if not row.get("cross_split"))
    selected = prioritized[:maximum_pairs]
    outputs: list[Path] = []
    font = ImageFont.load_default()
    pair_height, image_width, image_height = 250, 320, 210

    for sheet_index in range(0, len(selected), pairs_per_sheet):
        batch = selected[sheet_index : sheet_index + pairs_per_sheet]
        sheet = Image.new("RGB", (image_width * 2, 40 + pair_height * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 12), "candidate near duplicates requiring review", fill="black", font=font)
        for row_index, candidate in enumerate(batch):
            top = 40 + row_index * pair_height
            first_id = Path(str(candidate["image_a"])).stem
            second_id = Path(str(candidate["image_b"])).stem
            with Image.open(image_paths[first_id]) as first_image:
                first = _fit_image(first_image, (image_width - 10, image_height))
            with Image.open(image_paths[second_id]) as second_image:
                second = _fit_image(second_image, (image_width - 10, image_height))
            _place_centered(sheet, first, (0, top + 34, image_width, top + pair_height))
            _place_centered(sheet, second, (image_width, top + 34, image_width * 2, top + pair_height))
            label = (
                f"{candidate['image_a']} [{candidate['split_a']}] vs "
                f"{candidate['image_b']} [{candidate['split_b']}] | "
                f"d={candidate['perceptual_distance']} | cross={candidate['cross_split']} | "
                f"file_exact={candidate['exact_duplicate']} pixel_exact={candidate['pixel_duplicate']}"
            )
            draw.text((8, top + 10), label, fill="black", font=font)
        output = destination / f"near_duplicate_candidates_{sheet_index // pairs_per_sheet + 1:02d}.png"
        sheet.save(output, format="PNG")
        outputs.append(output)
    return outputs


def create_high_overlap_sheet(
    overlap_pairs: Sequence[Mapping[str, Any]],
    instances_by_id: Mapping[str, Mapping[str, Any]],
    image_paths: Mapping[str, Path],
    output_directory: str | Path,
    maximum_examples: int = 12,
) -> list[Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    selected = sorted(overlap_pairs, key=lambda row: -float(row["iou"]))[:maximum_examples]
    if not selected:
        return []
    columns = 3
    tile_width, tile_height, header = 340, 370, 34
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, pair in enumerate(selected):
        first = instances_by_id[str(pair["instance_a"])]
        second = instances_by_id[str(pair["instance_b"])]
        image_id = str(pair["source_image_id"])
        with Image.open(image_paths[image_id]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for color, instance in zip(BOX_COLORS[:2], (first, second), strict=True):
            draw.rectangle(
                (
                    int(instance["xmin"]),
                    int(instance["ymin"]),
                    int(instance["xmax"]),
                    int(instance["ymax"]),
                ),
                outline=color,
                width=6,
            )
        fitted = _fit_image(image, (tile_width - 10, tile_height - header - 10))
        column = index % columns
        row_index = index // columns
        left, top = column * tile_width, row_index * tile_height
        title = f"{image_id} | IoU={float(pair['iou']):.4f} | {first['raw_label']} / {second['raw_label']}"
        draw_sheet.text((left + 6, top + 8), title, fill="black", font=font)
        _place_centered(sheet, fitted, (left + 5, top + header, left + tile_width - 5, top + tile_height - 5))
    output = destination / "high_overlap_candidates.png"
    sheet.save(output, format="PNG")
    return [output]


def create_descriptive_plots(
    *,
    output_directory: str | Path,
    classes: Sequence[str],
    class_counts: Mapping[str, int],
    split_class_counts: Sequence[Mapping[str, Any]],
    instances: Sequence[Mapping[str, Any]],
    objects_per_image: Sequence[int],
    cooccurrence_matrix: np.ndarray,
) -> list[Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(9, 5))
    values = [class_counts[label] for label in classes]
    axis.bar(classes, values, color="#4472c4")
    axis.set_ylabel("Annotated instances")
    axis.set_title("WTBD class distribution")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    output = destination / "class_distribution.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)

    by_split = {(row["split"], row["class"]): int(row["instance_count"]) for row in split_class_counts}
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25
    for index, split in enumerate(("train", "validation", "test")):
        axis.bar(
            x + (index - 1) * width,
            [by_split.get((split, label), 0) for label in classes],
            width,
            label=split,
        )
    axis.set_xticks(x, classes, rotation=25)
    axis.set_ylabel("Annotated instances")
    axis.set_title("WTBD class counts by official source-image split")
    axis.legend()
    figure.tight_layout()
    output = destination / "class_distribution_by_split.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)

    area_fractions = [float(row["bbox_area_fraction"]) for row in instances if not row["bbox_issues"]]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(area_fractions, bins=40, color="#70ad47", edgecolor="white")
    axis.set_xlabel("Bounding-box area / image area")
    axis.set_ylabel("Instances")
    axis.set_title("WTBD bounding-box area-fraction distribution")
    figure.tight_layout()
    output = destination / "bbox_area_distribution.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)

    class_area = [
        [
            float(row["bbox_area_fraction"])
            for row in instances
            if row.get("canonical_label_if_unambiguous") == label and not row["bbox_issues"]
        ]
        for label in classes
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(class_area, tick_labels=classes, showfliers=False)
    axis.set_ylabel("Bounding-box area / image area")
    axis.set_title("WTBD bounding-box area fraction by class")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    output = destination / "bbox_area_fraction_by_class.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)

    frequency = Counter(objects_per_image)
    figure, axis = plt.subplots(figsize=(8, 5))
    keys = sorted(frequency)
    axis.bar(keys, [frequency[key] for key in keys], color="#ed7d31")
    axis.set_xlabel("Annotated objects per source image")
    axis.set_ylabel("Source images")
    axis.set_title("WTBD objects per image")
    axis.set_xticks(keys)
    figure.tight_layout()
    output = destination / "objects_per_image.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(cooccurrence_matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_title("Source-image class co-occurrence")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, int(cooccurrence_matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Source images")
    figure.tight_layout()
    output = destination / "class_cooccurrence.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    outputs.append(output)
    return outputs
