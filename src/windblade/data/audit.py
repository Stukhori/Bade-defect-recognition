"""Deterministic WTBD audit helpers and shared contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from windblade.config import ResolvedConfig, save_resolved_config
from windblade.data.duplicates import (
    decoded_pixel_digest,
    dhash64,
    duplicate_groups,
    format_dhash,
    near_duplicate_candidates,
    sha256_file,
)
from windblade.data.visualization import (
    create_annotation_contact_sheets,
    create_descriptive_plots,
    create_high_overlap_sheet,
    create_near_duplicate_sheets,
)
from windblade.data.voc import (
    BoundingBox,
    VocAnnotation,
    VocParseError,
    inclusive_iou,
    parse_voc_xml,
    validate_bounding_box,
)
from windblade.results import read_json, write_json


EXPECTED_CLASS_COUNTS: dict[str, int] = {
    "craze": 259,
    "corrosion": 254,
    "surface_injure": 394,
    "thunderstrike": 92,
    "crack": 224,
    "hide_craze": 345,
}
CANONICAL_CLASSES = tuple(EXPECTED_CLASS_COUNTS)
VALID_SPLITS = ("train", "validation", "test")


class SplitFormatError(ValueError):
    """Raised when the official split file cannot be interpreted unambiguously."""


def natural_id_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def canonicalize_label(raw_label: str) -> str | None:
    """Map only unambiguous capitalization variants to a frozen canonical label."""

    by_casefold = {label.casefold(): label for label in CANONICAL_CLASSES}
    return by_casefold.get(raw_label.casefold())


def pair_image_and_annotation_ids(
    image_filenames: Iterable[str],
    annotation_filenames: Iterable[str],
) -> dict[str, Any]:
    image_ids: dict[str, list[str]] = defaultdict(list)
    annotation_ids: dict[str, list[str]] = defaultdict(list)
    for filename in image_filenames:
        image_ids[Path(filename).stem].append(filename)
    for filename in annotation_filenames:
        annotation_ids[Path(filename).stem].append(filename)
    image_set = set(image_ids)
    annotation_set = set(annotation_ids)
    return {
        "matched_ids": sorted(image_set & annotation_set, key=natural_id_key),
        "images_without_xml": sorted(image_set - annotation_set, key=natural_id_key),
        "xml_without_images": sorted(annotation_set - image_set, key=natural_id_key),
        "duplicate_image_ids": {
            key: sorted(values) for key, values in image_ids.items() if len(values) > 1
        },
        "duplicate_annotation_ids": {
            key: sorted(values) for key, values in annotation_ids.items() if len(values) > 1
        },
    }


def parse_official_split(path: str | Path) -> list[dict[str, str]]:
    """Parse the observed ImageID,Subset CSV syntax without inventing IDs."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ImageID", "Subset"]:
            raise SplitFormatError(
                f"expected split header ['ImageID', 'Subset']; found {reader.fieldnames!r}"
            )
        records: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("ImageID") or "").strip()
            raw_split = (row.get("Subset") or "").strip()
            if not filename or not raw_split:
                raise SplitFormatError(f"empty split field at line {line_number}")
            normalized = "validation" if raw_split == "val" else raw_split
            if normalized not in VALID_SPLITS:
                raise SplitFormatError(f"unknown split {raw_split!r} at line {line_number}")
            records.append(
                {
                    "source_image_id": Path(filename).stem,
                    "image_filename": filename,
                    "split": normalized,
                    "raw_split": raw_split,
                }
            )
    return records


def validate_split_records(
    records: Sequence[Mapping[str, str]],
    known_image_ids: Iterable[str],
) -> dict[str, Any]:
    known = set(known_image_ids)
    memberships: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    for record in records:
        image_id = str(record["source_image_id"])
        memberships[image_id].add(str(record["split"]))
        occurrences[image_id] += 1
    referenced = set(memberships)
    return {
        "unknown_ids": sorted(referenced - known, key=natural_id_key),
        "omitted_ids": sorted(known - referenced, key=natural_id_key),
        "duplicate_ids": sorted(
            (image_id for image_id, count in occurrences.items() if count > 1),
            key=natural_id_key,
        ),
        "overlap_ids": sorted(
            (image_id for image_id, values in memberships.items() if len(values) > 1),
            key=natural_id_key,
        ),
        "counts": {
            split: sum(1 for values in memberships.values() if values == {split})
            for split in VALID_SPLITS
        },
    }


def deterministic_csv_text(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return buffer.getvalue()


def write_deterministic_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(deterministic_csv_text(rows, fieldnames), encoding="utf-8", newline="")


def calculate_dataset_fingerprint(checksum_rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = sorted(
        (
            str(row["relative_path"]).replace("\\", "/"),
            str(row["sha256"]),
        )
        for row in checksum_rows
    )
    serialized = "".join(f"{path}\t{digest}\n" for path, digest in normalized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "min", "p05", "p25", "median", "p75", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


@dataclass(frozen=True)
class DatasetAuditResult:
    summary_path: Path
    documentation_path: Path
    status: str
    critical_errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _resolve_from_repository(repository_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repository_root / path


def _checksum_manifest(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(raw_root).as_posix().casefold(),
    ):
        rows.append(
            {
                "relative_path": path.relative_to(raw_root).as_posix(),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _group_details(
    groups: Sequence[Sequence[str]],
    labels_by_image: Mapping[str, set[str]],
    split_by_image: Mapping[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "image_ids": list(group),
            "labels": {image_id: sorted(labels_by_image.get(image_id, set())) for image_id in group},
            "splits": {image_id: split_by_image.get(image_id) for image_id in group},
        }
        for group in groups
    ]


def _bbox_statistics_rows(instances: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [row for row in instances if not row["bbox_issues"]]
    scopes = [("all", valid)] + [
        (
            label,
            [row for row in valid if row.get("canonical_label_if_unambiguous") == label],
        )
        for label in CANONICAL_CLASSES
    ]
    metrics = (
        "bbox_width",
        "bbox_height",
        "bbox_area_pixels",
        "bbox_area_fraction",
        "bbox_aspect_ratio",
    )
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    for scope, scoped in scopes:
        nested[scope] = {}
        for metric in metrics:
            summary = percentile_summary([float(row[metric]) for row in scoped])
            nested[scope][metric] = summary
            rows.append({"scope": scope, "metric": metric, **summary})
    return rows, nested


def _diagnostic_bbox_rows(
    instances: Sequence[Mapping[str, Any]],
    *,
    elongated_aspect_ratio_low: float,
    elongated_aspect_ratio_high: float,
    almost_full_area_fraction: float,
) -> list[dict[str, Any]]:
    valid = [row for row in instances if not row["bbox_issues"]]
    conditions = {
        "width_lt_16": lambda row: float(row["bbox_width"]) < 16,
        "height_lt_16": lambda row: float(row["bbox_height"]) < 16,
        "width_lt_32": lambda row: float(row["bbox_width"]) < 32,
        "height_lt_32": lambda row: float(row["bbox_height"]) < 32,
        "area_fraction_lt_0_001": lambda row: float(row["bbox_area_fraction"]) < 0.001,
        "area_fraction_lt_0_005": lambda row: float(row["bbox_area_fraction"]) < 0.005,
        "area_fraction_lt_0_01": lambda row: float(row["bbox_area_fraction"]) < 0.01,
        "aspect_ratio_lt_diagnostic_low": lambda row: float(row["bbox_aspect_ratio"])
        < elongated_aspect_ratio_low,
        "aspect_ratio_gt_diagnostic_high": lambda row: float(row["bbox_aspect_ratio"])
        > elongated_aspect_ratio_high,
        "area_fraction_gt_almost_full": lambda row: float(row["bbox_area_fraction"])
        > almost_full_area_fraction,
    }
    rows: list[dict[str, Any]] = []
    for scope in ("all", *CANONICAL_CLASSES):
        scoped = (
            valid
            if scope == "all"
            else [row for row in valid if row.get("canonical_label_if_unambiguous") == scope]
        )
        row: dict[str, Any] = {"class": scope, "valid_instance_count": len(scoped)}
        row.update({name: sum(1 for item in scoped if condition(item)) for name, condition in conditions.items()})
        rows.append(row)
    return rows


def _render_audit_document(
    *,
    summary: Mapping[str, Any],
    source: Mapping[str, Any],
    class_count_rows: Sequence[Mapping[str, Any]],
    split_count_rows: Sequence[Mapping[str, Any]],
    raw_root_entries: Sequence[str],
    metadata_root: Path,
    figures_root: Path,
) -> str:
    image_stats = summary["images"]
    annotation_stats = summary["annotations"]
    split = summary["split"]
    duplicates = summary["duplicates"]
    objects = summary["objects_per_image"]
    bbox = summary["bbox"]
    overlap = bbox["overlap"]
    tiny = bbox["very_small_counts"]["all"]
    total_instances = annotation_stats["instance_count"]

    class_lines = ["| Class | Actual | Expected | Difference | Match |", "|---|---:|---:|---:|:---:|"]
    class_lines.extend(
        f"| {row['class']} | {row['actual_count']} | {row['published_expected_count']} | "
        f"{row['difference']} | {row['matches_expected']} |"
        for row in class_count_rows
    )
    split_totals = {
        split_name: sum(
            int(row["instance_count"]) for row in split_count_rows if row["split"] == split_name
        )
        for split_name in VALID_SPLITS
    }
    split_lines = ["| Split | Source images | Proportion | Instances |", "|---|---:|---:|---:|"]
    total_images = image_stats["count"] or 1
    for split_name in VALID_SPLITS:
        count = split[f"{split_name}_images"]
        split_lines.append(
            f"| {split_name} | {count} | {count / total_images:.4f} | {split_totals[split_name]} |"
        )
    critical_lines = [f"- {item}" for item in summary["audit"]["critical_errors"]] or ["- None."]
    warning_lines = [f"- {item}" for item in summary["audit"]["warnings"]] or ["- None."]
    global_area = bbox["statistics"]["all"]["bbox_area_fraction"]
    global_aspect = bbox["statistics"]["all"]["bbox_aspect_ratio"]

    return f"""# Phase 2 — WTBD Dataset Audit

## 1. Source and provenance

- Dataset: WTBD — Wind Turbine Blade Defect dataset.
- Official source: Springer Nature Figshare, DOI `{source['dataset_doi']}` (version {source['dataset_version']}).
- Article DOI: `{source['article_doi']}`.
- License: {source['license']}.
- Acquisition: {source['acquisition_method']} on {source['acquisition_date_utc']}.
- Archive: `{source['original_download_filename']}`, {source['archive_byte_size']} bytes.
- Archive SHA-256: `{source['archive_sha256']}`.
- Dataset fingerprint: `{summary['dataset_fingerprint']}`.

## 2. Published expectations

The audit tested 1,065 JPEG images, 1,065 primary PASCAL VOC XML files, 1,568 primary objects, six supplied categories, 1024×1024 resolution, and the published per-class counts. It did not treat the separate second-annotator XML directory as additional primary objects.

## 3. Raw file structure

The official archive extracted one top-level directory. Its root entries are:

{chr(10).join(f'- `{entry}`' for entry in raw_root_entries)}

The release includes `Annotations/`, `annotation_second_person/`, `JPEGImages/`, the official split CSV, class definitions, upstream scripts, requirements, and a supplied feature-visualization image. All raw files remain unmodified and Git-ignored. Checksums are in `data/metadata/wtbd/raw_file_checksums.csv`.

## 4. Image integrity

- JPEG files: {image_stats['count']}.
- Successfully decoded: {image_stats['readable']}.
- Failed decoding: {image_stats['unreadable']}.
- Zero-byte images: {image_stats['zero_byte']}.
- Resolution counts: `{json.dumps(image_stats['resolution_counts'], sort_keys=True)}`.
- XML/image dimension disagreements: {image_stats['xml_dimension_mismatch_count']}.

Every image was decoded with Pillow; dimensions, mode, byte checksum, decoded-pixel checksum, and dHash were recorded rather than trusted from filenames.

## 5. Annotation integrity

- Primary XML files: {annotation_stats['xml_count']}.
- Second-annotator XML files: {annotation_stats['second_annotator_xml_count']}.
- Parsed primary objects: {annotation_stats['instance_count']}.
- Invalid bounding boxes: {annotation_stats['invalid_bbox_count']}.
- Images without primary XML: {len(annotation_stats['unmatched_images'])}.
- Primary XML without images: {len(annotation_stats['unmatched_xml'])}.
- XML parse failures: {annotation_stats['parse_error_count']}.

Geometry is summarized with inclusive PASCAL VOC coordinates because the supplied `calculate_kappa.py` explicitly uses `+1` widths and areas. The supplied t-SNE script instead uses exclusive Python slices; that inconsistency is an upstream reference, not a Phase 3 preprocessing decision.

## 6. Label taxonomy

Exact raw-label counts are recorded in `raw_label_counts.csv`. Canonicalization is limited to unambiguous capitalization variants; raw XML is unchanged.

{chr(10).join(class_lines)}

## 7. Class distribution

The six classes are imbalanced; `thunderstrike` has the fewest supplied annotations. This is a descriptive dataset property and does not imply physical severity or model difficulty. See `class_counts.csv` and `figures/phase2/class_distribution.png`.

## 8. Source-image/object structure

- Mean objects/image: {objects['mean']:.6f}.
- Median: {objects['median']:.6f}; minimum: {objects['minimum']}; maximum: {objects['maximum']}.
- Exactly 1 object: {objects['exactly_1']}; 2: {objects['exactly_2']}; 3: {objects['exactly_3']}; 4+: {objects['four_or_more']}.
- Images with multiple defect classes: {objects['multiple_class_images']}.
- Images with repeated instances of the same class: {objects['multiple_same_class_images']}.

The source-image co-occurrence matrix is in `class_cooccurrence.csv`. Co-occurrence is not interpreted causally.

## 9. Bounding-box characteristics

For area fraction, min={global_area['min']:.8f}, p05={global_area['p05']:.8f}, median={global_area['median']:.8f}, p95={global_area['p95']:.8f}, max={global_area['max']:.8f}. For aspect ratio, min={global_aspect['min']:.6f}, median={global_aspect['median']:.6f}, max={global_aspect['max']:.6f}.

Diagnostic counts (not exclusion rules): width <16: {tiny['width_lt_16']}; height <16: {tiny['height_lt_16']}; width <32: {tiny['width_lt_32']}; height <32: {tiny['height_lt_32']}; area fraction <0.1%: {tiny['area_fraction_lt_0_001']}; <0.5%: {tiny['area_fraction_lt_0_005']}; <1%: {tiny['area_fraction_lt_0_01']}; aspect ratio <0.1: {tiny['aspect_ratio_lt_diagnostic_low']}; aspect ratio >10: {tiny['aspect_ratio_gt_diagnostic_high']}; area fraction >50%: {tiny['area_fraction_gt_almost_full']}.

Within-image box pairs: {overlap['total_pairs']}; IoU >0: {overlap['iou_gt_0']}; IoU ≥0.25: {overlap['iou_ge_0_25']}; IoU ≥0.50: {overlap['iou_ge_0_50']}; IoU ≥0.75: {overlap['iou_ge_0_75']}. No box was filtered or merged.

## 10. Official split

- Source: `{split['source']}`.
- Raw format: `{split['format']}`.
- SHA-256: `{split['sha256']}`.

{chr(10).join(split_lines)}

Overlap IDs: {split['overlap_count']}; duplicate rows: {split['duplicate_id_count']}; omitted IDs: {split['omitted_count']}; unknown IDs: {split['unknown_count']}.

Future crops such as `123_0`, `123_1`, and `123_2` must all inherit image `123`'s official split. Individual crops must never be randomly split.

## 11. Duplicate audit

- Exact file-duplicate groups: {duplicates['exact_file_groups']}.
- Exact decoded-pixel groups: {duplicates['exact_pixel_groups']}.
- Cross-split exact duplicate pairs: {duplicates['cross_split_exact_duplicates']}.
- Non-exact dHash candidates at distance ≤{duplicates['near_duplicate_distance_threshold']}: {duplicates['near_duplicate_candidates']}.
- Cross-split non-exact candidates: {duplicates['cross_split_near_duplicate_candidates']}.

Perceptual matches are **candidate near duplicates requiring review**, not established duplicates. Pair metadata are in `duplicate_candidates.csv`; visual sheets prioritize cross-split candidates.

## 12. Visual annotation review

- Per-class annotation contact sheets: `{figures_root.relative_to(figures_root.parents[1]).as_posix()}/annotation_examples/`.
- Near-duplicate review sheets: `{figures_root.relative_to(figures_root.parents[1]).as_posix()}/near_duplicates/`.
- High-overlap review: `{figures_root.relative_to(figures_root.parents[1]).as_posix()}/high_overlap/`.
- Dataset plots: `{figures_root.relative_to(figures_root.parents[1]).as_posix()}/`.

Sampling is deterministic with the Phase 2 visualization seed and is for quality control only. No audit crop was added to a model-ready dataset.

## 13. Implications for Crop-Based Classification

WTBD supplies {total_instances} potential future defect crops across {image_stats['count']} source images. Crop-relevant concerns for Phase 3 include variable box area/aspect ratio, {tiny['area_fraction_lt_0_01']} boxes below 1% of image area, multiple objects/classes in some source images, {overlap['iou_gt_0']} overlapping pairs, and split-level class imbalance. All crops from one source image must remain grouped. These facts neither validate nor invalidate crop classification and do not determine a context margin.

## Upstream preprocessing reference

The included `preprocessing_demo.py` reads common image extensions with OpenCV, resizes to 1024×1024 with `cv2.INTER_AREA`, and writes new files; it states no normalization. The included t-SNE code extracts annotation ROIs, resizes them to 64×128, and computes HOG/LBP features. None of these choices were executed or adopted for this project. Phase 3 retains authority over crop and preprocessing design.

## 14. Critical errors

{chr(10).join(critical_lines)}

## 15. Warnings

{chr(10).join(warning_lines)}

## 16. Phase 2 exit-gate status

**{summary['audit']['status'].upper()}**. All counts and conclusions in this document were generated from the machine-readable audit, not manually transcribed. No model was trained, no model weights were downloaded, no final classification crops were created, and no Phase 3 preprocessing choice was frozen.
"""


def run_wtbd_audit(config: ResolvedConfig, repository_root: str | Path) -> DatasetAuditResult:
    """Run the complete read-only WTBD forensic audit and write derived evidence."""

    root = Path(repository_root).resolve()
    config_data = config.as_dict()
    dataset_config = config_data["dataset"]
    audit_config = config_data.get("audit")
    if not isinstance(audit_config, Mapping):
        raise ValueError("dataset audit configuration requires an 'audit' mapping")
    if dataset_config["name"] != "wtbd":
        raise ValueError("dataset audit requires dataset.name = wtbd")
    if config_data["training"]["epochs"] != 0:
        raise ValueError("Phase 2 audit must not train; training.epochs must be 0")

    raw_root = _resolve_from_repository(root, dataset_config["raw_root"])
    release_root = _resolve_from_repository(root, dataset_config["release_root"])
    image_directory = release_root / dataset_config["image_directory"]
    annotation_directory = release_root / dataset_config["annotation_directory"]
    second_annotation_directory = release_root / dataset_config["second_annotation_directory"]
    split_path = release_root / dataset_config["split_file"]
    metadata_root = _resolve_from_repository(root, dataset_config["metadata_root"])
    figures_root = _resolve_from_repository(root, audit_config["figures_root"])
    documentation_path = root / "docs" / "phase2_dataset_audit.md"
    metadata_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, metadata_root / "audit_resolved_config.yaml")

    source_path = metadata_root / "source.json"
    if not source_path.is_file():
        raise FileNotFoundError("source.json is missing; official acquisition must run first")
    source = read_json(source_path)

    checksum_rows = _checksum_manifest(raw_root)
    write_deterministic_csv(
        metadata_root / "raw_file_checksums.csv",
        checksum_rows,
        ["relative_path", "file_size_bytes", "sha256"],
    )
    dataset_fingerprint = calculate_dataset_fingerprint(checksum_rows)
    checksum_by_relative = {row["relative_path"]: row for row in checksum_rows}

    image_paths = sorted(
        (
            path
            for path in image_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
        ),
        key=lambda path: natural_id_key(path.stem),
    )
    xml_paths = sorted(annotation_directory.glob("*.xml"), key=lambda path: natural_id_key(path.stem))
    second_xml_paths = sorted(
        second_annotation_directory.glob("*.xml"), key=lambda path: natural_id_key(path.stem)
    )
    pairing = pair_image_and_annotation_ids(
        [path.name for path in image_paths], [path.name for path in xml_paths]
    )

    annotations: dict[str, VocAnnotation] = {}
    parse_errors: dict[str, str] = {}
    xml_filename_mismatches: list[dict[str, str]] = []
    for xml_path in xml_paths:
        try:
            annotation = parse_voc_xml(xml_path)
        except VocParseError as exc:
            parse_errors[xml_path.name] = str(exc)
            continue
        annotations[xml_path.stem] = annotation
        if Path(annotation.filename).stem != xml_path.stem:
            xml_filename_mismatches.append(
                {
                    "xml": xml_path.name,
                    "declared_filename": annotation.filename,
                }
            )

    split_records = parse_official_split(split_path)
    split_validation = validate_split_records(split_records, [path.stem for path in image_paths])
    split_by_image: dict[str, str] = {}
    for record in split_records:
        split_by_image.setdefault(record["source_image_id"], record["split"])
    split_membership_rows = [
        {"source_image_id": image_id, "split": split_by_image[image_id]}
        for image_id in sorted(split_by_image, key=natural_id_key)
    ]
    write_deterministic_csv(
        metadata_root / "split_membership.csv",
        split_membership_rows,
        ["source_image_id", "split"],
    )

    image_rows: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    image_path_by_id: dict[str, Path] = {}
    resolution_counts: Counter[str] = Counter()
    decode_errors: dict[str, str] = {}
    zero_byte_images: list[str] = []
    xml_dimension_mismatches: list[str] = []
    for image_path in image_paths:
        image_id = image_path.stem
        image_path_by_id.setdefault(image_id, image_path)
        size_bytes = image_path.stat().st_size
        if size_bytes == 0:
            zero_byte_images.append(image_path.name)
        decoded = False
        width: int | None = None
        height: int | None = None
        mode: str | None = None
        pixel_sha256: str | None = None
        perceptual_hash: int | None = None
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
            pixel_sha256 = decoded_pixel_digest(image_path)
            perceptual_hash = dhash64(image_path)
            decoded = True
            resolution_counts[f"{width}x{height}"] += 1
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            decode_errors[image_path.name] = f"{type(exc).__name__}: {exc}"

        annotation = annotations.get(image_id)
        if decoded and annotation is not None and (annotation.width != width or annotation.height != height):
            xml_dimension_mismatches.append(image_id)
        relative = image_path.relative_to(raw_root).as_posix()
        file_sha256 = checksum_by_relative[relative]["sha256"]
        image_row = {
            "source_image_id": image_id,
            "filename": image_path.name,
            "extension": image_path.suffix,
            "file_size_bytes": size_bytes,
            "sha256": file_sha256,
            "decoded_width": width,
            "decoded_height": height,
            "image_mode": mode,
            "decoding_succeeded": decoded,
            "decoding_error": decode_errors.get(image_path.name),
            "number_of_annotated_objects": len(annotation.objects) if annotation else None,
            "official_split": split_by_image.get(image_id),
            "xml_width": annotation.width if annotation else None,
            "xml_height": annotation.height if annotation else None,
            "xml_dimensions_match": (
                annotation.width == width and annotation.height == height
                if decoded and annotation is not None
                else None
            ),
            "pixel_sha256": pixel_sha256,
            "dhash64": format_dhash(perceptual_hash) if perceptual_hash is not None else None,
        }
        image_rows.append(image_row)
        if decoded and pixel_sha256 is not None and perceptual_hash is not None:
            duplicate_records.append(
                {
                    **image_row,
                    "dhash_int": perceptual_hash,
                }
            )

    write_deterministic_csv(
        metadata_root / "images.csv",
        image_rows,
        list(image_rows[0].keys()) if image_rows else ["source_image_id"],
    )

    instances: list[dict[str, Any]] = []
    raw_label_counts: Counter[str] = Counter()
    invalid_boxes: list[dict[str, Any]] = []
    instances_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id in sorted(annotations, key=natural_id_key):
        annotation = annotations[image_id]
        image_row = next((row for row in image_rows if row["source_image_id"] == image_id), None)
        image_width = int(image_row["decoded_width"]) if image_row and image_row["decoded_width"] else annotation.width
        image_height = int(image_row["decoded_height"]) if image_row and image_row["decoded_height"] else annotation.height
        for object_index, obj in enumerate(annotation.objects):
            raw_label_counts[obj.raw_label] += 1
            canonical = canonicalize_label(obj.raw_label)
            issues = validate_bounding_box(obj.bbox, image_width, image_height)
            width = obj.bbox.width_inclusive
            height = obj.bbox.height_inclusive
            area = obj.bbox.area_inclusive
            area_fraction = area / float(image_width * image_height)
            instance = {
                "instance_id": f"{image_id}_{object_index}",
                "source_image_id": image_id,
                "image_filename": annotation.filename,
                "object_index": object_index,
                "raw_label": obj.raw_label,
                "canonical_label_if_unambiguous": canonical,
                "xmin": obj.bbox.xmin,
                "ymin": obj.bbox.ymin,
                "xmax": obj.bbox.xmax,
                "ymax": obj.bbox.ymax,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area_pixels": area,
                "bbox_area_fraction": area_fraction,
                "bbox_aspect_ratio": width / height if height else None,
                "bbox_center_x": obj.bbox.center_x,
                "bbox_center_y": obj.bbox.center_y,
                "official_split": split_by_image.get(image_id),
                "pose": obj.pose,
                "truncated": obj.truncated,
                "difficult": obj.difficult,
                "bbox_issues": ";".join(issues),
            }
            instances.append(instance)
            instances_by_image[image_id].append(instance)
            if issues:
                invalid_boxes.append({"instance_id": instance["instance_id"], "issues": list(issues)})

    write_deterministic_csv(
        metadata_root / "instances.csv",
        instances,
        list(instances[0].keys()) if instances else ["instance_id"],
    )
    raw_label_rows = [
        {
            "raw_label": raw_label,
            "canonical_label_if_unambiguous": canonicalize_label(raw_label),
            "instance_count": count,
        }
        for raw_label, count in sorted(raw_label_counts.items(), key=lambda item: (item[0].casefold(), item[0]))
    ]
    write_deterministic_csv(
        metadata_root / "raw_label_counts.csv",
        raw_label_rows,
        ["raw_label", "canonical_label_if_unambiguous", "instance_count"],
    )

    canonical_counts = Counter(
        row["canonical_label_if_unambiguous"]
        for row in instances
        if row["canonical_label_if_unambiguous"] is not None
    )
    class_count_rows = [
        {
            "class": label,
            "actual_count": canonical_counts[label],
            "published_expected_count": EXPECTED_CLASS_COUNTS[label],
            "difference": canonical_counts[label] - EXPECTED_CLASS_COUNTS[label],
            "matches_expected": canonical_counts[label] == EXPECTED_CLASS_COUNTS[label],
        }
        for label in CANONICAL_CLASSES
    ]
    write_deterministic_csv(
        metadata_root / "class_counts.csv",
        class_count_rows,
        ["class", "actual_count", "published_expected_count", "difference", "matches_expected"],
    )

    labels_by_image: dict[str, set[str]] = {
        image_id: {
            str(row["canonical_label_if_unambiguous"])
            for row in rows
            if row["canonical_label_if_unambiguous"] is not None
        }
        for image_id, rows in instances_by_image.items()
    }
    object_counts = [len(instances_by_image.get(path.stem, [])) for path in image_paths]
    multiple_class_images = sum(1 for labels in labels_by_image.values() if len(labels) > 1)
    multiple_same_class_images = 0
    for rows in instances_by_image.values():
        counts = Counter(row["canonical_label_if_unambiguous"] for row in rows)
        if any(count > 1 for label, count in counts.items() if label is not None):
            multiple_same_class_images += 1
    objects_statistics = {
        "mean": statistics.fmean(object_counts) if object_counts else 0.0,
        "median": statistics.median(object_counts) if object_counts else 0.0,
        "minimum": min(object_counts) if object_counts else 0,
        "maximum": max(object_counts) if object_counts else 0,
        "exactly_1": object_counts.count(1),
        "exactly_2": object_counts.count(2),
        "exactly_3": object_counts.count(3),
        "four_or_more": sum(1 for count in object_counts if count >= 4),
        "multiple_class_images": multiple_class_images,
        "multiple_same_class_images": multiple_same_class_images,
    }

    cooccurrence = np.zeros((len(CANONICAL_CLASSES), len(CANONICAL_CLASSES)), dtype=np.int64)
    for labels in labels_by_image.values():
        for first_index, first in enumerate(CANONICAL_CLASSES):
            if first not in labels:
                continue
            for second_index, second in enumerate(CANONICAL_CLASSES):
                if second in labels:
                    cooccurrence[first_index, second_index] += 1
    cooccurrence_rows = [
        {"class": label, **{other: int(cooccurrence[index, other_index]) for other_index, other in enumerate(CANONICAL_CLASSES)}}
        for index, label in enumerate(CANONICAL_CLASSES)
    ]
    write_deterministic_csv(
        metadata_root / "class_cooccurrence.csv",
        cooccurrence_rows,
        ["class", *CANONICAL_CLASSES],
    )

    bbox_rows, bbox_nested = _bbox_statistics_rows(instances)
    write_deterministic_csv(
        metadata_root / "bbox_statistics.csv",
        bbox_rows,
        ["scope", "metric", "count", "min", "p05", "p25", "median", "p75", "p95", "max"],
    )
    very_small_rows = _diagnostic_bbox_rows(
        instances,
        elongated_aspect_ratio_low=float(audit_config["elongated_aspect_ratio_low"]),
        elongated_aspect_ratio_high=float(audit_config["elongated_aspect_ratio_high"]),
        almost_full_area_fraction=float(audit_config["almost_full_area_fraction"]),
    )
    write_deterministic_csv(
        metadata_root / "very_small_bbox_counts.csv",
        very_small_rows,
        list(very_small_rows[0].keys()),
    )

    overlap_rows: list[dict[str, Any]] = []
    total_within_image_pairs = 0
    for image_id in sorted(instances_by_image, key=natural_id_key):
        rows = [row for row in instances_by_image[image_id] if not row["bbox_issues"]]
        for first, second in itertools.combinations(rows, 2):
            total_within_image_pairs += 1
            first_box = BoundingBox(first["xmin"], first["ymin"], first["xmax"], first["ymax"])
            second_box = BoundingBox(second["xmin"], second["ymin"], second["xmax"], second["ymax"])
            iou = inclusive_iou(first_box, second_box)
            if iou > 0:
                overlap_rows.append(
                    {
                        "source_image_id": image_id,
                        "instance_a": first["instance_id"],
                        "instance_b": second["instance_id"],
                        "label_a": first["raw_label"],
                        "label_b": second["raw_label"],
                        "official_split": split_by_image.get(image_id),
                        "iou": iou,
                    }
                )
    overlap_rows.sort(key=lambda row: (-float(row["iou"]), natural_id_key(str(row["source_image_id"]))))
    write_deterministic_csv(
        metadata_root / "overlap_pairs.csv",
        overlap_rows,
        ["source_image_id", "instance_a", "instance_b", "label_a", "label_b", "official_split", "iou"],
    )
    overlap_statistics = {
        "total_pairs": total_within_image_pairs,
        "iou_gt_0": len(overlap_rows),
        "iou_ge_0_25": sum(float(row["iou"]) >= 0.25 for row in overlap_rows),
        "iou_ge_0_50": sum(float(row["iou"]) >= 0.50 for row in overlap_rows),
        "iou_ge_0_75": sum(float(row["iou"]) >= 0.75 for row in overlap_rows),
    }

    split_class_rows: list[dict[str, Any]] = []
    for split_name in VALID_SPLITS:
        for label in CANONICAL_CLASSES:
            split_class_rows.append(
                {
                    "split": split_name,
                    "class": label,
                    "source_image_count": sum(
                        1
                        for image_id, labels in labels_by_image.items()
                        if split_by_image.get(image_id) == split_name and label in labels
                    ),
                    "instance_count": sum(
                        1
                        for row in instances
                        if row["official_split"] == split_name
                        and row["canonical_label_if_unambiguous"] == label
                    ),
                }
            )
    write_deterministic_csv(
        metadata_root / "split_class_counts.csv",
        split_class_rows,
        ["split", "class", "source_image_count", "instance_count"],
    )

    file_groups = duplicate_groups(duplicate_records, "sha256")
    pixel_groups = duplicate_groups(duplicate_records, "pixel_sha256")
    candidate_rows = near_duplicate_candidates(
        duplicate_records,
        maximum_distance=int(audit_config["near_duplicate_distance"]),
    )
    write_deterministic_csv(
        metadata_root / "duplicate_candidates.csv",
        candidate_rows,
        [
            "image_a",
            "image_b",
            "exact_duplicate",
            "pixel_duplicate",
            "perceptual_distance",
            "split_a",
            "split_b",
            "cross_split",
        ],
    )
    nonexact_candidates = [
        row for row in candidate_rows if not row["exact_duplicate"] and not row["pixel_duplicate"]
    ]
    cross_split_exact = [
        row
        for row in candidate_rows
        if row["cross_split"] and (row["exact_duplicate"] or row["pixel_duplicate"])
    ]
    cross_split_nonexact = [row for row in nonexact_candidates if row["cross_split"]]

    upstream_purposes = {
        "calculate_kappa.py": "two-annotator IoU matching and Cohen's kappa calculation; not executed",
        "generate_split.py": "seed-42 source-image split generator; inspected, not executed",
        "preprocessing_demo.py": "1024x1024 OpenCV INTER_AREA resize demonstration; not adopted",
        "tsne_analysis.py": "upstream HOG/LBP t-SNE feature visualization; not executed",
        "train_val_test_split.txt": "official source-image split CSV",
        "class_definitions.txt": "six raw class names",
        "requirements.txt": "upstream script dependency list; not installed",
        "Fig6_Feature_Visualization.png": "upstream supplied feature visualization; preserved",
    }
    upstream_rows: list[dict[str, Any]] = []
    for filename, purpose in upstream_purposes.items():
        path = release_root / filename
        relative = path.relative_to(raw_root).as_posix()
        checksum = checksum_by_relative.get(relative)
        upstream_rows.append(
            {
                "filename": filename,
                "present": path.is_file(),
                "file_size_bytes": checksum["file_size_bytes"] if checksum else None,
                "sha256": checksum["sha256"] if checksum else None,
                "purpose": purpose,
            }
        )
    write_deterministic_csv(
        metadata_root / "upstream_files.csv",
        upstream_rows,
        ["filename", "present", "file_size_bytes", "sha256", "purpose"],
    )

    critical_errors: list[str] = []
    warnings: list[str] = []
    if source.get("repository") != "Springer Nature Figshare":
        critical_errors.append("Dataset provenance does not identify Springer Nature Figshare.")
    configured_archive_path = Path(str(source.get("archive_relative_path", "")))
    archive_path = configured_archive_path if configured_archive_path.is_absolute() else root / configured_archive_path
    try:
        archive_relative_to_raw = archive_path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        archive_relative_to_raw = ""
    if source.get("archive_sha256") != checksum_by_relative.get(archive_relative_to_raw, {}).get("sha256"):
        critical_errors.append("Preserved archive checksum no longer matches source.json.")
    if len(image_paths) != 1065:
        critical_errors.append(f"Image count is {len(image_paths)}, expected 1065.")
    if len(xml_paths) != 1065:
        critical_errors.append(f"Primary XML count is {len(xml_paths)}, expected 1065.")
    if pairing["images_without_xml"] or pairing["xml_without_images"]:
        critical_errors.append("Image/XML pairing is incomplete in one or both directions.")
    if pairing["duplicate_image_ids"] or pairing["duplicate_annotation_ids"]:
        critical_errors.append("Duplicate image or primary annotation IDs were found.")
    if parse_errors:
        critical_errors.append(f"{len(parse_errors)} primary XML files failed parsing.")
    if xml_filename_mismatches:
        critical_errors.append(f"{len(xml_filename_mismatches)} XML filename fields disagree with XML IDs.")
    if decode_errors:
        critical_errors.append(f"{len(decode_errors)} images failed decoding.")
    if zero_byte_images:
        critical_errors.append(f"{len(zero_byte_images)} images are zero bytes.")
    if xml_dimension_mismatches:
        critical_errors.append(f"{len(xml_dimension_mismatches)} images disagree with XML dimensions.")
    if len(instances) != 1568:
        critical_errors.append(f"Primary instance count is {len(instances)}, expected 1568.")
    unexpected_labels = [row for row in raw_label_rows if row["canonical_label_if_unambiguous"] is None]
    if unexpected_labels:
        critical_errors.append(f"{len(unexpected_labels)} raw label strings have no unambiguous canonical mapping.")
    if not all(bool(row["matches_expected"]) for row in class_count_rows):
        critical_errors.append("At least one canonical class count differs from the published expectation.")
    if invalid_boxes:
        critical_errors.append(f"{len(invalid_boxes)} bounding boxes fail geometric validation.")
    if any(split_validation[key] for key in ("unknown_ids", "omitted_ids", "duplicate_ids", "overlap_ids")):
        critical_errors.append("The official split contains unknown, omitted, duplicate, or overlapping IDs.")
    if cross_split_exact:
        critical_errors.append(f"{len(cross_split_exact)} exact file/pixel duplicate pairs cross official splits.")
    if resolution_counts != Counter({"1024x1024": 1065}):
        warnings.append(f"Observed resolution distribution differs from 1065×1024x1024: {dict(resolution_counts)}")
    capitalization_variants = [
        row
        for row in raw_label_rows
        if row["canonical_label_if_unambiguous"] is not None
        and row["raw_label"] != row["canonical_label_if_unambiguous"]
    ]
    if capitalization_variants:
        warnings.append(f"{len(capitalization_variants)} raw label capitalization variants require recorded mapping.")
    if file_groups or pixel_groups:
        warnings.append(
            f"Found {len(file_groups)} exact-file and {len(pixel_groups)} exact-pixel duplicate groups; none were removed."
        )
    if nonexact_candidates:
        warnings.append(
            f"Found {len(nonexact_candidates)} non-exact dHash candidates at distance <= {audit_config['near_duplicate_distance']} requiring human review."
        )
    if cross_split_nonexact:
        warnings.append(f"{len(cross_split_nonexact)} non-exact near-duplicate candidates cross official splits.")
    if very_small_rows[0]["area_fraction_lt_0_01"]:
        warnings.append(
            f"{very_small_rows[0]['area_fraction_lt_0_01']} valid boxes occupy less than 1% of the source image."
        )
    if (
        very_small_rows[0]["aspect_ratio_lt_diagnostic_low"]
        or very_small_rows[0]["aspect_ratio_gt_diagnostic_high"]
    ):
        warnings.append(
            f"Diagnostic elongation flags: {very_small_rows[0]['aspect_ratio_lt_diagnostic_low']} boxes have "
            f"aspect ratio < {audit_config['elongated_aspect_ratio_low']} and "
            f"{very_small_rows[0]['aspect_ratio_gt_diagnostic_high']} exceed "
            f"{audit_config['elongated_aspect_ratio_high']}."
        )
    if very_small_rows[0]["area_fraction_gt_almost_full"]:
        warnings.append(
            f"{very_small_rows[0]['area_fraction_gt_almost_full']} boxes exceed the diagnostic "
            f"area-fraction threshold {audit_config['almost_full_area_fraction']}."
        )
    if overlap_statistics["iou_gt_0"]:
        warnings.append(f"{overlap_statistics['iou_gt_0']} within-image annotation pairs overlap (IoU > 0).")
    if multiple_class_images:
        warnings.append(f"{multiple_class_images} source images contain multiple canonical defect classes.")

    split_file_relative = split_path.relative_to(raw_root).as_posix()
    split_sha256 = checksum_by_relative[split_file_relative]["sha256"]
    duplicate_summary = {
        "exact_file_groups": len(file_groups),
        "exact_pixel_groups": len(pixel_groups),
        "exact_file_group_details": _group_details(file_groups, labels_by_image, split_by_image),
        "exact_pixel_group_details": _group_details(pixel_groups, labels_by_image, split_by_image),
        "cross_split_exact_duplicates": len(cross_split_exact),
        "near_duplicate_distance_threshold": int(audit_config["near_duplicate_distance"]),
        "near_duplicate_candidates": len(nonexact_candidates),
        "cross_split_near_duplicate_candidates": len(cross_split_nonexact),
    }
    status = "pass" if not critical_errors else "incomplete"
    summary = {
        "schema_version": "1.0",
        "dataset": "WTBD",
        "dataset_doi": source["dataset_doi"],
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": config.config_hash,
        "images": {
            "count": len(image_paths),
            "readable": len(image_paths) - len(decode_errors),
            "unreadable": len(decode_errors),
            "zero_byte": len(zero_byte_images),
            "resolution_counts": dict(sorted(resolution_counts.items())),
            "xml_dimension_mismatch_count": len(xml_dimension_mismatches),
        },
        "annotations": {
            "xml_count": len(xml_paths),
            "second_annotator_xml_count": len(second_xml_paths),
            "unmatched_images": pairing["images_without_xml"],
            "unmatched_xml": pairing["xml_without_images"],
            "duplicate_image_ids": pairing["duplicate_image_ids"],
            "duplicate_annotation_ids": pairing["duplicate_annotation_ids"],
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors,
            "xml_filename_mismatches": xml_filename_mismatches,
            "instance_count": len(instances),
            "invalid_bbox_count": len(invalid_boxes),
            "invalid_bboxes": invalid_boxes,
            "coordinate_convention": "PASCAL VOC inclusive (+1 width/height/area)",
        },
        "classes": {
            "raw_labels": {label: raw_label_counts[label] for label in sorted(raw_label_counts)},
            "canonical_counts": {label: canonical_counts[label] for label in CANONICAL_CLASSES},
            "published_counts": EXPECTED_CLASS_COUNTS,
            "expected_counts_match": all(bool(row["matches_expected"]) for row in class_count_rows),
        },
        "split": {
            "source": split_file_relative,
            "format": "CSV with header ImageID,Subset; raw subsets train,val,test",
            "sha256": split_sha256,
            "train_images": split_validation["counts"]["train"],
            "validation_images": split_validation["counts"]["validation"],
            "test_images": split_validation["counts"]["test"],
            "overlap_count": len(split_validation["overlap_ids"]),
            "duplicate_id_count": len(split_validation["duplicate_ids"]),
            "omitted_count": len(split_validation["omitted_ids"]),
            "unknown_count": len(split_validation["unknown_ids"]),
            "details": split_validation,
        },
        "duplicates": duplicate_summary,
        "objects_per_image": objects_statistics,
        "bbox": {
            "statistics": bbox_nested,
            "very_small_counts": {row["class"]: row for row in very_small_rows},
            "diagnostic_thresholds": {
                "elongated_aspect_ratio_low": float(audit_config["elongated_aspect_ratio_low"]),
                "elongated_aspect_ratio_high": float(audit_config["elongated_aspect_ratio_high"]),
                "almost_full_area_fraction": float(audit_config["almost_full_area_fraction"]),
            },
            "overlap": overlap_statistics,
        },
        "upstream": {
            "second_annotation_directory": dataset_config["second_annotation_directory"],
            "preprocessing_reference": {
                "target_size": [1024, 1024],
                "interpolation": "cv2.INTER_AREA",
                "normalization": None,
                "adopted_by_project": False,
            },
            "split_generator": {
                "seed": 42,
                "nominal_proportions": [0.70, 0.15, 0.15],
                "executed_by_project": False,
            },
        },
        "audit": {
            "critical_errors": critical_errors,
            "warnings": warnings,
            "status": status,
        },
    }
    summary_path = metadata_root / "audit_summary.json"
    write_json(summary_path, summary)
    write_deterministic_csv(
        metadata_root / "audit_findings.csv",
        [
            *({"severity": "critical_error", "message": message} for message in critical_errors),
            *({"severity": "warning", "message": message} for message in warnings),
        ],
        ["severity", "message"],
    )

    create_annotation_contact_sheets(
        instances,
        image_path_by_id,
        figures_root / "annotation_examples",
        CANONICAL_CLASSES,
        int(audit_config["annotation_examples_per_class"]),
        int(audit_config["visualization_seed"]),
    )
    create_near_duplicate_sheets(
        candidate_rows,
        image_path_by_id,
        figures_root / "near_duplicates",
        int(audit_config["near_duplicate_pairs_per_sheet"]),
        int(audit_config["near_duplicate_max_pairs"]),
    )
    instances_by_id = {str(row["instance_id"]): row for row in instances}
    create_high_overlap_sheet(
        [row for row in overlap_rows if float(row["iou"]) >= 0.50],
        instances_by_id,
        image_path_by_id,
        figures_root / "high_overlap",
    )
    create_descriptive_plots(
        output_directory=figures_root,
        classes=CANONICAL_CLASSES,
        class_counts=canonical_counts,
        split_class_counts=split_class_rows,
        instances=instances,
        objects_per_image=object_counts,
        cooccurrence_matrix=cooccurrence,
    )

    raw_root_entries = [
        f"{path.name}/" if path.is_dir() else path.name
        for path in sorted(release_root.iterdir(), key=lambda item: item.name.casefold())
    ]
    document = _render_audit_document(
        summary=summary,
        source=source,
        class_count_rows=class_count_rows,
        split_count_rows=split_class_rows,
        raw_root_entries=raw_root_entries,
        metadata_root=metadata_root,
        figures_root=figures_root,
    )
    documentation_path.write_text(document, encoding="utf-8", newline="\n")
    return DatasetAuditResult(
        summary_path=summary_path,
        documentation_path=documentation_path,
        status=status,
        critical_errors=tuple(critical_errors),
        warnings=tuple(warnings),
    )
