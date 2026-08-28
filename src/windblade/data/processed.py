"""Build and validate the versioned Phase 3 WTBD crop dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from windblade.config import ResolvedConfig, canonical_config_json
from windblade.data.crops import (
    CropValidationError,
    calculate_square_crop,
    create_training_crop_qc,
    generate_crop_png,
)
from windblade.utils import atomic_write_text


LABELS = ("craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
EXPECTED_CLASS_COUNTS = {
    "craze": 169,
    "corrosion": 178,
    "surface_injure": 264,
    "thunderstrike": 60,
    "crack": 131,
    "hide_craze": 263,
}
EXPECTED_SOURCE_SPLITS = {"train": 510, "validation": 101, "test": 109}


class ProcessedDatasetError(RuntimeError):
    """Raised when the frozen Phase 3 dataset contract is violated."""


MANIFEST_FIELDS = (
    "instance_id",
    "source_image_id",
    "source_filename",
    "object_index",
    "raw_label",
    "canonical_label",
    "class_id",
    "split",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_width",
    "bbox_height",
    "source_width",
    "source_height",
    "crop_xmin",
    "crop_ymin",
    "crop_xmax",
    "crop_ymax",
    "crop_side",
    "context_multiplier",
    "minimum_crop_side",
    "minimum_side_applied",
    "boundary_shifted",
    "max_side_clipped",
    "output_width",
    "output_height",
    "output_relative_path",
    "source_image_sha256",
    "processed_image_sha256",
    "defect_occupancy",
    "resize_scale_factor",
)


def _natural_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_processed_fingerprint(
    *,
    config_sha256: str,
    curation_manifest_sha256: str,
    manifest_content: str,
    checksum_content: str,
) -> str:
    """Fingerprint the complete Phase 3 policy, lineage, metadata, and pixels."""

    digest = hashlib.sha256()
    for tag, content in (
        ("phase3_config", config_sha256),
        ("phase2_curation_manifest", curation_manifest_sha256),
        ("crop_manifest", manifest_content),
        ("crop_checksums", checksum_content),
    ):
        digest.update(tag.encode("utf-8") + b"\0" + content.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _repo_path(repository_root: Path, configured: str) -> Path:
    return (repository_root / configured).resolve()


def _load_label_map(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    mapping = {str(row["canonical_label"]): int(row["class_id"]) for row in labels}
    if tuple(label for label, _ in sorted(mapping.items(), key=lambda item: item[1])) != LABELS:
        raise ProcessedDatasetError("classification label map violates the frozen class order")
    if mapping != LABEL_TO_ID:
        raise ProcessedDatasetError("classification label map contains invalid class IDs")
    return mapping


def verify_phase2_inputs(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Verify every frozen upstream invariant before generating crops."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    phase3 = data["crop_dataset"]
    curation = data["curation"]
    summary_path = _repo_path(root, curation["summary_file"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    interpreted = summary["curated_interpretation"]
    expected_fingerprint = phase3["expected_raw_fingerprint"]
    failures: list[str] = []
    if summary.get("status") != "PASS" or summary.get("blockers"):
        failures.append("Phase 2 strict curated status is not PASS")
    if summary.get("source_dataset_fingerprint") != expected_fingerprint:
        failures.append("raw dataset fingerprint changed")
    if summary.get("curation_version") != phase3["curated_version"]:
        failures.append("curation version changed")
    if int(interpreted.get("included_images", -1)) != int(phase3["expected_source_images"]):
        failures.append("curated source-image count changed")
    if int(interpreted.get("object_count", -1)) != int(phase3["expected_instances"]):
        failures.append("curated object count changed")
    if interpreted.get("split_counts") != EXPECTED_SOURCE_SPLITS:
        failures.append("curated source-image split counts changed")
    if interpreted.get("class_counts") != EXPECTED_CLASS_COUNTS:
        failures.append("curated class counts changed")
    if int(interpreted.get("pending_cross_split_near_duplicate_pairs", -1)) != 0:
        failures.append("a pending retained cross-split candidate exists")
    if int(interpreted.get("confirmed_same_scene_groups_crossing_splits", -1)) != 0:
        failures.append("a reviewed same-scene component crosses retained splits")
    if failures:
        raise ProcessedDatasetError("; ".join(failures))
    return summary


def _assert_safe_derived_root(output_root: Path, repository_root: Path) -> None:
    processed_root = (repository_root / "data" / "processed").resolve()
    if output_root == processed_root or processed_root not in output_root.parents:
        raise ProcessedDatasetError(f"refusing to rebuild unsafe output path: {output_root}")


def _prepare_output_root(output_root: Path, repository_root: Path) -> None:
    _assert_safe_derived_root(output_root, repository_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True, exist_ok=False)


def _statistics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metrics = {
        "bbox_width": lambda row: float(row["bbox_width"]),
        "bbox_height": lambda row: float(row["bbox_height"]),
        "bbox_area": lambda row: float(row["bbox_width"]) * float(row["bbox_height"]),
        "crop_side": lambda row: float(row["crop_side"]),
        "bbox_area_source_fraction": lambda row: (
            float(row["bbox_width"])
            * float(row["bbox_height"])
            / (float(row["source_width"]) * float(row["source_height"]))
        ),
        "defect_occupancy": lambda row: float(row["defect_occupancy"]),
        "resize_scale_factor": lambda row: float(row["resize_scale_factor"]),
    }
    output: list[dict[str, Any]] = []
    for scope in ("all", *LABELS):
        selected = list(rows) if scope == "all" else [row for row in rows if row["canonical_label"] == scope]
        for metric, getter in metrics.items():
            values = np.asarray([getter(row) for row in selected], dtype=np.float64)
            percentiles = np.percentile(values, [0, 5, 25, 50, 75, 95, 100], method="linear")
            output.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "count": len(values),
                    "min": f"{percentiles[0]:.12g}",
                    "p05": f"{percentiles[1]:.12g}",
                    "p25": f"{percentiles[2]:.12g}",
                    "median": f"{percentiles[3]:.12g}",
                    "p75": f"{percentiles[4]:.12g}",
                    "p95": f"{percentiles[5]:.12g}",
                    "max": f"{percentiles[6]:.12g}",
                }
            )
    counts = {
        "minimum_side_applied": sum(str(row["minimum_side_applied"]).lower() == "true" for row in rows),
        "boundary_shifted": sum(str(row["boundary_shifted"]).lower() == "true" for row in rows),
        "max_side_clipped": sum(str(row["max_side_clipped"]).lower() == "true" for row in rows),
    }
    return output, counts


def _validate_generated(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    included_sources: Mapping[str, str],
) -> None:
    instance_ids = [str(row["instance_id"]) for row in rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ProcessedDatasetError("duplicate instance IDs in processed manifest")
    expected_files = {str(row["output_relative_path"]).replace("\\", "/") for row in rows}
    actual_files = {
        path.relative_to(output_root).as_posix() for path in (output_root / "images").glob("*.png")
    }
    if expected_files != actual_files:
        raise ProcessedDatasetError("processed image store has missing or extra PNG files")
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        source_id = str(row["source_image_id"])
        if source_id not in included_sources:
            raise ProcessedDatasetError(f"excluded source entered processed dataset: {source_id}")
        if row["split"] != included_sources[source_id]:
            raise ProcessedDatasetError(f"split inheritance failed for source {source_id}")
        source_splits.setdefault(source_id, set()).add(str(row["split"]))
        if row["canonical_label"] not in LABEL_TO_ID:
            raise ProcessedDatasetError("invalid canonical label")
        if int(row["class_id"]) != LABEL_TO_ID[str(row["canonical_label"])]:
            raise ProcessedDatasetError("class ID does not match frozen label map")
        if int(row["crop_xmax"]) - int(row["crop_xmin"]) != int(row["crop_side"]):
            raise ProcessedDatasetError("manifest crop width is not crop_side")
        if int(row["crop_ymax"]) - int(row["crop_ymin"]) != int(row["crop_side"]):
            raise ProcessedDatasetError("manifest crop height is not crop_side")
        with Image.open(output_root / str(row["output_relative_path"])) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (224, 224):
                raise ProcessedDatasetError("generated crop is not a 224x224 RGB PNG")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ProcessedDatasetError("a source image crosses processed splits")


def build_processed_dataset(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Rebuild the complete deterministic Phase 3 crop dataset."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    phase3, crop_config, output_config = data["crop_dataset"], data["crop"], data["output"]
    phase2_summary = verify_phase2_inputs(config, root)
    _load_label_map(_repo_path(root, phase3["label_map_file"]))

    curation_manifest_path = _repo_path(root, data["curation"]["manifest_file"])
    curated_instances_path = root / "data" / "metadata" / "wtbd" / "curated_instances.csv"
    image_inventory_path = root / "data" / "metadata" / "wtbd" / "images.csv"
    curation_rows = read_csv(curation_manifest_path)
    included = {
        row["sample_id"]: row["curated_split"]
        for row in curation_rows
        if row["include"].lower() == "true"
    }
    if len(included) != int(phase3["expected_source_images"]):
        raise ProcessedDatasetError("included Phase 2 manifest source count changed")
    inventory = {row["source_image_id"]: row for row in read_csv(image_inventory_path)}
    instances = read_csv(curated_instances_path)
    instances.sort(key=lambda row: (_natural_id(row["source_image_id"]), int(row["object_index"])))
    if len(instances) != int(phase3["expected_instances"]):
        raise ProcessedDatasetError("curated instance count changed")

    release_root = _repo_path(root, data["dataset"]["release_root"])
    image_directory = release_root / data["dataset"]["image_directory"]
    output_root = _repo_path(root, phase3["output_root"])
    _prepare_output_root(output_root, root)
    source_paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for instance in instances:
        source_id = instance["source_image_id"]
        if source_id not in included:
            raise ProcessedDatasetError(f"curated instance references excluded source {source_id}")
        if source_id not in inventory:
            raise ProcessedDatasetError(f"source {source_id} is missing from image inventory")
        image_record = inventory[source_id]
        source_path = image_directory / instance["image_filename"]
        source_paths[source_id] = source_path
        source_width, source_height = int(image_record["decoded_width"]), int(image_record["decoded_height"])
        geometry = calculate_square_crop(
            xmin=int(instance["xmin"]),
            ymin=int(instance["ymin"]),
            xmax=int(instance["xmax"]),
            ymax=int(instance["ymax"]),
            image_width=source_width,
            image_height=source_height,
            context_multiplier=float(crop_config["context_multiplier"]),
            minimum_side=int(crop_config["minimum_side_pixels"]),
        )
        instance_id = instance["instance_id"]
        relative_path = Path("images") / f"{instance_id}.png"
        checksum = generate_crop_png(
            source_path,
            output_root / relative_path,
            geometry,
            output_size=(int(output_config["width"]), int(output_config["height"])),
        )
        bbox_width, bbox_height = int(instance["bbox_width"]), int(instance["bbox_height"])
        label = instance["canonical_label_if_unambiguous"]
        row: dict[str, Any] = {
            "instance_id": instance_id,
            "source_image_id": source_id,
            "source_filename": instance["image_filename"],
            "object_index": int(instance["object_index"]),
            "raw_label": instance["raw_label"],
            "canonical_label": label,
            "class_id": LABEL_TO_ID[label],
            "split": included[source_id],
            "bbox_xmin": int(instance["xmin"]),
            "bbox_ymin": int(instance["ymin"]),
            "bbox_xmax": int(instance["xmax"]),
            "bbox_ymax": int(instance["ymax"]),
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "source_width": source_width,
            "source_height": source_height,
            **geometry.as_dict(),
            "context_multiplier": float(crop_config["context_multiplier"]),
            "minimum_crop_side": int(crop_config["minimum_side_pixels"]),
            "output_width": int(output_config["width"]),
            "output_height": int(output_config["height"]),
            "output_relative_path": relative_path.as_posix(),
            "source_image_sha256": image_record["sha256"],
            "processed_image_sha256": checksum,
            "defect_occupancy": f"{bbox_width * bbox_height / (geometry.crop_side**2):.12g}",
            "resize_scale_factor": f"{int(output_config['width']) / geometry.crop_side:.12g}",
        }
        rows.append(row)

    _validate_generated(rows, output_root, included)
    class_counts = {label: sum(row["canonical_label"] == label for row in rows) for label in LABELS}
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise ProcessedDatasetError(f"processed class counts changed: {class_counts}")
    split_sources = {
        split: len({row["source_image_id"] for row in rows if row["split"] == split})
        for split in EXPECTED_SOURCE_SPLITS
    }
    if split_sources != EXPECTED_SOURCE_SPLITS:
        raise ProcessedDatasetError(f"source split counts changed: {split_sources}")
    split_instance_counts = {
        split: sum(row["split"] == split for row in rows) for split in EXPECTED_SOURCE_SPLITS
    }
    split_class_rows = [
        {
            "split": split,
            "canonical_label": label,
            "class_id": LABEL_TO_ID[label],
            "source_image_count": len(
                {row["source_image_id"] for row in rows if row["split"] == split and row["canonical_label"] == label}
            ),
            "instance_count": sum(
                row["split"] == split and row["canonical_label"] == label for row in rows
            ),
        }
        for split in EXPECTED_SOURCE_SPLITS
        for label in LABELS
    ]
    if any(row["instance_count"] == 0 for row in split_class_rows):
        raise ProcessedDatasetError("a class is absent from a processed split")

    manifest_content = csv_text(rows, MANIFEST_FIELDS)
    atomic_write_text(output_root / "manifest.csv", manifest_content)
    atomic_write_text(
        output_root / "manifest.json",
        json_text({"schema_version": "1.0", "dataset_version": phase3["version"], "instances": rows}),
    )
    checksum_rows = [
        {
            "relative_path": row["output_relative_path"],
            "instance_id": row["instance_id"],
            "sha256": row["processed_image_sha256"],
            "file_size_bytes": (output_root / str(row["output_relative_path"])).stat().st_size,
        }
        for row in rows
    ]
    checksum_content = csv_text(
        checksum_rows, ("relative_path", "instance_id", "sha256", "file_size_bytes")
    )
    atomic_write_text(output_root / "crop_checksum_manifest.csv", checksum_content)

    config_sha256 = hashlib.sha256(canonical_config_json(data).encode("utf-8")).hexdigest()
    curation_manifest_sha256 = sha256_file(curation_manifest_path)
    processed_fingerprint = calculate_processed_fingerprint(
        config_sha256=config_sha256,
        curation_manifest_sha256=curation_manifest_sha256,
        manifest_content=manifest_content,
        checksum_content=checksum_content,
    )

    statistics_rows, affected_counts = _statistics(rows)
    atomic_write_text(
        _repo_path(root, phase3["statistics_file"]),
        csv_text(
            statistics_rows,
            ("scope", "metric", "count", "min", "p05", "p25", "median", "p75", "p95", "max"),
        ),
    )
    atomic_write_text(
        _repo_path(root, phase3["split_counts_file"]),
        csv_text(
            split_class_rows,
            ("split", "canonical_label", "class_id", "source_image_count", "instance_count"),
        ),
    )

    split_rows = [
        {
            "instance_id": row["instance_id"],
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "canonical_label": row["canonical_label"],
            "class_id": row["class_id"],
        }
        for row in rows
    ]
    split_fields = ("instance_id", "source_image_id", "split", "canonical_label", "class_id")
    split_content = csv_text(split_rows, split_fields)
    atomic_write_text(_repo_path(root, phase3["instance_split_file"]), split_content)
    split_root = _repo_path(root, phase3["split_root"])
    split_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(split_root / "full_split.csv", split_content)

    all_stats = {row["metric"]: row for row in statistics_rows if row["scope"] == "all"}
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "dataset_version": phase3["version"],
        "upstream_raw_fingerprint": phase2_summary["source_dataset_fingerprint"],
        "phase2_curation_version": phase2_summary["curation_version"],
        "phase2_curation_manifest_sha256": curation_manifest_sha256,
        "phase3_config_sha256": config_sha256,
        "processed_dataset_fingerprint": processed_fingerprint,
        "source_image_count": len(included),
        "instance_count": len(rows),
        "class_counts": class_counts,
        "source_image_counts_by_split": split_sources,
        "instance_counts_by_split": split_instance_counts,
        "crop_policy": {
            "input_bbox_coordinates": "one_based_inclusive_voc",
            "crop_coordinates": "zero_based_half_open",
            "context_multiplier": float(crop_config["context_multiplier"]),
            "minimum_side_pixels": int(crop_config["minimum_side_pixels"]),
            "boundary_policy": crop_config["boundary_policy"],
            "padding": crop_config["padding"],
            "output": "224x224 RGB PNG",
            "resampling": output_config["resampling"],
        },
        "affected_counts": affected_counts,
        "global_statistics": all_stats,
        "phase2_gate_revalidated": True,
        "phase4_started": False,
    }
    atomic_write_text(output_root / "dataset_summary.json", json_text(summary))
    atomic_write_text(output_root / "resolved_config.yaml", config.to_yaml())
    atomic_write_text(
        output_root / "README.md",
        "# WTBD crops v1\n\n"
        "Deterministic Phase 3 metadata for one 224x224 RGB PNG per curated WTBD defect instance. "
        "Generated PNG pixels live in `images/` and are intentionally ignored by Git. Regenerate from "
        "the immutable WTBD raw release with `python scripts/build_wtbd_crops.py --config "
        "configs/crop_dataset.yaml`.\n",
    )

    figures_root = _repo_path(root, phase3["figures_root"])
    if figures_root.exists():
        shutil.rmtree(figures_root)
    create_training_crop_qc(
        rows,
        source_paths,
        output_root,
        figures_root,
        labels=LABELS,
        seed=int(data["qc"]["seed"]),
        examples_per_class=int(data["qc"]["random_examples_per_class"]),
        diagnostic_examples=int(data["qc"]["diagnostic_examples_per_sheet"]),
    )
    return summary


def validate_processed_dataset(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Validate an existing Phase 3 crop dataset without rewriting it."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    phase3 = data["crop_dataset"]
    verify_phase2_inputs(config, root)
    output_root = _repo_path(root, phase3["output_root"])
    summary = json.loads((output_root / "dataset_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(output_root / "manifest.csv")
    curation_rows = read_csv(_repo_path(root, data["curation"]["manifest_file"]))
    included = {
        row["sample_id"]: row["curated_split"]
        for row in curation_rows
        if row["include"].lower() == "true"
    }
    _validate_generated(rows, output_root, included)
    if len(rows) != int(phase3["expected_instances"]):
        raise ProcessedDatasetError("processed manifest count changed")
    for row in rows:
        path = output_root / row["output_relative_path"]
        if sha256_file(path) != row["processed_image_sha256"]:
            raise ProcessedDatasetError(f"processed checksum mismatch: {row['instance_id']}")
    if summary.get("status") != "PASS":
        raise ProcessedDatasetError("processed summary is not PASS")
    return summary
