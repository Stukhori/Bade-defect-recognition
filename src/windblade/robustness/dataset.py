"""Versioned canonical corrupted-image dataset generation and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from windblade.config import ResolvedConfig, calculate_config_hash
from windblade.data.processed import csv_text, json_text, read_csv, sha256_file
from windblade.deep.dataset import split_rows
from windblade.robustness.corruptions import apply_corruption, condition_specs, pillow_environment
from windblade.utils import atomic_write_text


MANIFEST_FIELDS = (
    "instance_id",
    "source_image_id",
    "true_class",
    "true_class_id",
    "corruption_family",
    "severity",
    "corruption_parameter",
    "clean_image_path",
    "clean_image_sha256",
    "clean_pixel_sha256",
    "corrupted_image_path",
    "corrupted_image_sha256",
    "corrupted_pixel_sha256",
    "width",
    "height",
    "mode",
    "transformation_config_hash",
)

CHECKSUM_FIELDS = (
    "corruption_family",
    "severity",
    "instance_id",
    "corrupted_image_sha256",
    "corrupted_pixel_sha256",
    "transformation_config_hash",
)


class RobustnessDatasetError(RuntimeError):
    """Raised when the immutable Phase 8 corruption dataset contract fails."""


def pixel_sha256(image: Image.Image) -> str:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(int(rgb.width).to_bytes(4, "big"))
    digest.update(int(rgb.height).to_bytes(4, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _repo_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_derived_root(path: Path, root: Path) -> None:
    allowed = (root / "data" / "processed").resolve()
    resolved = path.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RobustnessDatasetError(f"unsafe Phase 8 dataset output root: {resolved}")


def _prepare_output(path: Path, root: Path) -> None:
    _safe_derived_root(path, root)
    if path.exists():
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)


def _test_rows(config: ResolvedConfig, root: Path) -> list[dict[str, str]]:
    data = config.as_dict()
    rows = read_csv(root / data["dataset"]["base_manifest"])
    partitions = split_rows(rows)
    test_rows = partitions["test"]
    expected = data["dataset"]["expected_instances"]
    if {name: len(partitions[name]) for name in partitions} != {
        "train": int(expected["train"]),
        "validation": int(expected["validation"]),
        "test": int(expected["test"]),
    }:
        raise RobustnessDatasetError("frozen Phase 3 instance membership changed")
    if len({row["source_image_id"] for row in test_rows}) != int(data["dataset"]["expected_test_sources"]):
        raise RobustnessDatasetError("frozen Phase 3 test source-image membership changed")
    return test_rows


def clean_rows(config: ResolvedConfig, root: str | Path) -> list[dict[str, str]]:
    """Return the fixed test rows after verifying all clean files and checksums."""

    repository = Path(root).resolve()
    data = config.as_dict()
    base_root = repository / data["dataset"]["base_root"]
    rows = _test_rows(config, repository)
    for row in rows:
        path = base_root / row["output_relative_path"]
        if not path.is_file() or sha256_file(path) != row["processed_image_sha256"]:
            raise RobustnessDatasetError(f"clean Phase 3 image mismatch: {row['instance_id']}")
    return rows


def generate_corruption_dataset(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    """Generate each of twelve degraded conditions independently from clean."""

    repository = Path(root).resolve()
    data = config.as_dict()
    output_root = repository / data["dataset"]["output_root"]
    base_root = repository / data["dataset"]["base_root"]
    rows = clean_rows(config, repository)
    specs = condition_specs(data, include_clean=False)
    if len(specs) != int(data["phase8"]["expected_degraded_conditions"]):
        raise RobustnessDatasetError("configured degraded-condition count changed")
    _prepare_output(output_root, repository)
    manifest: list[dict[str, Any]] = []
    clean_pixel_hashes: dict[str, str] = {}
    for row in rows:
        clean_path = base_root / row["output_relative_path"]
        with Image.open(clean_path) as opened:
            clean = opened.convert("RGB").copy()
        if clean.size != (224, 224):
            raise RobustnessDatasetError(f"non-canonical clean image: {row['instance_id']}")
        clean_pixel_hashes[row["instance_id"]] = pixel_sha256(clean)
        for spec in specs:
            family = str(spec["corruption_family"])
            severity = str(spec["severity"])
            # The only transform input is this immutable clean object. No prior
            # corruption output is retained or accepted here.
            corrupted = apply_corruption(clean.copy(), family, severity, data)
            if corrupted.size != (224, 224) or corrupted.mode != "RGB":
                raise RobustnessDatasetError(f"invalid output for {row['instance_id']} {family}/{severity}")
            destination = output_root / "images" / family / severity / f"{row['instance_id']}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            corrupted.save(destination, format="PNG")
            manifest.append(
                {
                    "instance_id": row["instance_id"],
                    "source_image_id": row["source_image_id"],
                    "true_class": row["canonical_label"],
                    "true_class_id": int(row["class_id"]),
                    "corruption_family": family,
                    "severity": severity,
                    "corruption_parameter": spec["parameter"],
                    "clean_image_path": _repo_relative(clean_path, repository),
                    "clean_image_sha256": row["processed_image_sha256"],
                    "clean_pixel_sha256": clean_pixel_hashes[row["instance_id"]],
                    "corrupted_image_path": _repo_relative(destination, repository),
                    "corrupted_image_sha256": sha256_file(destination),
                    "corrupted_pixel_sha256": pixel_sha256(corrupted),
                    "width": corrupted.width,
                    "height": corrupted.height,
                    "mode": corrupted.mode,
                    "transformation_config_hash": spec["transformation_config_hash"],
                }
            )
    expected_count = int(data["phase8"]["expected_corrupted_images"])
    if len(manifest) != expected_count:
        raise RobustnessDatasetError(f"generated {len(manifest)} corruptions; expected {expected_count}")
    manifest.sort(key=lambda item: (item["corruption_family"], item["severity"], item["instance_id"]))
    manifest_text = csv_text(manifest, MANIFEST_FIELDS)
    checksum_rows = [{field: row[field] for field in CHECKSUM_FIELDS} for row in manifest]
    checksum_text = csv_text(checksum_rows, CHECKSUM_FIELDS)
    fingerprint = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    config_fingerprint = calculate_config_hash(data, length=64)
    conditions = {
        "clean": condition_specs(data, include_clean=True)[0],
        "degraded": specs,
        "conceptual_severity_only_within_family": True,
        "pillow_environment": pillow_environment(),
    }
    summary = {
        "status": "PASS",
        "dataset_version": data["dataset"]["output_version"],
        "phase3_processed_fingerprint": data["dataset"]["base_fingerprint"],
        "corruption_config_fingerprint": config_fingerprint,
        "robustness_dataset_fingerprint": fingerprint,
        "clean_reference_count": len(rows),
        "degraded_condition_count": len(specs),
        "corrupted_image_count": len(manifest),
        "all_outputs_224_rgb": True,
        "independent_clean_origin": True,
        "common_pixels_for_all_methods": True,
        "pillow_environment": pillow_environment(),
    }
    atomic_write_text(output_root / "manifest.csv", manifest_text)
    atomic_write_text(output_root / "corruption_checksum_manifest.csv", checksum_text)
    atomic_write_text(output_root / "checksum_manifest.csv", checksum_text)
    atomic_write_text(output_root / "conditions.json", json_text(conditions))
    atomic_write_text(output_root / "summary.json", json_text(summary))
    atomic_write_text(output_root / "resolved_config.yaml", config.to_yaml())
    atomic_write_text(
        output_root / "README.md",
        "# WTBD robustness v1\n\n"
        "The tracked manifests describe 1,944 losslessly stored PNG outputs derived independently "
        "from the 162 frozen Phase 3 test crops. The PNG payload directory is Git-ignored.\n",
    )
    return summary


def validate_corruption_dataset(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    data = config.as_dict()
    output_root = repository / data["dataset"]["output_root"]
    rows = read_csv(output_root / "manifest.csv")
    expected = int(data["phase8"]["expected_corrupted_images"])
    if len(rows) != expected:
        raise RobustnessDatasetError(f"manifest contains {len(rows)} rows; expected {expected}")
    observed_keys: set[tuple[str, str, str]] = set()
    checksum_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row["corruption_family"], row["severity"], row["instance_id"])
        if key in observed_keys:
            raise RobustnessDatasetError(f"duplicate corrupted condition row: {key}")
        observed_keys.add(key)
        path = repository / row["corrupted_image_path"]
        if not path.is_file() or sha256_file(path) != row["corrupted_image_sha256"]:
            raise RobustnessDatasetError(f"corrupted PNG checksum mismatch: {key}")
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "RGB" or pixel_sha256(image) != row["corrupted_pixel_sha256"]:
                raise RobustnessDatasetError(f"corrupted pixel contract mismatch: {key}")
        checksum_rows.append({field: row[field] for field in CHECKSUM_FIELDS})
    fixed_clean_rows = clean_rows(config, repository)
    expected_keys = {
        (str(spec["corruption_family"]), str(spec["severity"]), row["instance_id"])
        for spec in condition_specs(data, include_clean=False)
        for row in fixed_clean_rows
    }
    if observed_keys != expected_keys:
        raise RobustnessDatasetError("corruption condition membership changed")
    checksum_rows.sort(key=lambda item: (item["corruption_family"], item["severity"], item["instance_id"]))
    checksum_text = csv_text(checksum_rows, CHECKSUM_FIELDS)
    stored = (output_root / "corruption_checksum_manifest.csv").read_text(encoding="utf-8")
    if stored != checksum_text or (output_root / "checksum_manifest.csv").read_text(encoding="utf-8") != stored:
        raise RobustnessDatasetError("corruption checksum manifest is not canonical")
    fingerprint = hashlib.sha256(stored.encode("utf-8")).hexdigest()
    summary = __import__("json").loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if summary["robustness_dataset_fingerprint"] != fingerprint:
        raise RobustnessDatasetError("robustness dataset fingerprint mismatch")
    return summary


def create_training_qc(config: ResolvedConfig, root: str | Path) -> list[str]:
    """Render only deterministic training examples for corruption visual QC."""

    repository = Path(root).resolve()
    data = config.as_dict()
    rows = read_csv(repository / data["dataset"]["base_manifest"])
    train_rows = split_rows(rows)["train"]
    selected: list[dict[str, str]] = []
    for label in data["evaluation"]["classes"]:
        selected.append(next(row for row in train_rows if row["canonical_label"] == label))
    base_root = repository / data["dataset"]["base_root"]
    qc_root = repository / data["phase8"]["qc_root"]
    qc_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for family in data["corruptions"]["order"]:
        cells: list[list[Image.Image]] = []
        for row in selected:
            with Image.open(base_root / row["output_relative_path"]) as opened:
                clean = opened.convert("RGB").copy()
            cells.append([clean, *[apply_corruption(clean.copy(), family, severity, data) for severity in ("mild", "moderate", "severe")]])
        margin, label_height, label_width = 12, 28, 110
        sheet = Image.new("RGB", (label_width + 4 * 224 + 5 * margin, len(cells) * 224 + (len(cells) + 2) * margin + label_height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(("clean", "mild", "moderate", "severe")):
            draw.text((label_width + margin + column * (224 + margin), margin), label, fill="black")
        for row_index, images in enumerate(cells):
            y = margin + label_height + row_index * (224 + margin)
            for column, image in enumerate(images):
                sheet.paste(image, (label_width + margin + column * (224 + margin), y))
            draw.text((8, y + 105), selected[row_index]["canonical_label"], fill="black")
        destination = qc_root / f"{family}_training_qc.png"
        sheet.save(destination, format="PNG")
        outputs.append(_repo_relative(destination, repository))
    return outputs
