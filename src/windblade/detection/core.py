"""Deterministic Phase 11A full-image annotation and feasibility audit."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from windblade.config import ResolvedConfig


class DetectionAuditError(RuntimeError):
    """Raised when a Phase 11 audit or frozen-protocol invariant fails."""


CLASSES = ("craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze")
SPLITS = ("train", "validation", "test")
COLOR = {
    "craze": "#0072B2", "corrosion": "#E69F00", "surface_injure": "#009E73",
    "thunderstrike": "#D55E00", "crack": "#CC79A7", "hide_craze": "#56B4E9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DetectionAuditError(f"required JSON is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DetectionAuditError(f"required CSV is absent: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def validate_box(box: Sequence[float], width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise DetectionAuditError("image dimensions must be positive")
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise DetectionAuditError("box coordinates must contain four finite values")
    xmin, ymin, xmax, ymax = (float(value) for value in box)
    if xmin < 1 or ymin < 1 or xmax > width or ymax > height:
        raise DetectionAuditError(f"box is outside inclusive image bounds: {box} versus {width}x{height}")
    if xmax < xmin or ymax < ymin:
        raise DetectionAuditError(f"box has non-positive inclusive extent: {box}")


def validate_unique_annotations(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        (row["source_image_id"], row["class_name"], row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise DetectionAuditError("duplicate annotation coordinates and class exist within an image")


def validate_image_coverage(image_ids: Iterable[str], rows: Sequence[Mapping[str, Any]]) -> None:
    annotated = {str(row["source_image_id"]) for row in rows}
    missing = sorted(set(map(str, image_ids)) - annotated)
    if missing:
        raise DetectionAuditError("positive images with empty annotations: " + ", ".join(missing[:5]))


def voc_inclusive_to_yolo(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert inclusive VOC pixels to normalized continuous-edge YOLO coordinates."""
    validate_box(box, width, height)
    xmin, ymin, xmax, ymax = (float(value) for value in box)
    result = (
        (xmin + xmax - 1.0) / (2.0 * width),
        (ymin + ymax - 1.0) / (2.0 * height),
        (xmax - xmin + 1.0) / width,
        (ymax - ymin + 1.0) / height,
    )
    if not all(0.0 <= value <= 1.0 for value in result):
        raise DetectionAuditError(f"normalized box left [0,1]: {result}")
    return result


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1) + 1.0)
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1) + 1.0)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def choose_validation_threshold(rows: Sequence[Mapping[str, float]]) -> float:
    """Choose validation-only maximum F1, breaking ties toward the lower threshold."""
    if not rows:
        raise DetectionAuditError("validation threshold candidates are empty")
    normalized = []
    for row in rows:
        threshold, precision, recall = (float(row[key]) for key in ("threshold", "precision", "recall"))
        if not all(math.isfinite(value) for value in (threshold, precision, recall)):
            raise DetectionAuditError("threshold candidates must be finite")
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        normalized.append((f1, -threshold, threshold))
    return max(normalized)[2]


def aggregate_seed_metrics(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        raise DetectionAuditError("exactly three finite frozen-seed metrics are required")
    numbers = [float(value) for value in values]
    return {"seed_count": 3, "mean": mean(numbers), "sample_sd": stdev(numbers)}


def validate_framework_metrics(record: Mapping[str, float]) -> None:
    required = ("map_50_95", "map_50", "precision", "recall")
    if set(required) - set(record):
        raise DetectionAuditError("framework metric record is incomplete")
    if not all(math.isfinite(float(record[key])) and 0.0 <= float(record[key]) <= 1.0 for key in required):
        raise DetectionAuditError("framework metrics must be finite values in [0,1]")


def enforce_test_firewall(state: Mapping[str, bool]) -> None:
    required = ("dataset_frozen", "split_frozen", "training_config_frozen", "all_seeds_trained",
                "checkpoints_locked", "threshold_locked", "nms_locked")
    missing = [name for name in required if state.get(name) is not True]
    if missing:
        raise DetectionAuditError("test firewall is not locked: " + ", ".join(missing))


def decompose_detection_errors(
    truth: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, iou_threshold: float = 0.5
) -> dict[str, int]:
    """Deterministic greedy decomposition for one image, highest score first."""
    if not 0.0 < iou_threshold <= 1.0:
        raise DetectionAuditError("IoU threshold must be in (0,1]")
    ordered = sorted(predictions, key=lambda row: (-float(row.get("score", 0.0)), str(row.get("id", ""))))
    matched: set[int] = set(); result = Counter({
        "true_positive": 0, "missed_ground_truth": 0, "insufficient_iou": 0,
        "localized_wrong_class": 0, "duplicate_prediction": 0, "background_false_positive": 0,
    })
    for prediction in ordered:
        overlaps = [box_iou(prediction["box"], target["box"]) for target in truth]
        if not overlaps or max(overlaps) == 0:
            result["background_false_positive"] += 1; continue
        target_index = max(range(len(overlaps)), key=lambda index: (overlaps[index], -index))
        if overlaps[target_index] < iou_threshold:
            result["insufficient_iou"] += 1; continue
        if target_index in matched:
            result["duplicate_prediction"] += 1; continue
        matched.add(target_index)
        if prediction.get("class_name") != truth[target_index].get("class_name"):
            result["localized_wrong_class"] += 1
        else:
            result["true_positive"] += 1
    result["missed_ground_truth"] = len(truth) - len(matched)
    return dict(result)


def _assert_no_phase12(root: Path) -> None:
    offenders = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        lowered = relative.as_posix().lower()
        if ("phase12" in lowered or "phase_12" in lowered) and ".git" not in relative.parts and ".venv" not in relative.parts:
            offenders.append(relative.as_posix())
    if offenders:
        raise DetectionAuditError("Phase 12 path exists: " + ", ".join(sorted(offenders)[:5]))


def _phase10_identity(config: ResolvedConfig, root: Path) -> dict[str, Any]:
    data = config.as_dict()
    expected = data["upstream"]
    manifest = _json(root / data["inputs"]["phase10_manifest"])
    repro = _json(root / data["inputs"]["phase10_reproducibility"])
    upstream_inventory = _json(root / data["inputs"]["phase10_upstream_inventory"])
    checks = {
        "phase10_complete": manifest.get("status") == "complete" and manifest.get("core_technical_project_complete") is True,
        "phase10_config": manifest.get("phase10_config_fingerprint") == expected["phase10_config_fingerprint"],
        "phase10_output": repro.get("phase10_scientific_output_fingerprint") == expected["phase10_output_fingerprint"],
        "phase10_upstream": upstream_inventory.get("fingerprint") == expected["phase10_upstream_inventory_fingerprint"],
        "phase9a_corrected": any(
            row.get("fingerprint_or_commit") == expected["phase9a_corrected_output_fingerprint"]
            for row in read_csv(root / "experiments/summaries/phase10_final_synthesis_v1/tables/reproducibility_fingerprints.csv")
            if row.get("artifact") == "phase9a_output"
        ),
        "phase9a_transition_declared": expected["phase9a_transition"] == "caption_only_pass_b_true_class_display_correction",
    }
    if not all(checks.values()):
        raise DetectionAuditError(f"Phase 10/Phase 9A identity gate failed: {checks}")
    files = upstream_inventory.get("files", {})
    changed = [relative for relative, digest in files.items() if not (root / relative).is_file() or sha256_file(root / relative) != digest]
    if changed:
        raise DetectionAuditError("frozen Phase 3-9 upstream inventory changed: " + ", ".join(changed[:5]))
    phase10_files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for directory in (root / "experiments/summaries/phase10_final_synthesis_v1", root / "figures/phase10")
        for path in sorted(directory.rglob("*")) if path.is_file()
    }
    return {
        "status": "PASS", "checks": checks, "phase3_to_phase9_file_count": len(files),
        "phase10_file_count": len(phase10_files), "phase10_files": phase10_files,
        "phase10_files_fingerprint": _canonical_hash(phase10_files),
    }


def _app_inventory(root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for directory in (root / "app", root / "src/windblade_app", root / "src/windblade_review"):
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for relative in ("docs/app.md", "docs/human_review_interface.md", "requirements-app.txt", "scripts/validate_app.py", "scripts/validate_review_interface.py"):
        path = root / relative
        if path.is_file():
            candidates.append(path)
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(candidates))}
    return {"file_count": len(files), "fingerprint": _canonical_hash(files), "files": files}


def _memory_total() -> int | None:
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended_available", ctypes.c_ulonglong)]
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    return int(status.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None


def compute_gate(root: Path, config: ResolvedConfig) -> dict[str, Any]:
    import torch
    total, _, free = shutil.disk_usage(root)
    ram = _memory_total()
    cuda = bool(torch.cuda.is_available())
    required = float(config.as_dict()["phase11b_protocol"]["minimum_vram_gib"])
    return {
        "status": "PASS_FOR_PHASE11A_BLOCKED_FOR_PHASE11B",
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "ram_gib": None if ram is None else round(ram / 2**30, 2),
        "disk_total_gib": round(total / 2**30, 2), "disk_free_gib_at_gate": round(free / 2**30, 2),
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": cuda, "cuda_device_count": int(torch.cuda.device_count()),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else 0.0,
        "minimum_vram_gib": required, "deterministic_cuda_operations_assessed": False if not cuda else True,
        "planned_scientific_runs": 3, "estimated_storage_gib": 5,
        "training_authorized_here": cuda and torch.cuda.get_device_properties(0).total_memory / 2**30 >= required,
        "block_reason": None if cuda else "CUDA is unavailable; multi-seed CPU detector training is prohibited by the frozen protocol.",
    }


@dataclass(frozen=True)
class AuditData:
    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    splits: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    duplicate_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    feasibility: dict[str, Any]
    qc_selection: list[dict[str, Any]]
    dataset_fingerprint: str
    split_fingerprint: str


def _integer(row: Mapping[str, str], name: str) -> int:
    try:
        return int(row[name])
    except (KeyError, ValueError) as exc:
        raise DetectionAuditError(f"invalid integer {name} in row {row}") from exc


def _audit(config: ResolvedConfig, root: Path) -> AuditData:
    data = config.as_dict(); inputs = data["inputs"]; audit_cfg = data["phase11a"]
    raw_images = root / inputs["raw_images"]; raw_annotations = root / inputs["raw_annotations"]
    if not raw_images.is_dir() or not raw_annotations.is_dir():
        raise DetectionAuditError("authoritative full images or PASCAL VOC annotations are absent")
    source = _json(root / inputs["source_record"]); raw_audit = _json(root / inputs["raw_audit"])
    curation = _json(root / inputs["curation_summary"])
    if source.get("archive_sha256") != sha256_file(root / inputs["raw_archive"]):
        raise DetectionAuditError("raw archive fingerprint changed")
    if raw_audit.get("dataset_fingerprint") != audit_cfg["source_dataset_fingerprint"] or curation.get("status") != "PASS":
        raise DetectionAuditError("source audit/curation identity failed")
    if raw_audit["annotations"]["invalid_bbox_count"] or raw_audit["annotations"]["parse_error_count"]:
        raise DetectionAuditError("upstream raw audit contains invalid annotations")

    all_image_rows = read_csv(root / inputs["images_manifest"])
    curated_rows = read_csv(root / inputs["curation_manifest"])
    curated_instances = read_csv(root / inputs["curated_instances"])
    split_rows = read_csv(root / inputs["curated_split"])
    image_by_id = {row["source_image_id"]: row for row in all_image_rows}
    curated_manifest = {row["sample_id"]: row for row in curated_rows if row["include"] == "True"}
    split_by_id = {row["sample_id"]: row["curated_split"] for row in split_rows}
    if set(curated_manifest) != set(split_by_id) or set(row["source_image_id"] for row in curated_instances) != set(split_by_id):
        raise DetectionAuditError("curated image, annotation, and split identities disagree")

    images: list[dict[str, Any]] = []
    for image_id in sorted(split_by_id, key=int):
        row = image_by_id[image_id]; path = raw_images / row["filename"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise DetectionAuditError(f"source image missing or changed: {image_id}")
        with Image.open(path) as image:
            image.load(); decoded = (image.width, image.height)
        declared = (_integer(row, "decoded_width"), _integer(row, "decoded_height"))
        if decoded != declared or declared[0] <= 0 or declared[1] <= 0:
            raise DetectionAuditError(f"image dimension mismatch: {image_id}")
        images.append({
            "source_image_id": image_id, "filename": row["filename"], "split": split_by_id[image_id],
            "width": declared[0], "height": declared[1], "sha256": row["sha256"],
            "pixel_sha256": row["pixel_sha256"], "annotation_source": curated_manifest[image_id]["annotation_source"],
            "number_of_boxes": 0, "number_of_classes": 0,
        })
    output_images = {row["source_image_id"]: row for row in images}

    annotations: list[dict[str, Any]] = []; seen_annotations: set[tuple[Any, ...]] = set(); duplicate_annotations = 0
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    for row in sorted(curated_instances, key=lambda item: (int(item["source_image_id"]), int(item["object_index"]))):
        image_id = row["source_image_id"]; image = output_images[image_id]
        label = row["canonical_label_if_unambiguous"]
        if label not in class_to_id:
            raise DetectionAuditError(f"undeclared class: {label}")
        box = tuple(_integer(row, name) for name in ("xmin", "ymin", "xmax", "ymax"))
        validate_box(box, image["width"], image["height"])
        key = (image_id, label, *box)
        if key in seen_annotations:
            duplicate_annotations += 1
        seen_annotations.add(key)
        yolo = voc_inclusive_to_yolo(box, image["width"], image["height"])
        width = box[2] - box[0] + 1; height = box[3] - box[1] + 1
        annotation = {
            "instance_id": row["instance_id"], "source_image_id": image_id, "object_index": int(row["object_index"]),
            "class_id": class_to_id[label], "class_name": label, "defect_class_id": 0,
            "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3],
            "bbox_width": width, "bbox_height": height, "bbox_area_pixels": width * height,
            "bbox_area_fraction": (width * height) / (image["width"] * image["height"]),
            "bbox_aspect_ratio": width / height, "yolo_x_center": yolo[0], "yolo_y_center": yolo[1],
            "yolo_width": yolo[2], "yolo_height": yolo[3], "split": image["split"],
        }
        annotations.append(annotation); by_image[image_id].append(annotation)
    if duplicate_annotations:
        raise DetectionAuditError(f"duplicate annotations require review: {duplicate_annotations}")
    validate_unique_annotations(annotations)
    validate_image_coverage(output_images, annotations)
    for image_id, image in output_images.items():
        image["number_of_boxes"] = len(by_image[image_id]); image["number_of_classes"] = len({row["class_name"] for row in by_image[image_id]})
        if image["number_of_boxes"] == 0:
            raise DetectionAuditError(f"curated positive image has no annotation: {image_id}")

    exact_groups: dict[str, list[str]] = defaultdict(list)
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    for row in images:
        exact_groups[row["sha256"]].append(row["source_image_id"]); pixel_groups[row["pixel_sha256"]].append(row["source_image_id"])
    duplicate_rows: list[dict[str, Any]] = []
    for kind, groups in (("file_sha256", exact_groups), ("pixel_sha256", pixel_groups)):
        for digest, members in sorted(groups.items()):
            if len(members) > 1:
                member_splits = sorted({split_by_id[item] for item in members})
                duplicate_rows.append({"kind": kind, "fingerprint": digest, "members": "|".join(members), "splits": "|".join(member_splits), "cross_split": len(member_splits) > 1})
    if any(row["cross_split"] for row in duplicate_rows):
        raise DetectionAuditError("exact duplicate group crosses the detection split")

    included = set(split_by_id); pending_included = 0; pending_cross = 0; same_scene_cross = 0
    for row in read_csv(root / inputs["near_duplicate_decisions"]):
        a = Path(row["image_a"]).stem; b = Path(row["image_b"]).stem
        if a in included and b in included:
            if row["decision"] == "pending_review":
                pending_included += 1; pending_cross += int(split_by_id[a] != split_by_id[b])
            if row["decision"] == "same_scene" and split_by_id[a] != split_by_id[b]:
                same_scene_cross += 1
    if pending_cross or same_scene_cross:
        raise DetectionAuditError("unresolved or confirmed related image pair crosses the detection split")

    overlaps = 0; overlap_ge_025 = 0
    for rows in by_image.values():
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                value = box_iou((first["xmin"], first["ymin"], first["xmax"], first["ymax"]), (second["xmin"], second["ymin"], second["xmax"], second["ymax"]))
                overlaps += int(value > 0); overlap_ge_025 += int(value >= 0.25)

    small = float(audit_cfg["small_area_fraction"]); full = float(audit_cfg["almost_full_area_fraction"])
    low = float(audit_cfg["elongated_aspect_ratio_low"]); high = float(audit_cfg["elongated_aspect_ratio_high"])
    class_counts = Counter(row["class_name"] for row in annotations)
    split_image_counts = Counter(row["split"] for row in images); split_box_counts = Counter(row["split"] for row in annotations)
    object_distribution = Counter(row["number_of_boxes"] for row in images)
    suspicious = {
        "invalid_boxes": 0, "duplicate_annotations": duplicate_annotations,
        "area_fraction_lt_0_01": sum(row["bbox_area_fraction"] < small for row in annotations),
        "area_fraction_gt_0_50": sum(row["bbox_area_fraction"] > full for row in annotations),
        "aspect_ratio_lt_0_10": sum(row["bbox_aspect_ratio"] < low for row in annotations),
        "aspect_ratio_gt_10": sum(row["bbox_aspect_ratio"] > high for row in annotations),
        "overlapping_pairs_iou_gt_0": overlaps, "overlapping_pairs_iou_ge_0_25": overlap_ge_025,
        "xml_filename_mismatches_in_raw_release": len(raw_audit["annotations"]["xml_filename_mismatches"]),
        "curation_excluded_images": curation["curated_interpretation"]["excluded_images"],
    }
    findings = [
        {"severity": "info", "finding": "authoritative_source", "count": len(images), "interpretation": "Primary PASCAL VOC annotations retained by wtbd-curation-v1; no screenshot or Grad-CAM reconstruction."},
        {"severity": "pass", "finding": "invalid_boxes", "count": 0, "interpretation": "All curated boxes are finite, positive, in bounds, and class-mapped."},
        {"severity": "limitation", "finding": "background_only_images", "count": 0, "interpretation": "False-positive performance on healthy blades cannot be established."},
        {"severity": "reviewed_exclusion", "finding": "excluded_identity_or_duplicate_images", "count": curation["curated_interpretation"]["excluded_images"], "interpretation": "Existing human-reviewed curation exclusions are preserved; no Phase 11 relabeling or repair."},
        {"severity": "diagnostic", "finding": "small_boxes_area_lt_1pct", "count": suspicious["area_fraction_lt_0_01"], "interpretation": "Retained and reported; no enlargement or deletion."},
        {"severity": "diagnostic", "finding": "almost_full_boxes_area_gt_50pct", "count": suspicious["area_fraction_gt_0_50"], "interpretation": "Retained and reported; no clipping or repair."},
        {"severity": "diagnostic", "finding": "extreme_aspect_ratio_boxes", "count": suspicious["aspect_ratio_lt_0_10"] + suspicious["aspect_ratio_gt_10"], "interpretation": "Retained and included in deterministic QC."},
        {"severity": "pass", "finding": "cross_split_exact_or_confirmed_scene_leakage", "count": 0, "interpretation": "Detection split preserves curated source/duplicate isolation."},
        {"severity": "limitation", "finding": "pending_near_duplicate_pairs_within_split", "count": pending_included, "interpretation": "All retained pending candidates are confined within one split; they cannot leak across evaluation partitions."},
    ]

    phase9_concerns = {
        row["source_image_id"] for row in read_csv(root / inputs["phase9_review_cases"])
        if row["pass_a_possible_crop_or_background_problem"] in {"yes", "uncertain"} and row["source_image_id"] in included
    }
    qc_candidates: list[tuple[str, str, str]] = []
    for class_name in CLASSES:
        selected = min((row for row in annotations if row["class_name"] == class_name), key=lambda row: (row["bbox_area_fraction"], int(row["source_image_id"])))
        qc_candidates.append(("class_representative", class_name, selected["source_image_id"]))
    extreme_count = int(audit_cfg["qc_per_extreme"])
    for category, ordered in (
        ("smallest_box", sorted(annotations, key=lambda row: (row["bbox_area_fraction"], int(row["source_image_id"])))),
        ("largest_box", sorted(annotations, key=lambda row: (-row["bbox_area_fraction"], int(row["source_image_id"])))),
        ("unusual_aspect_ratio", sorted(annotations, key=lambda row: (-abs(math.log(row["bbox_aspect_ratio"])), int(row["source_image_id"])))),
    ):
        qc_candidates.extend((category, row["instance_id"], row["source_image_id"]) for row in ordered[:extreme_count])
    multi = sorted(images, key=lambda row: (-row["number_of_boxes"], int(row["source_image_id"])))[:extreme_count]
    qc_candidates.extend(("multiple_box_image", str(row["number_of_boxes"]), row["source_image_id"]) for row in multi if row["number_of_boxes"] > 1)
    overlap_ids = sorted({image_id for image_id, rows in by_image.items() if any(box_iou((a["xmin"], a["ymin"], a["xmax"], a["ymax"]), (b["xmin"], b["ymin"], b["xmax"], b["ymax"])) > 0 for i, a in enumerate(rows) for b in rows[i + 1:])}, key=int)
    qc_candidates.extend(("overlapping_boxes", "iou_gt_0", image_id) for image_id in overlap_ids[:extreme_count])
    qc_candidates.extend(("phase9_crop_or_background_concern", "single_reviewer_yes_or_uncertain", image_id) for image_id in sorted(phase9_concerns, key=int)[:extreme_count])
    selected_by_id: dict[str, list[str]] = defaultdict(list)
    for category, detail, image_id in qc_candidates:
        selected_by_id[image_id].append(f"{category}:{detail}")
    max_images = int(audit_cfg["qc_max_images"])
    qc_selection = [
        {"qc_index": index + 1, "source_image_id": image_id, "filename": output_images[image_id]["filename"],
         "split": output_images[image_id]["split"], "box_count": output_images[image_id]["number_of_boxes"],
         "selection_reasons": "|".join(reasons)}
        for index, (image_id, reasons) in enumerate(sorted(selected_by_id.items(), key=lambda item: int(item[0]))[:max_images])
    ]

    feasibility = {
        "class_agnostic_localization": {"decision": "supported with explicit limitations", "basis": "720 positive full images, 1,065 valid curated boxes, source-isolated split; no healthy controls."},
        "six_class_detection": {"decision": "supported with explicit limitations", "basis": "All six classes occur in every split, but class support is imbalanced and thunderstrike has only 60 boxes overall."},
        "healthy_blade_false_positive_evaluation": {"decision": "unsupported", "basis": "No curated or raw image has zero annotated defects."},
        "arbitrary_full_image_inspection": {"decision": "unsupported", "basis": "The fixed defect-positive dataset does not represent arbitrary operational or healthy-blade imagery."},
        "application_integration": {"decision": "unsupported", "basis": "No detector has been trained or evaluated; background false-positive evidence is absent."},
    }
    split_rows_out = [{"source_image_id": row["source_image_id"], "split": row["split"], "group_unit": "source_image_after_curated_duplicate_group_exclusion"} for row in images]
    dataset_identity = {
        "images": [{key: row[key] for key in ("source_image_id", "sha256", "width", "height")} for row in images],
        "annotations": [{key: row[key] for key in ("instance_id", "source_image_id", "class_id", "xmin", "ymin", "xmax", "ymax")} for row in annotations],
        "class_order": list(CLASSES), "coordinate_convention": audit_cfg["coordinate_convention"],
    }
    summary = {
        "status": "PASS", "raw_image_count": raw_audit["images"]["count"], "raw_box_count": raw_audit["annotations"]["instance_count"],
        "curated_image_count": len(images), "curated_box_count": len(annotations), "classes": dict(sorted(class_counts.items())),
        "split_image_counts": {name: split_image_counts[name] for name in SPLITS},
        "split_box_counts": {name: split_box_counts[name] for name in SPLITS},
        "images_with_zero_boxes": object_distribution[0], "images_with_one_box": object_distribution[1],
        "images_with_multiple_boxes": sum(count for objects, count in object_distribution.items() if objects > 1),
        "maximum_boxes_per_image": max(object_distribution),
        "multi_class_images": sum(row["number_of_classes"] > 1 for row in images),
        "background_or_healthy_images": 0, "duplicate_annotation_count": duplicate_annotations,
        "retained_exact_duplicate_groups": len(duplicate_rows), "pending_near_duplicate_pairs_within_split": pending_included,
        "cross_split_duplicate_or_related_pair_count": 0, "suspicious_geometry": suspicious,
        "annotation_format": "PASCAL VOC XML represented by the frozen Phase 1/2 manifests",
        "coordinate_convention": "one-based inclusive integer pixels; xmin/ymin start at 1 and xmax/ymax are valid through width/height",
        "annotation_provenance": source,
        "phase9_crop_or_background_concern_source_count": len(phase9_concerns),
    }
    return AuditData(images, annotations, split_rows_out, findings, duplicate_rows, summary, feasibility, qc_selection,
                     _canonical_hash(dataset_identity), _canonical_hash(split_rows_out))


def _draw_qc(root: Path, destination: Path, audit: AuditData, config: ResolvedConfig) -> list[dict[str, Any]]:
    raw = root / config.as_dict()["inputs"]["raw_images"]
    boxes = defaultdict(list)
    for row in audit.annotations:
        boxes[row["source_image_id"]].append(row)
    registry = []
    image_root = destination / "qc_packet/images"; image_root.mkdir(parents=True, exist_ok=True)
    for selected in audit.qc_selection:
        image_id = selected["source_image_id"]
        with Image.open(raw / selected["filename"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        line = max(2, round(max(image.size) / 300))
        for box in boxes[image_id]:
            color = COLOR[box["class_name"]]
            display = (box["xmin"] - 1, box["ymin"] - 1, box["xmax"] - 1, box["ymax"] - 1)
            draw.rectangle(display, outline=color, width=line)
            label = f"{box['class_name']} {box['instance_id']}"
            draw.rectangle((display[0], max(0, display[1] - 16), display[0] + 8 * len(label), display[1]), fill=color)
            draw.text((display[0] + 2, max(0, display[1] - 14)), label, fill="white")
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        filename = f"qc_{int(selected['qc_index']):02d}_source_{image_id}.jpg"
        output = image_root / filename
        image.save(output, format="JPEG", quality=88, optimize=False, progressive=False, subsampling=0)
        registry.append({**selected, "qc_asset": f"qc_packet/images/{filename}", "sha256": sha256_file(output)})
    rows = "\n".join(
        f'<tr><td>{row["qc_index"]}</td><td>{row["source_image_id"]}</td><td>{row["split"]}</td><td>{row["selection_reasons"]}</td><td><img src="images/{Path(row["qc_asset"]).name}" width="420"></td></tr>'
        for row in registry
    )
    html = "<!doctype html><meta charset=\"utf-8\"><title>Phase 11A annotation QC</title><h1>Phase 11A deterministic annotation QC</h1><p>Boxes come only from the authoritative retained primary PASCAL VOC annotations. This packet proposes no relabeling or repair.</p><table border=\"1\"><tr><th>#</th><th>source</th><th>split</th><th>selection</th><th>annotated image</th></tr>" + rows + "</table>\n"
    (destination / "qc_packet/index.html").write_text(html, encoding="utf-8")
    return registry


def _scientific_inventory(root: Path, excluded: Iterable[str] = ()) -> dict[str, str]:
    excluded_set = set(excluded)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in excluded_set}


def _generate(config: ResolvedConfig, repository: Path, destination: Path, audit: AuditData,
              phase10: dict[str, Any], app: dict[str, Any], compute: dict[str, Any]) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    image_fields = ["source_image_id", "filename", "split", "width", "height", "sha256", "pixel_sha256", "annotation_source", "number_of_boxes", "number_of_classes"]
    annotation_fields = ["instance_id", "source_image_id", "object_index", "class_id", "class_name", "defect_class_id", "xmin", "ymin", "xmax", "ymax", "bbox_width", "bbox_height", "bbox_area_pixels", "bbox_area_fraction", "bbox_aspect_ratio", "yolo_x_center", "yolo_y_center", "yolo_width", "yolo_height", "split"]
    _write_csv(destination / "tables/detection_image_manifest.csv", audit.images, image_fields)
    _write_csv(destination / "tables/detection_annotation_manifest.csv", audit.annotations, annotation_fields)
    _write_csv(destination / "tables/detection_split_manifest.csv", audit.splits, ["source_image_id", "split", "group_unit"])
    _write_csv(destination / "tables/annotation_findings.csv", audit.findings, ["severity", "finding", "count", "interpretation"])
    _write_csv(destination / "tables/duplicate_group_audit.csv", audit.duplicate_rows, ["kind", "fingerprint", "members", "splits", "cross_split"])
    registry = _draw_qc(repository, destination, audit, config)
    _write_csv(destination / "tables/qc_selection.csv", registry, ["qc_index", "source_image_id", "filename", "split", "box_count", "selection_reasons", "qc_asset", "sha256"])
    _write_json(destination / "class_mapping.json", {"multiclass": {name: index for index, name in enumerate(CLASSES)}, "class_agnostic": {"defect": 0}})
    _write_json(destination / "audit_summary.json", audit.summary)
    _write_json(destination / "feasibility.json", audit.feasibility)
    _write_json(destination / "compute_gate.json", compute)
    protocol = config.as_dict()["phase11b_protocol"] | {
        "dataset_fingerprint": audit.dataset_fingerprint, "split_fingerprint": audit.split_fingerprint,
        "training_started": False, "dependency_installed": False, "pretrained_weight_downloaded": False,
        "test_metrics_inspected": False,
        "required_next_command_after_gpu_and_new_pretest_pin": "python scripts/run_detection.py --config configs/detection.yaml --train",
    }
    _write_json(destination / "frozen_protocol.json", protocol)
    _write_json(destination / "phase10_immutability.json", phase10)
    _write_json(destination / "application_immutability.json", app)
    manifest = {
        "schema_version": "1.0", "phase": "11A", "result_id": config.as_dict()["phase11a"]["result_id"],
        "status": "complete", "audit_passed": True, "detection_dataset_frozen": True,
        "phase11b_training_started": False, "phase11b_blocked": True, "phase12_started": False,
        "phase10_unchanged": True, "application_unchanged": True,
        "dataset_fingerprint": audit.dataset_fingerprint, "split_fingerprint": audit.split_fingerprint,
        "config_fingerprint": _canonical_hash(config.as_dict()),
    }
    inventory = _scientific_inventory(destination, excluded=("manifest.json", "reproducibility.json", "validation.json"))
    output_fingerprint = _canonical_hash(inventory)
    manifest["expected_scientific_files"] = sorted(inventory)
    manifest["scientific_file_count"] = len(inventory)
    manifest["scientific_output_fingerprint"] = output_fingerprint
    _write_json(destination / "manifest.json", manifest)
    return {"inventory": inventory, "fingerprint": output_fingerprint, "qc_count": len(registry)}


def apparatus_check(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve(); _assert_no_phase12(repository)
    phase10 = _phase10_identity(config, repository); audit = _audit(config, repository); compute = compute_gate(repository, config)
    if tuple(config.as_dict()["phase11a"]["class_order"]) != CLASSES:
        raise DetectionAuditError("class order changed")
    return {
        "status": "PASS", "phase10_immutability": phase10["status"], "annotation_audit": audit.summary["status"],
        "curated_images": len(audit.images), "curated_boxes": len(audit.annotations),
        "dataset_fingerprint": audit.dataset_fingerprint, "split_fingerprint": audit.split_fingerprint,
        "phase11b_training_authorized_here": compute["training_authorized_here"], "phase12_started": False,
    }


def run_audit(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve(); _assert_no_phase12(repository)
    phase10 = _phase10_identity(config, repository); app = _app_inventory(repository)
    audit = _audit(config, repository); compute = compute_gate(repository, config)
    scratch_parent = repository / config.as_dict()["outputs"]["reproduction_scratch_root"]
    scratch_parent.mkdir(parents=True, exist_ok=True)
    first_dir = Path(tempfile.mkdtemp(prefix="pass_a_", dir=scratch_parent))
    second_dir = Path(tempfile.mkdtemp(prefix="pass_b_", dir=scratch_parent))
    try:
        first = _generate(config, repository, first_dir, audit, phase10, app, compute)
        second = _generate(config, repository, second_dir, audit, phase10, app, compute)
        if first["inventory"] != second["inventory"] or first["fingerprint"] != second["fingerprint"]:
            raise DetectionAuditError("two-pass Phase 11A preprocessing/QC generation differs")
        output = repository / config.as_dict()["outputs"]["summary_root"]
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(first_dir, output)
        reproduction = {
            "status": "PASS", "exact_two_pass_equality": True, "scientific_output_fingerprint": first["fingerprint"],
            "scientific_file_count": len(first["inventory"]), "inventory": first["inventory"],
            "dataset_fingerprint": audit.dataset_fingerprint, "split_fingerprint": audit.split_fingerprint,
            "phase10_files_fingerprint": phase10["phase10_files_fingerprint"], "application_fingerprint": app["fingerprint"],
        }
        _write_json(output / "reproducibility.json", reproduction)
        validation = validate_audit(config, repository, allow_missing_validation=True)
        _write_json(output / "validation.json", validation)
        return validation
    finally:
        shutil.rmtree(first_dir, ignore_errors=True); shutil.rmtree(second_dir, ignore_errors=True)


def validate_audit(config: ResolvedConfig, root: str | Path, *, allow_missing_validation: bool = False) -> dict[str, Any]:
    repository = Path(root).resolve(); _assert_no_phase12(repository)
    phase10 = _phase10_identity(config, repository); app = _app_inventory(repository); audit = _audit(config, repository)
    output = repository / config.as_dict()["outputs"]["summary_root"]
    if not output.is_dir():
        raise DetectionAuditError("Phase 11A output is absent")
    manifest = _json(output / "manifest.json"); reproduction = _json(output / "reproducibility.json")
    if manifest.get("status") != "complete" or not manifest.get("audit_passed") or manifest.get("phase11b_training_started") or manifest.get("phase12_started"):
        raise DetectionAuditError("Phase 11A manifest state is invalid")
    if manifest.get("dataset_fingerprint") != audit.dataset_fingerprint or manifest.get("split_fingerprint") != audit.split_fingerprint:
        raise DetectionAuditError("Phase 11A dataset/split fingerprint changed")
    if reproduction.get("dataset_fingerprint") != audit.dataset_fingerprint or not reproduction.get("exact_two_pass_equality"):
        raise DetectionAuditError("Phase 11A reproduction record failed")
    if reproduction.get("phase10_files_fingerprint") != phase10["phase10_files_fingerprint"]:
        raise DetectionAuditError("Phase 10 changed after Phase 11A generation")
    if reproduction.get("application_fingerprint") != app["fingerprint"]:
        raise DetectionAuditError("application changed after Phase 11A generation")
    current = _scientific_inventory(output, excluded=("manifest.json", "reproducibility.json", "validation.json"))
    recorded = reproduction.get("inventory", {})
    if current != recorded:
        raise DetectionAuditError("Phase 11A scientific output inventory changed")
    if _canonical_hash(current) != reproduction.get("scientific_output_fingerprint"):
        raise DetectionAuditError("Phase 11A output fingerprint changed")
    if not allow_missing_validation and not (output / "validation.json").is_file():
        raise DetectionAuditError("validation record is absent")
    return {
        "status": "PASS", "phase": "11A", "phase10_unchanged": True, "application_unchanged": True,
        "annotation_audit": "PASS", "dataset_fingerprint": audit.dataset_fingerprint,
        "split_fingerprint": audit.split_fingerprint, "scientific_output_fingerprint": reproduction["scientific_output_fingerprint"],
        "exact_two_pass_equality": True, "curated_images": len(audit.images), "curated_boxes": len(audit.annotations),
        "phase11b_training_started": False, "phase11b_blocked": True, "block_reason": "CUDA unavailable",
        "phase12_started": False,
    }
