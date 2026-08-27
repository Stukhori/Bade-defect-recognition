"""Versioned, non-destructive reconciliation for the audited WTBD release."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from windblade.config import ResolvedConfig, save_resolved_config
from windblade.data.audit import (
    CANONICAL_CLASSES,
    calculate_dataset_fingerprint,
    canonicalize_label,
    natural_id_key,
    write_deterministic_csv,
)
from windblade.data.duplicates import duplicate_groups, sha256_file
from windblade.data.voc import BoundingBox, VocAnnotation, inclusive_iou, parse_voc_xml
from windblade.results import read_json, write_json


class CurationError(RuntimeError):
    """Raised when curation evidence is incomplete or internally inconsistent."""


class IdentityStatus(StrEnum):
    CONSISTENT = "consistent"
    XML_NAME_CORRECT = "xml_name_correct"
    EMBEDDED_FILENAME_CORRECT = "embedded_filename_correct"
    ANNOTATION_REUSED = "annotation_reused_from_other_image"
    AMBIGUOUS = "ambiguous"
    UNUSABLE = "unusable"
    PENDING_REVIEW = "pending_review"


class AnnotationStatus(StrEnum):
    PRIMARY_CONFIRMED = "primary_confirmed"
    SECONDARY_CONFIRMED = "secondary_confirmed"
    BOTH_AGREE = "both_agree"
    PRIMARY_SECONDARY_DISAGREE = "primary_secondary_disagree"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    EXCLUDE = "exclude"


class DuplicateStatus(StrEnum):
    UNIQUE = "unique"
    EXACT_CANONICAL = "exact_duplicate_canonical"
    EXACT_REDUNDANT = "exact_duplicate_redundant"
    NEAR_REVIEW = "near_duplicate_review"
    NEAR_DISTINCT = "near_duplicate_distinct"
    NEAR_SAME_SCENE = "near_duplicate_same_scene"


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending_review"
    COMPLETED = "completed"
    POLICY_EXCLUDED_PENDING = "policy_excluded_pending_review"


class ReasonCode(StrEnum):
    NONE = "none"
    IDENTITY_PENDING = "identity_pending_review"
    MANUAL_IDENTITY_EXCLUSION = "manual_identity_exclusion"
    ANNOTATION_REUSED = "annotation_reused"
    UNUSABLE = "unusable"
    EXACT_REDUNDANT = "exact_duplicate_redundant"
    NEAR_SAME_SCENE_REDUNDANT = "near_duplicate_same_scene_redundant"


class IdentityDecision(StrEnum):
    PENDING = "pending_review"
    ACCEPT_XML_NAME = "accept_xml_name"
    ACCEPT_EMBEDDED_FILENAME = "accept_embedded_filename"
    MARK_REUSED = "mark_annotation_reused"
    EXCLUDE_UNUSABLE = "exclude_unusable"


class AnnotationSource(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NONE = "none"


class NearDuplicateDecision(StrEnum):
    PENDING = "pending_review"
    DISTINCT_CAPTURE = "distinct_capture"
    SAME_SCENE = "same_scene"
    UNRELATED = "unrelated_false_positive"


MANIFEST_COLUMNS = [
    "sample_id",
    "image_filename",
    "primary_xml_filename",
    "secondary_xml_filename",
    "primary_declared_filename",
    "secondary_declared_filename",
    "official_split",
    "resolved_image_filename",
    "annotation_source",
    "identity_status",
    "annotation_status",
    "duplicate_group_id",
    "duplicate_status",
    "curated_split",
    "include",
    "reason_code",
    "review_status",
    "reviewer",
    "review_notes",
    "evidence_artifact",
    "source_dataset_fingerprint",
    "curation_version",
]

IDENTITY_DECISION_COLUMNS = [
    "sample_id",
    "decision",
    "resolved_image_filename",
    "annotation_source",
    "include",
    "notes",
    "reviewer",
]

NEAR_DECISION_COLUMNS = [
    "pair_id",
    "image_a",
    "image_b",
    "decision",
    "canonical_sample_id",
    "notes",
    "reviewer",
]


@dataclass(frozen=True)
class CurationResult:
    status: str
    manifest_path: Path
    summary_path: Path
    blockers: tuple[str, ...]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise CurationError(f"{field} must be a boolean; received {value!r}")


def _annotation_signature(annotation: VocAnnotation) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item.raw_label,
            item.bbox.xmin,
            item.bbox.ymin,
            item.bbox.xmax,
            item.bbox.ymax,
        )
        for item in annotation.objects
    )


def _box_text(annotation: VocAnnotation) -> str:
    return ";".join(
        f"{item.raw_label}[{item.bbox.xmin},{item.bbox.ymin},{item.bbox.xmax},{item.bbox.ymax}]"
        for item in annotation.objects
    )


def _label_counter(annotation: VocAnnotation) -> Counter[str]:
    return Counter(item.raw_label for item in annotation.objects)


def match_annotation_objects(primary: VocAnnotation, secondary: VocAnnotation) -> dict[str, Any]:
    """Greedily match same-class boxes by IoU and report transparent diagnostics."""

    candidates: list[tuple[float, int, int]] = []
    for first_index, first in enumerate(primary.objects):
        for second_index, second in enumerate(secondary.objects):
            if first.raw_label == second.raw_label:
                candidates.append((inclusive_iou(first.bbox, second.bbox), first_index, second_index))
    matches: list[tuple[float, int, int]] = []
    used_primary: set[int] = set()
    used_secondary: set[int] = set()
    for iou, first_index, second_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if first_index in used_primary or second_index in used_secondary:
            continue
        used_primary.add(first_index)
        used_secondary.add(second_index)
        matches.append((iou, first_index, second_index))
    ious = [item[0] for item in matches]
    return {
        "primary_object_count": len(primary.objects),
        "secondary_object_count": len(secondary.objects),
        "object_count_agreement": len(primary.objects) == len(secondary.objects),
        "class_multiset_agreement": _label_counter(primary) == _label_counter(secondary),
        "matched_same_class_boxes": len(matches),
        "unmatched_primary": len(primary.objects) - len(used_primary),
        "unmatched_secondary": len(secondary.objects) - len(used_secondary),
        "mean_matched_iou": float(np.mean(ious)) if ious else None,
        "minimum_matched_iou": min(ious) if ious else None,
        "exact_annotation_signature": _annotation_signature(primary) == _annotation_signature(secondary),
    }


def categorize_annotator_comparison(
    primary: VocAnnotation,
    secondary: VocAnnotation,
    comparison: Mapping[str, Any],
    strong_iou: float,
) -> str:
    if Path(primary.filename).name != Path(secondary.filename).name:
        return "different_image_identity"
    if not comparison["object_count_agreement"]:
        return "object_count_disagreement"
    if not comparison["class_multiset_agreement"]:
        return "class_disagreement"
    minimum_iou = comparison["minimum_matched_iou"]
    if minimum_iou is not None and float(minimum_iou) >= strong_iou:
        return "strong_agreement"
    if comparison["matched_same_class_boxes"]:
        return "box_disagreement"
    return "unresolved"


def _thumbnail(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float64) / 255.0


def image_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    raw_mae = float(np.mean(np.abs(first - second)))
    first_flat = first.ravel()
    second_flat = second.ravel()
    if float(first_flat.std()) == 0.0 or float(second_flat.std()) == 0.0:
        correlation = 1.0 if np.array_equal(first_flat, second_flat) else 0.0
    else:
        correlation = float(np.corrcoef(first_flat, second_flat)[0, 1])
    first_z = (first_flat - first_flat.mean()) / max(float(first_flat.std()), 1e-12)
    second_z = (second_flat - second_flat.mean()) / max(float(second_flat.std()), 1e-12)
    normalized_mae = float(np.mean(np.abs(first_z - second_z)))
    return {
        "thumbnail_raw_mae": raw_mae,
        "thumbnail_intensity_correlation": correlation,
        "thumbnail_normalized_mae": normalized_mae,
    }


def current_raw_fingerprint(raw_root: Path) -> str:
    rows = [
        {
            "relative_path": path.relative_to(raw_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(
            (item for item in raw_root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(raw_root).as_posix(),
        )
    ]
    return calculate_dataset_fingerprint(rows)


def verify_raw_fingerprint(raw_root: Path, expected: str) -> str:
    actual = current_raw_fingerprint(raw_root)
    if actual != expected:
        raise CurationError(f"raw fingerprint changed: expected {expected}, observed {actual}")
    return actual


def assert_output_paths_safe(raw_root: Path, paths: Iterable[Path]) -> None:
    raw = raw_root.resolve()
    for path in paths:
        resolved = path.resolve()
        if resolved == raw or raw in resolved.parents:
            raise CurationError(f"refusing to write curation output inside immutable raw root: {resolved}")


def _load_annotations(directory: Path) -> dict[str, VocAnnotation]:
    return {
        path.stem: parse_voc_xml(path)
        for path in sorted(directory.glob("*.xml"), key=lambda item: natural_id_key(item.stem))
    }


def _write_schema(path: Path, version: str) -> None:
    enum_types = [
        IdentityStatus,
        AnnotationStatus,
        DuplicateStatus,
        ReviewStatus,
        ReasonCode,
        IdentityDecision,
        AnnotationSource,
        NearDuplicateDecision,
    ]
    write_json(
        path,
        {
            "schema_version": "1.0",
            "curation_version": version,
            "manifest_columns": MANIFEST_COLUMNS,
            "identity_decision_columns": IDENTITY_DECISION_COLUMNS,
            "near_duplicate_decision_columns": NEAR_DECISION_COLUMNS,
            "enums": {enum.__name__: [item.value for item in enum] for enum in enum_types},
        },
    )


def build_review_evidence(
    config: ResolvedConfig,
    repository_root: str | Path,
    *,
    generate_images: bool = True,
) -> dict[str, Any]:
    """Generate identity/annotator/near-duplicate diagnostics without changing raw files."""

    root = Path(repository_root).resolve()
    cfg = config.as_dict()
    dataset = cfg["dataset"]
    policy = cfg["curation"]
    raw_root = _resolve(root, dataset["raw_root"])
    release_root = _resolve(root, dataset["release_root"])
    metadata_root = _resolve(root, dataset["metadata_root"])
    figures_root = _resolve(root, policy["identity_review_root"])
    outputs = [
        _resolve(root, policy["identity_diagnostics_file"]),
        _resolve(root, policy["annotator_comparison_file"]),
        _resolve(root, policy["near_duplicate_index_file"]),
        _resolve(root, policy["manual_review_file"]),
        _resolve(root, policy["near_duplicate_review_file"]),
        figures_root,
    ]
    assert_output_paths_safe(raw_root, outputs)
    fingerprint = verify_raw_fingerprint(raw_root, policy["expected_raw_fingerprint"])

    image_dir = release_root / dataset["image_directory"]
    primary_dir = release_root / dataset["annotation_directory"]
    secondary_dir = release_root / dataset["second_annotation_directory"]
    primary = _load_annotations(primary_dir)
    secondary = _load_annotations(secondary_dir)
    images = _read_csv(metadata_root / "images.csv")
    image_rows = {row["source_image_id"]: row for row in images}
    image_paths = {row["source_image_id"]: image_dir / row["filename"] for row in images}
    size = int(policy["image_similarity_thumbnail"])
    thumbnails: dict[str, np.ndarray] = {}

    def thumbnail(sample_id: str) -> np.ndarray:
        if sample_id not in thumbnails:
            thumbnails[sample_id] = _thumbnail(image_paths[sample_id], size)
        return thumbnails[sample_id]

    annotator_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    strong_iou = float(policy["annotator_strong_iou"])
    correlation_threshold = float(policy["image_similarity_correlation"])
    for sample_id in sorted(primary, key=natural_id_key):
        first = primary[sample_id]
        second = secondary[sample_id]
        comparison = match_annotation_objects(first, second)
        category = categorize_annotator_comparison(first, second, comparison, strong_iou)
        annotator_rows.append(
            {
                "sample_id": sample_id,
                "primary_declared_filename": first.filename,
                "secondary_declared_filename": second.filename,
                "filename_identity_agreement": Path(first.filename).name == Path(second.filename).name,
                **comparison,
                "agreement_category": category,
            }
        )
        declared_id = Path(first.filename).stem
        if declared_id == sample_id:
            continue
        evidence_flags: list[str] = []
        declared_primary = primary.get(declared_id)
        declared_secondary = secondary.get(declared_id)
        declared_exists = declared_id in image_paths
        similarity = {
            "thumbnail_raw_mae": None,
            "thumbnail_intensity_correlation": None,
            "thumbnail_normalized_mae": None,
        }
        if declared_exists:
            similarity = image_similarity(thumbnail(sample_id), thumbnail(declared_id))
            if similarity["thumbnail_intensity_correlation"] >= correlation_threshold:
                evidence_flags.append("xml_and_embedded_images_highly_correlated")
        if Path(second.filename).stem == declared_id:
            evidence_flags.append("annotators_agree_on_embedded_filename")
        if category == "strong_agreement":
            evidence_flags.append("annotators_agree_strongly_on_current_xml")
        current_vs_declared = (
            match_annotation_objects(first, declared_primary) if declared_primary is not None else None
        )
        secondary_vs_declared = (
            match_annotation_objects(second, declared_secondary) if declared_secondary is not None else None
        )
        if current_vs_declared and current_vs_declared["exact_annotation_signature"]:
            evidence_flags.append("primary_annotation_exactly_repeats_declared_sample")
        elif current_vs_declared and current_vs_declared["class_multiset_agreement"]:
            evidence_flags.append("primary_classes_match_declared_sample")
        recommendation = IdentityStatus.AMBIGUOUS.value
        confidence = "low"
        correlation = similarity["thumbnail_intensity_correlation"]
        if (
            declared_exists
            and correlation is not None
            and correlation >= correlation_threshold
            and category == "strong_agreement"
        ):
            recommendation = IdentityStatus.XML_NAME_CORRECT.value
            confidence = "high"
        diagnostics.append(
            {
                "sample_id": sample_id,
                "xml_named_image": f"{sample_id}.jpg",
                "embedded_filename_image": first.filename,
                "primary_declared_filename": first.filename,
                "secondary_declared_filename": second.filename,
                "declared_image_exists": declared_exists,
                "primary_object_count": len(first.objects),
                "secondary_object_count": len(second.objects),
                "primary_classes": ";".join(item.raw_label for item in first.objects),
                "secondary_classes": ";".join(item.raw_label for item in second.objects),
                "primary_boxes": _box_text(first),
                "secondary_boxes": _box_text(second),
                "primary_declared_dimensions": f"{first.width}x{first.height}",
                "secondary_declared_dimensions": f"{second.width}x{second.height}",
                "xml_named_image_dimensions": (
                    f"{image_rows[sample_id]['decoded_width']}x{image_rows[sample_id]['decoded_height']}"
                ),
                "embedded_filename_image_dimensions": (
                    f"{image_rows[declared_id]['decoded_width']}x{image_rows[declared_id]['decoded_height']}"
                    if declared_exists
                    else ""
                ),
                "annotator_agreement_category": category,
                "annotator_mean_iou": comparison["mean_matched_iou"],
                "declared_primary_object_count": len(declared_primary.objects) if declared_primary else None,
                "primary_vs_declared_exact_signature": (
                    current_vs_declared["exact_annotation_signature"] if current_vs_declared else None
                ),
                "primary_vs_declared_mean_iou": (
                    current_vs_declared["mean_matched_iou"] if current_vs_declared else None
                ),
                "secondary_vs_declared_exact_signature": (
                    secondary_vs_declared["exact_annotation_signature"] if secondary_vs_declared else None
                ),
                "secondary_vs_declared_mean_iou": (
                    secondary_vs_declared["mean_matched_iou"] if secondary_vs_declared else None
                ),
                **similarity,
                "recommended_identity_status": recommendation,
                "recommendation_confidence": confidence,
                "automatic_resolution_applied": False,
                "evidence_flags": ";".join(evidence_flags),
                "review_required": True,
            }
        )

    annotator_path = _resolve(root, policy["annotator_comparison_file"])
    diagnostics_path = _resolve(root, policy["identity_diagnostics_file"])
    write_deterministic_csv(annotator_path, annotator_rows, list(annotator_rows[0]))
    write_deterministic_csv(diagnostics_path, diagnostics, list(diagnostics[0]))

    manual_path = _resolve(root, policy["manual_review_file"])
    if not manual_path.exists():
        write_deterministic_csv(
            manual_path,
            [
                {
                    "sample_id": row["sample_id"],
                    "decision": IdentityDecision.PENDING.value,
                    "resolved_image_filename": "",
                    "annotation_source": AnnotationSource.NONE.value,
                    "include": False,
                    "notes": "",
                    "reviewer": "",
                }
                for row in diagnostics
            ],
            IDENTITY_DECISION_COLUMNS,
        )

    instances = _read_csv(metadata_root / "instances.csv")
    labels_by_id: dict[str, set[str]] = defaultdict(set)
    counts_by_id: Counter[str] = Counter()
    for row in instances:
        label = row["canonical_label_if_unambiguous"]
        if label:
            labels_by_id[row["source_image_id"]].add(label)
        counts_by_id[row["source_image_id"]] += 1
    duplicate_candidates = _read_csv(metadata_root / "duplicate_candidates.csv")
    near_rows: list[dict[str, Any]] = []
    for index, row in enumerate(duplicate_candidates, start=1):
        first_id = Path(row["image_a"]).stem
        second_id = Path(row["image_b"]).stem
        exact = _parse_bool(row["exact_duplicate"], field="exact_duplicate") or _parse_bool(
            row["pixel_duplicate"], field="pixel_duplicate"
        )
        first_labels = labels_by_id[first_id]
        second_labels = labels_by_id[second_id]
        union = first_labels | second_labels
        similarity = image_similarity(thumbnail(first_id), thumbnail(second_id))
        near_rows.append(
            {
                "pair_id": f"pair-{index:04d}",
                **row,
                "annotation_class_jaccard": len(first_labels & second_labels) / len(union) if union else 1.0,
                "annotation_object_count_agreement": counts_by_id[first_id] == counts_by_id[second_id],
                **similarity,
                "recommended_decision": "exact_content" if exact else NearDuplicateDecision.PENDING.value,
                "review_status": ReviewStatus.COMPLETED.value if exact else ReviewStatus.PENDING.value,
            }
        )
    near_rows.sort(
        key=lambda row: (
            not _parse_bool(row["exact_duplicate"], field="exact_duplicate"),
            not _parse_bool(row["cross_split"], field="cross_split"),
            int(row["perceptual_distance"]),
            -float(row["thumbnail_intensity_correlation"]),
            row["pair_id"],
        )
    )
    for rank, row in enumerate(near_rows, start=1):
        row["priority_rank"] = rank
    near_path = _resolve(root, policy["near_duplicate_index_file"])
    near_columns = list(near_rows[0]) if near_rows else ["pair_id"]
    write_deterministic_csv(near_path, near_rows, near_columns)

    near_decision_path = _resolve(root, policy["near_duplicate_review_file"])
    if not near_decision_path.exists():
        nonexact = [
            row
            for row in near_rows
            if not _parse_bool(row["exact_duplicate"], field="exact_duplicate")
            and not _parse_bool(row["pixel_duplicate"], field="pixel_duplicate")
        ]
        write_deterministic_csv(
            near_decision_path,
            [
                {
                    "pair_id": row["pair_id"],
                    "image_a": row["image_a"],
                    "image_b": row["image_b"],
                    "decision": NearDuplicateDecision.PENDING.value,
                    "canonical_sample_id": "",
                    "notes": "",
                    "reviewer": "",
                }
                for row in nonexact
            ],
            NEAR_DECISION_COLUMNS,
        )

    review_index: list[dict[str, Any]] = []
    if generate_images:
        from windblade.data.visualization import create_identity_review_sheets

        review_index = create_identity_review_sheets(
            diagnostics,
            primary,
            secondary,
            image_paths,
            figures_root,
            int(policy["identity_samples_per_sheet"]),
        )
        write_deterministic_csv(
            figures_root / "index.csv",
            review_index,
            list(review_index[0]) if review_index else ["sample_id"],
        )
        write_json(
            figures_root / "index.json",
            {"schema_version": "1.0", "items": review_index},
        )

    aggregate = Counter(row["agreement_category"] for row in annotator_rows)
    possible_reuse = sum(
        1
        for row in diagnostics
        if "annotation_exactly_repeats_declared_sample" in str(row["evidence_flags"])
    )
    summary = {
        "schema_version": "1.0",
        "curation_version": policy["version"],
        "config_hash": config.config_hash,
        "source_dataset_fingerprint": fingerprint,
        "identity_mismatch_count": len(diagnostics),
        "automatic_identity_resolutions_applied": 0,
        "identity_review_artifact_count": len(review_index),
        "annotator_comparison": {
            "sample_count": len(annotator_rows),
            "categories": dict(sorted(aggregate.items())),
            "mismatch_rows_with_exact_primary_reuse_signature": possible_reuse,
        },
        "duplicate_review": {
            "candidate_rows": len(near_rows),
            "nonexact_pending": sum(
                1
                for row in near_rows
                if not _parse_bool(row["exact_duplicate"], field="exact_duplicate")
                and not _parse_bool(row["pixel_duplicate"], field="pixel_duplicate")
            ),
            "cross_split_nonexact_pending": sum(
                1
                for row in near_rows
                if _parse_bool(row["cross_split"], field="cross_split")
                and not _parse_bool(row["exact_duplicate"], field="exact_duplicate")
                and not _parse_bool(row["pixel_duplicate"], field="pixel_duplicate")
            ),
        },
    }
    write_json(metadata_root / "review_summary.json", summary)
    if verify_raw_fingerprint(raw_root, fingerprint) != fingerprint:
        raise AssertionError("unreachable fingerprint mismatch")
    return summary


def _validate_decision_columns(rows: Sequence[Mapping[str, str]], expected: Sequence[str], name: str) -> None:
    if not rows:
        return
    missing = [field for field in expected if field not in rows[0]]
    if missing:
        raise CurationError(f"{name} is missing columns: {', '.join(missing)}")


def apply_identity_decisions(
    base_rows: list[dict[str, Any]],
    decisions: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Merge validated human identity decisions into manifest rows deterministically."""

    _validate_decision_columns(decisions, IDENTITY_DECISION_COLUMNS, "manual review file")
    by_id = {row["sample_id"]: dict(row) for row in base_rows}
    seen: set[str] = set()
    for decision_row in decisions:
        sample_id = decision_row["sample_id"]
        if sample_id in seen:
            raise CurationError(f"duplicate manual identity decision for sample {sample_id}")
        seen.add(sample_id)
        if sample_id not in by_id:
            raise CurationError(f"manual identity decision references unknown sample {sample_id}")
        decision = IdentityDecision(decision_row["decision"])
        row = by_id[sample_id]
        if decision is IdentityDecision.PENDING:
            continue
        reviewer = decision_row["reviewer"].strip()
        if not reviewer:
            raise CurationError(f"completed decision for {sample_id} requires reviewer")
        include = _parse_bool(decision_row["include"], field=f"{sample_id}.include")
        annotation_source = AnnotationSource(decision_row["annotation_source"])
        if decision in {IdentityDecision.ACCEPT_XML_NAME, IdentityDecision.ACCEPT_EMBEDDED_FILENAME}:
            resolved = decision_row["resolved_image_filename"].strip()
            if not resolved or annotation_source is AnnotationSource.NONE:
                raise CurationError(f"accepted identity decision for {sample_id} requires image and annotation source")
            expected = (
                row["image_filename"]
                if decision is IdentityDecision.ACCEPT_XML_NAME
                else row["primary_declared_filename"]
            )
            if Path(resolved).name != Path(str(expected)).name:
                raise CurationError(f"resolved image for {sample_id} does not match decision {decision.value}")
            row["resolved_image_filename"] = resolved
            row["identity_status"] = (
                IdentityStatus.XML_NAME_CORRECT.value
                if decision is IdentityDecision.ACCEPT_XML_NAME
                else IdentityStatus.EMBEDDED_FILENAME_CORRECT.value
            )
            row["annotation_status"] = (
                AnnotationStatus.PRIMARY_CONFIRMED.value
                if annotation_source is AnnotationSource.PRIMARY
                else AnnotationStatus.SECONDARY_CONFIRMED.value
            )
            row["annotation_source"] = annotation_source.value
            row["include"] = include
            row["reason_code"] = ReasonCode.NONE.value if include else ReasonCode.MANUAL_IDENTITY_EXCLUSION.value
        elif decision is IdentityDecision.MARK_REUSED:
            row["identity_status"] = IdentityStatus.ANNOTATION_REUSED.value
            row["annotation_status"] = AnnotationStatus.EXCLUDE.value
            row["annotation_source"] = AnnotationSource.NONE.value
            row["include"] = False
            row["reason_code"] = ReasonCode.ANNOTATION_REUSED.value
        else:
            row["identity_status"] = IdentityStatus.UNUSABLE.value
            row["annotation_status"] = AnnotationStatus.EXCLUDE.value
            row["annotation_source"] = AnnotationSource.NONE.value
            row["include"] = False
            row["reason_code"] = ReasonCode.UNUSABLE.value
        row["review_status"] = ReviewStatus.COMPLETED.value
        row["reviewer"] = reviewer
        row["review_notes"] = decision_row["notes"].strip()
    return [by_id[key] for key in sorted(by_id, key=natural_id_key)]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            canonical = min((left, right), key=natural_id_key)
            self.parent[left] = canonical
            self.parent[right] = canonical


def _exact_group_map(images: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, bool]]:
    records = [
        {
            "source_image_id": row["source_image_id"],
            "sha256": row["sha256"],
            "pixel_sha256": row["pixel_sha256"],
        }
        for row in images
    ]
    combined = duplicate_groups(records, "sha256")
    seen = {tuple(group) for group in combined}
    for group in duplicate_groups(records, "pixel_sha256"):
        if tuple(group) not in seen:
            combined.append(group)
    combined.sort(key=lambda group: natural_id_key(group[0]))
    result: dict[str, tuple[str, bool]] = {}
    for index, group in enumerate(combined, start=1):
        canonical = min(group, key=natural_id_key)
        group_id = f"exact-{index:03d}"
        for sample_id in group:
            result[sample_id] = (group_id, sample_id == canonical)
    return result


def _apply_near_decisions(
    rows: list[dict[str, Any]],
    candidates: Sequence[Mapping[str, str]],
    decisions: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    _validate_decision_columns(decisions, NEAR_DECISION_COLUMNS, "near-duplicate review file")
    candidate_by_pair = {row["pair_id"]: row for row in candidates}
    union = _UnionFind()
    completed = 0
    seen: set[str] = set()
    for decision_row in decisions:
        pair_id = decision_row["pair_id"]
        if pair_id in seen:
            raise CurationError(f"duplicate near-duplicate decision for {pair_id}")
        seen.add(pair_id)
        if pair_id not in candidate_by_pair:
            raise CurationError(f"near-duplicate decision references unknown pair {pair_id}")
        decision = NearDuplicateDecision(decision_row["decision"])
        if decision is NearDuplicateDecision.PENDING:
            continue
        if not decision_row["reviewer"].strip():
            raise CurationError(f"completed near-duplicate decision {pair_id} requires reviewer")
        completed += 1
        if decision is NearDuplicateDecision.SAME_SCENE:
            candidate = candidate_by_pair[pair_id]
            union.union(Path(candidate["image_a"]).stem, Path(candidate["image_b"]).stem)
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in union.parent:
        groups[union.find(sample_id)].append(sample_id)
    by_id = {row["sample_id"]: row for row in rows}
    for index, group in enumerate(sorted(groups.values(), key=lambda g: natural_id_key(min(g, key=natural_id_key))), start=1):
        canonical = min(group, key=natural_id_key)
        group_id = f"near-reviewed-{index:03d}"
        for sample_id in group:
            row = by_id[sample_id]
            if row["duplicate_status"] in {DuplicateStatus.EXACT_CANONICAL.value, DuplicateStatus.EXACT_REDUNDANT.value}:
                continue
            row["duplicate_group_id"] = group_id
            row["duplicate_status"] = DuplicateStatus.NEAR_SAME_SCENE.value
            if sample_id != canonical:
                row["include"] = False
                row["curated_split"] = ""
                row["reason_code"] = ReasonCode.NEAR_SAME_SCENE_REDUNDANT.value
    return rows, completed


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["curation manifest is empty"]
    missing = [field for field in MANIFEST_COLUMNS if field not in rows[0]]
    if missing:
        errors.append(f"manifest missing columns: {', '.join(missing)}")
        return errors
    valid_identity = {item.value for item in IdentityStatus}
    valid_annotation = {item.value for item in AnnotationStatus}
    valid_duplicate = {item.value for item in DuplicateStatus}
    included_by_group: dict[str, set[str]] = defaultdict(set)
    resolved_included: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["sample_id"])
        if row["identity_status"] not in valid_identity:
            errors.append(f"{sample_id}: invalid identity_status")
        if row["annotation_status"] not in valid_annotation:
            errors.append(f"{sample_id}: invalid annotation_status")
        if row["duplicate_status"] not in valid_duplicate:
            errors.append(f"{sample_id}: invalid duplicate_status")
        included = row["include"] if isinstance(row["include"], bool) else _parse_bool(row["include"], field="include")
        if included:
            if row["identity_status"] in {IdentityStatus.PENDING_REVIEW.value, IdentityStatus.AMBIGUOUS.value}:
                errors.append(f"{sample_id}: unresolved identity is included")
            if not row["curated_split"]:
                errors.append(f"{sample_id}: included row lacks curated split")
            if row["annotation_status"] in {AnnotationStatus.EXCLUDE.value, AnnotationStatus.MANUAL_REVIEW_REQUIRED.value}:
                errors.append(f"{sample_id}: included row lacks confirmed annotation")
            resolved_included[str(row["resolved_image_filename"])].append(sample_id)
            group = str(row["duplicate_group_id"])
            if group:
                included_by_group[group].add(str(row["curated_split"]))
    for group, splits in included_by_group.items():
        if len(splits) > 1:
            errors.append(f"duplicate group {group} crosses curated splits: {sorted(splits)}")
    for filename, ids in resolved_included.items():
        if filename and len(ids) > 1:
            errors.append(f"resolved image {filename} is included by multiple samples: {ids}")
    return errors


def curated_statistics(
    manifest_rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic curated counts from included manifest rows only."""

    included_ids = {
        str(row["sample_id"])
        for row in manifest_rows
        if (
            row["include"]
            if isinstance(row["include"], bool)
            else _parse_bool(row["include"], field="include")
        )
    }
    included_instances = [row for row in instance_rows if str(row["source_image_id"]) in included_ids]
    classes = Counter(str(row["canonical_label_if_unambiguous"]) for row in included_instances)
    splits = Counter(
        str(row["curated_split"])
        for row in manifest_rows
        if str(row["sample_id"]) in included_ids
    )
    return {
        "included_image_count": len(included_ids),
        "object_count": len(included_instances),
        "class_counts": {label: classes[label] for label in CANONICAL_CLASSES},
        "split_counts": {split: splits[split] for split in ("train", "validation", "test")},
    }


def build_curation(
    config: ResolvedConfig,
    repository_root: str | Path,
    *,
    output_directory: Path | None = None,
) -> CurationResult:
    root = Path(repository_root).resolve()
    cfg = config.as_dict()
    dataset = cfg["dataset"]
    policy = cfg["curation"]
    raw_root = _resolve(root, dataset["raw_root"])
    release_root = _resolve(root, dataset["release_root"])
    metadata_root = output_directory.resolve() if output_directory else _resolve(root, dataset["metadata_root"])
    manifest_path = metadata_root / Path(policy["manifest_file"]).name
    summary_path = metadata_root / Path(policy["summary_file"]).name
    schema_path = metadata_root / Path(policy["schema_file"]).name
    assert_output_paths_safe(raw_root, [metadata_root, manifest_path, summary_path, schema_path])
    fingerprint = verify_raw_fingerprint(raw_root, policy["expected_raw_fingerprint"])
    metadata_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, metadata_root / "curation_resolved_config.yaml")
    _write_schema(schema_path, policy["version"])

    audit_metadata = _resolve(root, dataset["metadata_root"])
    images = _read_csv(audit_metadata / "images.csv")
    instances = _read_csv(audit_metadata / "instances.csv")
    diagnostics = _read_csv(_resolve(root, policy["identity_diagnostics_file"]))
    diagnostic_by_id = {row["sample_id"]: row for row in diagnostics}
    primary_dir = release_root / dataset["annotation_directory"]
    secondary_dir = release_root / dataset["second_annotation_directory"]
    primary = _load_annotations(primary_dir)
    secondary = _load_annotations(secondary_dir)
    exact_map = _exact_group_map(images)
    near_candidates = _read_csv(_resolve(root, policy["near_duplicate_index_file"]))
    near_ids = {
        Path(value).stem
        for row in near_candidates
        if not _parse_bool(row["exact_duplicate"], field="exact_duplicate")
        and not _parse_bool(row["pixel_duplicate"], field="pixel_duplicate")
        for value in (row["image_a"], row["image_b"])
    }
    rows: list[dict[str, Any]] = []
    for image in sorted(images, key=lambda row: natural_id_key(row["source_image_id"])):
        sample_id = image["source_image_id"]
        first = primary[sample_id]
        second = secondary[sample_id]
        mismatch = Path(first.filename).stem != sample_id
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "image_filename": image["filename"],
            "primary_xml_filename": f"{sample_id}.xml",
            "secondary_xml_filename": f"{sample_id}.xml",
            "primary_declared_filename": first.filename,
            "secondary_declared_filename": second.filename,
            "official_split": image["official_split"],
            "resolved_image_filename": "" if mismatch else image["filename"],
            "annotation_source": AnnotationSource.NONE.value if mismatch else AnnotationSource.PRIMARY.value,
            "identity_status": IdentityStatus.PENDING_REVIEW.value if mismatch else IdentityStatus.CONSISTENT.value,
            "annotation_status": AnnotationStatus.MANUAL_REVIEW_REQUIRED.value if mismatch else AnnotationStatus.PRIMARY_CONFIRMED.value,
            "duplicate_group_id": "",
            "duplicate_status": DuplicateStatus.NEAR_REVIEW.value if sample_id in near_ids else DuplicateStatus.UNIQUE.value,
            "curated_split": "" if mismatch else image["official_split"],
            "include": not mismatch,
            "reason_code": ReasonCode.IDENTITY_PENDING.value if mismatch else ReasonCode.NONE.value,
            "review_status": ReviewStatus.POLICY_EXCLUDED_PENDING.value if mismatch else ReviewStatus.NOT_REQUIRED.value,
            "reviewer": "",
            "review_notes": "",
            "evidence_artifact": (
                f"figures/phase2/identity_review/index.csv#{sample_id}"
                if mismatch
                else "data/metadata/wtbd/second_annotator_comparison.csv"
            ),
            "source_dataset_fingerprint": fingerprint,
            "curation_version": policy["version"],
        }
        if mismatch and sample_id not in diagnostic_by_id:
            raise CurationError(f"identity mismatch {sample_id} lacks diagnostics")
        rows.append(row)

    identity_decisions = _read_csv(_resolve(root, policy["manual_review_file"]))
    rows = apply_identity_decisions(rows, identity_decisions)
    by_id = {row["sample_id"]: row for row in rows}
    for sample_id, (group_id, canonical) in exact_map.items():
        row = by_id[sample_id]
        row["duplicate_group_id"] = group_id
        row["duplicate_status"] = (
            DuplicateStatus.EXACT_CANONICAL.value if canonical else DuplicateStatus.EXACT_REDUNDANT.value
        )
        if not canonical:
            row["include"] = False
            row["curated_split"] = ""
            row["reason_code"] = ReasonCode.EXACT_REDUNDANT.value

    near_decisions = _read_csv(_resolve(root, policy["near_duplicate_review_file"]))
    rows, completed_near = _apply_near_decisions(rows, near_candidates, near_decisions)
    for row in rows:
        included = bool(row["include"])
        row["curated_split"] = row["official_split"] if included else ""
    validation_errors = validate_manifest(rows)
    if validation_errors:
        raise CurationError("curation manifest validation failed: " + "; ".join(validation_errors))
    write_deterministic_csv(manifest_path, rows, MANIFEST_COLUMNS)

    included_ids = {row["sample_id"] for row in rows if row["include"]}
    curated_instances = [row for row in instances if row["source_image_id"] in included_ids]
    write_deterministic_csv(
        metadata_root / "curated_instances.csv",
        curated_instances,
        list(curated_instances[0]) if curated_instances else list(instances[0]),
    )
    split_rows = [
        {"sample_id": row["sample_id"], "curated_split": row["curated_split"]}
        for row in rows
        if row["include"]
    ]
    write_deterministic_csv(metadata_root / "curated_split_membership.csv", split_rows, ["sample_id", "curated_split"])
    computed_statistics = curated_statistics(rows, instances)
    class_counts = Counter(computed_statistics["class_counts"])
    class_rows = [{"class": label, "instance_count": class_counts[label]} for label in CANONICAL_CLASSES]
    write_deterministic_csv(metadata_root / "curated_class_counts.csv", class_rows, ["class", "instance_count"])
    split_counts = Counter(computed_statistics["split_counts"])
    pending_identity = sum(
        1 for row in rows if row["identity_status"] == IdentityStatus.PENDING_REVIEW.value
    )
    pending_near = sum(
        1
        for row in near_decisions
        if row["decision"] == NearDuplicateDecision.PENDING.value
    )
    cross_pending_near = sum(
        1
        for decision in near_decisions
        if decision["decision"] == NearDuplicateDecision.PENDING.value
        and _parse_bool(candidate_by_pair[decision["pair_id"]]["cross_split"], field="cross_split")
    ) if (candidate_by_pair := {row["pair_id"]: row for row in near_candidates}) else 0
    blockers: list[str] = []
    if pending_identity:
        blockers.append(f"{pending_identity} identity decisions remain pending (all are policy-excluded)")
    if pending_near:
        blockers.append(
            f"{pending_near} non-exact near-duplicate decisions remain pending, including {cross_pending_near} cross-split pairs"
        )
    status = "PASS" if not blockers else "BLOCKED_PENDING_HUMAN_REVIEW"
    raw_summary = read_json(audit_metadata / "audit_summary.json")
    summary = {
        "schema_version": "1.0",
        "curation_version": policy["version"],
        "status": status,
        "source_dataset_fingerprint": fingerprint,
        "raw_official_release": {
            "image_count": raw_summary["images"]["count"],
            "object_count": raw_summary["annotations"]["instance_count"],
            "class_counts": raw_summary["classes"]["canonical_counts"],
            "split_counts": raw_summary["split"]["details"]["counts"],
            "identity_mismatch_count": len(raw_summary["annotations"]["xml_filename_mismatches"]),
            "exact_duplicate_groups": raw_summary["duplicates"]["exact_file_groups"],
            "nonexact_near_duplicate_candidates": raw_summary["duplicates"]["near_duplicate_candidates"],
        },
        "curated_interpretation": {
            "included_images": len(included_ids),
            "excluded_images": len(rows) - len(included_ids),
            "object_count": len(curated_instances),
            "class_counts": {label: class_counts[label] for label in CANONICAL_CLASSES},
            "split_counts": {split: split_counts[split] for split in ("train", "validation", "test")},
            "excluded_exact_duplicates": sum(
                1 for row in rows if row["duplicate_status"] == DuplicateStatus.EXACT_REDUNDANT.value
            ),
            "pending_identity_rows": pending_identity,
            "pending_near_duplicate_pairs": pending_near,
            "pending_cross_split_near_duplicate_pairs": cross_pending_near,
            "completed_near_duplicate_decisions": completed_near,
            "included_unresolved_identity_rows": 0,
            "included_exact_duplicate_groups_crossing_splits": 0,
            "confirmed_same_scene_groups_crossing_splits": 0,
        },
        "blockers": blockers,
        "raw_data_modified": False,
        "published_counts_forced": False,
        "phase_3_started": False,
    }
    write_json(summary_path, summary)
    write_deterministic_csv(
        metadata_root / "curation_blockers.csv",
        (
            [
                {"blocker_type": "identity", "record_id": row["sample_id"], "detail": row["reason_code"]}
                for row in rows
                if row["identity_status"] == IdentityStatus.PENDING_REVIEW.value
            ]
            + [
                {
                    "blocker_type": "near_duplicate",
                    "record_id": row["pair_id"],
                    "detail": "cross_split" if _parse_bool(candidate_by_pair[row["pair_id"]]["cross_split"], field="cross_split") else "within_split",
                }
                for row in near_decisions
                if row["decision"] == NearDuplicateDecision.PENDING.value
            ]
        ),
        ["blocker_type", "record_id", "detail"],
    )
    verify_raw_fingerprint(raw_root, fingerprint)
    return CurationResult(status, manifest_path, summary_path, tuple(blockers))


def validate_existing_curation(config: ResolvedConfig, repository_root: str | Path) -> CurationResult:
    root = Path(repository_root).resolve()
    cfg = config.as_dict()
    dataset = cfg["dataset"]
    policy = cfg["curation"]
    raw_root = _resolve(root, dataset["raw_root"])
    fingerprint = verify_raw_fingerprint(raw_root, policy["expected_raw_fingerprint"])
    manifest_path = _resolve(root, policy["manifest_file"])
    summary_path = _resolve(root, policy["summary_file"])
    rows = _read_csv(manifest_path)
    errors = validate_manifest(rows)
    if errors:
        raise CurationError("existing manifest failed validation: " + "; ".join(errors))
    if any(row["source_dataset_fingerprint"] != fingerprint for row in rows):
        raise CurationError("manifest fingerprint does not match immutable raw release")
    summary = read_json(summary_path)
    return CurationResult(
        str(summary["status"]),
        manifest_path,
        summary_path,
        tuple(str(item) for item in summary.get("blockers", [])),
    )
