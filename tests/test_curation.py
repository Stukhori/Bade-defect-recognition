from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from windblade.data.curation import (
    AnnotationSource,
    AnnotationStatus,
    CurationError,
    DuplicateStatus,
    IdentityDecision,
    IdentityStatus,
    MANIFEST_COLUMNS,
    NearDuplicateDecision,
    ReasonCode,
    ReviewStatus,
    _apply_near_decisions,
    _exact_group_map,
    apply_identity_decisions,
    assert_output_paths_safe,
    build_same_scene_components,
    classify_pending_near_duplicates,
    curated_statistics,
    current_raw_fingerprint,
    match_annotation_objects,
    missing_classes_by_split,
    same_scene_component_violations,
    validate_manifest,
    verify_raw_fingerprint,
)
from windblade.data.voc import parse_voc_xml


def manifest_row(sample_id: str, split: str = "train") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "image_filename": f"{sample_id}.jpg",
        "primary_xml_filename": f"{sample_id}.xml",
        "secondary_xml_filename": f"{sample_id}.xml",
        "primary_declared_filename": f"{sample_id}.jpg",
        "secondary_declared_filename": f"{sample_id}.jpg",
        "official_split": split,
        "resolved_image_filename": f"{sample_id}.jpg",
        "annotation_source": AnnotationSource.PRIMARY.value,
        "identity_status": IdentityStatus.CONSISTENT.value,
        "annotation_status": AnnotationStatus.PRIMARY_CONFIRMED.value,
        "duplicate_group_id": "",
        "duplicate_status": DuplicateStatus.UNIQUE.value,
        "curated_split": split,
        "include": True,
        "reason_code": ReasonCode.NONE.value,
        "review_status": ReviewStatus.NOT_REQUIRED.value,
        "reviewer": "",
        "review_notes": "",
        "evidence_artifact": "evidence.csv",
        "source_dataset_fingerprint": "fingerprint",
        "curation_version": "test-v1",
    }


def xml_text(filename: str, label: str, box: tuple[int, int, int, int]) -> str:
    xmin, ymin, xmax, ymax = box
    return f"""<annotation><filename>{filename}</filename><size><width>32</width><height>32</height><depth>3</depth></size><object><name>{label}</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>"""


def test_manifest_schema_and_enums_are_centralized() -> None:
    assert len(MANIFEST_COLUMNS) == len(set(MANIFEST_COLUMNS))
    assert "source_dataset_fingerprint" in MANIFEST_COLUMNS
    assert IdentityStatus("xml_name_correct") is IdentityStatus.XML_NAME_CORRECT
    assert AnnotationStatus("manual_review_required") is AnnotationStatus.MANUAL_REVIEW_REQUIRED
    assert DuplicateStatus("exact_duplicate_redundant") is DuplicateStatus.EXACT_REDUNDANT
    with pytest.raises(ValueError):
        IdentityStatus("invented_status")


def test_manual_review_merge_is_deterministic_and_requires_reviewer() -> None:
    base = manifest_row("743")
    base.update(
        {
            "primary_declared_filename": "10.jpg",
            "resolved_image_filename": "",
            "annotation_source": AnnotationSource.NONE.value,
            "identity_status": IdentityStatus.PENDING_REVIEW.value,
            "annotation_status": AnnotationStatus.MANUAL_REVIEW_REQUIRED.value,
            "curated_split": "",
            "include": False,
            "reason_code": ReasonCode.IDENTITY_PENDING.value,
        }
    )
    decision = {
        "sample_id": "743",
        "decision": IdentityDecision.ACCEPT_XML_NAME.value,
        "resolved_image_filename": "743.jpg",
        "annotation_source": AnnotationSource.PRIMARY.value,
        "include": "true",
        "notes": "reviewed four-panel sheet",
        "reviewer": "reviewer-1",
    }
    first = apply_identity_decisions([base], [decision])
    second = apply_identity_decisions([base], [decision])
    assert first == second
    assert first[0]["identity_status"] == IdentityStatus.XML_NAME_CORRECT.value
    assert first[0]["include"] is True
    with pytest.raises(CurationError, match="requires reviewer"):
        apply_identity_decisions([base], [{**decision, "reviewer": ""}])


def test_pending_identity_is_excluded_and_never_valid_when_included() -> None:
    row = manifest_row("743")
    row["identity_status"] = IdentityStatus.PENDING_REVIEW.value
    row["annotation_status"] = AnnotationStatus.MANUAL_REVIEW_REQUIRED.value
    row["include"] = False
    row["curated_split"] = ""
    assert validate_manifest([row]) == []
    row["include"] = True
    row["curated_split"] = "train"
    errors = validate_manifest([row])
    assert any("unresolved identity is included" in item for item in errors)
    assert any("lacks confirmed annotation" in item for item in errors)


def test_output_guard_rejects_raw_mutation_paths(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert_output_paths_safe(raw, [tmp_path / "metadata" / "manifest.csv"])
    with pytest.raises(CurationError, match="immutable raw root"):
        assert_output_paths_safe(raw, [raw / "derived.csv"])


def test_exact_duplicate_canonicalization_uses_lowest_natural_id() -> None:
    images = [
        {"source_image_id": "640", "sha256": "a", "pixel_sha256": "p"},
        {"source_image_id": "547", "sha256": "a", "pixel_sha256": "p"},
        {"source_image_id": "700", "sha256": "b", "pixel_sha256": "q"},
    ]
    groups = _exact_group_map(images)
    assert groups["547"] == ("exact-001", True)
    assert groups["640"] == ("exact-001", False)
    assert "700" not in groups


def test_cross_split_duplicate_group_is_rejected() -> None:
    first = manifest_row("547", "train")
    second = manifest_row("640", "validation")
    for row in (first, second):
        row["duplicate_group_id"] = "exact-001"
        row["duplicate_status"] = DuplicateStatus.EXACT_CANONICAL.value
    errors = validate_manifest([first, second])
    assert errors == ["duplicate group exact-001 crosses curated splits: ['train', 'validation']"]


def test_same_scene_decision_excludes_redundant_sample() -> None:
    rows = [manifest_row("10", "train"), manifest_row("743", "test")]
    candidates = [{"pair_id": "pair-0001", "image_a": "10.jpg", "image_b": "743.jpg"}]
    decisions = [
        {
            "pair_id": "pair-0001",
            "image_a": "10.jpg",
            "image_b": "743.jpg",
            "decision": NearDuplicateDecision.SAME_SCENE.value,
            "canonical_sample_id": "10",
            "notes": "confirmed",
            "reviewer": "reviewer-1",
        }
    ]
    resolved, completed, components = _apply_near_decisions(rows, candidates, decisions)
    by_id = {row["sample_id"]: row for row in resolved}
    assert completed == 1
    assert len(components) == 1
    assert by_id["10"]["include"] is True
    assert by_id["743"]["include"] is False
    assert by_id["743"]["reason_code"] == ReasonCode.NEAR_SAME_SCENE_REDUNDANT.value


def test_same_scene_canonical_must_be_deterministic_lowest_id() -> None:
    rows = [manifest_row("10", "train"), manifest_row("743", "test")]
    candidates = [{"pair_id": "pair-0001", "image_a": "10.jpg", "image_b": "743.jpg"}]
    decisions = [
        {
            "pair_id": "pair-0001",
            "image_a": "10.jpg",
            "image_b": "743.jpg",
            "decision": NearDuplicateDecision.SAME_SCENE.value,
            "canonical_sample_id": "743",
            "notes": "confirmed",
            "reviewer": "reviewer-1",
        }
    ]
    with pytest.raises(CurationError, match="computed lowest ID 10"):
        _apply_near_decisions(rows, candidates, decisions)


def pending_decision(pair_id: str, first: str, second: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "image_a": f"{first}.jpg",
        "image_b": f"{second}.jpg",
        "decision": NearDuplicateDecision.PENDING.value,
        "canonical_sample_id": "",
        "notes": "",
        "reviewer": "",
    }


def test_pending_pair_with_excluded_sample_is_not_a_blocker() -> None:
    first = manifest_row("1", "train")
    second = manifest_row("2", "test")
    second["include"] = False
    second["curated_split"] = ""
    candidates = [{"pair_id": "p1", "image_a": "1.jpg", "image_b": "2.jpg"}]
    result = classify_pending_near_duplicates([first, second], candidates, [pending_decision("p1", "1", "2")])
    assert result["involving_excluded_images"] == ["p1"]
    assert result["cross_split_retained_pairs"] == []


def test_pending_within_split_pair_is_not_a_blocker() -> None:
    rows = [manifest_row("1", "train"), manifest_row("2", "train")]
    candidates = [{"pair_id": "p1", "image_a": "1.jpg", "image_b": "2.jpg"}]
    result = classify_pending_near_duplicates(rows, candidates, [pending_decision("p1", "1", "2")])
    assert result["within_split_retained_pairs"] == ["p1"]
    assert result["cross_split_retained_pairs"] == []


def test_pending_cross_split_retained_pair_is_a_blocker() -> None:
    rows = [manifest_row("1", "train"), manifest_row("2", "test")]
    candidates = [{"pair_id": "p1", "image_a": "1.jpg", "image_b": "2.jpg"}]
    result = classify_pending_near_duplicates(rows, candidates, [pending_decision("p1", "1", "2")])
    assert result["cross_split_retained_pairs"] == ["p1"]


def test_transitive_same_scene_component_keeps_only_lowest_id() -> None:
    candidates = [
        {"pair_id": "p1", "image_a": "10.jpg", "image_b": "30.jpg"},
        {"pair_id": "p2", "image_a": "20.jpg", "image_b": "30.jpg"},
    ]
    decisions = [
        {**pending_decision("p1", "10", "30"), "decision": "same_scene", "canonical_sample_id": "10", "reviewer": "r"},
        {**pending_decision("p2", "20", "30"), "decision": "same_scene", "canonical_sample_id": "10", "reviewer": "r"},
    ]
    rows = [manifest_row("10", "test"), manifest_row("20", "train"), manifest_row("30", "validation")]
    resolved, completed, components = _apply_near_decisions(rows, candidates, decisions)
    by_id = {row["sample_id"]: row for row in resolved}
    assert completed == 2
    assert components[0].canonical_sample_id == "10"
    assert components[0].members == ("10", "20", "30")
    assert by_id["10"]["include"] is True
    assert by_id["20"]["include"] is False
    assert by_id["30"]["include"] is False
    assert by_id["20"]["reason_code"] == ReasonCode.NEAR_SAME_SCENE_REDUNDANT.value
    assert same_scene_component_violations(resolved, components) == []


def test_completed_unrelated_cross_split_pair_remains_included() -> None:
    rows = [manifest_row("1", "train"), manifest_row("2", "test")]
    candidates = [{"pair_id": "p1", "image_a": "1.jpg", "image_b": "2.jpg"}]
    decision = {
        **pending_decision("p1", "1", "2"),
        "decision": NearDuplicateDecision.UNRELATED.value,
        "reviewer": "r",
    }
    resolved, completed, components = _apply_near_decisions(rows, candidates, [decision])
    assert completed == 1
    assert components == ()
    assert all(row["include"] for row in resolved)


def test_all_classes_must_remain_in_every_split() -> None:
    labels = ("craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze")
    counts = {(split, label): 1 for split in ("train", "validation", "test") for label in labels}
    assert missing_classes_by_split(counts) == []
    counts[("test", "thunderstrike")] = 0
    assert missing_classes_by_split(counts) == ["test:thunderstrike"]


def test_curated_statistics_are_deterministic_and_exclude_rows() -> None:
    included = manifest_row("1", "train")
    excluded = manifest_row("2", "test")
    excluded["include"] = False
    excluded["curated_split"] = ""
    instances = [
        {"source_image_id": "1", "canonical_label_if_unambiguous": "craze"},
        {"source_image_id": "1", "canonical_label_if_unambiguous": "crack"},
        {"source_image_id": "2", "canonical_label_if_unambiguous": "corrosion"},
    ]
    first = curated_statistics([excluded, included], list(reversed(instances)))
    second = curated_statistics([included, excluded], instances)
    assert first == second
    assert first["included_image_count"] == 1
    assert first["object_count"] == 2
    assert first["class_counts"]["corrosion"] == 0


def test_second_annotator_comparison_reports_iou(tmp_path: Path) -> None:
    primary_path = tmp_path / "primary.xml"
    secondary_path = tmp_path / "secondary.xml"
    primary_path.write_text(xml_text("7.jpg", "craze", (1, 1, 20, 20)), encoding="utf-8")
    secondary_path.write_text(xml_text("7.jpg", "craze", (2, 2, 20, 20)), encoding="utf-8")
    comparison = match_annotation_objects(parse_voc_xml(primary_path), parse_voc_xml(secondary_path))
    assert comparison["object_count_agreement"] is True
    assert comparison["class_multiset_agreement"] is True
    assert comparison["matched_same_class_boxes"] == 1
    assert 0.8 < comparison["mean_matched_iou"] < 1.0


def test_raw_fingerprint_is_stable_and_detects_change(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image_path = raw / "sample.png"
    Image.new("RGB", (4, 4), "white").save(image_path)
    first = current_raw_fingerprint(raw)
    assert verify_raw_fingerprint(raw, first) == first
    Image.new("RGB", (4, 4), "black").save(image_path)
    with pytest.raises(CurationError, match="raw fingerprint changed"):
        verify_raw_fingerprint(raw, first)
