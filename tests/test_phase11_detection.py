from __future__ import annotations

import json
from pathlib import Path

import pytest

from windblade.config import load_config
from windblade.detection import core


ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/detection.yaml")


@pytest.fixture(scope="module")
def audit():
    return core._audit(config(), ROOT)


def test_voc_inclusive_coordinate_conversion() -> None:
    assert core.voc_inclusive_to_yolo((1, 1, 10, 20), 10, 20) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_image_dimension_validation() -> None:
    with pytest.raises(core.DetectionAuditError, match="dimensions"):
        core.validate_box((1, 1, 1, 1), 0, 10)


def test_invalid_and_out_of_bounds_boxes_are_rejected() -> None:
    with pytest.raises(core.DetectionAuditError, match="non-positive"):
        core.validate_box((5, 5, 4, 9), 10, 10)
    with pytest.raises(core.DetectionAuditError, match="outside"):
        core.validate_box((1, 1, 11, 10), 10, 10)
    with pytest.raises(core.DetectionAuditError, match="finite"):
        core.validate_box((1, 1, float("nan"), 9), 10, 10)


def test_class_map_and_class_agnostic_conversion(audit) -> None:
    assert sorted({row["class_name"] for row in audit.annotations}) == sorted(core.CLASSES)
    assert {row["defect_class_id"] for row in audit.annotations} == {0}
    assert {row["class_id"] for row in audit.annotations} == set(range(6))


def test_duplicate_annotation_handling() -> None:
    row = {"source_image_id": "1", "class_name": "crack", "xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}
    core.validate_unique_annotations([row])
    with pytest.raises(core.DetectionAuditError, match="duplicate annotation"):
        core.validate_unique_annotations([row, dict(row)])


def test_empty_annotations_and_multi_box_images(audit) -> None:
    core.validate_image_coverage((row["source_image_id"] for row in audit.images), audit.annotations)
    with pytest.raises(core.DetectionAuditError, match="empty annotations"):
        core.validate_image_coverage(["missing"], audit.annotations)
    assert audit.summary["images_with_multiple_boxes"] > 0
    assert audit.summary["maximum_boxes_per_image"] >= 2


def test_source_group_and_exact_duplicate_split_isolation(audit) -> None:
    assert len({row["source_image_id"] for row in audit.splits}) == len(audit.splits) == 720
    assert not any(row["cross_split"] for row in audit.duplicate_rows)
    assert audit.summary["cross_split_duplicate_or_related_pair_count"] == 0


def test_frozen_split_is_deterministic(audit) -> None:
    second = core._audit(config(), ROOT)
    assert audit.splits == second.splits
    assert audit.split_fingerprint == second.split_fingerprint


def test_dataset_fingerprint_is_deterministic_and_sensitive(audit) -> None:
    second = core._audit(config(), ROOT)
    assert audit.dataset_fingerprint == second.dataset_fingerprint
    assert core._canonical_hash({"value": audit.dataset_fingerprint}) != core._canonical_hash({"value": "changed"})


def test_checkpoint_fingerprinting(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint-fixture")
    first = core.sha256_file(checkpoint)
    checkpoint.write_bytes(b"changed-checkpoint-fixture")
    assert core.sha256_file(checkpoint) != first


def test_validation_only_threshold_selection_and_tie_break() -> None:
    rows = [
        {"threshold": 0.2, "precision": 0.8, "recall": 0.8},
        {"threshold": 0.4, "precision": 0.8, "recall": 0.8},
        {"threshold": 0.6, "precision": 0.9, "recall": 0.5},
    ]
    assert core.choose_validation_threshold(rows) == 0.2


def test_test_firewall_enforcement() -> None:
    locked = {name: True for name in ("dataset_frozen", "split_frozen", "training_config_frozen", "all_seeds_trained", "checkpoints_locked", "threshold_locked", "nms_locked")}
    core.enforce_test_firewall(locked)
    locked["threshold_locked"] = False
    with pytest.raises(core.DetectionAuditError, match="threshold_locked"):
        core.enforce_test_firewall(locked)


def test_iou_matching() -> None:
    assert core.box_iou((0, 0, 9, 9), (0, 0, 9, 9)) == 1.0
    assert core.box_iou((0, 0, 4, 4), (5, 5, 9, 9)) == 0.0


def test_framework_ap_metric_validation() -> None:
    core.validate_framework_metrics({"map_50_95": 0.4, "map_50": 0.7, "precision": 0.6, "recall": 0.5})
    with pytest.raises(core.DetectionAuditError, match=r"\[0,1\]"):
        core.validate_framework_metrics({"map_50_95": 1.1, "map_50": 0.7, "precision": 0.6, "recall": 0.5})


def test_error_decomposition() -> None:
    truth = [{"box": (0, 0, 9, 9), "class_name": "crack"}, {"box": (20, 20, 29, 29), "class_name": "craze"}]
    predictions = [
        {"id": "a", "box": (0, 0, 9, 9), "class_name": "crack", "score": 0.9},
        {"id": "b", "box": (0, 0, 9, 9), "class_name": "crack", "score": 0.8},
        {"id": "c", "box": (20, 20, 29, 29), "class_name": "corrosion", "score": 0.7},
        {"id": "d", "box": (40, 40, 49, 49), "class_name": "crack", "score": 0.6},
    ]
    result = core.decompose_detection_errors(truth, predictions)
    assert result == {"true_positive": 1, "missed_ground_truth": 0, "insufficient_iou": 0, "localized_wrong_class": 1, "duplicate_prediction": 1, "background_false_positive": 1}


def test_three_seed_aggregation() -> None:
    result = core.aggregate_seed_metrics([0.4, 0.5, 0.6])
    assert result["seed_count"] == 3
    assert result["mean"] == pytest.approx(0.5)
    assert result["sample_sd"] == pytest.approx(0.1)
    with pytest.raises(core.DetectionAuditError, match="three"):
        core.aggregate_seed_metrics([0.4, 0.5])


def test_qualitative_selection_is_deterministic(audit) -> None:
    second = core._audit(config(), ROOT)
    assert audit.qc_selection == second.qc_selection
    assert len(audit.qc_selection) <= config().as_dict()["phase11a"]["qc_max_images"]


def test_two_pass_generation_is_exact(tmp_path: Path, audit) -> None:
    phase10 = {"status": "PASS", "phase10_files_fingerprint": "fixture"}
    app = {"file_count": 0, "fingerprint": "fixture", "files": {}}
    compute = {"status": "fixture", "training_authorized_here": False}
    first = core._generate(config(), ROOT, tmp_path / "first", audit, phase10, app, compute)
    second = core._generate(config(), ROOT, tmp_path / "second", audit, phase10, app, compute)
    assert first["inventory"] == second["inventory"]
    assert first["fingerprint"] == second["fingerprint"]


def test_phase10_immutability_gate_passes() -> None:
    result = core._phase10_identity(config(), ROOT)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_phase12_paths_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "docs/phase12_external_validation").mkdir(parents=True)
    with pytest.raises(core.DetectionAuditError, match="Phase 12"):
        core._assert_no_phase12(tmp_path)


def test_generated_manifest_declares_no_training_when_present() -> None:
    path = ROOT / "experiments/summaries/phase11_detection_audit_v1/manifest.json"
    if not path.is_file():
        pytest.skip("canonical Phase 11A output is generated after apparatus tests")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["phase11b_training_started"] is False
    assert manifest["phase12_started"] is False
