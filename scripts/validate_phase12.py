"""Validate the historical Phase 12A detector-integration apparatus without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml


class Phase12ValidationError(RuntimeError):
    """Raised when the frozen Phase 12A apparatus does not match its contract."""


EXPECTED_INTERNAL_CLASSES = {0: "item"}
EXPECTED_APPLICATION_CLASSES = {0: "defect_region_proposal"}
EXPECTED_DEPENDENCIES = {
    "streamlit": "1.62.0",
    "streamlit-cropper": "0.3.1",
    "numpy": "2.4.6",
    "Pillow": "12.3.0",
    "matplotlib": "3.11.1",
    "PyYAML": "6.0.3",
    "joblib": "1.5.3",
    "scikit-image": "0.26.0",
    "scikit-learn": "1.9.0",
    "torch": "2.13.0+cpu",
    "torchvision": "0.28.0+cpu",
    "ultralytics": "8.3.150",
}
EXPECTED_INFERENCE = {
    "image_size": 640,
    "confidence_threshold": 0.39,
    "nms_iou": 0.7,
    "class_agnostic_nms": True,
    "maximum_detections": 300,
    "device": "cpu",
    "save": False,
    "plots": False,
    "verbose": False,
    "model_ensembling": False,
    "checkpoint_switching": False,
    "user_adjustable_threshold": False,
    "user_adjustable_nms": False,
}
EXPECTED_CANDIDATE = {
    "detector": "YOLO11n",
    "package": "ultralytics",
    "package_version": "8.3.150",
    "task": "detect",
    "seed": 17,
    "selected_epoch": 83,
    "original_drive_path": "runs/seed_17/weights/epoch82.pt",
    "proposed_repository_path": "experiments/results/phase11b_yolo11n_v1/final/seed_17/epoch82.pt",
    "bytes": 16085716,
    "sha256": "793547a5ec31954d8e909b2f5c63f378374134353a8d1e0cefdd452a5365eefa",
    "selection_source": "validation_only",
    "selection_policy": "maximum_validation_map_50_95",
    "held_out_test_used_for_selection": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase12ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    _require(path.is_file(), f"missing required file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file else json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise Phase12ValidationError(f"invalid structured file: {path}") from error
    _require(isinstance(value, dict), f"required mapping is not an object: {path}")
    return value


def validate_config_contract(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "1.0", "configuration schema mismatch")
    apparatus = config.get("apparatus", {})
    _require(apparatus == {
        "name": "phase12a_detector_application_integration",
        "status": "compatibility_gate_passed_apparatus_only",
        "application_implementation": False,
        "external_deployment": False,
    }, "apparatus boundary mismatch")

    candidate = config.get("deployment_candidate", {})
    for key, expected in EXPECTED_CANDIDATE.items():
        _require(candidate.get(key) == expected, f"deployment candidate {key} mismatch")
    _require(candidate.get("checkpoint_internal_classes") == EXPECTED_INTERNAL_CLASSES, "checkpoint internal classes mismatch")
    _require(candidate.get("application_semantic_classes") == EXPECTED_APPLICATION_CLASSES, "application semantic classes mismatch")
    _require(candidate.get("class_mapping_scope") == "presentation_only", "class mapping must be presentation-only")
    _require(candidate.get("class_mapping_must_not_change") == [
        "coordinates", "scores", "thresholding", "nms", "model_execution",
    ], "class mapping isolation mismatch")
    candidates = candidate.get("validation_candidates")
    _require(candidates == [
        {"seed": 17, "validation_map_50_95": 0.43109},
        {"seed": 29, "validation_map_50_95": 0.42675},
        {"seed": 43, "validation_map_50_95": 0.40643},
    ], "validation candidate evidence mismatch")
    _require(max(candidates, key=lambda row: row["validation_map_50_95"])["seed"] == 17, "deployment candidate is not validation-selected")

    runtime = config.get("runtime_compatibility", {})
    _require(runtime.get("record") == "provenance/phase12_runtime_compatibility.json", "compatibility record path mismatch")
    _require(runtime.get("python") == "3.11.15", "probe Python mismatch")
    _require(runtime.get("device") == "cpu" and runtime.get("cuda_enabled") is False, "probe device mismatch")
    _require(runtime.get("telemetry_disabled") is True, "telemetry must be disabled")
    _require(runtime.get("dependencies") == EXPECTED_DEPENDENCIES, "runtime dependencies mismatch")
    _require(config.get("inference") == EXPECTED_INFERENCE, "frozen inference controls mismatch")

    design = config.get("application_design", {})
    _require(design.get("feature_name") == "Experimental automatic region proposals", "feature name mismatch")
    _require(design.get("proposals_user_reviewable") is True, "proposals must remain reviewable")
    _require(all(design.get(key) is True for key in (
        "retain_prepared_crop", "retain_manual_single_region", "retain_manual_multi_region",
        "separate_detector_and_classifier_scores",
    )), "existing workflows or score separation changed")
    _require(design.get("proposal_crop_pipeline") == "frozen_contextual_crop_then_frozen_six_class_mobilenet", "proposal crop pipeline mismatch")
    _require(design.get("zero_proposal_message") == "No region proposal exceeded the frozen threshold.", "zero-proposal message mismatch")
    _require(design.get("zero_proposals_mean_healthy") is False, "zero proposals must not mean healthy")
    _require(design.get("expose_internal_label_item") is False, "internal class label must not be exposed")

    limitations = config.get("limitations", {})
    _require(limitations.get("healthy_or_background_only_training_images") == 0, "healthy/background limitation mismatch")
    _require(limitations.get("healthy_blade_false_positive_behavior") == "unknown", "healthy false-positive limitation mismatch")
    _require(limitations.get("unsupported_claims") == [
        "safety", "severity", "progression", "remaining_life", "production_readiness",
    ], "unsupported-claims boundary mismatch")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=True)
    if result.returncode:
        raise Phase12ValidationError(result.stderr.strip() or "Git validation failed")
    return result.stdout.strip()


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / "configs/application_phase12.yaml"
    config = _load_mapping(config_path, yaml_file=True)
    validate_config_contract(config)

    for identity in config["frozen_inputs"].values():
        path = (root / identity["path"]).resolve()
        _require(path.is_relative_to(root), "frozen input escapes repository")
        _require(sha256_file(path) == identity["sha256"], f"frozen input hash mismatch: {identity['path']}")

    receipt = _load_mapping(root / config["frozen_inputs"]["phase11b_selection_receipt"]["path"])
    checkpoints = receipt.get("checkpoints")
    _require(isinstance(checkpoints, list) and len(checkpoints) == 3, "selection receipt checkpoint count mismatch")
    receipt_candidates = [{"seed": row.get("seed"), "validation_map_50_95": row.get("validation_map_50_95")} for row in checkpoints]
    _require(receipt_candidates == config["deployment_candidate"]["validation_candidates"], "configuration is not bound to receipt validation evidence")
    selected = checkpoints[0]
    candidate = config["deployment_candidate"]
    _require(
        selected.get("seed") == candidate["seed"]
        and selected.get("epoch") == candidate["selected_epoch"]
        and selected.get("drive_relative_path") == candidate["original_drive_path"]
        and selected.get("size_bytes") == candidate["bytes"]
        and selected.get("sha256") == candidate["sha256"],
        "deployment candidate does not match frozen validation receipt",
    )

    record_path = root / config["runtime_compatibility"]["record"]
    record = _load_mapping(record_path)
    _require(record.get("schema_version") == "1.0" and record.get("status") == "PASS", "runtime compatibility status mismatch")
    _require(record.get("apparatus_config_sha256") == sha256_file(config_path), "compatibility record configuration identity mismatch")
    _require(record.get("environment", {}).get("dependencies") == EXPECTED_DEPENDENCIES, "compatibility dependency record mismatch")
    _require(record.get("environment", {}).get("cpu_only") is True, "compatibility CPU record mismatch")
    checkpoint = record.get("checkpoint", {})
    _require(checkpoint.get("bytes") == candidate["bytes"] and checkpoint.get("sha256") == candidate["sha256"], "compatibility checkpoint identity mismatch")
    _require(checkpoint.get("task") == "detect", "compatibility task mismatch")
    _require(checkpoint.get("checkpoint_internal_classes") == {"0": "item"}, "compatibility internal class mismatch")
    _require(checkpoint.get("application_semantic_classes") == {"0": "defect_region_proposal"}, "compatibility semantic class mismatch")
    smoke = record.get("synthetic_smoke", {})
    _require(smoke.get("status") == "PASS" and smoke.get("project_data_used") is False, "synthetic smoke gate mismatch")
    _require(smoke.get("prediction_details_recorded") is False and smoke.get("timing_recorded") is False, "forbidden synthetic output recorded")
    _require(smoke.get("controls") == {
        "device": "cpu", "imgsz": 640, "conf": 0.39, "iou": 0.7,
        "agnostic_nms": True, "max_det": 300, "save": False,
        "plots": False, "verbose": False,
    }, "synthetic smoke controls mismatch")

    freeze = config["application_v2_freeze"]
    proposed = candidate["proposed_repository_path"]
    baseline_checkpoint = subprocess.run(
        ["git", "cat-file", "-e", f"{freeze['baseline_commit']}:{proposed}"],
        cwd=root, check=False, capture_output=True, text=True,
    )
    _require(
        baseline_checkpoint.returncode != 0,
        "Phase 12A baseline unexpectedly contained the deployment checkpoint",
    )
    for path, expected in freeze["git_objects"].items():
        _require(_git(root, "rev-parse", f"{freeze['baseline_commit']}:{path}") == expected, f"Application v2 baseline mismatch: {path}")

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "phase": "12A",
        "checkpoint_sha256": candidate["sha256"],
        "checkpoint_internal_classes": candidate["checkpoint_internal_classes"],
        "application_semantic_classes": candidate["application_semantic_classes"],
        "application_implementation": False,
        "external_deployment": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
