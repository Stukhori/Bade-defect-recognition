"""Validate the Phase 12B application integration without running detector inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

if __package__:
    from scripts.validate_phase12 import sha256_file, validate as validate_phase12a
else:
    from validate_phase12 import sha256_file, validate as validate_phase12a
from windblade_demo.constants import APPLICATION_VERSION
from windblade_demo.detector import (
    DETECTOR_CHECKPOINT, DETECTOR_CHECKPOINT_BYTES, DETECTOR_CHECKPOINT_SHA256,
    DETECTOR_INTERNAL_CLASSES, DETECTOR_PACKAGE_VERSION, DETECTOR_SEMANTIC_LABEL,
    INFERENCE_ARGUMENTS, PROPOSAL_LIMITATION_NOTICE, ZERO_PROPOSAL_MESSAGE,
    verify_checkpoint,
)


class Phase12BValidationError(RuntimeError):
    pass


CONFIG_PATH = Path("configs/application_phase12b.yaml")
CONFIG_SHA256 = "956fcb74f266e9084e31d7770dcd156a0001515240afaf1c4b9681967c141957"
PHASE12A_CONFIG_SHA256 = "8f084ceecfa911c6fe333d20c2d55a8cd8b738f0ae0dd2d5f00e5dd26f89dde1"
PHASE12A_RECORD_SHA256 = "d1dcd62c5e72d0d7fa7761c36be381c65d9cf9a8a64e14b29d59064ea3496a2f"
FROZEN_HASHES = {
    "configs/detection_phase11b.yaml": "fc0ab33a25bafb5b92da88f67343bca9bbcecb6c715d867b06f4ac74f90cff1b",
    "provenance/phase11b_selection_receipt.json": "6c236e9d7220b443f17a628a3d8f621afc56be3777949eeca47f462879e46509",
    "provenance/phase11b_final_test_metrics.json": "e6bac76aa5a2e3d68ed7b93f5be180228addba1f1f90ddafab398d198496bcbb",
}
EXPECTED_REQUIREMENTS = {
    "ultralytics==8.3.150",
    'torch==2.13.0+cpu; sys_platform != "darwin"',
    'torchvision==0.28.0+cpu; sys_platform != "darwin"',
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase12BValidationError(message)


def _tracked(root: Path, relative: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative.as_posix()], cwd=root,
        check=False, capture_output=True, text=True,
    )
    return result.returncode == 0


def _mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file else json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise Phase12BValidationError(f"Invalid structured file: {path}") from exc
    _require(isinstance(value, dict), f"Expected mapping: {path}")
    return value


def validate(root: Path, *, require_tracked: bool = True) -> dict[str, Any]:
    root = root.resolve()
    historical = validate_phase12a(root)
    _require(historical["phase"] == "12A" and historical["status"] == "PASS", "historical Phase 12A validation failed")
    _require(sha256_file(root / "configs/application_phase12.yaml") == PHASE12A_CONFIG_SHA256, "historical Phase 12A configuration changed")
    _require(sha256_file(root / "provenance/phase12_runtime_compatibility.json") == PHASE12A_RECORD_SHA256, "historical Phase 12A compatibility record changed")

    config_path = root / CONFIG_PATH
    _require(sha256_file(config_path) == CONFIG_SHA256, "Phase 12B configuration hash mismatch")
    config = _mapping(config_path, yaml_file=True)
    _require(config.get("schema_version") == "1.0", "Phase 12B schema mismatch")
    _require(config.get("apparatus") == {
        "name": "phase12b_detector_application_integration",
        "status": "implemented_and_locally_validated",
        "application_version": "3.0.0",
        "external_deployment": False,
    }, "Phase 12B apparatus boundary mismatch")
    checkpoint = config.get("checkpoint", {})
    _require(checkpoint.get("path") == DETECTOR_CHECKPOINT.as_posix(), "checkpoint path mismatch")
    _require(checkpoint.get("bytes") == DETECTOR_CHECKPOINT_BYTES, "checkpoint size contract mismatch")
    _require(checkpoint.get("sha256") == DETECTOR_CHECKPOINT_SHA256, "checkpoint SHA contract mismatch")
    _require(checkpoint.get("seed") == 17 and checkpoint.get("selected_epoch") == 83, "checkpoint selection metadata mismatch")
    _require(checkpoint.get("task") == "detect", "checkpoint task mismatch")
    _require(checkpoint.get("checkpoint_internal_classes") == DETECTOR_INTERNAL_CLASSES, "checkpoint internal classes mismatch")
    _require(checkpoint.get("application_semantic_classes") == {0: "defect_region_proposal"}, "application semantic classes mismatch")
    _require(checkpoint.get("mapping_scope") == "presentation_only", "class mapping scope mismatch")
    runtime = config.get("runtime", {})
    _require(runtime.get("package_version") == DETECTOR_PACKAGE_VERSION, "detector package version mismatch")
    _require(runtime.get("torch") == "2.13.0+cpu" and runtime.get("torchvision") == "0.28.0+cpu", "CPU dependency pins mismatch")
    _require(runtime.get("device") == "cpu", "detector device mismatch")
    _require(runtime.get("external_downloads") is False and runtime.get("telemetry") is False, "external integration boundary mismatch")
    _require(runtime.get("filesystem_prediction_outputs") is False, "prediction output boundary mismatch")
    inference = config.get("inference", {})
    expected_config_inference = {
        "image_size": 640, "confidence_threshold": 0.39, "nms_iou": 0.7,
        "class_agnostic_nms": True, "maximum_detections": 300, "device": "cpu",
        "save": False, "plots": False, "verbose": False, "ensemble": False,
        "user_adjustable_controls": False,
    }
    _require(inference == expected_config_inference, "frozen inference controls mismatch")
    application = config.get("application", {})
    _require(application.get("feature_name") == "Experimental automatic region proposals", "feature name mismatch")
    _require(application.get("presentation_label") == DETECTOR_SEMANTIC_LABEL, "proposal label mismatch")
    _require(application.get("zero_proposal_message") == ZERO_PROPOSAL_MESSAGE, "zero-proposal wording mismatch")
    _require(application.get("human_review_required_before_classification") is True, "review gate mismatch")
    _require(application.get("detector_classifier_scores_separate") is True, "score separation mismatch")
    _require(application.get("retain_modes") == ["prepared_crop", "manual_single_region", "manual_multi_region"], "manual workflow contract mismatch")
    _require(application.get("never_claim_healthy_or_defect_free") is True, "healthy-claim boundary mismatch")

    checked_checkpoint = verify_checkpoint(root)
    _require(checked_checkpoint == (root / DETECTOR_CHECKPOINT).resolve(), "checkpoint resolution mismatch")
    required_paths = [CONFIG_PATH, DETECTOR_CHECKPOINT] + [Path(value) for value in config["implementation"].values()]
    for relative in required_paths:
        _require((root / relative).is_file(), f"missing Phase 12B file: {relative}")
    if require_tracked:
        _require(all(_tracked(root, path) for path in required_paths), "Phase 12B files are not all tracked")

    for relative, expected in FROZEN_HASHES.items():
        _require(sha256_file(root / relative) == expected, f"frozen input changed: {relative}")
    app_source = (root / "app/app.py").read_text(encoding="utf-8")
    _require(APPLICATION_VERSION == "3.0.0", "application version mismatch")
    _require("Auto detection" in app_source, "auto-detection workflow is absent")
    _require("Detect and classify defects" in app_source, "primary auto-detection action is absent")
    _require("st.info(ZERO_PROPOSAL_MESSAGE)" in app_source, "zero-proposal message is absent")
    _require("I reviewed the selected regions" in app_source, "review gate copy is absent")
    _require("experimental" not in app_source.lower(), "experimental wording appears in the UI source")
    _require('"item"' not in app_source and "'item'" not in app_source, "internal detector label appears in the UI source")
    detector_source = (root / "src/windblade_demo/detector.py").read_text(encoding="utf-8")
    _require("PROPOSAL_LIMITATION_NOTICE" in detector_source, "proposal limitation contract is absent")
    requirement_lines = {
        line.strip() for line in (root / "app/requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    _require(EXPECTED_REQUIREMENTS <= requirement_lines, "deployment dependency pins are incomplete")
    local_requirements = (root / "requirements-app.txt").read_text(encoding="utf-8").splitlines()
    _require(
        EXPECTED_REQUIREMENTS <= set(local_requirements),
        "local detector or CPU dependency pins are absent",
    )

    provenance_path = root / config["implementation"]["provenance"]
    provenance = _mapping(provenance_path)
    _require(provenance.get("status") == "PASS" and provenance.get("phase") == "12B", "Phase 12B provenance status mismatch")
    _require(provenance.get("configuration_sha256") == CONFIG_SHA256, "Phase 12B provenance configuration mismatch")
    smoke = provenance.get("synthetic_smoke", {})
    _require(smoke == {
        "status": "PASS", "source": "deterministic_in_memory_rgb",
        "project_data_used": False, "prediction_details_recorded": False,
        "timing_recorded": False,
    }, "Phase 12B synthetic smoke record mismatch")
    _require(provenance.get("external_deployment") is False, "external deployment must remain false")

    return {
        "schema_version": "1.0", "status": "PASS", "phase": "12B",
        "application_version": APPLICATION_VERSION,
        "checkpoint_sha256": DETECTOR_CHECKPOINT_SHA256,
        "checkpoint_bytes": DETECTOR_CHECKPOINT_BYTES,
        "inference_arguments": dict(INFERENCE_ARGUMENTS),
        "external_deployment": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate(root, require_tracked=not args.allow_untracked), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
