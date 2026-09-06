from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from scripts.validate_phase12 import (
    Phase12ValidationError,
    validate,
    validate_config_contract,
)
from windblade.final_synthesis import core as phase10_core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/application_phase12.yaml"


def apparatus() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase12a_apparatus_passes_without_loading_detector() -> None:
    result = validate(ROOT)
    assert result == {
        "schema_version": "1.0",
        "status": "PASS",
        "phase": "12A",
        "checkpoint_sha256": "793547a5ec31954d8e909b2f5c63f378374134353a8d1e0cefdd452a5365eefa",
        "checkpoint_internal_classes": {0: "item"},
        "application_semantic_classes": {0: "defect_region_proposal"},
        "application_implementation": False,
        "external_deployment": False,
    }


@pytest.mark.parametrize("mapping", [
    {},
    {0: "defect"},
    {1: "item"},
    {0: "item", 1: "item"},
])
def test_internal_class_identity_fails_closed(mapping: dict) -> None:
    changed = deepcopy(apparatus())
    changed["deployment_candidate"]["checkpoint_internal_classes"] = mapping
    with pytest.raises(Phase12ValidationError, match="internal classes"):
        validate_config_contract(changed)


@pytest.mark.parametrize("mapping", [
    {},
    {0: "item"},
    {0: "defect"},
    {1: "defect_region_proposal"},
])
def test_application_semantic_mapping_fails_closed(mapping: dict) -> None:
    changed = deepcopy(apparatus())
    changed["deployment_candidate"]["application_semantic_classes"] = mapping
    with pytest.raises(Phase12ValidationError, match="semantic classes"):
        validate_config_contract(changed)


def test_mapping_is_presentation_only_and_internal_label_is_hidden() -> None:
    config = apparatus()
    candidate = config["deployment_candidate"]
    assert candidate["class_mapping_scope"] == "presentation_only"
    assert candidate["class_mapping_must_not_change"] == [
        "coordinates", "scores", "thresholding", "nms", "model_execution",
    ]
    assert config["application_design"]["expose_internal_label_item"] is False
    assert config["application_design"]["zero_proposals_mean_healthy"] is False


@pytest.mark.parametrize(("key", "value"), [
    ("image_size", 641),
    ("confidence_threshold", 0.4),
    ("nms_iou", 0.6),
    ("class_agnostic_nms", False),
    ("maximum_detections", 301),
    ("device", "cuda"),
    ("user_adjustable_threshold", True),
    ("user_adjustable_nms", True),
])
def test_inference_controls_fail_closed(key: str, value: object) -> None:
    changed = deepcopy(apparatus())
    changed["inference"][key] = value
    with pytest.raises(Phase12ValidationError, match="inference controls"):
        validate_config_contract(changed)


def test_candidate_is_selected_only_from_frozen_validation_evidence() -> None:
    config = apparatus()
    candidate = config["deployment_candidate"]
    assert candidate["selection_source"] == "validation_only"
    assert candidate["held_out_test_used_for_selection"] is False
    assert max(candidate["validation_candidates"], key=lambda row: row["validation_map_50_95"])["seed"] == 17
    changed = deepcopy(config)
    changed["deployment_candidate"]["held_out_test_used_for_selection"] = True
    with pytest.raises(Phase12ValidationError, match="held_out_test_used_for_selection"):
        validate_config_contract(changed)


def test_compatibility_record_contains_identity_and_pass_state_only() -> None:
    record = json.loads((ROOT / "provenance/phase12_runtime_compatibility.json").read_text(encoding="utf-8"))
    serialized = json.dumps(record).lower()
    assert record["status"] == "PASS"
    assert record["synthetic_smoke"]["prediction_details_recorded"] is False
    assert record["synthetic_smoke"]["timing_recorded"] is False
    assert "coordinate" not in serialized
    assert "elapsed" not in serialized
    assert "latency" not in serialized


def test_checkpoint_was_not_present_in_historical_phase12a_baseline() -> None:
    candidate = apparatus()["deployment_candidate"]
    record = json.loads((ROOT / "provenance/phase12_runtime_compatibility.json").read_text(encoding="utf-8"))
    assert record["checkpoint_copied_to_repository"] is False
    assert record["checkpoint_modified"] is False
    assert validate(ROOT)["phase"] == "12A"


def test_application_v2_git_objects_are_frozen() -> None:
    config = apparatus()
    assert config["apparatus"]["application_implementation"] is False
    assert config["apparatus"]["external_deployment"] is False
    assert set(config["application_v2_freeze"]["git_objects"]) == {
        "app", "src/windblade_demo", ".streamlit", "requirements-app.txt",
    }


def test_phase10_guard_allows_only_declared_phase12_apparatus(tmp_path: Path) -> None:
    allowed = (
        "configs/application_phase12.yaml",
        "docs/phase12_integration.md",
        "provenance/phase12_runtime_compatibility.json",
        "scripts/validate_phase12.py",
        "tests/test_phase12_apparatus.py",
        "configs/application_phase12b.yaml",
        "provenance/phase12b_implementation.json",
        "scripts/validate_phase12b.py",
        "tests/test_phase12b_integration.py",
    )
    for relative in allowed:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    phase10_core._assert_no_optional_phase_paths(tmp_path)
    forbidden = tmp_path / "src/phase12_application.py"
    forbidden.parent.mkdir(parents=True)
    forbidden.touch()
    with pytest.raises(phase10_core.FinalSynthesisError, match="Phase 12"):
        phase10_core._assert_no_optional_phase_paths(tmp_path)
