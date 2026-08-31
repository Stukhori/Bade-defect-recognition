from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from windblade.config import calculate_config_hash, load_config
from windblade.data.processed import LABELS, read_csv
from windblade.error_analysis.core import (
    EVENTS, bool_value, event_category, natural_instance_key, occupancy_bin,
    select_exemplars, transition_tables,
)
from windblade.error_analysis.gradcam import (
    annotation_box,
    gradcam_map,
    render_annotation,
    resolve_module,
    validate_target_identity,
)
from windblade.error_analysis.review import pass_b_caption_mismatches, render_pass_b_index
from windblade.error_analysis.phase9b import (
    REVIEWER_ATTESTATION,
    _form_record,
    build_crosstabs,
    build_joined_review_tables,
    build_response_summary,
    validate_attestation,
    validate_corrected_packet,
    validate_mapping_rows,
)
from windblade.error_analysis.runner import _completed_review_data_exist
from windblade_review.schema import load_pass_schema


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/error_analysis.yaml")


@pytest.mark.parametrize(
    ("clean_correct", "degraded_correct", "clean_prediction", "degraded_prediction", "expected"),
    [
        (True, True, "craze", "craze", "stable_correct"),
        (True, False, "craze", "crack", "harmful_flip"),
        (False, True, "crack", "craze", "beneficial_flip"),
        (False, False, "crack", "corrosion", "changed_wrong"),
        (False, False, "crack", "crack", "stable_wrong"),
    ],
)
def test_frozen_event_definitions(clean_correct: bool, degraded_correct: bool, clean_prediction: str, degraded_prediction: str, expected: str) -> None:
    assert event_category(clean_correct, degraded_correct, clean_prediction, degraded_prediction, "brightness_severe") == expected
    assert event_category(clean_correct, degraded_correct, clean_prediction, degraded_prediction, "clean") == "clean_only"


def test_event_set_is_exhaustive_and_boolean_parser_is_strict() -> None:
    assert EVENTS == ("clean_only", "stable_correct", "harmful_flip", "beneficial_flip", "changed_wrong", "stable_wrong")
    assert bool_value(True) and bool_value("true") and not bool_value("False")
    with pytest.raises(Exception): bool_value("yes")


def test_frozen_occupancy_bins_cover_boundaries_without_overlap() -> None:
    bins = CONFIG.as_dict()["geometry"]["occupancy_bins"]
    assert [occupancy_bin(value, bins) for value in (0.0, 0.099, 0.10, 0.249, 0.25, 0.499, 0.50, 1.0)] == [
        "lt_0.10", "lt_0.10", "0.10_to_lt_0.25", "0.10_to_lt_0.25",
        "0.25_to_lt_0.50", "0.25_to_lt_0.50", "ge_0.50", "ge_0.50",
    ]


def test_phase9_contract_freezes_methods_conditions_quotas_and_layers() -> None:
    data = CONFIG.as_dict()
    assert data["evaluation"]["methods"] == ["hog", "lbp", "resnet18", "mobilenet_v3_small"]
    assert data["evaluation"]["seeds"] == [17, 29, 43]
    assert len(data["evaluation"]["conditions"]) == 13
    assert sum(data["selection"]["quotas"].values()) == data["selection"]["target_distinct_cases"] == 60
    assert data["gradcam"]["target_layers"] == {"resnet18": "layer4.1", "mobilenet_v3_small": "features.12"}
    assert calculate_config_hash(data, length=64) == calculate_config_hash(CONFIG.as_dict(), length=64)


class TinyCamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(value)).flatten(1))


def test_gradcam_hooks_gradients_is_finite_deterministic_and_does_not_mutate_parameters() -> None:
    torch.manual_seed(7)
    model = TinyCamModel().eval()
    tensor = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    layer = resolve_module(model, "features.0")
    first, shape, logits = gradcam_map(model, layer, tensor, 1)
    second, second_shape, second_logits = gradcam_map(model, layer, tensor, 1)
    assert shape == second_shape == (1, 4, 16, 16)
    assert first.shape == (224, 224) and np.isfinite(first).all()
    assert np.array_equal(first, second) and np.array_equal(logits, second_logits)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_missing_gradcam_layer_is_rejected() -> None:
    with pytest.raises(Exception): resolve_module(TinyCamModel(), "not.a.layer")


def test_gradcam_target_roles_require_frozen_true_and_predicted_classes() -> None:
    prediction = {
        "true_class_id": 2,
        "true_label": "surface_injure",
        "predicted_class_id": 0,
        "predicted_label": "craze",
    }
    validate_target_identity("true_class", 2, "surface_injure", prediction)
    validate_target_identity("predicted_class", 0, "craze", prediction)
    with pytest.raises(Exception, match="true_class"):
        validate_target_identity("true_class", 0, "craze", prediction)
    with pytest.raises(Exception, match="predicted_class"):
        validate_target_identity("predicted_class", 2, "surface_injure", prediction)
    with pytest.raises(Exception, match="index/label"):
        validate_target_identity("true_class", 2, "craze", prediction)


def test_annotation_box_is_resized_from_frozen_crop_geometry() -> None:
    meta = {"bbox_xmin": 20, "bbox_ymin": 30, "bbox_xmax": 120, "bbox_ymax": 130, "crop_xmin": 10, "crop_ymin": 20, "crop_side": 200}
    assert annotation_box(meta) == (11, 11, 123, 123)
    rendered = render_annotation(Image.new("RGB", (224, 224), "white"), annotation_box(meta))
    assert rendered.size == (224, 224) and rendered.mode == "RGB"


def _selection_rows() -> list[dict]:
    rows = []
    for method in ("resnet18", "mobilenet_v3_small"):
        for sample_index, true_label in enumerate(LABELS):
            true_id = LABELS.index(true_label)
            for condition in ("clean", "brightness_severe"):
                for seed_index, seed in enumerate((17, 29, 43)):
                    clean_correct = sample_index % 2 == 0
                    if condition == "clean":
                        correct = clean_correct
                        event = "clean_only"
                    else:
                        correct = False
                        event = "harmful_flip" if clean_correct else "stable_wrong"
                    predicted_id = true_id if correct else (true_id + 1 + (seed_index if sample_index == 5 else 0)) % 6
                    rows.append({"method": method, "seed": seed, "condition_id": condition, "instance_id": f"{100 + sample_index}_0", "source_image_id": str(100 + sample_index), "true_label": true_label, "true_class_id": true_id, "corruption_family": "clean" if condition == "clean" else "brightness", "severity": "clean" if condition == "clean" else "severe", "correct": correct, "clean_correct": clean_correct, "predicted_label": LABELS[predicted_id], "event_category": event})
    return rows


def test_exemplar_selection_is_deterministic_and_never_duplicates_sample_condition() -> None:
    first_candidates, first, first_summary = select_exemplars(_selection_rows(), CONFIG.as_dict())
    second_candidates, second, second_summary = select_exemplars(_selection_rows(), CONFIG.as_dict())
    assert first_candidates == second_candidates and first == second and first_summary == second_summary
    keys = {(row["instance_id"], row["condition_id"]) for row in first}
    assert len(keys) == len(first)
    assert [row["review_id"] for row in first] == [f"P9A-{index:03d}" for index in range(1, len(first) + 1)]


def test_transition_counts_include_exact_denominator() -> None:
    rows = []
    for event in EVENTS[1:]:
        rows.append({"method": "hog", "seed": "not_applicable", "condition_id": "jpeg_severe", "corruption_family": "jpeg", "severity": "severe", "event_category": event})
    per_seed, aggregate = transition_tables(rows)
    assert len(per_seed) == len(aggregate) == 1
    assert per_seed[0]["denominator"] == 5
    assert sum(per_seed[0][event] for event in EVENTS[1:]) == 5


def test_natural_instance_sort_is_numeric() -> None:
    assert sorted(["10_0", "2_1", "2_0"], key=natural_instance_key) == ["2_0", "2_1", "10_0"]


def test_pass_b_caption_uses_target_label_not_prediction_label(tmp_path: Path) -> None:
    review_root = tmp_path / "human_review_packet"
    mapping = [{
        "review_id": "P9A-001",
        "method": "resnet18",
        "condition_id": "clean",
        "selection_event": "clean_consensus_error",
        "eligibility_rule": "consensus",
        "true_label": "surface_injure",
    }]
    common = {
        "review_id": "P9A-001",
        "seed": 17,
        "input_state": "clean",
        "predicted_label": "craze",
    }
    gradcam = [
        {
            **common,
            "target_role": "true_class",
            "target_label": "surface_injure",
            "overlay_path": "figures/true_class_surface_injure_overlay.png",
        },
        {
            **common,
            "target_role": "predicted_class",
            "target_label": "craze",
            "overlay_path": "figures/predicted_class_craze_overlay.png",
        },
    ]
    page, _ = render_pass_b_index(mapping, gradcam, tmp_path, review_root)
    assert "true_class: surface_injure" in page
    assert "predicted_class: craze" in page
    assert "true_class: craze" not in page
    assert pass_b_caption_mismatches(page, gradcam, tmp_path, review_root) == []


def test_all_canonical_gradcam_targets_and_pass_b_captions_match_frozen_predictions() -> None:
    summary = ROOT / "experiments/summaries/phase9_error_analysis_v1"
    manifest = read_csv(summary / "gradcam/gradcam_manifest.csv")
    errors = read_csv(summary / "error_manifest.csv")
    frozen = {
        (row["method"], row["seed"], row["condition_id"], row["instance_id"]): row
        for row in errors
    }
    incorrect_review_ids: set[str] = set()
    true_targets = predicted_targets = 0
    for row in manifest:
        prediction = frozen[
            (row["method"], row["seed"], row["input_condition_id"], row["instance_id"])
        ]
        validate_target_identity(
            row["target_role"], int(row["target_class_id"]), row["target_label"], prediction
        )
        if row["target_role"] == "true_class":
            true_targets += 1
        else:
            predicted_targets += 1
        if row["true_label"] != row["predicted_label"]:
            incorrect_review_ids.add(row["review_id"])
    assert len(manifest) == 507
    assert true_targets == 330
    assert predicted_targets == 177
    assert len(incorrect_review_ids) == 52
    page = (summary / "human_review_packet/pass_b/index.html").read_text(encoding="utf-8")
    assert pass_b_caption_mismatches(
        page, manifest, ROOT, summary / "human_review_packet"
    ) == []


def test_phase9b_attestation_must_be_exact_and_confirm_nonautomatic_authorship() -> None:
    phase9b = CONFIG.as_dict()["phase9b"]
    assert phase9b["reviewer_attestation"] == REVIEWER_ATTESTATION
    validate_attestation(phase9b)
    with pytest.raises(Exception, match="exact Phase 9B"):
        validate_attestation({**phase9b, "reviewer_attestation": "approximately true"})
    with pytest.raises(Exception, match="not confirmed"):
        validate_attestation({**phase9b, "reviewer_attestation_confirmed": False})
    with pytest.raises(Exception, match="corrected Pass B"):
        validate_attestation({**phase9b, "reviewer_confirmed_corrected_pass_b": False})


def test_phase9b_mapping_requires_exact_order_and_frozen_case_metadata() -> None:
    expected = ("P9A-001", "P9A-002")
    selected = [
        {
            "review_id": review_id,
            "instance_id": str(index),
            "source_image_id": str(index),
            "method": "resnet18",
            "condition_id": "clean",
            "selection_event": "clean_consensus_error",
            "eligibility_rule": "consensus",
            "true_label": "craze",
        }
        for index, review_id in enumerate(expected, start=1)
    ]
    mapping = [dict(row) for row in selected]
    validate_mapping_rows(mapping, selected, expected)
    with pytest.raises(Exception, match="order"):
        validate_mapping_rows(list(reversed(mapping)), selected, expected)
    changed = [dict(row) for row in mapping]
    changed[0]["true_label"] = "crack"
    with pytest.raises(Exception, match="true_label"):
        validate_mapping_rows(changed, selected, expected)


def test_phase9b_form_gate_requires_complete_valid_forms_without_rewriting(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    for pass_name in ("pass_a", "pass_b"):
        source = ROOT / (
            "experiments/summaries/phase9_error_analysis_v1/human_review_packet/"
            f"{pass_name}/{pass_name}_review_form.csv"
        )
        destination = summary / f"human_review_packet/{pass_name}/{pass_name}_review_form.csv"
        destination.parent.mkdir(parents=True)
        shutil.copy2(source, destination)
    expected_ids = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    for pass_name, expected_total in (("pass_a", 300), ("pass_b", 240)):
        path = summary / f"human_review_packet/{pass_name}/{pass_name}_review_form.csv"
        before = path.read_bytes()
        _, record, source_bytes, _ = _form_record(ROOT, summary, pass_name, expected_ids)
        assert record["answered_required"] == expected_total
        assert record["complete"] and record["allowed_choices_valid"]
        assert source_bytes == before == path.read_bytes()

    pass_b = summary / "human_review_packet/pass_b/pass_b_review_form.csv"
    with pass_b.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["activation_primarily_inside_annotation"] = ""
    with pass_b.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(Exception, match="incomplete"):
        _form_record(ROOT, summary, "pass_b", expected_ids)


def test_corrected_pass_b_archive_matches_every_repository_evidence_file() -> None:
    expected_ids = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    record = validate_corrected_packet(ROOT, CONFIG.as_dict()["phase9b"], expected_ids)
    assert record["status"] == "PASS"
    assert record["archive_file_count"] == 1681
    assert record["nonform_files_compared"] == 1680
    assert record["nonform_hash_mismatches"] == record["nonform_missing"] == 0
    assert not record["superseded_packet_used"]


def _synthetic_phase9b_inputs() -> dict:
    pass_a_schema = load_pass_schema(ROOT / "configs/error_analysis.yaml", "pass_a")
    pass_b_schema = load_pass_schema(ROOT / "configs/error_analysis.yaml", "pass_b")
    events = (
        "clean_consensus_error",
        "severe_harmful_flip",
        "severe_stable_wrong",
        "severe_stable_correct",
        "beneficial_flip",
        "seed_disagreement",
    )
    pass_a = []
    pass_b = []
    selected = []
    mapping = []
    errors = []
    for index in range(1, 61):
        review_id = f"P9A-{index:03d}"
        event = events[(index - 1) % len(events)]
        clean = event == "clean_consensus_error"
        condition = "clean" if clean else "brightness_severe"
        true_label = LABELS[(index - 1) % len(LABELS)]
        method = "resnet18" if index % 2 else "mobilenet_v3_small"
        selected_row = {
            "review_id": review_id,
            "instance_id": f"{index}_0",
            "source_image_id": str(index),
            "method": method,
            "condition_id": condition,
            "corruption_family": "clean" if clean else "brightness",
            "severity": "clean" if clean else "severe",
            "selection_event": event,
            "eligibility_rule": "consensus" if event != "seed_disagreement" else "seed_disagreement",
            "true_label": true_label,
            "satisfying_seed_count": "3",
        }
        selected.append(selected_row)
        mapping.append(
            {
                **selected_row,
                "clean_asset": f"pass_a/assets/{review_id}/clean.png",
                "degraded_asset": "" if clean else f"pass_a/assets/{review_id}/degraded.png",
            }
        )
        pass_a.append(
            {
                "review_id": review_id,
                **{
                    field.name: (field.choices[(index - 1) % len(field.choices)] if field.required else "")
                    for field in pass_a_schema.fields
                },
            }
        )
        pass_b.append(
            {
                "review_id": review_id,
                **{
                    field.name: (field.choices[(index - 1) % len(field.choices)] if field.required else "")
                    for field in pass_b_schema.fields
                },
            }
        )
        for seed_index, seed in enumerate((17, 29, 43)):
            correct = seed_index != 2
            predicted = true_label if correct else LABELS[(LABELS.index(true_label) + 1) % len(LABELS)]
            errors.append(
                {
                    "method": method,
                    "seed": str(seed),
                    "condition_id": condition,
                    "instance_id": f"{index}_0",
                    "true_label": true_label,
                    "predicted_class_id": str(LABELS.index(predicted)),
                    "predicted_label": predicted,
                    "correct": str(correct),
                    "event_category": "clean_only" if clean else "harmful_flip",
                    "prediction_changed_from_clean": str(not clean and not correct),
                }
            )
    return {
        "pass_a_schema": pass_a_schema,
        "pass_b_schema": pass_b_schema,
        "pass_a": tuple(pass_a),
        "pass_b": tuple(pass_b),
        "selected": selected,
        "mapping": mapping,
        "error_rows": errors,
    }


def test_phase9b_join_summaries_and_crosstabs_are_deterministic_and_reconcile() -> None:
    inputs = _synthetic_phase9b_inputs()
    first_cases, first_predictions = build_joined_review_tables(inputs)
    second_cases, second_predictions = build_joined_review_tables(inputs)
    assert first_cases == second_cases and first_predictions == second_predictions
    assert len(first_cases) == 60 and len(first_predictions) == 180
    assert {row["analysis_group"] for row in first_cases} >= {
        "harmful_flip", "beneficial_flip", "stable_correct", "stable_wrong", "seed_disagreement"
    }
    summary = build_response_summary(inputs)
    required = [row for row in summary if row["required"] == "true"]
    for pass_name, schema in (("pass_a", inputs["pass_a_schema"]), ("pass_b", inputs["pass_b_schema"])):
        for field in schema.required_fields:
            assert sum(int(row["count"]) for row in required if row["pass_name"] == pass_name and row["field"] == field) == 60
    case_tabs = build_crosstabs(inputs, first_cases, ("analysis_group",), "review_case")
    prediction_tabs = build_crosstabs(inputs, first_predictions, ("correct",), "case_seed_prediction")
    assert case_tabs and prediction_tabs
    assert all(int(row["denominator"]) > 0 for row in case_tabs + prediction_tabs)


def test_phase9a_regeneration_is_guarded_after_human_review_completion() -> None:
    summary = ROOT / "experiments/summaries/phase9_error_analysis_v1"
    assert _completed_review_data_exist(summary)


def test_phase9_scientific_modules_expose_no_fitting_or_training_api() -> None:
    for relative in (
        "src/windblade/error_analysis/core.py",
        "src/windblade/error_analysis/runner.py",
        "src/windblade/error_analysis/phase9b.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert ".fit(" not in source
        assert "optimizer.step(" not in source
        assert "torch.optim" not in source
