from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from windblade.config import calculate_config_hash, load_config
from windblade.data.processed import LABELS
from windblade.error_analysis.core import (
    EVENTS, bool_value, event_category, natural_instance_key, occupancy_bin,
    select_exemplars, transition_tables,
)
from windblade.error_analysis.gradcam import annotation_box, gradcam_map, render_annotation, resolve_module


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


def test_phase9_scientific_modules_expose_no_fitting_or_training_api() -> None:
    for relative in ("src/windblade/error_analysis/core.py", "src/windblade/error_analysis/runner.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert ".fit(" not in source
        assert "optimizer.step(" not in source
        assert "torch.optim" not in source
