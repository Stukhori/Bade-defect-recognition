from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torchvision.models import mobilenet_v3_small, resnet18

from windblade.config import load_config
from windblade.data.processed import LABELS
from windblade.data_efficiency import (
    CLASS_COUNTS,
    FRACTIONS,
    INSTANCE_COUNTS,
    METHODS,
    REDUCED_FRACTIONS,
    SEEDS,
    SOURCE_COUNTS,
    DataEfficiencyError,
    _head_fingerprint,
    _validate_phase7_config,
    _write_cnn_run,
    calculate_learning_curve_outputs,
    load_and_validate_subsets,
)
from windblade.deep.dataset import balanced_subset_class_weights
from windblade.deep.mobilenet import model_from_official_state as mobile_from_state
from windblade.deep.resnet import model_from_official_state as resnet_from_state


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT / "configs/data_efficiency.yaml")


def _metric(value: float) -> dict:
    return {
        "macro_f1": value,
        "balanced_accuracy": value + 0.01,
        "accuracy": value - 0.01,
        "macro_precision": value,
        "macro_recall": value + 0.01,
        "per_class": {
            label: {"class_id": index, "precision": value, "recall": value, "f1": value, "support": 1}
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix_counts": [[0] * 6 for _ in range(6)],
        "confusion_matrix_row_normalized": [[0.0] * 6 for _ in range(6)],
    }


def test_all_frozen_subset_manifests_reproduce_contract():
    subsets = load_and_validate_subsets(_config(), ROOT)
    assert len(subsets) == 12
    for seed in SEEDS:
        prior = set()
        for fraction in FRACTIONS:
            subset = subsets[(seed, fraction)]
            assert len(subset["source_ids"]) == SOURCE_COUNTS[fraction]
            assert len(subset["rows"]) == INSTANCE_COUNTS[fraction]
            assert tuple(subset["class_counts"][label] for label in LABELS) == CLASS_COUNTS[fraction]
            assert {row["split"] for row in subset["rows"]} == {"train"}
            if prior:
                assert prior < subset["source_ids"]
            prior = subset["source_ids"]


def test_subset_class_weights_use_only_active_training_rows():
    subsets = load_and_validate_subsets(_config(), ROOT)
    for seed in SEEDS:
        for fraction in FRACTIONS:
            rows = subsets[(seed, fraction)]["rows"]
            weights = balanced_subset_class_weights(rows)
            expected = torch.tensor(
                [len(rows) / (6 * count) for count in CLASS_COUNTS[fraction]], dtype=torch.float32
            )
            assert torch.equal(weights, expected)
            with pytest.raises(ValueError):
                balanced_subset_class_weights([*rows, {**rows[0], "split": "validation"}])


def test_phase7_config_enforces_upstream_frozen_values_and_no_search():
    config = _config()
    _validate_phase7_config(config.as_dict(), ROOT)
    assert config.as_dict()["data_efficiency"]["hyperparameter_search"] is False
    changed = config.as_dict()
    changed["data_efficiency"]["frozen_configs"]["hog"] = "configs/traditional_baselines.yaml"
    with pytest.raises(DataEfficiencyError):
        _validate_phase7_config(changed, ROOT)


def test_cnn_runner_has_no_warm_start_or_search_inputs():
    parameters = inspect.signature(_write_cnn_run).parameters
    assert "checkpoint" not in parameters
    assert "learning_rate" not in parameters
    assert "weight_decay" not in parameters
    assert "search" not in parameters


def test_same_seed_fresh_heads_match_across_independent_construction():
    resnet_state = resnet18(weights=None).state_dict()
    mobile_state = mobilenet_v3_small(weights=None).state_dict()
    for seed in SEEDS:
        resnet_heads = [_head_fingerprint(resnet_from_state(resnet_state, seed=seed), "resnet18") for _ in REDUCED_FRACTIONS]
        mobile_heads = [_head_fingerprint(mobile_from_state(mobile_state, seed=seed), "mobilenet_v3_small") for _ in REDUCED_FRACTIONS]
        assert len(set(resnet_heads)) == 1
        assert len(set(mobile_heads)) == 1


def test_known_aggregation_formulas_and_traditional_full_sd():
    values = {0.25: 0.4, 0.50: 0.6, 0.75: 0.7, 1.0: 0.8}
    metrics = {}
    for method in METHODS:
        for fraction in FRACTIONS:
            count = 1 if method in {"hog", "lbp"} and fraction == 1.0 else 3
            metrics[(method, fraction)] = [_metric(values[fraction] + offset) for offset in ([0.0] if count == 1 else [-0.1, 0.0, 0.1])]
    result = calculate_learning_curve_outputs(metrics)
    hog_full = next(row for row in result["learning_curve_summary"] if row["method"] == "hog" and row["training_fraction"] == 1.0)
    assert hog_full["macro_f1_sample_sd"] is None
    resnet_quarter = next(row for row in result["learning_curve_summary"] if row["method"] == "resnet18" and row["training_fraction"] == 0.25)
    assert resnet_quarter["macro_f1_mean"] == pytest.approx(0.4)
    assert resnet_quarter["macro_f1_sample_sd"] == pytest.approx(0.1)
    retention = next(row for row in result["performance_retention"] if row["method"] == "hog" and row["training_fraction"] == 0.5)
    assert retention["performance_retention"] == pytest.approx(0.75)
    marginal = [row["macro_f1_gain"] for row in result["marginal_gains"] if row["method"] == "hog"]
    assert marginal == pytest.approx([0.2, 0.1, 0.1])
    threshold = next(row for row in result["threshold_95_percent"] if row["method"] == "hog")
    assert threshold["threshold_fraction"] == 1.0
    auc = next(row for row in result["normalized_learning_curve_auc"] if row["method"] == "hog")
    assert auc["normalized_macro_f1_learning_curve_auc"] == pytest.approx((0.125 + 0.1625 + 0.1875) / 0.75)
