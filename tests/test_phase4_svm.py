from __future__ import annotations

import numpy as np
import pytest
import yaml

from windblade.models.svm import (
    ModelSelectionError,
    fit_train_only,
    generate_svm_grid,
    select_configuration,
)
from windblade.traditional import _freeze_selection, _validation_grid


SVM_CONFIG = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"],
}


def _records():
    return [
        {
            "C": row["C"],
            "gamma": row["gamma"],
            "validation_macro_f1": 0.5,
            "validation_balanced_accuracy": 0.5,
            "validation_macro_recall": 0.5,
        }
        for row in generate_svm_grid(SVM_CONFIG)
    ]


def test_grid_has_exact_predeclared_eight_candidates():
    grid = generate_svm_grid(SVM_CONFIG)
    assert len(grid) == 8
    assert {(row["C"], row["gamma"]) for row in grid} == {
        (c, gamma) for c in (0.1, 1.0, 10.0, 100.0) for gamma in ("scale", "auto")
    }


def test_bad_grid_is_rejected():
    with pytest.raises(ModelSelectionError):
        generate_svm_grid({"C": [1], "gamma": ["scale"]})


def test_tie_breaking_prefers_lower_c_then_scale():
    selected = select_configuration(_records())
    assert selected["C"] == 0.1
    assert selected["gamma"] == "scale"


def test_selection_uses_metrics_in_declared_order():
    records = _records()
    records[-1]["validation_macro_f1"] = 0.6
    assert select_configuration(records)["C"] == 100.0
    records[-2]["validation_macro_f1"] = 0.6
    records[-2]["validation_balanced_accuracy"] = 0.7
    assert select_configuration(records)["gamma"] == "scale"


def test_test_metrics_cannot_enter_selection():
    records = _records()
    records[0]["test_macro_f1"] = 0.0
    records[-1]["test_macro_f1"] = 1.0
    assert select_configuration(records)["C"] == 0.1


def test_scaler_is_fit_only_to_passed_training_matrix():
    X_train = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    y_train = np.asarray([0, 0, 1, 1])
    model = fit_train_only(X_train, y_train, C=1, gamma="scale")
    assert np.array_equal(model.named_steps["scaler"].mean_, X_train.mean(axis=0))
    assert model.named_steps["svm"].probability is False
    assert model.named_steps["svm"].class_weight == "balanced"


def test_validation_search_accepts_only_train_and_validation_inputs():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(36, 5))
    labels = np.asarray([0, 1, 2] * 12)
    train = np.arange(24)
    validation = np.arange(24, 36)
    rows, predictions = _validation_grid(
        "toy",
        features,
        train,
        validation,
        labels,
        [f"i{index}" for index in range(36)],
        [f"s{index}" for index in range(36)],
        SVM_CONFIG,
        ["a", "b", "c"],
    )
    assert len(rows) == 8
    assert len(predictions) == 8 * len(validation)
    assert {row["instance_id"] for row in predictions} == {
        f"i{index}" for index in validation
    }


def test_same_selected_model_reproduces_predictions():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(30, 4))
    labels = np.asarray([0, 1] * 15)
    first = fit_train_only(features, labels, C=10, gamma="auto").predict(features)
    second = fit_train_only(features, labels, C=10, gamma="auto").predict(features)
    assert np.array_equal(first, second)


def test_frozen_selected_config_serializes_and_reloads_identically(tmp_path):
    selected = {
        "C": 1.0,
        "gamma": "scale",
        "validation_macro_f1": 0.4,
        "validation_balanced_accuracy": 0.5,
        "validation_macro_recall": 0.5,
    }
    path = tmp_path / "frozen.yaml"
    first = _freeze_selection(
        path,
        method="toy+rbf_svm",
        selected=selected,
        feature_config={"fixed": True},
        feature_hash="feature",
        processed_fingerprint="dataset",
        grid_fingerprint="grid",
        tolerance=1e-12,
        git_commit="commit",
    )
    second = _freeze_selection(
        path,
        method="toy+rbf_svm",
        selected=selected,
        feature_config={"fixed": True},
        feature_hash="feature",
        processed_fingerprint="dataset",
        grid_fingerprint="grid",
        tolerance=1e-12,
        git_commit="commit",
    )
    assert first == second == yaml.safe_load(path.read_text(encoding="utf-8"))
