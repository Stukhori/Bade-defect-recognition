"""Frozen RBF-SVM protocol for handcrafted features."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ModelSelectionError(ValueError):
    """Raised when the predeclared SVM protocol is violated."""


def generate_svm_grid(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    grid = tuple(
        {"C": float(c_value), "gamma": str(gamma)}
        for c_value, gamma in product(config["C"], config["gamma"])
    )
    if len(grid) != 8:
        raise ModelSelectionError(f"frozen SVM grid requires 8 configurations; received {len(grid)}")
    expected = {
        (c_value, gamma)
        for c_value in (0.1, 1.0, 10.0, 100.0)
        for gamma in ("scale", "auto")
    }
    if {(row["C"], row["gamma"]) for row in grid} != expected:
        raise ModelSelectionError("SVM grid differs from the predeclared C/gamma values")
    return grid


def build_svm_pipeline(C: float, gamma: str) -> Pipeline:
    if gamma not in {"scale", "auto"}:
        raise ModelSelectionError(f"unsupported frozen gamma: {gamma}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(C),
                    gamma=gamma,
                    kernel="rbf",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def fit_train_only(
    X_train: np.ndarray, y_train: np.ndarray, *, C: float, gamma: str
) -> Pipeline:
    if len(X_train) != len(y_train):
        raise ModelSelectionError("training feature and label lengths differ")
    pipeline = build_svm_pipeline(C=C, gamma=gamma)
    pipeline.fit(X_train, y_train)
    return pipeline


def select_configuration(
    records: Sequence[Mapping[str, Any]], tolerance: float = 1.0e-12
) -> dict[str, Any]:
    """Apply the frozen validation-only metric and tie-breaking hierarchy."""

    if len(records) != 8:
        raise ModelSelectionError("selection requires all 8 validation configurations")
    if tolerance < 0:
        raise ModelSelectionError("selection tolerance cannot be negative")
    best: Mapping[str, Any] | None = None
    for candidate in records:
        if best is None:
            best = candidate
            continue
        candidate_metrics = (
            float(candidate["validation_macro_f1"]),
            float(candidate["validation_balanced_accuracy"]),
            float(candidate["validation_macro_recall"]),
        )
        best_metrics = (
            float(best["validation_macro_f1"]),
            float(best["validation_balanced_accuracy"]),
            float(best["validation_macro_recall"]),
        )
        decision = 0
        for candidate_value, best_value in zip(candidate_metrics, best_metrics, strict=True):
            if candidate_value > best_value + tolerance:
                decision = 1
                break
            if best_value > candidate_value + tolerance:
                decision = -1
                break
        if decision == 0:
            candidate_c, best_c = float(candidate["C"]), float(best["C"])
            if candidate_c < best_c:
                decision = 1
            elif candidate_c == best_c:
                candidate_gamma = str(candidate["gamma"])
                best_gamma = str(best["gamma"])
                if candidate_gamma == "scale" and best_gamma != "scale":
                    decision = 1
        if decision == 1:
            best = candidate
    if best is None:
        raise ModelSelectionError("no validation configuration available")
    return dict(best)
