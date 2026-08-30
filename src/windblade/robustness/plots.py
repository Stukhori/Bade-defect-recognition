"""Restrained Phase 8 scientific figures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windblade.data.processed import LABELS
from windblade.evaluation.reporting import plot_confusion
from windblade.robustness.aggregation import METHODS


DISPLAY = {
    "hog": "HOG + SVM",
    "lbp": "LBP + SVM",
    "resnet18": "ResNet-18",
    "mobilenet_v3_small": "MobileNetV3-Small",
}
COLORS = {"hog": "#4c78a8", "lbp": "#f58518", "resnet18": "#54a24b", "mobilenet_v3_small": "#e45756"}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_main_figures(
    aggregates: Mapping[str, Sequence[Mapping[str, Any]]],
    entries: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> list[str]:
    root = Path(output_root)
    summary = {(row["method"], row["condition_id"]): row for row in aggregates["robustness_summary"]}
    outputs: list[Path] = []
    for family, filename, title in (
        ("gaussian_blur", "blur_robustness.png", "Gaussian blur robustness"),
        ("resolution", "resolution_robustness.png", "Resolution-degradation robustness"),
        ("brightness", "brightness_robustness.png", "Brightness-reduction robustness"),
        ("jpeg", "jpeg_robustness.png", "JPEG-compression robustness"),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        x = np.arange(4)
        condition_ids = ("clean", *(f"{family}_{level}" for level in ("mild", "moderate", "severe")))
        for method in METHODS:
            rows = [summary[(method, condition)] for condition in condition_ids]
            values = [row["macro_f1"] for row in rows]
            deviations = [row["macro_f1_sample_sd"] or 0.0 for row in rows]
            if method in {"resnet18", "mobilenet_v3_small"}:
                axis.errorbar(x, values, yerr=deviations, marker="o", capsize=4, label=DISPLAY[method], color=COLORS[method])
            else:
                axis.plot(x, values, marker="o", label=DISPLAY[method], color=COLORS[method])
        axis.set_xticks(x, ("clean", "mild", "moderate", "severe"))
        axis.set_ylim(0, 1)
        axis.set_ylabel("Macro-F1")
        axis.set_title(title)
        axis.legend()
        destination = root / filename
        _save(figure, destination)
        outputs.append(destination)

    ordered_conditions = [f"{family}_{severity}" for family in ("gaussian_blur", "resolution", "brightness", "jpeg") for severity in ("mild", "moderate", "severe")]
    retention = np.asarray([[summary[(method, condition)]["retention_percent"] for condition in ordered_conditions] for method in METHODS])
    figure, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(retention, cmap="viridis", vmin=0, vmax=max(100.0, float(retention.max())))
    axis.set_xticks(range(12), [condition.replace("gaussian_blur", "blur").replace("_", "\n", 1) for condition in ordered_conditions])
    axis.set_yticks(range(4), [DISPLAY[method] for method in METHODS])
    axis.set_title("Clean-performance retention (%)")
    for row in range(4):
        for column in range(12):
            axis.text(column, row, f"{retention[row, column]:.1f}", ha="center", va="center", fontsize=8, color="white" if retention[row, column] < 55 else "black")
    figure.colorbar(image, ax=axis, label="Retention (%)")
    destination = root / "performance_retention_heatmap.png"
    _save(figure, destination)
    outputs.append(destination)

    severe = list(aggregates["severe_summary"])
    families = ("gaussian_blur", "resolution", "brightness", "jpeg")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(4)
    width = 0.19
    for index, method in enumerate(METHODS):
        values = [next(row["macro_f1"] for row in severe if row["method"] == method and row["corruption_family"] == family) for family in families]
        errors = [next(row["macro_f1_sample_sd"] or 0.0 for row in severe if row["method"] == method and row["corruption_family"] == family) for family in families]
        axis.bar(x + (index - 1.5) * width, values, width, yerr=errors if method not in {"hog", "lbp"} else None, capsize=3, label=DISPLAY[method], color=COLORS[method])
    axis.set_xticks(x, ("blur", "resolution", "brightness", "JPEG"))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Severe-condition macro-F1")
    axis.set_title("Severe simulated image-degradation comparison")
    axis.legend()
    destination = root / "severe_condition_comparison.png"
    _save(figure, destination)
    outputs.append(destination)

    overall = list(aggregates["overall_summary"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(range(4), [next(row["mean_degraded_condition_macro_f1"] for row in overall if row["method"] == method) for method in METHODS], color=[COLORS[method] for method in METHODS])
    axis.set_xticks(range(4), [DISPLAY[method] for method in METHODS], rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean macro-F1")
    axis.set_title("Mean across twelve predeclared degraded conditions")
    destination = root / "mean_degraded_condition_macro_f1.png"
    _save(figure, destination)
    outputs.append(destination)

    flips = {(row["method"], row["condition_id"]): row for row in aggregates["prediction_flip_rates"]}
    figure, axis = plt.subplots(figsize=(13, 5.5))
    for method in METHODS:
        values = [100.0 * flips[(method, condition)]["prediction_flip_rate"] for condition in ordered_conditions]
        errors = [100.0 * (flips[(method, condition)]["prediction_flip_rate_sample_sd"] or 0.0) for condition in ordered_conditions]
        axis.errorbar(range(12), values, yerr=errors if method not in {"hog", "lbp"} else None, marker="o", capsize=3, label=DISPLAY[method], color=COLORS[method])
    axis.set_xticks(range(12), [condition.replace("gaussian_blur", "blur").replace("_", "\n", 1) for condition in ordered_conditions])
    axis.set_ylabel("Prediction flip rate (%)")
    axis.set_title("Prediction changes relative to clean inference")
    axis.legend()
    destination = root / "prediction_flip_rates.png"
    _save(figure, destination)
    outputs.append(destination)

    per_class = list(aggregates["per_class_robustness"])
    severe_conditions = ("clean", "gaussian_blur_severe", "resolution_severe", "brightness_severe", "jpeg_severe")
    for method in ("resnet18", "mobilenet_v3_small"):
        matrix = np.asarray([[next(row["f1_mean"] for row in per_class if row["method"] == method and row["condition_id"] == condition and row["class"] == label) for condition in severe_conditions] for label in LABELS])
        figure, axis = plt.subplots(figsize=(9, 6))
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1)
        axis.set_xticks(range(5), ("clean", "blur", "resolution", "brightness", "JPEG"))
        axis.set_yticks(range(6), LABELS)
        axis.set_title(f"{DISPLAY[method]} per-class F1 at severe degradation")
        for row in range(6):
            for column in range(5):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white" if matrix[row, column] < 0.55 else "black")
        figure.colorbar(image, ax=axis, label="Mean F1")
        destination = root / f"per_class_severe_{method}.png"
        _save(figure, destination)
        outputs.append(destination)

    entry_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        entry_groups[(str(entry["method"]), str(entry["condition_id"]))].append(entry)
    for method in METHODS:
        for condition in severe_conditions:
            matrices = np.asarray([entry["result"]["metrics"]["confusion_matrix_row_normalized"] for entry in entry_groups[(method, condition)]], dtype=np.float64)
            destination = root / "confusion_matrices" / f"{method}_{condition}_mean_normalized.png"
            plot_confusion(matrices.mean(axis=0), LABELS, destination, normalized=True)
            outputs.append(destination)
    return [path.as_posix() for path in outputs]
