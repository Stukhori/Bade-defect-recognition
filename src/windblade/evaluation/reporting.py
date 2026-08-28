"""CSV and restrained scientific plots for traditional baselines."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windblade.utils import atomic_write_text


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(Path(path), buffer.getvalue())


def write_matrix_csv(path: str | Path, matrix: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
    rows = []
    for label, values in zip(labels, matrix, strict=True):
        rows.append({"true_label": label, **{name: value for name, value in zip(labels, values, strict=True)}})
    write_csv(path, rows, ("true_label", *labels))


def plot_confusion(
    matrix: Sequence[Sequence[float]], labels: Sequence[str], output: str | Path, *, normalized: bool
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Row-normalized confusion matrix" if normalized else "Confusion matrix counts")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{values[row, column]:.2f}" if normalized else str(int(values[row, column]))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_validation_grid(rows: Sequence[Mapping[str, Any]], method: str, output: str | Path) -> None:
    selected = [row for row in rows if row["method"] == method]
    labels = [f"C={row['C']:g}\n{row['gamma']}" for row in selected]
    values = [float(row["validation_macro_f1"]) for row in selected]
    colors = ["#2f78c4" if not row["selected"] else "#e07a1f" for row in selected]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(range(len(values)), values, color=colors)
    axis.set_ylim(0, 1)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Validation macro-F1")
    axis.set_title(f"{method.upper()} fixed SVM validation grid")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_baseline_comparison(metrics: Mapping[str, Mapping[str, float]], output: str | Path) -> None:
    names = list(metrics)
    measures = ("macro_f1", "balanced_accuracy", "accuracy")
    x = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 5))
    for index, measure in enumerate(measures):
        axis.bar(x + (index - 1) * width, [metrics[name][measure] for name in names], width, label=measure)
    axis.set_ylim(0, 1)
    axis.set_xticks(x, [name.upper() + " + SVM" for name in names])
    axis.set_ylabel("Test score")
    axis.set_title("Traditional baseline comparison")
    axis.legend()
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
