"""Deterministic quantitative Phase 9A figures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windblade.data.processed import LABELS


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "windblade-phase9a"})
    plt.close(fig)
    return path.as_posix()


def _matrix_figure(values: Sequence[Mapping[str, Any]], title: str, path: Path) -> str:
    counts = np.zeros((6, 6), dtype=int)
    normalized = np.zeros((6, 6), dtype=float)
    for row in values:
        i, j = int(row["true_class_id"]), int(row["predicted_class_id"])
        counts[i, j], normalized[i, j] = int(row["count"]), float(row["row_normalized"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, matrix, label, fmt in ((axes[0], counts, "Raw count", "d"), (axes[1], normalized, "Row-normalized", ".2f")):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=(counts.max() if fmt == "d" else 1))
        for i in range(6):
            for j in range(6): axis.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7)
        axis.set_xticks(range(6), LABELS, rotation=45, ha="right"); axis.set_yticks(range(6), LABELS)
        axis.set_xlabel("Predicted class"); axis.set_ylabel("True class"); axis.set_title(label)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(title); fig.tight_layout()
    return _save(fig, path)


def create_figures(
    confusion: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]],
    difficulty: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], figures_root: Path,
) -> list[str]:
    outputs: list[str] = []
    groups: dict[tuple[str, Any, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in confusion: groups[(str(row["method"]), row["seed"], str(row["condition_id"]))].append(row)
    for (method, seed, condition), values in groups.items():
        if condition != "clean" and not condition.endswith("_severe"): continue
        seed_label = "deterministic" if seed == "not_applicable" else f"seed_{seed}"
        path = figures_root / "confusion_matrices" / method / seed_label / f"{condition}.png"
        outputs.append(_matrix_figure(values, f"{method} / {seed_label} / {condition}", path))

    severe = [row for row in transitions if row["severity"] == "severe"]
    fig, axis = plt.subplots(figsize=(11, 6))
    labels = [f"{row['method']}\n{row['corruption_family']}" for row in severe]
    values = [float(row["harmful_flip_mean"]) for row in severe]
    axis.bar(range(len(values)), values, color="#9c2f45"); axis.set_xticks(range(len(values)), labels, rotation=60, ha="right")
    axis.set_ylabel("Harmful flips (mean across seeds where applicable)"); axis.set_title("Severe-corruption harmful flips")
    fig.tight_layout(); outputs.append(_save(fig, figures_root / "severe_harmful_flips.png"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for axis, method in zip(axes.flat, ("hog", "lbp", "resnet18", "mobilenet_v3_small"), strict=True):
        values = [row for row in difficulty if row["method"] == method]
        x = np.arange(6); width = 0.36
        axis.bar(x - width / 2, [float(row["clean_recall_mean"]) for row in values], width, label="clean recall")
        axis.bar(x + width / 2, [float(row["mean_severe_recall"]) for row in values], width, label="mean severe recall")
        axis.set_xticks(x, LABELS, rotation=40, ha="right"); axis.set_ylim(0, 1); axis.set_title(method); axis.legend(fontsize=8)
    fig.suptitle("Observed class recall indicators"); fig.tight_layout(); outputs.append(_save(fig, figures_root / "class_difficulty_recall.png"))

    selected_geometry = [row for row in geometry if row["condition_id"] == "clean" and row["seed"] in (17, "not_applicable") and row["geometry_variable"] == "occupancy_bin"]
    fig, axis = plt.subplots(figsize=(11, 6))
    keys = sorted({str(row["geometry_value"]) for row in selected_geometry})
    x = np.arange(len(keys)); width = 0.18
    for index, method in enumerate(("hog", "lbp", "resnet18", "mobilenet_v3_small")):
        lookup = {str(row["geometry_value"]): float(row["error_rate"]) for row in selected_geometry if row["method"] == method}
        axis.bar(x + (index - 1.5) * width, [lookup.get(key, 0.0) for key in keys], width, label=method)
    axis.set_xticks(x, keys); axis.set_ylabel("Clean error rate"); axis.set_title("Clean errors by frozen defect-occupancy bin (CNN seed 17 shown)"); axis.legend()
    fig.tight_layout(); outputs.append(_save(fig, figures_root / "geometry_occupancy_clean_error.png"))
    return outputs
