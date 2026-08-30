"""Metric derivation and three-seed aggregation for Phase 8."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from windblade.data.processed import LABELS


METHODS = ("hog", "lbp", "resnet18", "mobilenet_v3_small")


def robustness_derivations(clean_macro_f1: float, corrupted_macro_f1: float) -> dict[str, float]:
    if clean_macro_f1 <= 0:
        raise ValueError("clean macro-F1 must be positive")
    retention = corrupted_macro_f1 / clean_macro_f1
    return {
        "absolute_drop": corrupted_macro_f1 - clean_macro_f1,
        "retention": retention,
        "retention_percent": retention * 100.0,
        "relative_loss": 1.0 - retention,
        "relative_loss_percent": (1.0 - retention) * 100.0,
    }


def prediction_transitions(
    true_ids: Sequence[int], clean_predictions: Sequence[int], corrupted_predictions: Sequence[int]
) -> dict[str, Any]:
    true = np.asarray(true_ids, dtype=np.int64)
    clean = np.asarray(clean_predictions, dtype=np.int64)
    corrupted = np.asarray(corrupted_predictions, dtype=np.int64)
    if not (len(true) == len(clean) == len(corrupted)) or len(true) == 0:
        raise ValueError("prediction transition inputs must have equal nonzero length")
    clean_correct = clean == true
    corrupted_correct = corrupted == true
    changed = clean != corrupted
    return {
        "prediction_flip_count": int(changed.sum()),
        "prediction_flip_rate": float(changed.mean()),
        "harmful_flip_count": int((clean_correct & ~corrupted_correct).sum()),
        "beneficial_flip_count": int((~clean_correct & corrupted_correct).sum()),
        "correct_to_correct": int((clean_correct & corrupted_correct).sum()),
        "correct_to_incorrect": int((clean_correct & ~corrupted_correct).sum()),
        "incorrect_to_correct": int((~clean_correct & corrupted_correct).sum()),
        "incorrect_to_incorrect": int((~clean_correct & ~corrupted_correct).sum()),
    }


def _mean_sd(values: Sequence[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) == 3 else None


def _prediction_ids(entry: Mapping[str, Any]) -> list[int]:
    return [int(row["predicted_class_id"]) for row in entry["result"]["predictions"]]


def aggregate_evaluations(
    entries: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate deterministic traditional results and three CNN seeds."""

    spec_by_id = {str(spec["condition_id"]): dict(spec) for spec in specs}
    expected = {"clean", *[key for key in spec_by_id if key != "clean"]}
    by_identity: dict[tuple[str, int | None], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for entry in entries:
        method = str(entry["method"])
        seed = entry.get("seed")
        condition_id = str(entry["condition_id"])
        if method not in METHODS or condition_id not in spec_by_id:
            raise ValueError("unexpected Phase 8 evaluation identity")
        by_identity[(method, int(seed) if seed is not None else None)][condition_id] = entry
    expected_identities = {
        ("hog", None),
        ("lbp", None),
        *((method, seed) for method in ("resnet18", "mobilenet_v3_small") for seed in (17, 29, 43)),
    }
    if set(by_identity) != expected_identities or any(set(values) != expected for values in by_identity.values()):
        raise ValueError("incomplete Phase 8 model-condition matrix")

    per_seed: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    transition_seed_rows: list[dict[str, Any]] = []
    for (method, seed), condition_entries in sorted(by_identity.items(), key=lambda item: (METHODS.index(item[0][0]), item[0][1] or 0)):
        clean_entry = condition_entries["clean"]
        clean_metrics = clean_entry["result"]["metrics"]
        clean_predictions = _prediction_ids(clean_entry)
        clean_by_id = {row["instance_id"]: row for row in clean_entry["result"]["predictions"]}
        for spec in specs:
            condition_id = str(spec["condition_id"])
            entry = condition_entries[condition_id]
            metrics = entry["result"]["metrics"]
            derived = robustness_derivations(float(clean_metrics["macro_f1"]), float(metrics["macro_f1"]))
            true_ids = [int(row["true_class_id"]) for row in entry["result"]["predictions"]]
            transitions = prediction_transitions(true_ids, clean_predictions, _prediction_ids(entry))
            row = {
                "method": method,
                "seed": seed,
                "condition_id": condition_id,
                "corruption_family": spec["corruption_family"],
                "severity": spec["severity"],
                "parameter": spec["parameter"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                **derived,
                **transitions,
            }
            per_seed.append(row)
            transition_seed_rows.append(
                {key: row[key] for key in (
                    "method", "seed", "condition_id", "corruption_family", "severity",
                    "prediction_flip_count", "prediction_flip_rate", "harmful_flip_count",
                    "beneficial_flip_count", "correct_to_correct", "correct_to_incorrect",
                    "incorrect_to_correct", "incorrect_to_incorrect",
                )}
            )
            if condition_id == "clean":
                continue
            for prediction in entry["result"]["predictions"]:
                clean_prediction = clean_by_id[prediction["instance_id"]]
                clean_correct = bool(clean_prediction["correct"])
                corrupted_correct = bool(prediction["correct"])
                instance = {
                    "method": method,
                    "seed": seed,
                    "instance_id": prediction["instance_id"],
                    "source_image_id": prediction["source_image_id"],
                    "true_label": prediction["true_label"],
                    "corruption_family": spec["corruption_family"],
                    "severity": spec["severity"],
                    "clean_prediction": clean_prediction["predicted_label"],
                    "corrupted_prediction": prediction["predicted_label"],
                    "clean_correct": clean_correct,
                    "corrupted_correct": corrupted_correct,
                    "prediction_changed": clean_prediction["predicted_class_id"] != prediction["predicted_class_id"],
                    "harmful_flip": clean_correct and not corrupted_correct,
                    "beneficial_flip": (not clean_correct) and corrupted_correct,
                }
                for label in LABELS:
                    key = f"logit_{label}"
                    instance[key] = prediction.get(key)
                instance_rows.append(instance)

    group_entries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        group_entries[(row["method"], row["condition_id"])].append(row)
    summary: list[dict[str, Any]] = []
    flip_summary: list[dict[str, Any]] = []
    transition_summary: list[dict[str, Any]] = []
    for method in METHODS:
        for spec in specs:
            rows = group_entries[(method, str(spec["condition_id"]))]
            expected_replicates = 1 if method in {"hog", "lbp"} else 3
            if len(rows) != expected_replicates:
                raise ValueError("incorrect Phase 8 replicate count")
            aggregate: dict[str, Any] = {
                "method": method,
                "condition_id": spec["condition_id"],
                "corruption_family": spec["corruption_family"],
                "severity": spec["severity"],
                "parameter": spec["parameter"],
                "replicate_count": len(rows),
                "standard_deviation": "N/A" if len(rows) == 1 else "sample (ddof=1)",
            }
            for key in (
                "macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall",
                "absolute_drop", "retention", "retention_percent", "relative_loss",
                "relative_loss_percent", "prediction_flip_rate", "prediction_flip_count",
                "harmful_flip_count", "beneficial_flip_count",
            ):
                mean, sd = _mean_sd([float(row[key]) for row in rows])
                aggregate[key] = mean
                aggregate[f"{key}_sample_sd"] = sd
            summary.append(aggregate)
            flip_summary.append({key: aggregate[key] for key in (
                "method", "condition_id", "corruption_family", "severity", "prediction_flip_rate",
                "prediction_flip_rate_sample_sd", "prediction_flip_count", "harmful_flip_count",
                "beneficial_flip_count",
            )})
            transition_row = {
                "method": method,
                "condition_id": spec["condition_id"],
                "corruption_family": spec["corruption_family"],
                "severity": spec["severity"],
            }
            for key in ("correct_to_correct", "correct_to_incorrect", "incorrect_to_correct", "incorrect_to_incorrect"):
                mean, sd = _mean_sd([float(row[key]) for row in rows])
                transition_row[f"{key}_mean"] = mean
                transition_row[f"{key}_sample_sd"] = sd
            transition_summary.append(transition_row)

    summary_by_key = {(row["method"], row["condition_id"]): row for row in summary}
    family_summary: list[dict[str, Any]] = []
    severe_summary: list[dict[str, Any]] = []
    overall_summary: list[dict[str, Any]] = []
    for method in METHODS:
        clean = summary_by_key[(method, "clean")]
        all_degraded = [row for row in summary if row["method"] == method and row["condition_id"] != "clean"]
        overall_summary.append(
            {
                "method": method,
                "clean_macro_f1": clean["macro_f1"],
                "mean_degraded_condition_macro_f1": float(np.mean([row["macro_f1"] for row in all_degraded])),
                "mean_degraded_condition_retention": float(np.mean([row["retention"] for row in all_degraded])),
                "mean_degraded_condition_retention_percent": float(np.mean([row["retention_percent"] for row in all_degraded])),
            }
        )
        for family in ("gaussian_blur", "resolution", "brightness", "jpeg"):
            levels = {severity: summary_by_key[(method, f"{family}_{severity}")] for severity in ("mild", "moderate", "severe")}
            family_row = {
                "method": method,
                "corruption_family": family,
                "clean_macro_f1": clean["macro_f1"],
                "mild_macro_f1": levels["mild"]["macro_f1"],
                "moderate_macro_f1": levels["moderate"]["macro_f1"],
                "severe_macro_f1": levels["severe"]["macro_f1"],
                "mean_degraded_macro_f1": float(np.mean([levels[level]["macro_f1"] for level in ("mild", "moderate", "severe")])),
                "severe_retention": levels["severe"]["retention"],
                "severe_retention_percent": levels["severe"]["retention_percent"],
                "mean_degraded_retention": float(np.mean([levels[level]["retention"] for level in ("mild", "moderate", "severe")])),
                "mean_degraded_retention_percent": float(np.mean([levels[level]["retention_percent"] for level in ("mild", "moderate", "severe")])),
            }
            family_summary.append(family_row)
            severe_summary.append(
                {
                    "method": method,
                    "corruption_family": family,
                    "macro_f1": levels["severe"]["macro_f1"],
                    "macro_f1_sample_sd": levels["severe"]["macro_f1_sample_sd"],
                    "retention": levels["severe"]["retention"],
                    "retention_percent": levels["severe"]["retention_percent"],
                    "prediction_flip_rate": levels["severe"]["prediction_flip_rate"],
                    "harmful_flip_count": levels["severe"]["harmful_flip_count"],
                    "beneficial_flip_count": levels["severe"]["beneficial_flip_count"],
                }
            )

    per_class: list[dict[str, Any]] = []
    entry_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        entry_groups[(str(entry["method"]), str(entry["condition_id"]))].append(entry)
    for method in METHODS:
        for spec in specs:
            condition_id = str(spec["condition_id"])
            condition_entries = entry_groups[(method, condition_id)]
            clean_entries = entry_groups[(method, "clean")]
            for label in LABELS:
                values = [float(entry["result"]["metrics"]["per_class"][label]["f1"]) for entry in condition_entries]
                clean_values = [float(entry["result"]["metrics"]["per_class"][label]["f1"]) for entry in clean_entries]
                value_mean, value_sd = _mean_sd(values)
                clean_mean, _ = _mean_sd(clean_values)
                per_class.append(
                    {
                        "method": method,
                        "condition_id": condition_id,
                        "corruption_family": spec["corruption_family"],
                        "severity": spec["severity"],
                        "class": label,
                        "support": condition_entries[0]["result"]["metrics"]["per_class"][label]["support"],
                        "f1_mean": value_mean,
                        "f1_sample_sd": value_sd,
                        "clean_f1_mean": clean_mean,
                        "class_f1_drop": value_mean - clean_mean,
                    }
                )
    return {
        "per_seed_results": per_seed,
        "robustness_summary": summary,
        "family_summary": family_summary,
        "severe_summary": severe_summary,
        "overall_summary": overall_summary,
        "per_class_robustness": per_class,
        "prediction_flip_rates": flip_summary,
        "error_transitions_per_seed": transition_seed_rows,
        "error_transitions": transition_summary,
        "instance_robustness": instance_rows,
    }
