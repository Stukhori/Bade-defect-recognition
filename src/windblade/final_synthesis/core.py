"""Phase 10 aggregation, paired bootstrap inference, reporting, and validation.

This module reads frozen Phase 3--9 artifacts only.  It never instantiates a
model, creates a prediction, or changes an upstream scientific file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windblade.config import ResolvedConfig, calculate_config_hash, load_config
from windblade.data.processed import LABELS, read_csv, sha256_file, validate_processed_dataset
from windblade.data.subsets import validate_training_subsets
from windblade.evaluation.metrics import classification_metrics
from windblade.traditional import validate_traditional_results
from windblade.resnet_experiment import validate_resnet18_results
from windblade.mobilenet_experiment import validate_mobilenet_results
from windblade.data_efficiency import validate_data_efficiency_results
from windblade.robustness.runner import validate_robustness_results
from windblade.error_analysis.runner import validate_error_analysis
from windblade.error_analysis.phase9b import validate_phase9b
from windblade.utils import atomic_write_text


METHODS = ("hog", "lbp", "resnet18", "mobilenet_v3_small")
CNN_METHODS = ("resnet18", "mobilenet_v3_small")
METHOD_NAMES = {
    "hog": "HOG + SVM",
    "lbp": "LBP + SVM",
    "resnet18": "ResNet-18",
    "mobilenet_v3_small": "MobileNetV3-Small",
}
METHOD_COLORS = {
    "hog": "#0072B2",
    "lbp": "#009E73",
    "resnet18": "#D55E00",
    "mobilenet_v3_small": "#CC79A7",
}
SCIENTIFIC_EXCLUSIONS = frozenset({"reproducibility.json", "validation.json"})


class FinalSynthesisError(RuntimeError):
    """Raised when a Phase 10 contract or validation gate fails."""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _hash_mapping(mapping: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(mapping.items())), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_table(root: Path, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    table_root = root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    csv_path = table_root / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field)) for field in fields})
    atomic_write_text(table_root / f"{name}.json", _json_text({"fields": list(fields), "rows": list(rows)}))


def _read_prediction(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = sorted(read_csv(path), key=lambda row: row["instance_id"])
    if len(rows) != 162:
        raise FinalSynthesisError(f"expected 162 predictions in {path}")
    ids = [row["instance_id"] for row in rows]
    true = np.asarray([int(row["true_class_id"]) for row in rows], dtype=np.int64)
    predicted = np.asarray([int(row["predicted_class_id"]) for row in rows], dtype=np.int64)
    if len(set(ids)) != 162 or not np.all((true >= 0) & (true < len(LABELS))):
        raise FinalSynthesisError(f"invalid prediction identities in {path}")
    return ids, true, predicted


def load_clean_predictions(root: Path) -> tuple[list[str], np.ndarray, dict[str, list[np.ndarray]]]:
    paths: dict[str, list[Path]] = {
        "hog": [root / "experiments/summaries/phase4_traditional_v1/hog/test_predictions.csv"],
        "lbp": [root / "experiments/summaries/phase4_traditional_v1/lbp/test_predictions.csv"],
        "resnet18": [
            root / f"experiments/summaries/phase5_resnet18_v1/final/seed_{seed}/test_predictions.csv"
            for seed in (17, 29, 43)
        ],
        "mobilenet_v3_small": [
            root / f"experiments/summaries/phase6_mobilenet_v3_small_v1/final/seed_{seed}/test_predictions.csv"
            for seed in (17, 29, 43)
        ],
    }
    canonical_ids: list[str] | None = None
    canonical_true: np.ndarray | None = None
    predictions: dict[str, list[np.ndarray]] = {}
    for method in METHODS:
        predictions[method] = []
        for path in paths[method]:
            ids, true, predicted = _read_prediction(path)
            if canonical_ids is None:
                canonical_ids, canonical_true = ids, true
            elif ids != canonical_ids or not np.array_equal(true, canonical_true):
                raise FinalSynthesisError("clean prediction pairing or labels changed")
            predictions[method].append(predicted)
    assert canonical_ids is not None and canonical_true is not None
    return canonical_ids, canonical_true, predictions


def bootstrap_indices(y_true: Sequence[int], *, resamples: int, seed: int) -> np.ndarray:
    """Generate paired class-stratified bootstrap positions deterministically."""

    true = np.asarray(y_true, dtype=np.int64)
    if true.ndim != 1 or len(true) == 0 or resamples <= 0:
        raise ValueError("bootstrap inputs must be a nonempty vector and positive resample count")
    labels = sorted(int(value) for value in np.unique(true))
    if labels != list(range(len(LABELS))):
        raise ValueError("bootstrap requires all six frozen classes")
    groups = [np.flatnonzero(true == label) for label in labels]
    rng = np.random.default_rng(seed)
    result = np.empty((resamples, len(true)), dtype=np.int64)
    for row in range(resamples):
        result[row] = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
    return result


def _metric_vector(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    counts = np.bincount(true * 6 + predicted, minlength=36).reshape(6, 6).astype(np.float64)
    support = counts.sum(axis=1)
    predicted_count = counts.sum(axis=0)
    diagonal = np.diag(counts)
    recall = np.divide(diagonal, support, out=np.zeros(6), where=support != 0)
    precision = np.divide(diagonal, predicted_count, out=np.zeros(6), where=predicted_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) != 0)
    overall = np.asarray(
        [f1.mean(), diagonal.sum() / counts.sum(), recall.mean(), precision.mean(), recall.mean()],
        dtype=np.float64,
    )
    return np.concatenate([overall, precision, recall, f1])


def _bootstrap_metrics(
    true: np.ndarray, predictions: Mapping[str, Sequence[np.ndarray]], indices: np.ndarray
) -> dict[str, np.ndarray]:
    result = {method: np.empty((len(indices), 23), dtype=np.float64) for method in METHODS}
    for resample, positions in enumerate(indices):
        sampled_true = true[positions]
        for method in METHODS:
            seed_metrics = [_metric_vector(sampled_true, predicted[positions]) for predicted in predictions[method]]
            result[method][resample] = np.mean(seed_metrics, axis=0)
    return result


def _percentile(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted values; provided for validation although Phase 10 omits p-values."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def validate_statistical_plan(config: ResolvedConfig) -> dict[str, Any]:
    data = config.as_dict()
    plan = data.get("statistical_plan", {})
    bootstrap = plan.get("bootstrap", {})
    failures: list[str] = []
    checks = {
        "phase": data.get("project", {}).get("phase") == 10,
        "primary_metric": plan.get("primary_metric") == "macro_f1" == data["evaluation"]["primary_metric"],
        "frozen_before_generation": plan.get("frozen_before_generation") is True,
        "paired": plan.get("pairing") == "same_resampled_test_instances_for_every_method_and_comparison",
        "bootstrap_method": bootstrap.get("method") == "paired_class_stratified_nonparametric_percentile",
        "resamples": int(bootstrap.get("resamples", 0)) >= 1000,
        "fixed_seed": isinstance(bootstrap.get("seed"), int),
        "confidence": float(bootstrap.get("confidence_level", 0)) == 0.95,
        "stratified": bootstrap.get("stratification") == "true_class",
        "common_indices": bootstrap.get("common_indices_across_methods") is True,
        "cnn_aggregation": plan.get("cnn_treatment") == "calculate_each_seed_metric_within_each_resample_then_average_three_seed_metrics",
        "deterministic_sd": "seed_sd_not_applicable" in str(plan.get("deterministic_treatment")),
        "missing_rejected": str(plan.get("missing_value_policy", "")).startswith("reject_"),
        "p_values_omitted": plan.get("p_values") == "omitted",
        "six_pairs": len(plan.get("pairwise_comparisons", [])) == 6,
        "interpretation_limits": len(plan.get("interpretation_limits", [])) >= 5,
    }
    for key, passed in checks.items():
        if not passed:
            failures.append(key)
    expected_pairs = {tuple(pair) for pair in combinations(METHODS, 2)}
    if {tuple(pair) for pair in plan.get("pairwise_comparisons", [])} != expected_pairs:
        failures.append("pairwise_comparison_family")
    if failures:
        raise FinalSynthesisError("invalid frozen statistical plan: " + ", ".join(failures))
    return {"status": "PASS", "checks": checks, "config_fingerprint": calculate_config_hash(data, length=64)}


def _upstream_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    explicit = [
        "configs/crop_dataset.yaml", "configs/traditional_baselines.yaml", "configs/resnet18_baseline.yaml",
        "configs/mobilenet_v3_small_baseline.yaml", "configs/data_efficiency.yaml", "configs/robustness.yaml",
        "configs/error_analysis.yaml", "data/processed/wtbd_crops_v1/manifest.csv",
        "data/processed/wtbd_crops_v1/crop_checksum_manifest.csv", "data/splits/wtbd_crops_v1_split.csv",
        "data/processed/wtbd_robustness_v1/manifest.csv", "data/processed/wtbd_robustness_v1/corruption_checksum_manifest.csv",
        "data/processed/wtbd_robustness_v1/conditions.json", "data/processed/wtbd_robustness_v1/summary.json",
        "phase9a_corrected_pass_b_review_2026-08-31.zip",
    ]
    for relative in explicit:
        path = root / relative
        if not path.is_file():
            raise FinalSynthesisError(f"required upstream file is absent: {relative}")
        files.add(path)
    for relative in (
        "experiments/summaries/phase4_traditional_v1", "experiments/summaries/phase5_resnet18_v1",
        "experiments/summaries/phase6_mobilenet_v3_small_v1", "experiments/summaries/phase7_data_efficiency_v1",
        "experiments/summaries/phase8_robustness_v1", "experiments/summaries/phase9_error_analysis_v1",
        "experiments/audits", "figures/phase3", "figures/phase4", "figures/phase5", "figures/phase6",
        "figures/phase7", "figures/phase8", "figures/phase9",
    ):
        directory = root / relative
        if not directory.is_dir():
            raise FinalSynthesisError(f"required upstream directory is absent: {relative}")
        files.update(path for path in directory.rglob("*") if path.is_file())
    for relative in (
        "experiments/results/phase4_traditional_v1/hog/model.joblib",
        "experiments/results/phase4_traditional_v1/lbp/model.joblib",
    ):
        path = root / relative
        if path.is_file():
            files.add(path)
    for architecture, phase in (("resnet18", 5), ("mobilenet_v3_small", 6)):
        result_id = "phase5_resnet18_v1" if phase == 5 else "phase6_mobilenet_v3_small_v1"
        for seed in (17, 29, 43):
            path = root / f"experiments/results/{result_id}/final/seed_{seed}/best_state_dict.pt"
            if not path.is_file():
                raise FinalSynthesisError(f"required frozen checkpoint is absent: {path.relative_to(root)}")
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def collect_upstream_inventory(root: Path) -> dict[str, Any]:
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in _upstream_files(root)}
    return {"file_count": len(hashes), "fingerprint": _hash_mapping(hashes), "files": hashes}


def _assert_no_optional_phase_paths(root: Path) -> None:
    # Phase 11 is an authorized, scientifically separate downstream experiment
    # once the Phase 10 freeze exists. Historical Phase 10 validation therefore
    # rejects only still-unauthorized Phase 12 work.
    forbidden_tokens = ("phase12", "phase_12")
    offending = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if any(token in path.relative_to(root).as_posix().lower() for token in forbidden_tokens)
        and ".git" not in path.parts and ".venv" not in path.parts and ".uv-cache" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and "cache" not in path.relative_to(root).parts
    ]
    if offending:
        raise FinalSynthesisError("forbidden Phase 12 paths exist: " + ", ".join(offending[:5]))


def _validate_upstream_identity(config: ResolvedConfig, root: Path) -> dict[str, Any]:
    data = config.as_dict()
    upstream = data["upstream"]
    _assert_no_optional_phase_paths(root)
    phase7 = _json(root / "experiments/summaries/phase7_data_efficiency_v1/manifest.json")
    phase8 = _json(root / "experiments/summaries/phase8_robustness_v1/manifest.json")
    phase9_preflight = _json(root / "experiments/summaries/phase9_error_analysis_v1/preflight.json")
    phase9_repro = _json(root / "experiments/summaries/phase9_error_analysis_v1/reproducibility.json")
    phase9b_repro = _json(root / "experiments/summaries/phase9_error_analysis_v1/phase9b/reproducibility.json")
    phase9b_manifest = _json(root / "experiments/summaries/phase9_error_analysis_v1/phase9b/manifest.json")
    checks = {
        "phase7_complete": phase7.get("status") == "completed" and phase7.get("result_id") == upstream["phase7_result_id"],
        "phase8_complete": phase8.get("status") == "completed" and phase8.get("result_id") == upstream["phase8_result_id"],
        "processed_fingerprint": phase8.get("base_dataset_fingerprint") == upstream["processed_dataset_fingerprint"],
        "phase8_config_fingerprint": phase8.get("corruption_config_fingerprint") == upstream["phase8_corruption_config_fingerprint"],
        "phase8_data_fingerprint": phase8.get("robustness_dataset_fingerprint") == upstream["phase8_robustness_dataset_fingerprint"],
        "phase9a_input": phase9_preflight.get("input_fingerprint") == upstream["phase9a_input_fingerprint"],
        "phase9a_output": phase9_repro.get("output_fingerprint") == upstream["phase9a_output_fingerprint"],
        "phase9b_output": phase9b_repro.get("phase9b_derived_output_fingerprint") == upstream["phase9b_output_fingerprint"],
        "phase9_complete": phase9b_manifest.get("phase9_complete") is True and phase9b_manifest.get("phase9_frozen") is True,
        "pass_a": sha256_file(root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet/pass_a/pass_a_review_form.csv") == upstream["pass_a_sha256"],
        "pass_b": sha256_file(root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet/pass_b/pass_b_review_form.csv") == upstream["pass_b_sha256"],
        "mapping": sha256_file(root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet/id_mapping/review_id_mapping.csv") == upstream["mapping_sha256"],
    }
    if not all(checks.values()):
        raise FinalSynthesisError("upstream identity gate failed: " + ", ".join(key for key, ok in checks.items() if not ok))
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", upstream["phase9_freeze_commit"], "HEAD"],
            cwd=root, check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FinalSynthesisError("Phase 9 freeze commit is not an ancestor of HEAD") from exc
    return {"status": "PASS", "checks": checks}


def _run_upstream_validators(root: Path) -> dict[str, Any]:
    crop = load_config(root / "configs/crop_dataset.yaml")
    error = load_config(root / "configs/error_analysis.yaml")
    results = {
        "phase3_dataset": validate_processed_dataset(crop, root)["status"],
        "phase3_subsets": "PASS" if validate_training_subsets(crop, root)["nesting_validated"] else "FAIL",
        "phase4": validate_traditional_results(load_config(root / "configs/traditional_baselines.yaml"), root)["status"],
        "phase5": validate_resnet18_results(load_config(root / "configs/resnet18_baseline.yaml"), root)["status"],
        "phase6": validate_mobilenet_results(load_config(root / "configs/mobilenet_v3_small_baseline.yaml"), root)["status"],
        "phase7": validate_data_efficiency_results(load_config(root / "configs/data_efficiency.yaml"), root)["status"],
        "phase8": validate_robustness_results(load_config(root / "configs/robustness.yaml"), root)["status"],
        "phase9a": validate_error_analysis(error, root)["status"],
        "phase9b": validate_phase9b(error, root)["status"],
    }
    if any(value != "PASS" for value in results.values()):
        raise FinalSynthesisError("an upstream validator failed")
    return {"status": "PASS", "validators": results}


@dataclass(frozen=True)
class CleanAnalysis:
    comparison: list[dict[str, Any]]
    per_class: list[dict[str, Any]]
    pairwise: list[dict[str, Any]]
    confusion: dict[str, Any]
    bootstrap_indices: np.ndarray


def _clean_analysis(config: ResolvedConfig, root: Path) -> CleanAnalysis:
    data = config.as_dict()
    plan = data["statistical_plan"]
    ids, true, predictions = load_clean_predictions(root)
    indices = bootstrap_indices(true, resamples=int(plan["bootstrap"]["resamples"]), seed=int(plan["bootstrap"]["seed"]))
    boot = _bootstrap_metrics(true, predictions, indices)
    confidence = float(plan["bootstrap"]["confidence_level"])
    comparison: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    metric_offsets = {"macro_f1": 0, "accuracy": 1, "balanced_accuracy": 2, "macro_precision": 3, "macro_recall": 4}
    for method in METHODS:
        seed_metrics = [classification_metrics(true, predicted, LABELS) for predicted in predictions[method]]
        aggregate: dict[str, float] = {}
        aggregate_sd: dict[str, float | None] = {}
        row: dict[str, Any] = {
            "method": method, "method_name": METHOD_NAMES[method], "designation": "deterministic" if method in ("hog", "lbp") else "three frozen seeds",
            "test_instances": len(ids), "test_sources": 109, "cnn_seed_count": 0 if method in ("hog", "lbp") else 3,
            "seed_sd_definition": "N/A" if method in ("hog", "lbp") else "sample SD (ddof=1)",
            "bootstrap_unit": "test instance, stratified by true class", "bootstrap_resamples": len(indices),
        }
        for metric, offset in metric_offsets.items():
            values = np.asarray([float(item[metric]) for item in seed_metrics])
            aggregate[metric] = float(values.mean())
            aggregate_sd[metric] = None if len(values) == 1 else float(values.std(ddof=1))
            low, high = _percentile(boot[method][:, offset], confidence)
            row[f"{metric}"] = aggregate[metric]
            row[f"{metric}_seed_sd"] = aggregate_sd[metric]
            row[f"{metric}_bootstrap_ci_low"] = low
            row[f"{metric}_bootstrap_ci_high"] = high
        comparison.append(row)
        matrices = np.asarray([item["confusion_matrix_counts"] for item in seed_metrics], dtype=float)
        confusion[method] = {
            "method_name": METHOD_NAMES[method], "class_order": list(LABELS),
            "seed_count": len(seed_metrics), "counts_per_seed": [item["confusion_matrix_counts"] for item in seed_metrics],
            "mean_counts": matrices.mean(axis=0).tolist(),
        }
        for class_id, label in enumerate(LABELS):
            class_row: dict[str, Any] = {
                "method": method, "method_name": METHOD_NAMES[method], "class_id": class_id, "class": label,
                "support": int(seed_metrics[0]["per_class"][label]["support"]),
                "cnn_seed_count": 0 if method in ("hog", "lbp") else 3,
                "seed_sd_definition": "N/A" if method in ("hog", "lbp") else "sample SD (ddof=1)",
            }
            for metric, base in (("precision", 5), ("recall", 11), ("f1", 17)):
                values = np.asarray([float(item["per_class"][label][metric]) for item in seed_metrics])
                low, high = _percentile(boot[method][:, base + class_id], confidence)
                class_row[metric] = float(values.mean())
                class_row[f"{metric}_seed_sd"] = None if len(values) == 1 else float(values.std(ddof=1))
                class_row[f"{metric}_bootstrap_ci_low"] = low
                class_row[f"{metric}_bootstrap_ci_high"] = high
            per_class.append(class_row)
    pairwise: list[dict[str, Any]] = []
    for method_a, method_b in (tuple(pair) for pair in plan["pairwise_comparisons"]):
        point_a = next(row["macro_f1"] for row in comparison if row["method"] == method_a)
        point_b = next(row["macro_f1"] for row in comparison if row["method"] == method_b)
        differences = boot[method_b][:, 0] - boot[method_a][:, 0]
        low, high = _percentile(differences, confidence)
        pairwise.append({
            "method_a": method_a, "method_a_name": METHOD_NAMES[method_a], "method_b": method_b,
            "method_b_name": METHOD_NAMES[method_b], "direction": f"{method_b} minus {method_a}",
            "macro_f1_a": point_a, "macro_f1_b": point_b, "difference_b_minus_a": point_b - point_a,
            "bootstrap_ci_low": low, "bootstrap_ci_high": high, "confidence_level": confidence,
            "sample_unit": "paired frozen test instance", "denominator": 162, "pairing": "yes",
            "cnn_seeds_a": 3 if method_a in CNN_METHODS else 0, "cnn_seeds_b": 3 if method_b in CNN_METHODS else 0,
            "p_value": None, "multiplicity": "no p-values; pointwise estimation interval; no familywise significance claim",
            "interpretation_limit": "fixed-test-sample uncertainty; CNN seed means retain all three seeds",
        })
    return CleanAnalysis(comparison, per_class, pairwise, confusion, indices)


def _data_efficiency_rows(root: Path) -> list[dict[str, Any]]:
    rows = read_csv(root / "experiments/summaries/phase7_data_efficiency_v1/aggregate/learning_curve_summary.csv")
    thresholds = {row["method"]: row for row in read_csv(root / "experiments/summaries/phase7_data_efficiency_v1/aggregate/threshold_95_percent.csv")}
    aucs = {row["method"]: row["normalized_macro_f1_learning_curve_auc"] for row in read_csv(root / "experiments/summaries/phase7_data_efficiency_v1/aggregate/normalized_learning_curve_auc.csv")}
    full = {row["method"]: float(row["macro_f1_mean"]) for row in rows if float(row["training_fraction"]) == 1.0}
    output = []
    for row in rows:
        value = float(row["macro_f1_mean"]); reference = full[row["method"]]
        output.append({
            **row, "method_name": METHOD_NAMES[row["method"]], "absolute_change_from_full": value - reference,
            "relative_change_from_full": value / reference - 1.0, "retention_percent": 100.0 * value / reference,
            "smallest_fraction_reaching_95_percent": float(thresholds[row["method"]]["threshold_fraction"]),
            "normalized_learning_curve_auc": float(aucs[row["method"]]), "inference": "descriptive only",
        })
    return output


def _robustness_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    for row in read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/robustness_summary.csv"):
        summary.append({**row, "method_name": METHOD_NAMES[row["method"]], "condition_design": "fixed synthetic grid point", "inference": "descriptive only"})
    severe = []
    for row in read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/severe_summary.csv"):
        severe.append({**row, "method_name": METHOD_NAMES[row["method"]], "condition_design": "fixed severe synthetic condition", "inference": "descriptive only"})
    return summary, severe


def _error_human_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = "experiments/summaries/phase9_error_analysis_v1/phase9b/tables/response_summary.csv"
    for row in read_csv(root / source):
        rows.append({
            "section": "human_review", "metric": row["field"], "category": row["response"],
            "value": int(row["count"]), "denominator": int(row["denominator"]), "unit": "cases",
            "percentage": float(row["percentage"]), "scope": f"single reviewer; {row['pass_name']}", "source": source,
        })
    cross_seed = read_csv(root / "experiments/summaries/phase9_error_analysis_v1/tables/cross_seed_distributions.csv")
    for row in cross_seed:
        if row["condition_id"] == "clean" and row["measure"] in {"unanimous_failure", "majority_only_failure", "seed_disagreement"} and row["value"] == "True":
            rows.append({
                "section": "clean_error_stability", "metric": row["measure"], "category": row["method"],
                "value": int(row["count"]), "denominator": int(row["denominator"]), "unit": "test instances",
                "percentage": 100.0 * int(row["count"]) / int(row["denominator"]), "scope": "descriptive across three frozen CNN seeds",
                "source": "experiments/summaries/phase9_error_analysis_v1/tables/cross_seed_distributions.csv",
            })
    agreement = read_csv(root / "experiments/summaries/phase9_error_analysis_v1/tables/cross_method_agreement.csv")
    for category in ("misclassified_by_all_four", "both_cnns_correct_both_handcrafted_wrong", "missed_by_both_cnns", "model_specific_failure"):
        identities = {row["instance_id"] for row in agreement if row["agreement_rule"] == "strict" and row["scope"] == "clean" and row["category"] == category}
        rows.append({
            "section": "cross_method_clean", "metric": category, "category": "strict CNN consensus", "value": len(identities),
            "denominator": 162, "unit": "test instances", "percentage": 100.0 * len(identities) / 162,
            "scope": "descriptive frozen clean predictions", "source": "experiments/summaries/phase9_error_analysis_v1/tables/cross_method_agreement.csv",
        })
    return rows


def _experimental_summary(root: Path) -> list[dict[str, Any]]:
    manifest = read_csv(root / "data/processed/wtbd_crops_v1/manifest.csv")
    counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "validation", "test")}
    sources = {split: len({row["source_image_id"] for row in manifest if row["split"] == split}) for split in counts}
    rows = []
    for split in ("train", "validation", "test"):
        rows.append({"dataset": "WTBD curated crop benchmark", "split": split, "source_images": sources[split], "instances": counts[split], "classes": 6, "split_unit": "source image", "processed_fingerprint": "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991"})
    return rows


def _tradeoff_rows(clean: Sequence[Mapping[str, Any]], data_efficiency: Sequence[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    overall = {row["method"]: row for row in read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/overall_summary.csv")}
    severe = read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/severe_summary.csv")
    phase7 = {row["method"]: row for row in data_efficiency if float(row["training_fraction"]) == 1.0}
    clean_map = {row["method"]: row for row in clean}
    rows = []
    for method in METHODS:
        severe_rows = [row for row in severe if row["method"] == method]
        rows.append({
            "method": method, "method_name": METHOD_NAMES[method], "clean_macro_f1": clean_map[method]["macro_f1"],
            "clean_seed_sd": clean_map[method]["macro_f1_seed_sd"],
            "mean_degraded_macro_f1": float(overall[method]["mean_degraded_condition_macro_f1"]),
            "mean_degraded_retention_percent": float(overall[method]["mean_degraded_condition_retention_percent"]),
            "mean_severe_retention_percent": float(np.mean([float(row["retention_percent"]) for row in severe_rows])),
            "fraction_reaching_95_percent_full": phase7[method]["smallest_fraction_reaching_95_percent"],
            "normalized_learning_curve_auc": phase7[method]["normalized_learning_curve_auc"],
            "training_randomness_replication": "N/A deterministic" if method in ("hog", "lbp") else "three frozen seeds",
            "interpretation": "multidimensional observed trade-off; no composite rank or universal winner",
        })
    return rows


def _reproducibility_rows(config: ResolvedConfig, root: Path, upstream_inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = config.as_dict(); upstream = data["upstream"]
    records = [
        ("phase9_freeze_commit", upstream["phase9_freeze_commit"], "Git"),
        ("phase3_processed_dataset", upstream["processed_dataset_fingerprint"], "Phase 3 manifest and crops"),
        ("phase8_corruption_config", upstream["phase8_corruption_config_fingerprint"], "Phase 8"),
        ("phase8_robustness_dataset", upstream["phase8_robustness_dataset_fingerprint"], "Phase 8"),
        ("phase9a_input", upstream["phase9a_input_fingerprint"], "Phase 9A"),
        ("phase9a_output", upstream["phase9a_output_fingerprint"], "Phase 9A"),
        ("phase9b_config", upstream["phase9b_config_fingerprint"], "Phase 9B"),
        ("phase9b_output", upstream["phase9b_output_fingerprint"], "Phase 9B"),
        ("pass_a_form", upstream["pass_a_sha256"], "human review"),
        ("pass_b_form", upstream["pass_b_sha256"], "human review"),
        ("review_id_mapping", upstream["mapping_sha256"], "human review"),
        ("phase10_config", calculate_config_hash(data, length=64), "Phase 10"),
        ("complete_upstream_inventory", upstream_inventory["fingerprint"], f"{upstream_inventory['file_count']} frozen files"),
    ]
    return [{"artifact": name, "fingerprint_or_commit": value, "scope": scope} for name, value, scope in records]


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": "windblade-phase10"})
    plt.close(fig)


def _figures(
    root: Path, output: Path, clean: CleanAnalysis, data_efficiency: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]], severe: Sequence[Mapping[str, Any]], human: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    figure_root = output / "figures"
    names = [METHOD_NAMES[method] for method in METHODS]
    colors = [METHOD_COLORS[method] for method in METHODS]
    values = [next(row["macro_f1"] for row in clean.comparison if row["method"] == method) for method in METHODS]
    lows = [next(row["macro_f1_bootstrap_ci_low"] for row in clean.comparison if row["method"] == method) for method in METHODS]
    highs = [next(row["macro_f1_bootstrap_ci_high"] for row in clean.comparison if row["method"] == method) for method in METHODS]
    fig, ax = plt.subplots(figsize=(8.5, 5)); x = np.arange(4)
    ax.bar(x, values, color=colors); ax.errorbar(x, values, yerr=[np.asarray(values)-lows, np.asarray(highs)-values], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, names, rotation=12); ax.set_ylim(0, 1); ax.set_ylabel("Test macro-F1")
    ax.set_title("Clean performance (95% paired stratified bootstrap CI)"); fig.tight_layout()
    path = figure_root / "clean_macro_f1_bootstrap_ci.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "Bars are frozen point estimates; error bars are 95% paired class-stratified percentile bootstrap intervals across 162 test instances, with CNN metrics averaged across all three frozen seeds within each resample.", "sources": ["tables/clean_method_comparison.csv"]}

    fig, ax = plt.subplots(figsize=(10, 5.5)); width = 0.19; class_x = np.arange(6)
    for index, method in enumerate(METHODS):
        rows = [row for row in clean.per_class if row["method"] == method]
        ax.bar(class_x + (index-1.5)*width, [row["f1"] for row in rows], width, label=METHOD_NAMES[method], color=METHOD_COLORS[method])
    ax.set_xticks(class_x, [label.replace("_", " ") for label in LABELS], rotation=15); ax.set_ylim(0, 1); ax.set_ylabel("Clean per-class F1"); ax.legend(ncol=2); ax.set_title("Frozen clean per-class performance"); fig.tight_layout()
    path = figure_root / "clean_per_class_f1.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "CNN bars are three-seed means; deterministic methods are single point estimates. No error bars are shown in this figure; exact seed SD and bootstrap intervals are tabulated.", "sources": ["tables/clean_per_class_performance.csv"]}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in METHODS:
        rows = sorted((row for row in data_efficiency if row["method"] == method), key=lambda row: float(row["training_fraction"]))
        xvals = [100*float(row["training_fraction"]) for row in rows]; yvals = [float(row["macro_f1_mean"]) for row in rows]
        errors = [np.nan if row["macro_f1_sample_sd"] in (None, "") else float(row["macro_f1_sample_sd"]) for row in rows]
        ax.errorbar(xvals, yvals, yerr=errors, marker="o", capsize=3, label=METHOD_NAMES[method], color=METHOD_COLORS[method])
    ax.set_xlabel("Labeled training source images (%)"); ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Frozen data-efficiency grid"); fig.tight_layout()
    path = figure_root / "data_efficiency_learning_curves.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "Error bars are sample SD across three predeclared reduced-data replicates; full-data HOG/LBP SD is N/A and is not drawn as zero.", "sources": ["tables/data_efficiency_summary.csv"]}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True); severity_order = {"clean":0,"mild":1,"moderate":2,"severe":3}
    for ax, family in zip(axes.flat, ("gaussian_blur", "resolution", "brightness", "jpeg")):
        for method in METHODS:
            rows = [row for row in robustness if row["method"] == method and (row["corruption_family"] == family or row["condition_id"] == "clean")]
            rows = sorted(rows, key=lambda row: severity_order[row["severity"]])
            ax.plot(range(4), [float(row["macro_f1"]) for row in rows], marker="o", color=METHOD_COLORS[method], label=METHOD_NAMES[method])
        ax.set_xticks(range(4), ["clean","mild","moderate","severe"]); ax.set_title(family.replace("_", " ")); ax.set_ylim(0,1)
    axes[0,0].set_ylabel("Macro-F1"); axes[1,0].set_ylabel("Macro-F1"); axes[1,0].set_xlabel("Severity"); axes[1,1].set_xlabel("Severity")
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4); fig.suptitle("Fixed synthetic corruption grid", y=.98); fig.tight_layout(rect=(0,.07,1,.96))
    path = figure_root / "robustness_curves.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "CNN curves are three-seed means; HOG/LBP are deterministic. Conditions are fixed synthetic design points, not random operational environments.", "sources": ["tables/robustness_retention_summary.csv"]}

    matrix = np.asarray([[float(next(row["retention_percent"] for row in severe if row["method"]==method and row["corruption_family"]==family)) for family in ("gaussian_blur","resolution","brightness","jpeg")] for method in METHODS])
    fig, ax = plt.subplots(figsize=(8,4.8)); image = ax.imshow(matrix, vmin=0, vmax=100, cmap="cividis")
    ax.set_xticks(range(4), ["Blur","Resolution","Brightness","JPEG"]); ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{matrix[i,j]:.1f}%",ha="center",va="center",color="white" if matrix[i,j] < 55 else "black")
    fig.colorbar(image, ax=ax, label="Severe retention (%)"); ax.set_title("Severe-condition retention on declared grid"); fig.tight_layout()
    path = figure_root / "severe_retention_heatmap.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "Retention is corrupted macro-F1 divided by each method's clean macro-F1. It is a descriptive ratio on four fixed severe conditions.", "sources": ["tables/severe_corruption_summary.csv"]}

    per_class = read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/per_class_robustness.csv")
    selected = [row for row in per_class if row["severity"] == "severe"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, family in zip(axes.flat, ("gaussian_blur","resolution","brightness","jpeg")):
        x = np.arange(6); width=.19
        for index, method in enumerate(METHODS):
            rows = [row for row in selected if row["method"]==method and row["corruption_family"]==family]
            ax.bar(x+(index-1.5)*width,[float(row["f1_mean"]) for row in rows],width,color=METHOD_COLORS[method],label=METHOD_NAMES[method])
        ax.set_xticks(x,[label.replace("_"," ") for label in LABELS],rotation=25); ax.set_title(family.replace("_"," ")); ax.set_ylim(0,1)
    axes[0,0].set_ylabel("Severe per-class F1"); axes[1,0].set_ylabel("Severe per-class F1")
    handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=4); fig.tight_layout(rect=(0,.08,1,1))
    path = figure_root / "severe_per_class_f1.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "CNN bars are three-seed means; deterministic bars are point estimates. Small class supports constrain interpretation.", "sources": ["experiments/summaries/phase8_robustness_v1/aggregate/per_class_robustness.csv"]}

    keys = [("dataset_label_visually_plausible","yes","Label plausible"),("activation_primarily_inside_annotation","inside","Activation inside"),("activation_primarily_inside_annotation","partial","Activation partial"),("pattern_consistent_across_cnn_seeds","yes","Seed consistent"),("pattern_consistent_across_cnn_seeds","partly","Seed partly")]
    values = [next(float(row["percentage"]) for row in human if row["metric"]==metric and row["category"]==category) for metric,category,_ in keys]
    fig, ax = plt.subplots(figsize=(8.5,4.8)); ax.barh([label for _,_,label in keys], values, color="#56B4E9"); ax.set_xlim(0,100); ax.set_xlabel("Reviewed cases (%)"); ax.set_title("Phase 9 single-reviewer descriptive judgments"); fig.tight_layout()
    path = figure_root / "phase9_human_review_summary.png"; _save_figure(fig, path)
    registry[path.name] = {"caption": "Percentages are one reviewer's post-hoc judgments on 60 deliberately selected cases; they are not objective ground truth or population estimates.", "sources": ["tables/error_human_review_summary.csv"]}
    return registry


def _output_inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in SCIENTIFIC_EXCLUSIONS
    }


def _generate_pass(config: ResolvedConfig, root: Path, output: Path, upstream_gate: Mapping[str, Any], upstream_inventory: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    data = config.as_dict(); clean = _clean_analysis(config, root)
    experimental = _experimental_summary(root); efficiency = _data_efficiency_rows(root)
    robustness, severe = _robustness_rows(root); human = _error_human_rows(root)
    tradeoffs = _tradeoff_rows(clean.comparison, efficiency, root)
    reproducibility = _reproducibility_rows(config, root, upstream_inventory)
    tables: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
    def register(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows: raise FinalSynthesisError(f"empty canonical table: {name}")
        fields = list(rows[0]);
        if any(list(row) != fields for row in rows): raise FinalSynthesisError(f"inconsistent fields in {name}")
        tables[name] = (rows, fields); _write_table(output, name, rows, fields)
    register("experimental_data_summary", experimental)
    register("clean_method_comparison", clean.comparison)
    register("clean_per_class_performance", clean.per_class)
    register("data_efficiency_summary", efficiency)
    register("robustness_retention_summary", robustness)
    register("severe_corruption_summary", severe)
    register("paired_macro_f1_differences", clean.pairwise)
    register("error_human_review_summary", human)
    register("cross_phase_tradeoffs", tradeoffs)
    register("reproducibility_fingerprints", reproducibility)
    atomic_write_text(output / "clean_confusion_matrices.json", _json_text(clean.confusion))
    with (output / "bootstrap_indices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["resample_id", "sample_positions"])
        for index, positions in enumerate(clean.bootstrap_indices): writer.writerow([index, ";".join(str(int(value)) for value in positions)])
    atomic_write_text(output / "statistical_plan.json", _json_text(data["statistical_plan"]))
    atomic_write_text(output / "upstream_validation.json", _json_text(upstream_gate))
    atomic_write_text(output / "upstream_inventory.json", _json_text(upstream_inventory))
    figures = _figures(root, output, clean, efficiency, robustness, severe, human)
    for record in figures.values():
        record["source_hashes"] = {
            source: sha256_file(output / source) if (output / source).is_file() else sha256_file(root / source)
            for source in record["sources"]
        }
    atomic_write_text(output / "figure_registry.json", _json_text(figures))
    registry = {
        "lineage": "frozen dataset/splits -> frozen predictions/logits -> frozen phase metrics -> Phase 10 derivation -> canonical table/figure",
        "tables": {
            name: {"csv": f"tables/{name}.csv", "json": f"tables/{name}.json", "rows": len(rows), "upstream_sources": _table_sources(name)}
            for name, (rows, _) in tables.items()
        },
        "figures": figures,
    }
    atomic_write_text(output / "result_registry.json", _json_text(registry))
    runtime = {
        "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(), "operating_system_release": platform.release(), "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ("numpy","matplotlib","scikit-learn","Pillow","PyYAML","torch","torchvision")},
        "bootstrap_seed": data["statistical_plan"]["bootstrap"]["seed"], "bootstrap_resamples": data["statistical_plan"]["bootstrap"]["resamples"],
        "generation_commit": _git(root, "rev-parse", "HEAD"),
    }
    atomic_write_text(output / "runtime.json", _json_text(runtime))
    summary = {
        "status": "completed", "phase": 10, "result_id": data["outputs"]["result_id"],
        "primary_metric": "macro_f1", "p_values": "omitted", "canonical_table_count": len(tables),
        "canonical_figure_count": len(figures), "bootstrap_resamples": len(clean.bootstrap_indices),
        "clean_results": {row["method"]: row["macro_f1"] for row in clean.comparison},
        "mean_degraded_results": {row["method"]: float(row["mean_degraded_condition_macro_f1"]) for row in read_csv(root / "experiments/summaries/phase8_robustness_v1/aggregate/overall_summary.csv")},
        "phase11_started": False, "phase12_started": False, "core_technical_project_complete": True,
    }
    atomic_write_text(output / "summary.json", _json_text(summary))
    expected_files = sorted([f"tables/{name}.{extension}" for name in tables for extension in ("csv","json")] + [f"figures/{name}" for name in figures] + ["bootstrap_indices.csv","clean_confusion_matrices.json","statistical_plan.json","upstream_validation.json","upstream_inventory.json","figure_registry.json","result_registry.json","runtime.json","summary.json","manifest.json"])
    manifest = {
        "schema_version": "1.0", "phase": 10, "status": "complete", "result_id": data["outputs"]["result_id"],
        "phase9_frozen_before_start": True, "statistical_plan_frozen_before_generation": True,
        "model_training_count": 0, "new_prediction_count": 0, "p_values_calculated": 0,
        "canonical_tables": list(tables), "canonical_figures": list(figures), "expected_scientific_files": expected_files,
        "phase11_started": False, "phase12_started": False, "core_technical_project_complete": True,
        "phase10_config_fingerprint": calculate_config_hash(data, length=64),
    }
    atomic_write_text(output / "manifest.json", _json_text(manifest))
    observed = sorted(_output_inventory(output))
    if observed != expected_files:
        raise FinalSynthesisError(f"generated inventory mismatch: expected {len(expected_files)}, observed {len(observed)}")
    return {"summary": summary, "manifest": manifest, "inventory": _output_inventory(output)}


def _table_sources(name: str) -> list[str]:
    mapping = {
        "experimental_data_summary": ["data/processed/wtbd_crops_v1/manifest.csv"],
        "clean_method_comparison": ["Phase 4/5/6 frozen clean test predictions"],
        "clean_per_class_performance": ["Phase 4/5/6 frozen clean test predictions"],
        "data_efficiency_summary": ["experiments/summaries/phase7_data_efficiency_v1/aggregate"],
        "robustness_retention_summary": ["experiments/summaries/phase8_robustness_v1/aggregate/robustness_summary.csv"],
        "severe_corruption_summary": ["experiments/summaries/phase8_robustness_v1/aggregate/severe_summary.csv"],
        "paired_macro_f1_differences": ["Phase 4/5/6 frozen paired clean test predictions", "bootstrap_indices.csv"],
        "error_human_review_summary": ["experiments/summaries/phase9_error_analysis_v1", "experiments/summaries/phase9_error_analysis_v1/phase9b"],
        "cross_phase_tradeoffs": ["canonical clean, data-efficiency, and robustness sources"],
        "reproducibility_fingerprints": ["frozen Phase 3--9 manifests, forms, mapping, and complete upstream inventory"],
    }
    return mapping[name]


def apparatus_check(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    plan = validate_statistical_plan(config)
    upstream = _validate_upstream_identity(config, repository)
    ids, true, predictions = load_clean_predictions(repository)
    if len(ids) != 162 or any(len(values) != (3 if method in CNN_METHODS else 1) for method, values in predictions.items()):
        raise FinalSynthesisError("clean prediction matrix is incomplete")
    indices = bootstrap_indices(true, resamples=5, seed=config.as_dict()["statistical_plan"]["bootstrap"]["seed"])
    for row in indices:
        if [int(np.sum(true[row] == label)) for label in range(6)] != [27, 30, 33, 9, 14, 49]:
            raise FinalSynthesisError("class-stratified bootstrap integrity failed")
    return {"status": "PASS", "statistical_plan": plan, "upstream_identity": upstream, "paired_test_instances": len(ids), "phase11_started": False, "phase12_started": False}


def run_final_synthesis(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve(); data = config.as_dict()
    apparatus = apparatus_check(config, repository)
    upstream_validators = _run_upstream_validators(repository)
    upstream_gate = {"apparatus": apparatus, "validators": upstream_validators}
    upstream_inventory = collect_upstream_inventory(repository)
    scratch = repository / data["outputs"]["reproduction_scratch_root"]
    pass_a = scratch / "pass_a"; pass_b = scratch / "pass_b"
    first = _generate_pass(config, repository, pass_a, upstream_gate, upstream_inventory)
    second = _generate_pass(config, repository, pass_b, upstream_gate, upstream_inventory)
    if first["inventory"] != second["inventory"]:
        differing = sorted(set(first["inventory"]) | set(second["inventory"]))
        differing = [path for path in differing if first["inventory"].get(path) != second["inventory"].get(path)]
        raise FinalSynthesisError("two-pass regeneration differed: " + ", ".join(differing[:10]))
    canonical = repository / data["outputs"]["summary_root"]
    if canonical.exists():
        shutil.rmtree(canonical)
    shutil.copytree(pass_a, canonical)
    figure_root = repository / data["outputs"]["figures_root"]
    if figure_root.exists():
        shutil.rmtree(figure_root)
    shutil.copytree(pass_a / "figures", figure_root)
    shutil.rmtree(canonical / "figures")
    # Canonical figure paths live outside the summary root; hashes remain part of the scientific fingerprint.
    canonical_inventory = {k: v for k, v in first["inventory"].items() if not k.startswith("figures/")}
    canonical_inventory.update({f"figures/{path.name}": sha256_file(path) for path in sorted(figure_root.glob("*.png"))})
    fingerprint = _hash_mapping(canonical_inventory)
    reproduction = {
        "status": "PASS", "passes": 2, "exact_file_hash_equality": True,
        "scientific_file_count": len(canonical_inventory), "phase10_scientific_output_fingerprint": fingerprint,
        "bootstrap_indices_sha256": canonical_inventory["bootstrap_indices.csv"], "inventory": canonical_inventory,
        "environmental_timestamps_removed": True, "temporary_absolute_paths_excluded": True,
    }
    atomic_write_text(canonical / "reproducibility.json", _json_text(reproduction))
    validation = validate_final_synthesis(config, repository, run_upstream=False)
    atomic_write_text(canonical / "validation.json", _json_text(validation))
    shutil.rmtree(scratch)
    return {"status": "PASS", "result_id": data["outputs"]["result_id"], "output_fingerprint": fingerprint, "scientific_file_count": len(canonical_inventory), "two_pass_reproduction": "PASS", "canonical_tables": 10, "canonical_figures": 7, "phase11_started": False, "phase12_started": False}


def _canonical_inventory(root: Path, summary: Path, figures: Path) -> dict[str, str]:
    inventory = {
        path.relative_to(summary).as_posix(): sha256_file(path)
        for path in sorted(summary.rglob("*")) if path.is_file() and path.name not in SCIENTIFIC_EXCLUSIONS
    }
    inventory.update({f"figures/{path.name}": sha256_file(path) for path in sorted(figures.glob("*.png"))})
    return inventory


def validate_final_synthesis(config: ResolvedConfig, root: str | Path, *, run_upstream: bool = True) -> dict[str, Any]:
    repository = Path(root).resolve(); data = config.as_dict()
    validate_statistical_plan(config); _validate_upstream_identity(config, repository); _assert_no_optional_phase_paths(repository)
    summary = repository / data["outputs"]["summary_root"]; figures = repository / data["outputs"]["figures_root"]
    if not summary.is_dir() or not figures.is_dir(): raise FinalSynthesisError("Phase 10 outputs are absent")
    manifest = _json(summary / "manifest.json"); reproduction = _json(summary / "reproducibility.json")
    if manifest.get("status") != "complete" or manifest.get("phase11_started") or manifest.get("phase12_started"):
        raise FinalSynthesisError("Phase 10 manifest state is invalid")
    inventory = _canonical_inventory(repository, summary, figures)
    if inventory != reproduction.get("inventory") or _hash_mapping(inventory) != reproduction.get("phase10_scientific_output_fingerprint"):
        raise FinalSynthesisError("Phase 10 scientific output fingerprint changed")
    expected = set(manifest["expected_scientific_files"])
    actual_pass_layout = {path if not path.startswith("figures/") else path for path in inventory}
    if expected != actual_pass_layout:
        raise FinalSynthesisError("Phase 10 exact output inventory changed")
    stored_upstream = _json(summary / "upstream_inventory.json")
    current_upstream = collect_upstream_inventory(repository)
    if stored_upstream != current_upstream:
        raise FinalSynthesisError("a frozen Phase 3--9 artifact changed")
    for name in manifest["canonical_tables"]:
        csv_path = summary / f"tables/{name}.csv"; json_path = summary / f"tables/{name}.json"
        csv_rows = read_csv(csv_path); payload = _json(json_path)
        if payload["fields"] != list(csv_rows[0]) or len(payload["rows"]) != len(csv_rows):
            raise FinalSynthesisError(f"CSV/JSON table mismatch: {name}")
        for csv_row, json_row in zip(csv_rows, payload["rows"]):
            if csv_row != {field: _cell(json_row.get(field)) for field in payload["fields"]}:
                raise FinalSynthesisError(f"CSV/JSON value mismatch: {name}")
    figure_registry = _json(summary / "figure_registry.json")
    if set(figure_registry) != {path.name for path in figures.glob("*.png")}:
        raise FinalSynthesisError("figure registry mismatch")
    for record in figure_registry.values():
        for source, expected_hash in record["source_hashes"].items():
            source_path = summary / source if (summary / source).is_file() else repository / source
            if sha256_file(source_path) != expected_hash: raise FinalSynthesisError("figure/table source mismatch")
    indices = read_csv(summary / "bootstrap_indices.csv")
    if len(indices) != int(data["statistical_plan"]["bootstrap"]["resamples"]): raise FinalSynthesisError("bootstrap row count changed")
    _, true, _ = load_clean_predictions(repository)
    for row in indices:
        positions = np.asarray([int(value) for value in row["sample_positions"].split(";")], dtype=np.int64)
        if len(positions) != 162 or [int(np.sum(true[positions] == label)) for label in range(6)] != [27,30,33,9,14,49]:
            raise FinalSynthesisError("stored bootstrap stratification changed")
    upstream_status = _run_upstream_validators(repository) if run_upstream else {"status": "NOT_RUN"}
    return {
        "status": "PASS", "phase10_complete": True, "phase10_frozen": True,
        "output_fingerprint": reproduction["phase10_scientific_output_fingerprint"],
        "scientific_file_count": len(inventory), "canonical_table_count": len(manifest["canonical_tables"]),
        "canonical_figure_count": len(manifest["canonical_figures"]), "two_pass_reproduction": reproduction["status"],
        "upstream_artifacts_unchanged": True, "upstream_validators": upstream_status,
        "table_json_consistency": "PASS", "figure_data_consistency": "PASS",
        "bootstrap_integrity": "PASS", "p_values": "omitted", "phase11_started": False, "phase12_started": False,
    }
