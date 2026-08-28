"""Canonical Phase 4 HOG/LBP plus RBF-SVM experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import yaml

from windblade.config import ResolvedConfig, calculate_config_hash
from windblade.data.processed import LABELS, json_text, read_csv, validate_processed_dataset
from windblade.data.subsets import validate_training_subsets
from windblade.environment import capture_environment, capture_git_provenance
from windblade.evaluation.metrics import classification_metrics
from windblade.evaluation.reporting import (
    plot_baseline_comparison,
    plot_confusion,
    plot_validation_grid,
    write_csv,
    write_matrix_csv,
)
from windblade.evaluation.timing import median_batch_latency, prediction_timing
from windblade.features.cache import cache_key, load_or_extract_features
from windblade.features.hog import extract_hog, hog_config_hash
from windblade.features.lbp import extract_spatial_lbp, lbp_config_hash
from windblade.models.svm import fit_train_only, generate_svm_grid, select_configuration
from windblade.utils import atomic_write_text, format_utc, utc_now


class TraditionalExperimentError(RuntimeError):
    """Raised when a Phase 4 scientific safeguard fails."""


@dataclass(frozen=True)
class FeatureBundle:
    family: str
    features: np.ndarray
    config_hash: str
    fingerprint: str
    cache_metadata: dict[str, Any]
    extractor: Callable[[str | Path], np.ndarray]


GRID_FIELDS = (
    "method",
    "candidate_id",
    "C",
    "gamma",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "fit_seconds",
    "validation_prediction_seconds",
    "selected",
)


def _version(name: str) -> str:
    return metadata.version(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_id(method: str, C: float, gamma: str) -> str:
    c_text = f"{C:g}".replace(".", "p")
    return f"{method}_C{c_text}_gamma_{gamma}"


def _validate_phase4_config(data: Mapping[str, Any]) -> None:
    expected_fingerprint = "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991"
    failures: list[str] = []
    if data["project"]["phase"] != 4:
        failures.append("project phase is not 4")
    if data["dataset"]["processed_version"] != "wtbd_crops_v1":
        failures.append("processed dataset version changed")
    if data["dataset"]["processed_fingerprint"] != expected_fingerprint:
        failures.append("processed dataset fingerprint changed")
    if tuple(data["classes"]["order"]) != LABELS:
        failures.append("class order changed")
    if int(data["hog"]["expected_dimensions"]) != 6084:
        failures.append("HOG dimensionality changed")
    if int(data["lbp"]["expected_dimensions"]) != 1372:
        failures.append("LBP dimensionality changed")
    if data["svm"]["kernel"] != "rbf" or data["svm"]["class_weight"] != "balanced":
        failures.append("SVM family or class weighting changed")
    if bool(data["svm"]["probability"]):
        failures.append("SVM probability estimation must remain disabled")
    generate_svm_grid(data["svm"])
    if failures:
        raise TraditionalExperimentError("; ".join(failures))


def _safe_clean(path: Path, allowed_parent: Path) -> None:
    resolved, parent = path.resolve(), allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise TraditionalExperimentError(f"refusing to clean unsafe output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _validation_grid(
    method: str,
    features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    labels: np.ndarray,
    instance_ids: Sequence[str],
    source_ids: Sequence[str],
    svm_config: Mapping[str, Any],
    class_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate only train/validation inputs; no test input is accepted."""

    X_train, y_train = features[train_indices], labels[train_indices]
    X_validation, y_validation = features[validation_indices], labels[validation_indices]
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for candidate in generate_svm_grid(svm_config):
        candidate_id = _candidate_id(method, candidate["C"], candidate["gamma"])
        started = time.perf_counter()
        model = fit_train_only(
            X_train, y_train, C=float(candidate["C"]), gamma=str(candidate["gamma"])
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = model.predict(X_validation)
        prediction_seconds = time.perf_counter() - started
        metrics = classification_metrics(y_validation, predicted, class_names)
        rows.append(
            {
                "method": method,
                "candidate_id": candidate_id,
                "C": float(candidate["C"]),
                "gamma": candidate["gamma"],
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_macro_precision": metrics["macro_precision"],
                "validation_macro_recall": metrics["macro_recall"],
                "validation_macro_f1": metrics["macro_f1"],
                "fit_seconds": fit_seconds,
                "validation_prediction_seconds": prediction_seconds,
                "selected": False,
            }
        )
        for index, predicted_id in zip(validation_indices, predicted, strict=True):
            true_id = int(labels[index])
            predictions.append(
                {
                    "method": method,
                    "candidate_id": candidate_id,
                    "instance_id": instance_ids[index],
                    "source_image_id": source_ids[index],
                    "true_class_id": true_id,
                    "true_label": class_names[true_id],
                    "predicted_class_id": int(predicted_id),
                    "predicted_label": class_names[int(predicted_id)],
                    "correct": bool(true_id == int(predicted_id)),
                }
            )
    return rows, predictions


def _grid_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    scientific = [
        {
            key: row[key]
            for key in (
                "method",
                "candidate_id",
                "C",
                "gamma",
                "validation_accuracy",
                "validation_balanced_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
                "selected",
            )
        }
        for row in rows
    ]
    encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _freeze_selection(
    path: Path,
    *,
    method: str,
    selected: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    feature_hash: str,
    processed_fingerprint: str,
    grid_fingerprint: str,
    tolerance: float,
    git_commit: str | None,
) -> dict[str, Any]:
    expected = {
        "schema_version": "1.0",
        "phase": 4,
        "method": method,
        "processed_dataset_fingerprint": processed_fingerprint,
        "feature_config": dict(feature_config),
        "feature_config_hash": feature_hash,
        "svm": {
            "kernel": "rbf",
            "class_weight": "balanced",
            "probability": False,
            "C": float(selected["C"]),
            "gamma": str(selected["gamma"]),
        },
        "selection": {
            "split": "validation",
            "primary_metric": "macro_f1",
            "validation_macro_f1": float(selected["validation_macro_f1"]),
            "validation_balanced_accuracy": float(selected["validation_balanced_accuracy"]),
            "validation_macro_recall": float(selected["validation_macro_recall"]),
            "numeric_tolerance": tolerance,
            "validation_grid_fingerprint": grid_fingerprint,
        },
        "git_commit": git_commit,
    }
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        comparison = dict(existing)
        comparison.pop("selection_timestamp_utc", None)
        if comparison != expected:
            raise TraditionalExperimentError(f"existing frozen selection disagrees: {path}")
        return existing
    record = {**expected, "selection_timestamp_utc": format_utc(utc_now())}
    atomic_write_text(
        path,
        yaml.safe_dump(record, sort_keys=True, allow_unicode=True, default_flow_style=False),
    )
    return record


def _evaluate_selected(
    *,
    method: str,
    bundle: FeatureBundle,
    selected: Mapping[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    labels: np.ndarray,
    instance_ids: Sequence[str],
    source_ids: Sequence[str],
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    method_root: Path,
    figures_root: Path,
    timing_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    X_train, y_train = bundle.features[train_indices], labels[train_indices]
    X_test, y_test = bundle.features[test_indices], labels[test_indices]
    started = time.perf_counter()
    model = fit_train_only(
        X_train, y_train, C=float(selected["C"]), gamma=str(selected["gamma"])
    )
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(X_test)
    metrics = classification_metrics(y_test, predicted, class_names)

    # Required deterministic repeat of only the already-selected model. This
    # cannot influence selection and is not an alternative test configuration.
    repeated_model = fit_train_only(
        X_train, y_train, C=float(selected["C"]), gamma=str(selected["gamma"])
    )
    repeated_prediction = repeated_model.predict(X_test)
    repeated_metrics = classification_metrics(y_test, repeated_prediction, class_names)
    if not np.array_equal(predicted, repeated_prediction) or metrics != repeated_metrics:
        raise TraditionalExperimentError(f"{method} selected model is not deterministically reproducible")

    inference = prediction_timing(
        model,
        X_test,
        warmup_runs=int(timing_config["warmup_runs"]),
        repeats=int(timing_config["inference_repeats"]),
    )
    test_paths = [image_paths[index] for index in test_indices]
    feature_timing = median_batch_latency(
        lambda: [bundle.extractor(path) for path in test_paths],
        sample_count=len(test_paths),
        warmup_runs=int(timing_config["warmup_runs"]),
        repeats=int(timing_config["feature_repeats"]),
    )
    method_root.mkdir(parents=True, exist_ok=True)
    model_path = method_root / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    model_sha256 = _sha256(model_path)
    model_size = model_path.stat().st_size
    predictions = []
    for index, predicted_id in zip(test_indices, predicted, strict=True):
        true_id = int(labels[index])
        predictions.append(
            {
                "instance_id": instance_ids[index],
                "source_image_id": source_ids[index],
                "true_class_id": true_id,
                "true_label": class_names[true_id],
                "predicted_class_id": int(predicted_id),
                "predicted_label": class_names[int(predicted_id)],
                "correct": bool(true_id == int(predicted_id)),
            }
        )
    write_csv(
        method_root / "test_predictions.csv",
        predictions,
        (
            "instance_id",
            "source_image_id",
            "true_class_id",
            "true_label",
            "predicted_class_id",
            "predicted_label",
            "correct",
        ),
    )
    atomic_write_text(method_root / "test_metrics.json", json_text(metrics))
    write_matrix_csv(
        method_root / "confusion_matrix_counts.csv",
        metrics["confusion_matrix_counts"],
        class_names,
    )
    write_matrix_csv(
        method_root / "confusion_matrix_normalized.csv",
        metrics["confusion_matrix_row_normalized"],
        class_names,
    )
    plot_confusion(
        metrics["confusion_matrix_counts"],
        class_names,
        figures_root / f"confusion_{method}_counts.png",
        normalized=False,
    )
    plot_confusion(
        metrics["confusion_matrix_row_normalized"],
        class_names,
        figures_root / f"confusion_{method}_normalized.png",
        normalized=True,
    )
    efficiency = {
        "method": method,
        "feature_dimensions": int(bundle.features.shape[1]),
        "feature_config_hash": bundle.config_hash,
        "feature_fingerprint": bundle.fingerprint,
        "initial_full_dataset_feature_extraction_seconds": bundle.cache_metadata[
            "initial_extraction_seconds"
        ],
        "test_feature_extraction_median_seconds_per_image": feature_timing[
            "median_seconds_per_image"
        ],
        "svm_fit_seconds": fit_seconds,
        "prediction_median_seconds_per_image": inference["median_seconds_per_image"],
        "combined_median_seconds_per_image": feature_timing["median_seconds_per_image"]
        + inference["median_seconds_per_image"],
        "model_size_bytes": model_size,
        "model_sha256": model_sha256,
        "feature_timing": feature_timing,
        "prediction_timing": inference,
        "deterministic_repeat_predictions_identical": True,
        "deterministic_repeat_metrics_identical": True,
    }
    atomic_write_text(method_root / "efficiency.json", json_text(efficiency))
    atomic_write_text(
        method_root / "model_metadata.json",
        json_text(
            {
                "method": method,
                "model_file": model_path.name,
                "model_sha256": model_sha256,
                "model_size_bytes": model_size,
                "feature_config_hash": bundle.config_hash,
                "feature_fingerprint": bundle.fingerprint,
                "processed_dataset_fingerprint": bundle.cache_metadata[
                    "processed_dataset_fingerprint"
                ],
                "selected_C": float(selected["C"]),
                "selected_gamma": selected["gamma"],
                "training_instances": len(train_indices),
                "test_instances": len(test_indices),
            }
        ),
    )
    return metrics, efficiency


def _publish_versioned_summary(output_root: Path, summary_root: Path) -> None:
    if summary_root.exists():
        shutil.rmtree(summary_root)
    for source in output_root.rglob("*"):
        if not source.is_file() or source.suffix == ".joblib" or source.name == "run.log":
            continue
        relative = source.relative_to(output_root)
        destination = summary_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_traditional_baselines(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Execute the canonical validation-freeze-test protocol exactly once."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    _validate_phase4_config(data)
    phase3_summary = validate_processed_dataset(config, root)
    validate_training_subsets(config, root)
    if phase3_summary["processed_dataset_fingerprint"] != data["dataset"]["processed_fingerprint"]:
        raise TraditionalExperimentError("Phase 3 processed fingerprint changed before Phase 4")

    traditional = data["traditional"]
    output_root = (root / traditional["output_root"]).resolve()
    summary_root = (root / traditional["summary_root"]).resolve()
    figures_root = (root / traditional["figures_root"]).resolve()
    _safe_clean(output_root, (root / "experiments" / "results").resolve())
    _safe_clean(figures_root, (root / "figures").resolve())
    output_root.mkdir(parents=True, exist_ok=True)

    processed_root = root / "data" / "processed" / data["dataset"]["processed_version"]
    rows = read_csv(processed_root / "manifest.csv")
    if len(rows) != 1065:
        raise TraditionalExperimentError("Phase 4 manifest does not contain exactly 1,065 samples")
    instance_ids = [row["instance_id"] for row in rows]
    source_ids = [row["source_image_id"] for row in rows]
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    splits = [row["split"] for row in rows]
    image_paths = [processed_root / row["output_relative_path"] for row in rows]
    for row, image_path in zip(rows, image_paths, strict=True):
        if not image_path.is_file() or _sha256(image_path) != row["processed_image_sha256"]:
            raise TraditionalExperimentError(f"Phase 3 image/checksum mismatch: {row['instance_id']}")
    train_indices = np.asarray([index for index, split in enumerate(splits) if split == "train"])
    validation_indices = np.asarray(
        [index for index, split in enumerate(splits) if split == "validation"]
    )
    if (len(train_indices), len(validation_indices)) != (757, 146):
        raise TraditionalExperimentError("Phase 3 train/validation instance counts changed")

    versions = {
        "numpy": _version("numpy"),
        "Pillow": _version("Pillow"),
        "scikit-image": _version("scikit-image"),
        "scikit-learn": _version("scikit-learn"),
        "joblib": _version("joblib"),
    }
    processed_fingerprint = data["dataset"]["processed_fingerprint"]
    cache_root = (root / traditional["cache_root"]).resolve()
    bundles: dict[str, FeatureBundle] = {}
    for family, feature_config, extractor_function, hash_function in (
        ("hog", data["hog"], extract_hog, hog_config_hash),
        ("lbp", data["lbp"], extract_spatial_lbp, lbp_config_hash),
    ):
        config_hash = hash_function(feature_config, versions)
        key = cache_key(processed_fingerprint, family, config_hash, versions)
        extractor = lambda path, function=extractor_function, cfg=feature_config: function(path, cfg)
        features, cache_metadata = load_or_extract_features(
            cache_root=cache_root,
            family=family,
            key=key,
            instance_ids=instance_ids,
            image_paths=image_paths,
            labels=labels,
            source_ids=source_ids,
            extractor=extractor,
            expected_dimensions=int(feature_config["expected_dimensions"]),
            metadata={
                "processed_dataset_fingerprint": processed_fingerprint,
                "feature_config_hash": config_hash,
                "feature_config": feature_config,
                "library_versions": versions,
            },
        )
        if features.shape != (1065, int(feature_config["expected_dimensions"])):
            raise TraditionalExperimentError(f"{family} feature dimensions violate the frozen gate")
        bundles[family] = FeatureBundle(
            family=family,
            features=features,
            config_hash=config_hash,
            fingerprint=cache_metadata["feature_fingerprint"],
            cache_metadata=cache_metadata,
            extractor=extractor,
        )

    # Validation-only selection. This function has no test inputs.
    all_grid_rows: list[dict[str, Any]] = []
    validation_predictions: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    tolerance = float(data["selection"]["numeric_tolerance"])
    for family in ("hog", "lbp"):
        grid_rows, candidate_predictions = _validation_grid(
            family,
            bundles[family].features,
            train_indices,
            validation_indices,
            labels,
            instance_ids,
            source_ids,
            data["svm"],
            LABELS,
        )
        winning = select_configuration(grid_rows, tolerance=tolerance)
        for row in grid_rows:
            row["selected"] = row["candidate_id"] == winning["candidate_id"]
        selected[family] = next(row for row in grid_rows if row["selected"])
        all_grid_rows.extend(grid_rows)
        validation_predictions.extend(candidate_predictions)
    grid_fingerprint = _grid_fingerprint(all_grid_rows)
    write_csv(output_root / "validation_grid.csv", all_grid_rows, GRID_FIELDS)
    write_csv(
        output_root / "validation_predictions.csv",
        validation_predictions,
        (
            "method",
            "candidate_id",
            "instance_id",
            "source_image_id",
            "true_class_id",
            "true_label",
            "predicted_class_id",
            "predicted_label",
            "correct",
        ),
    )

    provenance = capture_git_provenance(root)
    frozen: dict[str, dict[str, Any]] = {}
    for family in ("hog", "lbp"):
        frozen_path = (root / traditional[f"frozen_{family}_config"]).resolve()
        frozen[family] = _freeze_selection(
            frozen_path,
            method=f"{family}+rbf_svm",
            selected=selected[family],
            feature_config=data[family],
            feature_hash=bundles[family].config_hash,
            processed_fingerprint=processed_fingerprint,
            grid_fingerprint=grid_fingerprint,
            tolerance=tolerance,
            git_commit=provenance["git_commit"],
        )
    selected_record = {
        "validation_grid_fingerprint": grid_fingerprint,
        "test_metrics_available_at_freeze_time": False,
        "methods": frozen,
    }
    atomic_write_text(output_root / "selected_hyperparameters.json", json_text(selected_record))
    if not all((root / traditional[f"frozen_{family}_config"]).is_file() for family in ("hog", "lbp")):
        raise TraditionalExperimentError("selected configurations were not frozen before test gate")

    # Test data are materialized only after the validation grid and immutable
    # selected-config files exist. Only the two selected configurations enter.
    test_indices = np.asarray([index for index, split in enumerate(splits) if split == "test"])
    if len(test_indices) != 162:
        raise TraditionalExperimentError("Phase 3 test instance count changed")
    metrics_by_method: dict[str, dict[str, Any]] = {}
    efficiency_by_method: dict[str, dict[str, Any]] = {}
    for family in ("hog", "lbp"):
        metrics, efficiency = _evaluate_selected(
            method=family,
            bundle=bundles[family],
            selected=selected[family],
            train_indices=train_indices,
            test_indices=test_indices,
            labels=labels,
            instance_ids=instance_ids,
            source_ids=source_ids,
            image_paths=image_paths,
            class_names=LABELS,
            method_root=output_root / family,
            figures_root=figures_root,
            timing_config=data["timing"],
        )
        metrics_by_method[family] = metrics
        efficiency_by_method[family] = efficiency

    summary_rows = [
        {
            "method": family,
            "selected_C": selected[family]["C"],
            "selected_gamma": selected[family]["gamma"],
            "validation_macro_f1": selected[family]["validation_macro_f1"],
            "test_accuracy": metrics_by_method[family]["accuracy"],
            "test_balanced_accuracy": metrics_by_method[family]["balanced_accuracy"],
            "test_macro_precision": metrics_by_method[family]["macro_precision"],
            "test_macro_recall": metrics_by_method[family]["macro_recall"],
            "test_macro_f1": metrics_by_method[family]["macro_f1"],
        }
        for family in ("hog", "lbp")
    ]
    write_csv(
        output_root / "summary_metrics.csv",
        summary_rows,
        (
            "method",
            "selected_C",
            "selected_gamma",
            "validation_macro_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_precision",
            "test_macro_recall",
            "test_macro_f1",
        ),
    )
    efficiency_rows = [
        {
            key: efficiency_by_method[family][key]
            for key in (
                "method",
                "feature_dimensions",
                "feature_config_hash",
                "feature_fingerprint",
                "initial_full_dataset_feature_extraction_seconds",
                "test_feature_extraction_median_seconds_per_image",
                "svm_fit_seconds",
                "prediction_median_seconds_per_image",
                "combined_median_seconds_per_image",
                "model_size_bytes",
                "model_sha256",
            )
        }
        for family in ("hog", "lbp")
    ]
    write_csv(output_root / "efficiency.csv", efficiency_rows, tuple(efficiency_rows[0]))
    for family in ("hog", "lbp"):
        plot_validation_grid(
            all_grid_rows, family, figures_root / f"validation_grid_{family}.png"
        )
    plot_baseline_comparison(
        {
            family: {
                "macro_f1": metrics_by_method[family]["macro_f1"],
                "balanced_accuracy": metrics_by_method[family]["balanced_accuracy"],
                "accuracy": metrics_by_method[family]["accuracy"],
            }
            for family in ("hog", "lbp")
        },
        figures_root / "traditional_baseline_comparison.png",
    )

    manifest = {
        "schema_version": "1.0",
        "phase": 4,
        "status": "completed",
        "result_id": traditional["result_id"],
        "created_utc": format_utc(utc_now()),
        "processed_dataset_version": data["dataset"]["processed_version"],
        "processed_dataset_fingerprint": processed_fingerprint,
        "phase3_gate_passed_before_training": True,
        "sample_counts": {"all": 1065, "train": 757, "validation": 146, "test": 162},
        "class_order": list(LABELS),
        "library_versions": versions,
        "configuration_hash": calculate_config_hash(data, length=64),
        "validation_grid_fingerprint": grid_fingerprint,
        "feature_config_hashes": {family: bundles[family].config_hash for family in bundles},
        "feature_fingerprints": {family: bundles[family].fingerprint for family in bundles},
        "selected_candidates": {
            family: {"C": selected[family]["C"], "gamma": selected[family]["gamma"]}
            for family in selected
        },
        "test_evaluated_candidates": 2,
        "rejected_test_evaluations": 0,
        "deterministic_repeat_passed": True,
        "data_efficiency_fractions_run": [1.0],
        "cnn_trained": False,
        "pretrained_weights_downloaded": False,
        "robustness_started": False,
        "phase5_started": False,
        "git": provenance,
        "environment": capture_environment("cpu", root),
    }
    atomic_write_text(output_root / "manifest.json", json_text(manifest))
    atomic_write_text(output_root / "resolved_config.yaml", config.to_yaml())
    atomic_write_text(output_root / "run.log", "Phase 4 canonical run completed successfully.\n")
    _publish_versioned_summary(output_root, summary_root)
    return {
        "status": "PASS",
        "result_id": traditional["result_id"],
        "processed_dataset_fingerprint": processed_fingerprint,
        "validation_grid_fingerprint": grid_fingerprint,
        "selected": selected,
        "metrics": metrics_by_method,
        "efficiency": efficiency_by_method,
        "feature_config_hashes": {family: bundles[family].config_hash for family in bundles},
        "feature_fingerprints": {family: bundles[family].fingerprint for family in bundles},
        "deterministic_repeat_passed": True,
        "output_root": output_root,
        "summary_root": summary_root,
    }
