"""Frozen Phase 7 limited-labeled-data experiment and result validator."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from windblade.config import ResolvedConfig, load_config
from windblade.data.processed import LABELS, json_text, read_csv, validate_processed_dataset
from windblade.data.subsets import subset_fingerprint, validate_training_subsets
from windblade.deep.checkpoints import save_checkpoint, state_dict_fingerprint
from windblade.deep.dataset import (
    WTBDCropDataset,
    balanced_subset_class_weights,
    make_loader,
    split_rows,
)
from windblade.deep.determinism import resolve_device
from windblade.deep.mobilenet import load_official_model, model_from_official_state as mobilenet_from_state
from windblade.deep.resnet import load_official_backbone, model_from_official_state as resnet_from_state
from windblade.deep.training import aggregate_seed_metrics, run_epoch, train_with_validation
from windblade.environment import capture_environment, capture_git_provenance
from windblade.evaluation.metrics import classification_metrics
from windblade.evaluation.reporting import write_csv, write_matrix_csv
from windblade.mobilenet_experiment import validate_mobilenet_results
from windblade.models.svm import fit_train_only
from windblade.resnet_experiment import (
    HISTORY_FIELDS,
    PREDICTION_FIELDS,
    _records as cnn_records,
    validate_resnet18_results,
)
from windblade.traditional import validate_traditional_results
from windblade.utils import atomic_write_text, format_utc, utc_now


EXPECTED_FINGERPRINT = "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991"
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
REDUCED_FRACTIONS = FRACTIONS[:3]
SEEDS = (17, 29, 43)
SOURCE_COUNTS = {0.25: 128, 0.50: 255, 0.75: 383, 1.00: 510}
INSTANCE_COUNTS = {0.25: 252, 0.50: 440, 0.75: 608, 1.00: 757}
CLASS_COUNTS = {
    0.25: (39, 40, 66, 12, 28, 67),
    0.50: (70, 72, 111, 22, 51, 114),
    0.75: (98, 100, 151, 32, 73, 154),
    1.00: (123, 126, 185, 42, 93, 188),
}
METHODS = ("hog", "lbp", "resnet18", "mobilenet_v3_small")
CLASSICAL = ("hog", "lbp")
CNNS = ("resnet18", "mobilenet_v3_small")
TEST_PREDICTION_FIELDS = PREDICTION_FIELDS[:9]


class DataEfficiencyError(RuntimeError):
    """Raised when the frozen Phase 7 contract is violated."""


def _fraction_key(fraction: float) -> str:
    return f"{fraction:.2f}"


def _fraction_dir(fraction: float) -> str:
    return f"frac_{round(fraction * 100):03d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataEfficiencyError(f"expected YAML mapping: {path}")
    return value


def frozen_config_identities(config: ResolvedConfig, root: Path) -> dict[str, dict[str, Any]]:
    paths = config.as_dict()["data_efficiency"]["frozen_configs"]
    identities: dict[str, dict[str, Any]] = {}
    for method, relative in paths.items():
        path = (root / relative).resolve()
        identities[method] = {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
    return identities


def _validate_phase7_config(data: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    phase = data.get("data_efficiency", {})
    failures: list[str] = []
    expected_scalars = {
        "phase": data["project"]["phase"] == 7,
        "fingerprint": data["dataset"]["processed_fingerprint"] == EXPECTED_FINGERPRINT,
        "classes": tuple(data["classes"]["order"]) == LABELS,
        "fractions": tuple(float(value) for value in phase["fractions"]) == FRACTIONS,
        "reduced_fractions": tuple(float(value) for value in phase["reduced_fractions"]) == REDUCED_FRACTIONS,
        "seeds": tuple(int(value) for value in phase["seeds"]) == SEEDS,
        "source_unit": phase["subset_unit"] == "source_image",
        "nested": phase["nested"] is True,
        "fixed_partitions": phase["validation_fixed"] is True and phase["test_fixed"] is True,
        "no_search": phase["hyperparameter_search"] is False,
        "metric": phase["primary_metric"] == "macro_f1",
    }
    failures.extend(name for name, passed in expected_scalars.items() if not passed)
    for fraction in FRACTIONS:
        key = _fraction_key(fraction)
        if int(phase["source_counts"][key]) != SOURCE_COUNTS[fraction]:
            failures.append(f"source_count_{key}")
        if int(phase["instance_counts"][key]) != INSTANCE_COUNTS[fraction]:
            failures.append(f"instance_count_{key}")
        if tuple(int(v) for v in phase["class_counts"][key]) != CLASS_COUNTS[fraction]:
            failures.append(f"class_counts_{key}")
    expected_paths = {
        "hog": "configs/frozen/traditional_hog_svm.yaml",
        "lbp": "configs/frozen/traditional_lbp_svm.yaml",
        "resnet18": "configs/frozen/resnet18.yaml",
        "mobilenet_v3_small": "configs/frozen/mobilenet_v3_small.yaml",
    }
    if dict(phase["frozen_configs"]) != expected_paths:
        failures.append("frozen_config_paths")
    frozen = {name: _yaml(root / path) for name, path in expected_paths.items()}
    checks = {
        "hog": frozen["hog"]["method"] == "hog+rbf_svm"
        and frozen["hog"]["svm"] == {"C": 10.0, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf", "probability": False}
        and frozen["hog"]["feature_config_hash"] == "e0723cd80ec462644aec14e3827821d716d04ff375424b5de45ac5ddac4d5cf2",
        "lbp": frozen["lbp"]["method"] == "lbp+rbf_svm"
        and frozen["lbp"]["svm"] == {"C": 10.0, "class_weight": "balanced", "gamma": "scale", "kernel": "rbf", "probability": False}
        and frozen["lbp"]["feature_config_hash"] == "e952103e7c0664952a0b8c568141bd26d177333c572f7ec134d1270c2592d122",
    }
    for method in CNNS:
        training = frozen[method]["training"]
        expected_lr = 3e-4 if method == "resnet18" else 1e-4
        checks[method] = (
            float(training["learning_rate"]) == expected_lr
            and float(training["weight_decay"]) == 0.0
            and training["optimizer"] == "AdamW"
            and int(training["batch_size"]) == 32
            and int(training["validation_batch_size"]) == 64
            and int(training["max_epochs"]) == 30
            and int(training["patience"]) == 6
            and float(training["min_delta"]) == 1e-4
            and frozen[method]["input"]["augmentation"] == "none"
            and frozen[method]["model"]["fine_tune"] == "all"
        )
    failures.extend(f"frozen_{name}" for name, passed in checks.items() if not passed)
    if failures:
        raise DataEfficiencyError("Phase 7 frozen configuration gate failed: " + ", ".join(failures))
    return frozen


def load_and_validate_subsets(config: ResolvedConfig, root: Path) -> dict[tuple[int, float], dict[str, Any]]:
    summary = validate_training_subsets(config, root)
    processed_root = root / "data" / "processed" / "wtbd_crops_v1"
    manifest = read_csv(processed_root / "manifest.csv")
    by_id = {row["instance_id"]: row for row in manifest}
    train_sources = {row["source_image_id"] for row in manifest if row["split"] == "train"}
    validation_test_sources = {row["source_image_id"] for row in manifest if row["split"] != "train"}
    result: dict[tuple[int, float], dict[str, Any]] = {}
    for seed in SEEDS:
        previous: set[str] = set()
        for fraction in FRACTIONS:
            details = summary["seeds"][str(seed)][_fraction_key(fraction)]
            path = root / details["manifest_relative_path"]
            rows = read_csv(path)
            ids = [row["instance_id"] for row in rows]
            if len(ids) != len(set(ids)) or any(instance_id not in by_id for instance_id in ids):
                raise DataEfficiencyError(f"invalid subset identities: seed {seed}, fraction {fraction}")
            full_rows = [by_id[instance_id] for instance_id in ids]
            sources = {row["source_image_id"] for row in full_rows}
            counts = tuple(Counter(row["canonical_label"] for row in full_rows)[label] for label in LABELS)
            if (
                len(sources) != SOURCE_COUNTS[fraction]
                or len(full_rows) != INSTANCE_COUNTS[fraction]
                or counts != CLASS_COUNTS[fraction]
                or not sources <= train_sources
                or sources & validation_test_sources
                or any(row["split"] != "train" for row in full_rows)
                or (previous and not previous < sources)
            ):
                raise DataEfficiencyError(f"frozen subset gate failed: seed {seed}, fraction {fraction}")
            previous = sources
            content = path.read_text(encoding="utf-8")
            if subset_fingerprint(content) != details["sha256"]:
                raise DataEfficiencyError(f"subset fingerprint changed: {path}")
            result[(seed, fraction)] = {
                "rows": full_rows,
                "source_ids": sources,
                "fingerprint": details["sha256"],
                "manifest_relative_path": details["manifest_relative_path"],
                "class_counts": dict(zip(LABELS, counts, strict=True)),
            }
    return result


def verify_phase7_start_gate(config: ResolvedConfig, root: Path) -> dict[str, Any]:
    frozen = _validate_phase7_config(config.as_dict(), root)
    phase3 = validate_processed_dataset(config, root)
    subsets = load_and_validate_subsets(config, root)
    phase4 = validate_traditional_results(load_config(root / "configs/traditional_baselines.yaml"), root)
    phase5 = validate_resnet18_results(load_config(root / "configs/resnet18_baseline.yaml"), root)
    phase6 = validate_mobilenet_results(load_config(root / "configs/mobilenet_v3_small_baseline.yaml"), root)
    if phase3["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT:
        raise DataEfficiencyError("processed dataset fingerprint changed")
    if phase4["status"] != "PASS" or phase5["status"] != "PASS" or phase6["status"] != "PASS":
        raise DataEfficiencyError("an upstream result validator did not pass")
    return {
        "phase3": phase3,
        "phase4": phase4,
        "phase5": phase5,
        "phase6": phase6,
        "frozen": frozen,
        "subsets": subsets,
    }


def _run_root(base: Path, method: str, seed: int, fraction: float) -> Path:
    return base / method / f"seed_{seed}" / _fraction_dir(fraction)


def _scientific_history(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in rows]


def _head_fingerprint(model: torch.nn.Module, method: str) -> str:
    head = model.fc if method == "resnet18" else model.classifier[-1]
    return state_dict_fingerprint(head.state_dict())


def _prediction_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return cnn_records(rows)


def _classical_records(
    indices: Sequence[int], predicted: Sequence[int], rows: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for index, predicted_id in zip(indices, predicted, strict=True):
        row = rows[index]
        true_id = int(row["class_id"])
        output.append(
            {
                "method": "",
                "fraction": "",
                "seed": "",
                "instance_id": row["instance_id"],
                "source_image_id": row["source_image_id"],
                "true_class_id": true_id,
                "true_label": LABELS[true_id],
                "predicted_class_id": int(predicted_id),
                "predicted_label": LABELS[int(predicted_id)],
                "correct": true_id == int(predicted_id),
            }
        )
    return output


def _validate_complete_run(path: Path, method: str, seed: int, fraction: float) -> dict[str, Any] | None:
    marker = path / "complete.json"
    required = [path / "test_metrics.json", path / "test_predictions.csv", path / "validation_metrics.json"]
    if method in CNNS:
        required.extend([path / "validation_predictions.csv", path / "history.csv", path / "best_state_dict.json", path / "best_state_dict.pt"])
    if not marker.is_file() or not all(item.is_file() for item in required):
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        predictions = read_csv(path / "test_predictions.csv")
        if (
            record["method"] != method
            or int(record["seed"]) != seed
            or float(record["fraction"]) != fraction
            or record["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT
            or len(predictions) != 162
            or len({row["instance_id"] for row in predictions}) != 162
        ):
            return None
        if method in CNNS:
            metadata = json.loads((path / "best_state_dict.json").read_text(encoding="utf-8"))
            state = torch.load(path / "best_state_dict.pt", map_location="cpu", weights_only=True)
            if state_dict_fingerprint(state) != metadata["checkpoint_fingerprint"]:
                return None
        return record
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def _load_feature_cache(
    root: Path, method: str, rows: Sequence[Mapping[str, str]], frozen: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_root = root / "experiments" / "cache" / "traditional_features"
    expected_ids = [row["instance_id"] for row in rows]
    for metadata_path in sorted(cache_root.glob(f"{method}_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("processed_dataset_fingerprint") != EXPECTED_FINGERPRINT
            or metadata.get("feature_config_hash") != frozen["feature_config_hash"]
        ):
            continue
        array_path = metadata_path.with_suffix(".npz")
        if not array_path.is_file():
            continue
        with np.load(array_path, allow_pickle=False) as stored:
            ids = [str(value) for value in stored["instance_ids"].tolist()]
            features = np.asarray(stored["features"], dtype=np.float64)
        if ids != expected_ids:
            continue
        from windblade.features.cache import feature_matrix_fingerprint

        if feature_matrix_fingerprint(ids, features) != metadata["feature_fingerprint"]:
            continue
        if features.shape != (1065, int(frozen["feature_config"]["expected_dimensions"])):
            continue
        return features, metadata
    raise DataEfficiencyError(f"no fingerprint-valid frozen {method.upper()} feature cache exists")


def _write_classical_run(
    run_root: Path,
    *,
    method: str,
    seed: int,
    fraction: float,
    subset: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, str]],
    features: np.ndarray,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    prior = _validate_complete_run(run_root, method, seed, fraction)
    if prior is not None:
        print(f"{method.upper()} seed{seed} {fraction:.0%} validated; resume skip", flush=True)
        return prior
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)
    by_id = {row["instance_id"]: index for index, row in enumerate(all_rows)}
    train_indices = np.asarray([by_id[row["instance_id"]] for row in subset["rows"]], dtype=np.int64)
    validation_indices = np.asarray([index for index, row in enumerate(all_rows) if row["split"] == "validation"])
    test_indices = np.asarray([index for index, row in enumerate(all_rows) if row["split"] == "test"])
    labels = np.asarray([int(row["class_id"]) for row in all_rows], dtype=np.int64)
    started = time.perf_counter()
    model = fit_train_only(
        features[train_indices],
        labels[train_indices],
        C=float(frozen["svm"]["C"]),
        gamma=str(frozen["svm"]["gamma"]),
    )
    fit_seconds = time.perf_counter() - started
    validation_predicted = model.predict(features[validation_indices])
    validation_metrics = classification_metrics(labels[validation_indices], validation_predicted, LABELS)
    test_predicted = model.predict(features[test_indices])
    test_metrics = classification_metrics(labels[test_indices], test_predicted, LABELS)
    validation_records = _classical_records(validation_indices, validation_predicted, all_rows)
    test_records = _classical_records(test_indices, test_predicted, all_rows)
    for record in validation_records + test_records:
        record.update({"method": method, "fraction": fraction, "seed": seed})
    atomic_write_text(run_root / "validation_metrics.json", json_text(validation_metrics))
    atomic_write_text(run_root / "test_metrics.json", json_text(test_metrics))
    fields = (
        "method", "fraction", "seed", "instance_id", "source_image_id", "true_class_id",
        "true_label", "predicted_class_id", "predicted_label", "correct",
    )
    write_csv(run_root / "validation_predictions.csv", validation_records, fields)
    write_csv(run_root / "test_predictions.csv", test_records, fields)
    write_matrix_csv(run_root / "confusion_matrix_counts.csv", test_metrics["confusion_matrix_counts"], LABELS)
    metadata = {
        "schema_version": "1.0",
        "method": method,
        "fraction": fraction,
        "seed": seed,
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "source_subset_fingerprint": subset["fingerprint"],
        "source_images": SOURCE_COUNTS[fraction],
        "training_instances": INSTANCE_COUNTS[fraction],
        "training_class_counts": subset["class_counts"],
        "feature_config_hash": frozen["feature_config_hash"],
        "C": float(frozen["svm"]["C"]),
        "gamma": str(frozen["svm"]["gamma"]),
        "scaler_fit_scope": "active_training_subset_only",
        "class_weight": "balanced_from_active_subset_by_SVC",
        "validation_instances": 146,
        "test_instances": 162,
        "test_evaluations": 1,
        "fit_seconds": fit_seconds,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    atomic_write_text(run_root / "model_metadata.json", json_text(metadata))
    atomic_write_text(run_root / "complete.json", json_text(metadata))
    print(
        f"{method.upper()} seed{seed} {fraction:.0%} complete | "
        f"validation macro-F1={validation_metrics['macro_f1']:.6f} | "
        f"test macro-F1={test_metrics['macro_f1']:.6f} | fit={fit_seconds:.3f}s",
        flush=True,
    )
    return metadata


def _write_cnn_run(
    run_root: Path,
    *,
    method: str,
    seed: int,
    fraction: float,
    subset: Mapping[str, Any],
    processed_root: Path,
    validation_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    official_state: dict[str, torch.Tensor],
    official_fingerprint: str,
    frozen: Mapping[str, Any],
    git_commit: str | None,
    device: torch.device,
) -> dict[str, Any]:
    prior = _validate_complete_run(run_root, method, seed, fraction)
    if prior is not None:
        print(f"{method} seed{seed} {fraction:.0%} validated; resume skip", flush=True)
        return prior
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)
    factory = resnet_from_state if method == "resnet18" else mobilenet_from_state
    model = factory(official_state, seed=seed)
    initial_head_fingerprint = _head_fingerprint(model, method)
    class_weights = balanced_subset_class_weights(subset["rows"])
    training = frozen["training"]
    train_dataset = WTBDCropDataset(subset["rows"], processed_root)
    validation_dataset = WTBDCropDataset(validation_rows, processed_root)
    test_dataset = WTBDCropDataset(test_rows, processed_root)
    result = train_with_validation(
        model,
        make_loader(train_dataset, batch_size=32, shuffle=True, seed=seed),
        make_loader(validation_dataset, batch_size=64, shuffle=False, seed=seed),
        device=device,
        class_weights=class_weights,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_epochs=30,
        patience=6,
        min_delta=1e-4,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    _, test_metrics, test_records = run_epoch(
        result["model"],
        make_loader(test_dataset, batch_size=64, shuffle=False, seed=seed),
        criterion,
        device,
    )
    write_csv(run_root / "history.csv", result["history"], HISTORY_FIELDS)
    atomic_write_text(run_root / "validation_metrics.json", json_text(result["best_validation_metrics"]))
    atomic_write_text(run_root / "test_metrics.json", json_text(test_metrics))
    write_csv(run_root / "validation_predictions.csv", _prediction_records(result["best_validation_records"]), PREDICTION_FIELDS)
    write_csv(run_root / "test_predictions.csv", _prediction_records(test_records), PREDICTION_FIELDS)
    write_matrix_csv(run_root / "confusion_matrix_counts.csv", test_metrics["confusion_matrix_counts"], LABELS)
    architecture = "torchvision_resnet18" if method == "resnet18" else "torchvision_mobilenet_v3_small"
    metadata = {
        "schema_version": "1.0",
        "phase": 7,
        "architecture": architecture,
        "method": method,
        "fraction": fraction,
        "subset_seed": seed,
        "model_seed": seed,
        "seed": seed,
        "source_subset_fingerprint": subset["fingerprint"],
        "source_subset_manifest": subset["manifest_relative_path"],
        "source_images": SOURCE_COUNTS[fraction],
        "training_instances": INSTANCE_COUNTS[fraction],
        "training_class_counts": subset["class_counts"],
        "class_weights": {label: float(value) for label, value in zip(LABELS, class_weights.tolist(), strict=True)},
        "class_weight_formula": "N_subset/(6*N_subset,c), active training labels only",
        "frozen_hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "batch_size": 32,
            "validation_batch_size": 64,
            "max_epochs": 30,
            "patience": 6,
            "min_delta": 1e-4,
            "augmentation": "none",
            "fine_tuning": "all_parameters",
            "mixed_precision": False,
            "num_workers": 0,
        },
        "official_pretrained_weight_fingerprint": official_fingerprint,
        "fresh_official_pretrained_start": True,
        "warm_started_from_fraction": None,
        "initial_head_fingerprint": initial_head_fingerprint,
        "best_epoch": result["best_epoch"],
        "epochs_executed": result["epochs_executed"],
        "validation_metrics": result["best_validation_metrics"],
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "git_commit": git_commit,
        "training_seconds": result["training_seconds"],
        "validation_instances": 146,
        "test_instances": 162,
        "test_evaluations": 1,
    }
    checkpoint = save_checkpoint(run_root / "best_state_dict.pt", result["best_state_dict"], metadata)
    complete = {
        **metadata,
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "checkpoint_bytes": checkpoint["checkpoint_bytes"],
        "validation_macro_f1": result["best_validation_metrics"]["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    atomic_write_text(run_root / "complete.json", json_text(complete))
    print(
        f"{method} seed{seed} {fraction:.0%} complete | best epoch={result['best_epoch']} | "
        f"validation macro-F1={result['best_validation_metrics']['macro_f1']:.6f} | "
        f"test macro-F1={test_metrics['macro_f1']:.6f} | training={result['training_seconds']:.1f}s",
        flush=True,
    )
    return complete


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_metrics(root: Path, method: str) -> list[dict[str, Any]]:
    if method in CLASSICAL:
        return [_json(root / "experiments" / "summaries" / "phase4_traditional_v1" / method / "test_metrics.json")]
    phase = "phase5_resnet18_v1" if method == "resnet18" else "phase6_mobilenet_v3_small_v1"
    return [
        _json(root / "experiments" / "summaries" / phase / "final" / f"seed_{seed}" / "test_metrics.json")
        for seed in SEEDS
    ]


def calculate_learning_curve_outputs(
    metrics_by_method_fraction: Mapping[tuple[str, float], Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Calculate every predeclared Phase 7 aggregate from run-level metrics."""

    learning: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    absolute: list[dict[str, Any]] = []
    marginal: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    thunderstrike: list[dict[str, Any]] = []
    for method in METHODS:
        means: dict[float, float] = {}
        summaries: dict[float, dict[str, Any]] = {}
        for fraction in FRACTIONS:
            metrics = list(metrics_by_method_fraction[(method, fraction)])
            if fraction == 1.0 and method in CLASSICAL:
                if len(metrics) != 1:
                    raise DataEfficiencyError("traditional 100% endpoint must have exactly one canonical result")
                summary = {
                    "overall": {
                        key: {"mean": float(metrics[0][key]), "sample_sd": None}
                        for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall")
                    },
                    "per_class": {
                        label: {
                            key: {"mean": float(metrics[0]["per_class"][label][key]), "sample_sd": None}
                            for key in ("precision", "recall", "f1")
                        }
                        for label in LABELS
                    },
                }
            else:
                summary = aggregate_seed_metrics(metrics)
            summaries[fraction] = summary
            means[fraction] = summary["overall"]["macro_f1"]["mean"]
            learning.append(
                {
                    "method": method,
                    "training_fraction": fraction,
                    "source_images": SOURCE_COUNTS[fraction],
                    "training_instances": INSTANCE_COUNTS[fraction],
                    "macro_f1_mean": means[fraction],
                    "macro_f1_sample_sd": summary["overall"]["macro_f1"]["sample_sd"],
                    "balanced_accuracy_mean": summary["overall"]["balanced_accuracy"]["mean"],
                    "balanced_accuracy_sample_sd": summary["overall"]["balanced_accuracy"]["sample_sd"],
                    "accuracy_mean": summary["overall"]["accuracy"]["mean"],
                    "accuracy_sample_sd": summary["overall"]["accuracy"]["sample_sd"],
                    "replicate_count": len(metrics),
                    "standard_deviation": "N/A (single deterministic canonical result)"
                    if len(metrics) == 1 else "sample (ddof=1)",
                }
            )
            for label in LABELS:
                values = summary["per_class"][label]
                per_class.append(
                    {
                        "method": method,
                        "training_fraction": fraction,
                        "class": label,
                        "training_class_count": CLASS_COUNTS[fraction][LABELS.index(label)],
                        "f1_mean": values["f1"]["mean"],
                        "f1_sample_sd": values["f1"]["sample_sd"],
                        "precision_mean": values["precision"]["mean"],
                        "precision_sample_sd": values["precision"]["sample_sd"],
                        "recall_mean": values["recall"]["mean"],
                        "recall_sample_sd": values["recall"]["sample_sd"],
                    }
                )
                if label == "thunderstrike":
                    thunderstrike.append(dict(per_class[-1]))
        full = means[1.0]
        for fraction in REDUCED_FRACTIONS:
            retention.append(
                {
                    "method": method,
                    "training_fraction": fraction,
                    "full_data_macro_f1": full,
                    "macro_f1_mean": means[fraction],
                    "performance_retention": means[fraction] / full,
                    "performance_retention_percent": 100.0 * means[fraction] / full,
                }
            )
        for fraction in FRACTIONS:
            absolute.append(
                {
                    "method": method,
                    "training_fraction": fraction,
                    "macro_f1_mean": means[fraction],
                    "full_data_macro_f1": full,
                    "delta_to_full": means[fraction] - full,
                }
            )
        for start, end in zip(FRACTIONS[:-1], FRACTIONS[1:], strict=True):
            marginal.append(
                {
                    "method": method,
                    "from_fraction": start,
                    "to_fraction": end,
                    "macro_f1_gain": means[end] - means[start],
                }
            )
        threshold = next(fraction for fraction in FRACTIONS if means[fraction] >= 0.95 * full)
        thresholds.append(
            {
                "method": method,
                "threshold_fraction": threshold,
                "threshold_percent": int(threshold * 100),
                "criterion_macro_f1": 0.95 * full,
                "observed_macro_f1": means[threshold],
            }
        )
        normalized_auc = float(np.trapezoid([means[value] for value in FRACTIONS], FRACTIONS) / 0.75)
        auc_rows.append({"method": method, "normalized_macro_f1_learning_curve_auc": normalized_auc})
    return {
        "learning_curve_summary": learning,
        "per_class_learning_curves": per_class,
        "performance_retention": retention,
        "absolute_loss": absolute,
        "marginal_gains": marginal,
        "threshold_95_percent": thresholds,
        "normalized_learning_curve_auc": auc_rows,
        "thunderstrike_learning_curve": thunderstrike,
    }


def _write_reused_references(root: Path, output: Path) -> dict[str, Any]:
    reference_root = output / "reused_full_data"
    reference_root.mkdir(parents=True, exist_ok=True)
    references = {
        "phase4_hog": root / "experiments/summaries/phase4_traditional_v1/hog/test_metrics.json",
        "phase4_lbp": root / "experiments/summaries/phase4_traditional_v1/lbp/test_metrics.json",
        "phase5_resnet": root / "experiments/summaries/phase5_resnet18_v1/aggregate/test_summary.json",
        "phase6_mobilenet": root / "experiments/summaries/phase6_mobilenet_v3_small_v1/aggregate/test_summary.json",
    }
    result = {}
    for name, path in references.items():
        record = {
            "reused": True,
            "retrained_in_phase7": False,
            "source_path": path.relative_to(root).as_posix(),
            "source_sha256": _sha256(path),
            "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
            "content": _json(path),
        }
        atomic_write_text(reference_root / f"{name}_reference.json", json_text(record))
        result[name] = {key: record[key] for key in record if key != "content"}
    return result


def _aggregate_results(root: Path, output: Path) -> dict[str, Any]:
    metrics: dict[tuple[str, float], list[dict[str, Any]]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for fraction in REDUCED_FRACTIONS:
            items = []
            for seed in SEEDS:
                run_root = _run_root(output, method, seed, fraction)
                item = _json(run_root / "test_metrics.json")
                items.append(item)
                per_seed_rows.append(
                    {
                        "method": method,
                        "training_fraction": fraction,
                        "seed": seed,
                        "macro_f1": item["macro_f1"],
                        "balanced_accuracy": item["balanced_accuracy"],
                        "accuracy": item["accuracy"],
                        "macro_precision": item["macro_precision"],
                        "macro_recall": item["macro_recall"],
                    }
                )
                complete = _json(run_root / "complete.json")
                if method in CNNS:
                    compute_rows.append(
                        {
                            "method": method,
                            "training_fraction": fraction,
                            "seed": seed,
                            "source_images": complete["source_images"],
                            "training_instances": complete["training_instances"],
                            "epochs_executed": complete["epochs_executed"],
                            "best_epoch": complete["best_epoch"],
                            "best_validation_macro_f1": complete["validation_metrics"]["macro_f1"],
                            "total_training_seconds": complete["training_seconds"],
                            "checkpoint_fingerprint": complete["checkpoint_fingerprint"],
                            "checkpoint_bytes": complete["checkpoint_bytes"],
                        }
                    )
            metrics[(method, fraction)] = items
        metrics[(method, 1.0)] = _full_metrics(root, method)
        for index, item in enumerate(metrics[(method, 1.0)]):
            per_seed_rows.append(
                {
                    "method": method,
                    "training_fraction": 1.0,
                    "seed": "canonical" if method in CLASSICAL else SEEDS[index],
                    "macro_f1": item["macro_f1"],
                    "balanced_accuracy": item["balanced_accuracy"],
                    "accuracy": item["accuracy"],
                    "macro_precision": item["macro_precision"],
                    "macro_recall": item["macro_recall"],
                }
            )
    outputs = calculate_learning_curve_outputs(metrics)
    aggregate = output / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    fields = {
        "learning_curve_summary": ("method", "training_fraction", "source_images", "training_instances", "macro_f1_mean", "macro_f1_sample_sd", "balanced_accuracy_mean", "balanced_accuracy_sample_sd", "accuracy_mean", "accuracy_sample_sd", "replicate_count", "standard_deviation"),
        "per_class_learning_curves": ("method", "training_fraction", "class", "training_class_count", "f1_mean", "f1_sample_sd", "precision_mean", "precision_sample_sd", "recall_mean", "recall_sample_sd"),
        "performance_retention": ("method", "training_fraction", "full_data_macro_f1", "macro_f1_mean", "performance_retention", "performance_retention_percent"),
        "absolute_loss": ("method", "training_fraction", "macro_f1_mean", "full_data_macro_f1", "delta_to_full"),
        "marginal_gains": ("method", "from_fraction", "to_fraction", "macro_f1_gain"),
        "threshold_95_percent": ("method", "threshold_fraction", "threshold_percent", "criterion_macro_f1", "observed_macro_f1"),
        "normalized_learning_curve_auc": ("method", "normalized_macro_f1_learning_curve_auc"),
        "thunderstrike_learning_curve": ("method", "training_fraction", "class", "training_class_count", "f1_mean", "f1_sample_sd", "precision_mean", "precision_sample_sd", "recall_mean", "recall_sample_sd"),
    }
    for name, rows in outputs.items():
        write_csv(aggregate / f"{name}.csv", rows, fields[name])
    write_csv(
        aggregate / "per_seed_results.csv",
        per_seed_rows,
        ("method", "training_fraction", "seed", "macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall"),
    )
    write_csv(
        aggregate / "compute_summary.csv",
        compute_rows,
        ("method", "training_fraction", "seed", "source_images", "training_instances", "epochs_executed", "best_epoch", "best_validation_macro_f1", "total_training_seconds", "checkpoint_fingerprint", "checkpoint_bytes"),
    )
    totals = {
        method: sum(float(row["total_training_seconds"]) for row in compute_rows if row["method"] == method)
        for method in CNNS
    }
    latency_sources = {
        "resnet18": "experiments/summaries/phase5_resnet18_v1/aggregate/efficiency.json",
        "mobilenet_v3_small": "experiments/summaries/phase6_mobilenet_v3_small_v1/aggregate/efficiency.json",
    }
    compute = {
        "new_reduced_cnn_training_seconds": totals,
        "new_reduced_cnn_training_seconds_total": sum(totals.values()),
        "inference_timing_repeated_in_phase7": False,
        "reused_inference_timing_sources": latency_sources,
    }
    atomic_write_text(aggregate / "compute_summary.json", json_text(compute))
    result = {**outputs, "per_seed_results": per_seed_rows, "compute_rows": compute_rows, "compute": compute}
    atomic_write_text(aggregate / "summary.json", json_text(result))
    return result


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_learning_metric(rows: Sequence[Mapping[str, Any]], metric: str, path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        x = [100 * float(row["training_fraction"]) for row in selected]
        y = [float(row[f"{metric}_mean"]) for row in selected]
        errors = [np.nan if row[f"{metric}_sample_sd"] in (None, "") else float(row[f"{metric}_sample_sd"]) for row in selected]
        axis.errorbar(x, y, yerr=errors, marker="o", capsize=4, linewidth=1.8, label=method)
    axis.set_xticks([25, 50, 75, 100])
    axis.set_xlabel("Labeled training source images (%)")
    axis.set_ylabel(title)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title(f"{title} versus labeled training-data fraction")
    _save_figure(figure, path)


def _plot_phase7(aggregate: Mapping[str, Any], figures: Path) -> None:
    learning = aggregate["learning_curve_summary"]
    _plot_learning_metric(learning, "macro_f1", figures / "macro_f1_learning_curves.png", "Test macro-F1")
    _plot_learning_metric(learning, "balanced_accuracy", figures / "balanced_accuracy_learning_curves.png", "Test balanced accuracy")
    _plot_learning_metric(learning, "accuracy", figures / "accuracy_learning_curves.png", "Test accuracy")

    for key, filename, ylabel, value_key in (
        ("performance_retention", "performance_retention.png", "Full-data performance retained (%)", "performance_retention_percent"),
        ("absolute_loss", "absolute_f1_loss.png", "Macro-F1 delta to full data", "delta_to_full"),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        for method in METHODS:
            rows = [row for row in aggregate[key] if row["method"] == method]
            axis.plot([100 * row["training_fraction"] for row in rows], [row[value_key] for row in rows], marker="o", label=method)
        axis.set_xticks([25, 50, 75, 100] if key == "absolute_loss" else [25, 50, 75])
        axis.set_xlabel("Labeled training source images (%)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        _save_figure(figure, figures / filename)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    transitions = ((0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
    x = np.arange(len(METHODS))
    width = 0.24
    for index, transition in enumerate(transitions):
        values = [next(row["macro_f1_gain"] for row in aggregate["marginal_gains"] if row["method"] == method and (row["from_fraction"], row["to_fraction"]) == transition) for method in METHODS]
        axis.bar(x + (index - 1) * width, values, width, label=f"{transition[0]:.0%}→{transition[1]:.0%}")
    axis.set_xticks(x, METHODS, rotation=15)
    axis.set_ylabel("Observed macro-F1 gain")
    axis.legend()
    axis.axhline(0, color="black", linewidth=0.8)
    _save_figure(figure, figures / "marginal_data_gains.png")

    figure, axis = plt.subplots(figsize=(7.5, 5))
    auc = aggregate["normalized_learning_curve_auc"]
    axis.bar([row["method"] for row in auc], [row["normalized_macro_f1_learning_curve_auc"] for row in auc])
    axis.set_ylabel("Normalized macro-F1 learning-curve AUC")
    axis.tick_params(axis="x", rotation=15)
    axis.set_ylim(0, 1)
    _save_figure(figure, figures / "normalized_learning_curve_auc.png")

    for method in METHODS:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for label in LABELS:
            rows = [row for row in aggregate["per_class_learning_curves"] if row["method"] == method and row["class"] == label]
            axis.plot([100 * row["training_fraction"] for row in rows], [row["f1_mean"] for row in rows], marker="o", label=label)
        axis.set_xticks([25, 50, 75, 100])
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Labeled training source images (%)")
        axis.set_ylabel("Test per-class F1")
        axis.legend(ncol=2)
        axis.grid(alpha=0.25)
        _save_figure(figure, figures / f"per_class_{method}.png")

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for method in METHODS:
        rows = [row for row in aggregate["thunderstrike_learning_curve"] if row["method"] == method]
        yerr = [np.nan if row["f1_sample_sd"] is None else row["f1_sample_sd"] for row in rows]
        axis.errorbar([100 * row["training_fraction"] for row in rows], [row["f1_mean"] for row in rows], yerr=yerr, marker="o", capsize=4, label=method)
    axis.set_xticks([25, 50, 75, 100])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Labeled training source images (%)")
    axis.set_ylabel("Thunderstrike test F1")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, figures / "thunderstrike_learning_curve.png")


def _canonical_reproduction(
    *,
    method: str,
    output: Path,
    subset: Mapping[str, Any],
    processed_root: Path,
    validation_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    official_state: dict[str, torch.Tensor],
    official_fingerprint: str,
    frozen: Mapping[str, Any],
    git_commit: str | None,
    device: torch.device,
) -> dict[str, Any]:
    original_root = _run_root(output, method, 17, 0.25)
    rerun_root = output / "reproducibility" / method / "seed_17_frac_025"
    _write_cnn_run(
        rerun_root,
        method=method,
        seed=17,
        fraction=0.25,
        subset=subset,
        processed_root=processed_root,
        validation_rows=validation_rows,
        test_rows=test_rows,
        official_state=official_state,
        official_fingerprint=official_fingerprint,
        frozen=frozen,
        git_commit=git_commit,
        device=device,
    )
    original_complete, rerun_complete = _json(original_root / "complete.json"), _json(rerun_root / "complete.json")
    original_history, rerun_history = read_csv(original_root / "history.csv"), read_csv(rerun_root / "history.csv")
    checks = {
        "initial_head_fingerprint_identical": original_complete["initial_head_fingerprint"] == rerun_complete["initial_head_fingerprint"],
        "best_epoch_identical": original_complete["best_epoch"] == rerun_complete["best_epoch"],
        "scientific_history_identical": _scientific_history(original_history) == _scientific_history(rerun_history),
        "validation_predictions_identical": read_csv(original_root / "validation_predictions.csv") == read_csv(rerun_root / "validation_predictions.csv"),
        "test_predictions_identical": read_csv(original_root / "test_predictions.csv") == read_csv(rerun_root / "test_predictions.csv"),
        "checkpoint_fingerprint_identical": original_complete["checkpoint_fingerprint"] == rerun_complete["checkpoint_fingerprint"],
        "validation_metrics_identical": _json(original_root / "validation_metrics.json") == _json(rerun_root / "validation_metrics.json"),
        "test_metrics_identical": _json(original_root / "test_metrics.json") == _json(rerun_root / "test_metrics.json"),
    }
    record = {"method": method, "seed": 17, "fraction": 0.25, "timings_excluded": True, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    atomic_write_text(output / "reproducibility" / f"{method}.json", json_text(record))
    if record["status"] != "PASS":
        raise DataEfficiencyError(f"canonical {method} reproducibility failed")
    return record


def _publish(output: Path, summary: Path) -> None:
    if summary.exists():
        shutil.rmtree(summary)
    shutil.copytree(output, summary, ignore=shutil.ignore_patterns("*.pt"))


def run_data_efficiency(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Run the fixed 36-fit reduced-data matrix, then aggregate and reproduce canonicals."""

    root = Path(repository_root).resolve()
    gate = verify_phase7_start_gate(config, root)
    git = capture_git_provenance(root)
    if git["git_commit"] is None or git["git_dirty"]:
        raise DataEfficiencyError("scientific runs require the clean committed Phase 7 apparatus")
    data = config.as_dict()
    phase = data["data_efficiency"]
    output = (root / phase["output_root"]).resolve()
    summary = (root / phase["summary_root"]).resolve()
    figures = (root / phase["figures_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output / "resolved_config.yaml", config.to_yaml())
    references = _write_reused_references(root, output)
    manifest = {
        "schema_version": "1.0",
        "phase": 7,
        "status": "running",
        "result_id": phase["result_id"],
        "apparatus_commit": git["git_commit"],
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "fractions": list(FRACTIONS),
        "reduced_fractions": list(REDUCED_FRACTIONS),
        "seeds": list(SEEDS),
        "new_reduced_fits_planned": 36,
        "new_cnn_trainings_planned": 18,
        "full_data_training_runs": 0,
        "hyperparameter_search": False,
        "phase8_started": False,
        "frozen_config_identities": frozen_config_identities(config, root),
        "reused_full_data": references,
        "environment": capture_environment(str(resolve_device(data["runtime"]["device"])), root),
        "started_utc": format_utc(utc_now()),
    }
    atomic_write_text(output / "manifest.json", json_text(manifest))
    processed_root = root / "data" / "processed" / "wtbd_crops_v1"
    all_rows = read_csv(processed_root / "manifest.csv")
    split = split_rows(all_rows)

    for method in CLASSICAL:
        features, cache = _load_feature_cache(root, method, all_rows, gate["frozen"][method])
        phase4_manifest = _json(
            root / "experiments" / "summaries" / "phase4_traditional_v1" / "manifest.json"
        )
        expected_fingerprint = phase4_manifest["feature_fingerprints"][method]
        if cache["feature_fingerprint"] != expected_fingerprint:
            raise DataEfficiencyError(f"{method} cached feature fingerprint differs from frozen Phase 4")
        for seed in SEEDS:
            for fraction in REDUCED_FRACTIONS:
                _write_classical_run(
                    _run_root(output, method, seed, fraction),
                    method=method,
                    seed=seed,
                    fraction=fraction,
                    subset=gate["subsets"][(seed, fraction)],
                    all_rows=all_rows,
                    features=features,
                    frozen=gate["frozen"][method],
                )

    official: dict[str, tuple[dict[str, torch.Tensor], str]] = {}
    resnet_model, resnet_provenance = load_official_backbone()
    official["resnet18"] = (resnet_model.state_dict(), resnet_provenance["pretrained_backbone_fingerprint"])
    del resnet_model
    mobile_model, mobile_provenance = load_official_model()
    official["mobilenet_v3_small"] = (mobile_model.state_dict(), mobile_provenance["pretrained_mobilenet_fingerprint"])
    del mobile_model
    if official["resnet18"][1] != gate["frozen"]["resnet18"]["model"]["pretrained_backbone_fingerprint"]:
        raise DataEfficiencyError("official ResNet fingerprint changed")
    if official["mobilenet_v3_small"][1] != gate["frozen"]["mobilenet_v3_small"]["model"]["pretrained_mobilenet_fingerprint"]:
        raise DataEfficiencyError("official MobileNet fingerprint changed")
    device = resolve_device(data["runtime"]["device"])
    for method in CNNS:
        state, pretrained_fingerprint = official[method]
        for seed in SEEDS:
            initial_fingerprints = []
            for fraction in REDUCED_FRACTIONS:
                completed = _write_cnn_run(
                    _run_root(output, method, seed, fraction),
                    method=method,
                    seed=seed,
                    fraction=fraction,
                    subset=gate["subsets"][(seed, fraction)],
                    processed_root=processed_root,
                    validation_rows=split["validation"],
                    test_rows=split["test"],
                    official_state=state,
                    official_fingerprint=pretrained_fingerprint,
                    frozen=gate["frozen"][method],
                    git_commit=git["git_commit"],
                    device=device,
                )
                initial_fingerprints.append(completed["initial_head_fingerprint"])
            if len(set(initial_fingerprints)) != 1:
                raise DataEfficiencyError(f"same-seed {method} heads differ across fractions")

    reproductions = {}
    for method in CNNS:
        state, pretrained_fingerprint = official[method]
        reproductions[method] = _canonical_reproduction(
            method=method,
            output=output,
            subset=gate["subsets"][(17, 0.25)],
            processed_root=processed_root,
            validation_rows=split["validation"],
            test_rows=split["test"],
            official_state=state,
            official_fingerprint=pretrained_fingerprint,
            frozen=gate["frozen"][method],
            git_commit=git["git_commit"],
            device=device,
        )
    aggregate = _aggregate_results(root, output)
    _plot_phase7(aggregate, figures)
    manifest.update(
        {
            "status": "completed",
            "completed_utc": format_utc(utc_now()),
            "new_reduced_fits_completed": 36,
            "new_cnn_trainings_completed": 18,
            "primary_test_evaluations": 36,
            "canonical_reproducibility_reruns": 2,
            "reproducibility": reproductions,
            "same_seed_initial_heads_match": True,
            "full_data_results_reused": True,
            "inference_timing_repeated": False,
        }
    )
    atomic_write_text(output / "manifest.json", json_text(manifest))
    _publish(output, summary)
    result = validate_data_efficiency_results(config, root)
    return result


def validate_data_efficiency_results(
    config: ResolvedConfig, repository_root: str | Path
) -> dict[str, Any]:
    """Validate all tracked Phase 7 scientific artifacts without retraining."""

    root = Path(repository_root).resolve()
    gate = verify_phase7_start_gate(config, root)
    phase = config.as_dict()["data_efficiency"]
    summary = (root / phase["summary_root"]).resolve()
    figures = (root / phase["figures_root"]).resolve()
    if not summary.is_dir():
        raise DataEfficiencyError("Phase 7 summary directory does not exist")
    manifest = _json(summary / "manifest.json")
    expected_manifest = {
        "status": "completed",
        "phase": 7,
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "new_reduced_fits_completed": 36,
        "new_cnn_trainings_completed": 18,
        "primary_test_evaluations": 36,
        "canonical_reproducibility_reruns": 2,
        "same_seed_initial_heads_match": True,
        "full_data_results_reused": True,
        "full_data_training_runs": 0,
        "hyperparameter_search": False,
        "phase8_started": False,
        "inference_timing_repeated": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise DataEfficiencyError("Phase 7 manifest exit gate failed")
    if manifest["frozen_config_identities"] != frozen_config_identities(config, root):
        raise DataEfficiencyError("frozen configuration identity changed")
    expected_test_ids = {
        row["instance_id"]
        for row in read_csv(root / "data" / "processed" / "wtbd_crops_v1" / "manifest.csv")
        if row["split"] == "test"
    }
    expected_validation_ids = {
        row["instance_id"]
        for row in read_csv(root / "data" / "processed" / "wtbd_crops_v1" / "manifest.csv")
        if row["split"] == "validation"
    }
    metrics: dict[tuple[str, float], list[dict[str, Any]]] = {}
    run_count = 0
    cnn_count = 0
    initial_heads: dict[tuple[str, int], set[str]] = {}
    for method in METHODS:
        for fraction in REDUCED_FRACTIONS:
            items = []
            for seed in SEEDS:
                run_root = _run_root(summary, method, seed, fraction)
                complete = _json(run_root / "complete.json")
                test_predictions = read_csv(run_root / "test_predictions.csv")
                validation_predictions = read_csv(run_root / "validation_predictions.csv")
                test_ids = {row["instance_id"] for row in test_predictions}
                validation_ids = {row["instance_id"] for row in validation_predictions}
                subset = gate["subsets"][(seed, fraction)]
                if (
                    complete["method"] != method
                    or int(complete["seed"]) != seed
                    or float(complete["fraction"]) != fraction
                    or complete["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT
                    or complete["source_subset_fingerprint"] != subset["fingerprint"]
                    or int(complete["source_images"]) != SOURCE_COUNTS[fraction]
                    or int(complete["training_instances"]) != INSTANCE_COUNTS[fraction]
                    or dict(complete["training_class_counts"]) != subset["class_counts"]
                    or int(complete["test_evaluations"]) != 1
                    or test_ids != expected_test_ids
                    or validation_ids != expected_validation_ids
                    or len(test_predictions) != 162
                    or len(validation_predictions) != 146
                ):
                    raise DataEfficiencyError(f"invalid completed run: {method}, {seed}, {fraction}")
                if method in CNNS:
                    cnn_count += 1
                    weights = balanced_subset_class_weights(subset["rows"])
                    expected_weights = {label: float(value) for label, value in zip(LABELS, weights.tolist(), strict=True)}
                    if complete["class_weights"] != expected_weights or complete["warm_started_from_fraction"] is not None or not complete["fresh_official_pretrained_start"]:
                        raise DataEfficiencyError(f"invalid CNN start/weights: {method}, {seed}, {fraction}")
                    if not all(f"logit_{label}" in test_predictions[0] for label in LABELS):
                        raise DataEfficiencyError("CNN logits are missing")
                    initial_heads.setdefault((method, seed), set()).add(complete["initial_head_fingerprint"])
                items.append(_json(run_root / "test_metrics.json"))
                run_count += 1
            metrics[(method, fraction)] = items
        metrics[(method, 1.0)] = _full_metrics(root, method)
    if run_count != 36 or cnn_count != 18 or any(len(values) != 1 for values in initial_heads.values()):
        raise DataEfficiencyError("run-count or initial-head exit gate failed")
    expected_aggregate = calculate_learning_curve_outputs(metrics)
    stored_aggregate = _json(summary / "aggregate" / "summary.json")
    for key, value in expected_aggregate.items():
        if stored_aggregate[key] != value:
            raise DataEfficiencyError(f"aggregate does not reproduce: {key}")
    for method in CNNS:
        reproduction = _json(summary / "reproducibility" / f"{method}.json")
        if reproduction["status"] != "PASS" or not all(reproduction["checks"].values()):
            raise DataEfficiencyError(f"canonical {method} reproducibility did not pass")
    for record in manifest["reused_full_data"].values():
        source = root / record["source_path"]
        if not source.is_file() or _sha256(source) != record["source_sha256"]:
            raise DataEfficiencyError("an upstream reused result fingerprint changed")
    expected_figures = {
        "macro_f1_learning_curves.png",
        "balanced_accuracy_learning_curves.png",
        "accuracy_learning_curves.png",
        "performance_retention.png",
        "absolute_f1_loss.png",
        "marginal_data_gains.png",
        "normalized_learning_curve_auc.png",
        "per_class_resnet18.png",
        "per_class_mobilenet_v3_small.png",
        "per_class_hog.png",
        "per_class_lbp.png",
        "thunderstrike_learning_curve.png",
    }
    if not all((figures / name).is_file() for name in expected_figures):
        raise DataEfficiencyError("one or more required Phase 7 figures are missing")
    subset_fingerprints = {
        f"seed_{seed}_{_fraction_dir(fraction)}": gate["subsets"][(seed, fraction)]["fingerprint"]
        for seed in SEEDS
        for fraction in FRACTIONS
    }
    return {
        "status": "PASS",
        "result_id": phase["result_id"],
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "new_reduced_fits": run_count,
        "new_cnn_trainings": cnn_count,
        "subset_fingerprints": subset_fingerprints,
        "frozen_config_identities": manifest["frozen_config_identities"],
        "learning_curve_summary": expected_aggregate["learning_curve_summary"],
        "performance_retention": expected_aggregate["performance_retention"],
        "marginal_gains": expected_aggregate["marginal_gains"],
        "threshold_95_percent": expected_aggregate["threshold_95_percent"],
        "normalized_learning_curve_auc": expected_aggregate["normalized_learning_curve_auc"],
        "reproducibility": manifest["reproducibility"],
        "phase8_started": False,
    }
