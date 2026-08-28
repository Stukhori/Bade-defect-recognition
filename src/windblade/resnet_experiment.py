"""Canonical Phase 5 validation-freeze-test ResNet-18 experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from windblade.config import ResolvedConfig, load_config
from windblade.data.processed import LABELS, json_text, read_csv, validate_processed_dataset
from windblade.data.subsets import validate_training_subsets
from windblade.deep.checkpoints import load_checkpoint, save_checkpoint, state_dict_fingerprint
from windblade.deep.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WTBDCropDataset,
    balanced_class_weights,
    make_loader,
    split_rows,
)
from windblade.deep.determinism import resolve_device, seed_torch
from windblade.deep.resnet import (
    EXPECTED_RESNET18_PARAMETERS,
    WEIGHT_ENUM,
    load_official_backbone,
    model_from_official_state,
)
from windblade.deep.training import (
    aggregate_seed_metrics,
    hyperparameter_grid,
    inference_latency,
    run_epoch,
    select_candidate,
    train_with_validation,
)
from windblade.environment import capture_environment, capture_git_provenance
from windblade.evaluation.reporting import plot_confusion, write_csv, write_matrix_csv
from windblade.traditional import validate_traditional_results
from windblade.utils import atomic_write_text, format_utc, utc_now

EXPECTED_FINGERPRINT = "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991"
HISTORY_FIELDS = (
    "epoch", "train_loss", "train_accuracy", "train_macro_f1", "validation_loss",
    "validation_accuracy", "validation_balanced_accuracy", "validation_macro_precision",
    "validation_macro_recall", "validation_macro_f1", "learning_rate", "elapsed_seconds",
    "is_best_epoch",
)
PREDICTION_FIELDS = (
    "instance_id", "source_image_id", "true_class_id", "true_label",
    "predicted_class_id", "predicted_label", "correct",
    *(f"logit_{label}" for label in LABELS),
)


class ResNetExperimentError(RuntimeError):
    pass


def _safe_clean(path: Path, parent: Path) -> None:
    resolved, allowed = path.resolve(), parent.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ResNetExperimentError(f"unsafe output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _validate_config(data: Mapping[str, Any]) -> None:
    failures = []
    checks = {
        "phase": data["project"]["phase"] == 5,
        "fingerprint": data["dataset"]["processed_fingerprint"] == EXPECTED_FINGERPRINT,
        "architecture": data["model"]["architecture"] == "resnet18",
        "weights": data["model"]["weights"] == "IMAGENET1K_V1",
        "classes": tuple(data["classes"]["order"]) == LABELS,
        "parameters": int(data["model"]["expected_total_parameters"]) == EXPECTED_RESNET18_PARAMETERS,
        "input": data["input"]["size"] == [224, 224] and data["input"]["augmentation"] == "none",
        "normalization": tuple(data["input"]["normalization"]["mean"]) == IMAGENET_MEAN and tuple(data["input"]["normalization"]["std"]) == IMAGENET_STD,
        "training": int(data["training"]["batch_size"]) == 32 and int(data["training"]["validation_batch_size"]) == 64 and int(data["training"]["max_epochs"]) == 30 and int(data["training"]["num_workers"]) == 0,
        "search": data["search"]["seed"] == 17 and data["search"]["learning_rates"] == [1e-4, 3e-4] and data["search"]["weight_decays"] == [0.0, 1e-4],
        "seeds": data["final"]["seeds"] == [17, 29, 43],
        "policies": data["training"]["class_weight"] == "balanced" and not data["training"]["mixed_precision"] and data["training"]["scheduler"] == "none",
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise ResNetExperimentError("Phase 5 configuration gate failed: " + ", ".join(failures))


def verify_start_gate(config: ResolvedConfig, root: Path) -> dict[str, Any]:
    data = config.as_dict()
    _validate_config(data)
    phase3 = validate_processed_dataset(config, root)
    validate_training_subsets(config, root)
    phase4 = validate_traditional_results(load_config(root / "configs/traditional_baselines.yaml"), root)
    if phase3["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT or phase4["status"] != "PASS":
        raise ResNetExperimentError("Phase 4 or frozen Phase 3 gate failed")
    return {"phase3": phase3, "phase4": phase4}


def _records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda item: item["instance_id"]):
        true_id, predicted_id = int(row["true_class_id"]), int(row["predicted_class_id"])
        output.append(
            {
                "instance_id": row["instance_id"],
                "source_image_id": row["source_image_id"],
                "true_class_id": true_id,
                "true_label": LABELS[true_id],
                "predicted_class_id": predicted_id,
                "predicted_label": LABELS[predicted_id],
                "correct": true_id == predicted_id,
                **{f"logit_{label}": row["logits"][index] for index, label in enumerate(LABELS)},
            }
        )
    return output


def _write_training(root: Path, result: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "history.csv", result["history"], HISTORY_FIELDS)
    atomic_write_text(root / "validation_metrics.json", json_text(result["best_validation_metrics"]))
    write_csv(root / "validation_predictions.csv", _records(result["best_validation_records"]), PREDICTION_FIELDS)


def _checkpoint_metadata(
    *, seed: int, result: Mapping[str, Any], lr: float, decay: float, git_commit: str | None
) -> dict[str, Any]:
    return {
        "architecture": "torchvision_resnet18",
        "class_count": 6,
        "class_order": list(LABELS),
        "pretrained_weight_enum": WEIGHT_ENUM,
        "seed": seed,
        "epoch": result["best_epoch"],
        "learning_rate": lr,
        "weight_decay": decay,
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "git_commit": git_commit,
        "validation_metrics": result["best_validation_metrics"],
    }


def _grid_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    scientific = [{key: value for key, value in row.items() if key != "training_seconds"} for row in rows]
    payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _freeze(
    path: Path, selected: Mapping[str, Any], grid_fingerprint: str,
    backbone_fingerprint: str, git_commit: str | None,
) -> dict[str, Any]:
    record = {
        "schema_version": "1.0", "phase": 5,
        "processed_dataset_fingerprint": EXPECTED_FINGERPRINT,
        "model": {"architecture": "torchvision_resnet18", "pretrained_weight_enum": WEIGHT_ENUM, "pretrained_backbone_fingerprint": backbone_fingerprint, "num_classes": 6, "fine_tune": "all", "total_parameters": EXPECTED_RESNET18_PARAMETERS},
        "input": {"size": [224, 224], "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}, "augmentation": "none"},
        "training": {"optimizer": "AdamW", "learning_rate": selected["learning_rate"], "weight_decay": selected["weight_decay"], "betas": [0.9, 0.999], "eps": 1e-8, "batch_size": 32, "validation_batch_size": 64, "max_epochs": 30, "patience": 6, "min_delta": 1e-4, "class_weight": "balanced CrossEntropyLoss", "scheduler": None, "mixed_precision": False, "num_workers": 0},
        "selection": {"tuning_seed": 17, "metric": "validation_macro_f1", "validation_macro_f1": selected["validation_macro_f1"], "validation_balanced_accuracy": selected["validation_balanced_accuracy"], "validation_macro_recall": selected["validation_macro_recall"], "numeric_tolerance": 1e-12, "grid_fingerprint": grid_fingerprint},
        "final_seeds": [17, 29, 43], "git_commit": git_commit,
        "frozen_before_test": True, "frozen_utc": format_utc(utc_now()),
    }
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("frozen_utc", None)
        expected = dict(record)
        expected.pop("frozen_utc", None)
        if comparable != expected:
            raise ResNetExperimentError("existing frozen ResNet config disagrees with validation selection")
        return existing
    atomic_write_text(path, yaml.safe_dump(record, sort_keys=True, allow_unicode=True))
    return record


def _plot_tuning(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [f"lr={r['learning_rate']:g}\nwd={r['weight_decay']:g}" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [r["validation_macro_f1"] for r in rows], color=["#e07a1f" if r["selected"] else "#2f78c4" for r in rows])
    ax.set_ylim(0, 1); ax.set_ylabel("Best validation macro-F1"); ax.set_title("ResNet-18 predeclared tuning grid")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_curves(history: Sequence[Mapping[str, Any]], seed: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5)); epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_macro_f1"] for r in history], label="train")
    ax.plot(epochs, [r["validation_macro_f1"] for r in history], label="validation")
    ax.set_ylim(0, 1); ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1"); ax.set_title(f"Final ResNet-18 training, seed {seed}"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _publish(output: Path, summary: Path) -> None:
    if summary.exists():
        shutil.rmtree(summary)
    for source in output.rglob("*"):
        if not source.is_file() or source.suffix == ".pt" or source.name == "run.log":
            continue
        destination = summary / source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_resnet18_baseline(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(); data = config.as_dict(); gate = verify_start_gate(config, root)
    output = root / data["resnet18"]["output_root"]; summary = root / data["resnet18"]["summary_root"]; figures = root / data["resnet18"]["figures_root"]
    _safe_clean(output, root / "experiments/results"); _safe_clean(figures, root / "figures")
    provenance = capture_git_provenance(root); git_commit = provenance["git_commit"]
    processed_root = root / "data/processed/wtbd_crops_v1"
    rows = read_csv(processed_root / "manifest.csv"); partitions = split_rows(rows)
    # Decode and hash every Phase 3 crop before any run.
    datasets = {name: WTBDCropDataset(items, processed_root, verify_hashes=True) for name, items in partitions.items()}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            dataset[index]
    weights = balanced_class_weights(partitions["train"])
    seed_torch(17); official_model, pretrained = load_official_backbone()
    official_state = deepcopy(official_model.state_dict()); del official_model
    pretrained.update({"torch_version": torch.__version__, "torchvision_version": metadata.version("torchvision")})
    atomic_write_text(output / "pretrained_backbone.json", json_text(pretrained))
    device = resolve_device(data["runtime"]["device"])
    tuning_rows = []
    for index, candidate in enumerate(hyperparameter_grid(), 1):
        seed_torch(17)
        model = model_from_official_state(official_state, seed=17)
        result = train_with_validation(
            model,
            make_loader(datasets["train"], batch_size=32, shuffle=True, seed=17),
            make_loader(datasets["validation"], batch_size=64, shuffle=False, seed=17),
            device=device, class_weights=weights,
            learning_rate=candidate["learning_rate"], weight_decay=candidate["weight_decay"],
        )
        run_root = output / "tuning" / f"config_{index:02d}"; _write_training(run_root, result)
        checkpoint = save_checkpoint(run_root / "best_state_dict.pt", result["best_state_dict"], _checkpoint_metadata(seed=17, result=result, lr=candidate["learning_rate"], decay=candidate["weight_decay"], git_commit=git_commit))
        tuning_rows.append({"candidate_id": f"config_{index:02d}", **candidate, "best_epoch": result["best_epoch"], "validation_macro_f1": result["best_validation_metrics"]["macro_f1"], "validation_balanced_accuracy": result["best_validation_metrics"]["balanced_accuracy"], "validation_macro_recall": result["best_validation_metrics"]["macro_recall"], "training_seconds": result["training_seconds"], "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"], "selected": False})
    selected = select_candidate(tuning_rows)
    for row in tuning_rows:
        row["selected"] = row["candidate_id"] == selected["candidate_id"]
    grid_fp = _grid_fingerprint(tuning_rows)
    write_csv(output / "tuning/grid_results.csv", tuning_rows, tuple(tuning_rows[0]))
    frozen = _freeze(root / data["resnet18"]["frozen_config"], selected, grid_fp, pretrained["pretrained_backbone_fingerprint"], git_commit)
    atomic_write_text(output / "frozen/selected_hyperparameters.json", json_text(frozen))
    _plot_tuning(tuning_rows, figures / "tuning_validation_macro_f1.png")

    final_results: dict[int, dict[str, Any]] = {}
    for seed in (17, 29, 43):
        seed_torch(seed); model = model_from_official_state(official_state, seed=seed)
        result = train_with_validation(model, make_loader(datasets["train"], batch_size=32, shuffle=True, seed=seed), make_loader(datasets["validation"], batch_size=64, shuffle=False, seed=seed), device=device, class_weights=weights, learning_rate=float(selected["learning_rate"]), weight_decay=float(selected["weight_decay"]))
        run_root = output / "final" / f"seed_{seed}"; _write_training(run_root, result)
        checkpoint = save_checkpoint(run_root / "best_state_dict.pt", result["best_state_dict"], _checkpoint_metadata(seed=seed, result=result, lr=float(selected["learning_rate"]), decay=float(selected["weight_decay"]), git_commit=git_commit))
        result["checkpoint"] = checkpoint; final_results[seed] = result
        _plot_curves(result["history"], seed, figures / f"final_training_curves_seed{seed}.png")
    # The test loader is first constructed only after tuning freeze and all final checkpoints exist.
    for seed, result in final_results.items():
        checkpoint_path = output / "final" / f"seed_{seed}" / "best_state_dict.pt"
        if not checkpoint_path.is_file(): raise ResNetExperimentError("final checkpoint gate failed")
        state, _ = load_checkpoint(checkpoint_path, expected_dataset_fingerprint=EXPECTED_FINGERPRINT)
        result["model"].load_state_dict(state)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        _, metrics, records = run_epoch(result["model"], make_loader(datasets["test"], batch_size=64, shuffle=False, seed=seed), criterion, device)
        run_root = output / "final" / f"seed_{seed}"
        atomic_write_text(run_root / "test_metrics.json", json_text(metrics)); write_csv(run_root / "test_predictions.csv", _records(records), PREDICTION_FIELDS)
        write_matrix_csv(run_root / "confusion_matrix_counts.csv", metrics["confusion_matrix_counts"], LABELS); write_matrix_csv(run_root / "confusion_matrix_normalized.csv", metrics["confusion_matrix_row_normalized"], LABELS)
        plot_confusion(metrics["confusion_matrix_row_normalized"], LABELS, figures / f"confusion_seed{seed}.png", normalized=True)
        result["test_metrics"] = metrics; result["test_records"] = records
    aggregate = aggregate_seed_metrics([final_results[seed]["test_metrics"] for seed in (17, 29, 43)])
    atomic_write_text(output / "aggregate/test_summary.json", json_text(aggregate))
    write_matrix_csv(output / "aggregate/mean_normalized_confusion.csv", aggregate["mean_normalized_confusion"], LABELS)
    plot_confusion(aggregate["mean_normalized_confusion"], LABELS, figures / "confusion_mean_normalized.png", normalized=True)
    seed_rows = []
    for seed, result in final_results.items():
        metric = result["test_metrics"]
        seed_rows.append({"seed": seed, "best_epoch": result["best_epoch"], "validation_macro_f1": result["best_validation_metrics"]["macro_f1"], **{key: metric[key] for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall")}, "training_seconds": result["training_seconds"], "epochs_executed": result["epochs_executed"], "average_seconds_per_epoch": result["training_seconds"] / result["epochs_executed"], "checkpoint_fingerprint": result["checkpoint"]["checkpoint_fingerprint"], "checkpoint_bytes": result["checkpoint"]["checkpoint_bytes"]})
    write_csv(output / "aggregate/per_seed_summary.csv", seed_rows, tuple(seed_rows[0]))
    per_class_rows = [{"class_id": i, "class_label": label, **{f"{metric}_{stat}": aggregate["per_class"][label][metric][stat] for metric in ("precision", "recall", "f1") for stat in ("mean", "sample_sd")}} for i, label in enumerate(LABELS)]
    write_csv(output / "aggregate/per_class_summary.csv", per_class_rows, tuple(per_class_rows[0]))
    timing = inference_latency(final_results[17]["model"], datasets["test"][0]["image"], device)
    efficiency = {"device": str(device), "total_parameters": EXPECTED_RESNET18_PARAMETERS, "trainable_parameters": EXPECTED_RESNET18_PARAMETERS, "canonical_seed": 17, "checkpoint_bytes": final_results[17]["checkpoint"]["checkpoint_bytes"], "inference": timing, "training": [{key: row[key] for key in ("seed", "training_seconds", "epochs_executed", "best_epoch", "average_seconds_per_epoch")} for row in seed_rows]}
    atomic_write_text(output / "aggregate/efficiency.json", json_text(efficiency))

    # Full independent seed-17 training rerun and explicit required test reproduction.
    seed_torch(17); repro_model = model_from_official_state(official_state, seed=17)
    repro = train_with_validation(repro_model, make_loader(datasets["train"], batch_size=32, shuffle=True, seed=17), make_loader(datasets["validation"], batch_size=64, shuffle=False, seed=17), device=device, class_weights=weights, learning_rate=float(selected["learning_rate"]), weight_decay=float(selected["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)); _, repro_test_metrics, repro_records = run_epoch(repro_model, make_loader(datasets["test"], batch_size=64, shuffle=False, seed=17), criterion, device)
    scientific_history = lambda history: [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in history]
    reproducibility = {
        "status": "PASS",
        "best_epoch_identical": repro["best_epoch"] == final_results[17]["best_epoch"],
        "validation_predictions_identical": _records(repro["best_validation_records"]) == _records(final_results[17]["best_validation_records"]),
        "test_predictions_identical": _records(repro_records) == _records(final_results[17]["test_records"]),
        "test_metrics_identical": repro_test_metrics == final_results[17]["test_metrics"],
        "scientific_history_identical": scientific_history(repro["history"]) == scientific_history(final_results[17]["history"]),
        "checkpoint_fingerprint_identical": state_dict_fingerprint(repro["best_state_dict"]) == final_results[17]["checkpoint"]["checkpoint_fingerprint"],
    }
    if not all(value for key, value in reproducibility.items() if key != "status"):
        reproducibility["status"] = "FAIL"; atomic_write_text(output / "reproducibility.json", json_text(reproducibility)); raise ResNetExperimentError("canonical seed-17 reproducibility failed")
    atomic_write_text(output / "reproducibility.json", json_text(reproducibility))
    environment = capture_environment(str(device), root)
    environment.update({"requested_device": data["runtime"]["device"], "actual_device": str(device), "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    manifest = {"status": "completed", "phase": 5, "result_id": data["resnet18"]["result_id"], "phase4_start_gate": gate["phase4"], "processed_dataset_fingerprint": EXPECTED_FINGERPRINT, "sample_counts": {name: len(dataset) for name, dataset in datasets.items()}, "class_order": list(LABELS), "class_weights": weights.tolist(), "tuning_candidates": 4, "tuning_test_evaluations": 0, "selected_candidate": selected["candidate_id"], "grid_fingerprint": grid_fp, "frozen_before_test": True, "final_seeds": [17, 29, 43], "final_test_evaluations": 3, "reproducibility_test_evaluations": 1, "parameter_count": EXPECTED_RESNET18_PARAMETERS, "pretrained_backbone_fingerprint": pretrained["pretrained_backbone_fingerprint"], "no_augmentation": True, "data_efficiency_started": False, "robustness_started": False, "phase6_started": False, "environment": environment}
    atomic_write_text(output / "manifest.json", json_text(manifest)); atomic_write_text(output / "resolved_config.yaml", config.to_yaml()); atomic_write_text(output / "run.log", "Phase 5 canonical run completed successfully.\n")
    _publish(output, summary)
    return {"status": "PASS", "manifest": manifest, "tuning": tuning_rows, "selected": selected, "per_seed": seed_rows, "aggregate": aggregate, "efficiency": efficiency, "reproducibility": reproducibility, "summary_root": summary}


def validate_resnet18_results(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(); verify_start_gate(config, root); data = config.as_dict(); summary = root / data["resnet18"]["summary_root"]
    manifest = json.loads((summary / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest["processed_dataset_fingerprint"] != EXPECTED_FINGERPRINT or manifest["sample_counts"] != {"train": 757, "validation": 146, "test": 162} or manifest["tuning_candidates"] != 4 or manifest["tuning_test_evaluations"] != 0 or manifest["final_test_evaluations"] != 3 or manifest["final_seeds"] != [17, 29, 43] or manifest["parameter_count"] != EXPECTED_RESNET18_PARAMETERS or not manifest["frozen_before_test"] or manifest["phase6_started"]:
        raise ResNetExperimentError("Phase 5 manifest exit assertions failed")
    grid = read_csv(summary / "tuning/grid_results.csv")
    if len(grid) != 4 or len([row for row in grid if row["selected"].lower() == "true"]) != 1:
        raise ResNetExperimentError("Phase 5 tuning grid invalid")
    for seed in (17, 29, 43):
        run_root = summary / "final" / f"seed_{seed}"
        if len(read_csv(run_root / "test_predictions.csv")) != 162 or len(read_csv(run_root / "validation_predictions.csv")) != 146:
            raise ResNetExperimentError(f"seed {seed} prediction package invalid")
    repro = json.loads((summary / "reproducibility.json").read_text(encoding="utf-8"))
    if repro["status"] != "PASS": raise ResNetExperimentError("Phase 5 reproducibility failed")
    return {"status": "PASS", "result_id": manifest["result_id"], "pretrained_backbone_fingerprint": manifest["pretrained_backbone_fingerprint"], "tuning_candidates": 4, "final_seeds": [17, 29, 43], "reproducibility": "PASS"}
