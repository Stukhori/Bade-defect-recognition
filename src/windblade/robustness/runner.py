"""Canonical Phase 8 gate, execution, persistence, and regeneration validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from windblade.config import ResolvedConfig, calculate_config_hash, load_config
from windblade.data.processed import LABELS, json_text, read_csv, sha256_file, validate_processed_dataset
from windblade.data_efficiency import validate_data_efficiency_results
from windblade.environment import capture_environment
from windblade.evaluation.reporting import write_csv, write_matrix_csv
from windblade.mobilenet_experiment import validate_mobilenet_results
from windblade.resnet_experiment import validate_resnet18_results
from windblade.robustness.aggregation import METHODS, aggregate_evaluations
from windblade.robustness.corruptions import condition_specs, pillow_environment
from windblade.robustness.dataset import (
    clean_rows,
    create_training_qc,
    generate_corruption_dataset,
    validate_corruption_dataset,
)
from windblade.robustness.evaluation import (
    evaluate_cnn_condition,
    evaluate_traditional_condition,
    load_frozen_cnn,
    load_frozen_traditional,
    verify_clean_reproduction,
)
from windblade.robustness.plots import create_main_figures
from windblade.traditional import validate_traditional_results
from windblade.utils import atomic_write_text, format_utc, utc_now


class RobustnessRunError(RuntimeError):
    """Raised when a Phase 8 gate or result invariant fails."""


PREDICTION_FIELDS = (
    "instance_id", "source_image_id", "true_class_id", "true_label",
    "predicted_class_id", "predicted_label", "correct",
    *(f"logit_{label}" for label in LABELS),
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    if result.returncode:
        raise RobustnessRunError(f"Git gate failed: git {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_upstream_gates(config: ResolvedConfig, root: str | Path, *, require_clean_git: bool) -> dict[str, Any]:
    repository = Path(root).resolve()
    data = config.as_dict()
    phase3 = validate_processed_dataset(load_config(repository / "configs/crop_dataset.yaml"), repository)
    phase4 = validate_traditional_results(load_config(repository / "configs/traditional_baselines.yaml"), repository)
    phase5 = validate_resnet18_results(load_config(repository / "configs/resnet18_baseline.yaml"), repository)
    phase6 = validate_mobilenet_results(load_config(repository / "configs/mobilenet_v3_small_baseline.yaml"), repository)
    phase7 = validate_data_efficiency_results(load_config(repository / data["upstream"]["phase7_config"]), repository)
    if any(item.get("status") != "PASS" for item in (phase3, phase4, phase5, phase6, phase7)):
        raise RobustnessRunError("PHASE 8 NOT STARTED — FROZEN UPSTREAM GATE FAILED")
    if phase3["processed_dataset_fingerprint"] != data["dataset"]["base_fingerprint"]:
        raise RobustnessRunError("PHASE 8 NOT STARTED — FROZEN UPSTREAM GATE FAILED")
    state_text = (repository / "PROJECT_STATE.md").read_text(encoding="utf-8")
    required_statement = "Phase 7 is complete. Phases 0–6 remain frozen. Phase 8 has not started."
    if required_statement not in state_text:
        raise RobustnessRunError("PHASE 8 NOT STARTED — FROZEN UPSTREAM GATE FAILED")
    rows = clean_rows(config, repository)
    counts = {"total": 1065, "train": 757, "validation": 146, "test": len(rows), "test_sources": len({row["source_image_id"] for row in rows})}
    if counts != {"total": 1065, "train": 757, "validation": 146, "test": 162, "test_sources": 109}:
        raise RobustnessRunError("PHASE 8 NOT STARTED — FROZEN UPSTREAM GATE FAILED")
    artifact_paths = [
        data["models"]["hog"]["model"], data["models"]["lbp"]["model"],
        *(f"{data['models']['resnet18']['result_root']}/seed_{seed}/best_state_dict.pt" for seed in (17, 29, 43)),
        *(f"{data['models']['mobilenet_v3_small']['result_root']}/seed_{seed}/best_state_dict.pt" for seed in (17, 29, 43)),
    ]
    missing = [path for path in artifact_paths if not (repository / path).is_file()]
    if missing:
        raise RobustnessRunError("missing frozen artifact(s): " + ", ".join(missing))
    git_record: dict[str, Any] = {"checked": require_clean_git}
    if require_clean_git:
        if _git(repository, "status", "--porcelain", "--untracked-files=normal"):
            raise RobustnessRunError("Phase 8 scientific run requires a clean Git worktree")
        head, origin = _git(repository, "rev-parse", "HEAD"), _git(repository, "rev-parse", "origin/main")
        if head != origin:
            raise RobustnessRunError("Phase 8 scientific run requires main synchronized with origin/main")
        git_record.update({"head": head, "origin_main": origin, "synchronized": True, "clean": True})
    return {
        "status": "PASS",
        "phase3": phase3["status"], "phase4": phase4["status"], "phase5": phase5["status"],
        "phase6": phase6["status"], "phase7": phase7["status"],
        "processed_dataset_fingerprint": phase3["processed_dataset_fingerprint"],
        "counts": counts, "frozen_artifact_count": len(artifact_paths), "git": git_record,
    }


def _image_paths(config: Mapping[str, Any], root: Path, rows: Sequence[Mapping[str, str]], condition_id: str) -> list[Path]:
    if condition_id == "clean":
        base = root / config["dataset"]["base_root"]
        return [base / row["output_relative_path"] for row in rows]
    manifest = read_csv(root / config["dataset"]["output_root"] / "manifest.csv")
    selected = {row["instance_id"]: root / row["corrupted_image_path"] for row in manifest if f"{row['corruption_family']}_{row['severity']}" == condition_id}
    if set(selected) != {row["instance_id"] for row in rows}:
        raise RobustnessRunError(f"corruption membership mismatch: {condition_id}")
    return [selected[row["instance_id"]] for row in rows]


def _clean_gate_with_models(config: ResolvedConfig, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int], torch.nn.Module], list[dict[str, Any]]]:
    data = config.as_dict()
    rows = clean_rows(config, root)
    paths = _image_paths(data, root, rows, "clean")
    entries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    traditional: dict[str, Any] = {}
    for method in ("hog", "lbp"):
        model, feature_config = load_frozen_traditional(method, data, root)
        result = evaluate_traditional_condition(method, model, feature_config, rows, paths)
        reports.append(verify_clean_reproduction(method, None, result, data, root))
        entries.append({"method": method, "seed": None, "condition_id": "clean", "result": result})
        traditional[method] = (model, feature_config)
    device = torch.device(data["runtime"]["device"])
    cnn_models: dict[tuple[str, int], torch.nn.Module] = {}
    for method in ("resnet18", "mobilenet_v3_small"):
        for seed in (17, 29, 43):
            model = load_frozen_cnn(method, seed, data, root, device)
            result = evaluate_cnn_condition(model, rows, paths, device=device, batch_size=int(data["evaluation"]["batch_size"]))
            reports.append(verify_clean_reproduction(method, seed, result, data, root))
            entries.append({"method": method, "seed": seed, "condition_id": "clean", "result": result})
            cnn_models[(method, seed)] = model
    return entries, traditional, cnn_models, reports


def apparatus_check(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    upstream = verify_upstream_gates(config, repository, require_clean_git=False)
    entries, _, models, reports = _clean_gate_with_models(config, repository)
    del entries, models
    qc = create_training_qc(config, repository)
    return {
        "status": "PASS", "upstream": upstream, "clean_reproduction": reports,
        "training_only_qc": qc, "scientific_corruptions_generated": 0,
        "model_training_or_refitting": 0,
    }


def _safe_clean_summary(path: Path, root: Path) -> None:
    allowed = (root / "experiments" / "summaries").resolve()
    resolved = path.resolve()
    if resolved == allowed or allowed not in resolved.parents or resolved.name != "phase8_robustness_v1":
        raise RobustnessRunError(f"unsafe Phase 8 summary root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _write_evaluation_tree(root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        method, seed, condition = str(entry["method"]), entry.get("seed"), str(entry["condition_id"])
        method_root = root / method / (f"seed_{seed}" if seed is not None else "deterministic") / condition
        result = entry["result"]
        atomic_write_text(method_root / "metrics.json", json_text(result["metrics"]))
        write_csv(method_root / "predictions.csv", result["predictions"], PREDICTION_FIELDS)
        write_matrix_csv(method_root / "confusion_matrix_counts.csv", result["metrics"]["confusion_matrix_counts"], LABELS)
        write_matrix_csv(method_root / "confusion_matrix_normalized.csv", result["metrics"]["confusion_matrix_row_normalized"], LABELS)


def _write_aggregates(root: Path, aggregates: Mapping[str, Sequence[Mapping[str, Any]]], entries: Sequence[Mapping[str, Any]]) -> None:
    aggregate_root = root / "aggregate"
    fields = {
        "per_seed_results": ("method", "seed", "condition_id", "corruption_family", "severity", "parameter", "macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "absolute_drop", "retention", "retention_percent", "relative_loss", "relative_loss_percent", "prediction_flip_count", "prediction_flip_rate", "harmful_flip_count", "beneficial_flip_count", "correct_to_correct", "correct_to_incorrect", "incorrect_to_correct", "incorrect_to_incorrect"),
        "robustness_summary": ("method", "condition_id", "corruption_family", "severity", "parameter", "replicate_count", "standard_deviation", "macro_f1", "macro_f1_sample_sd", "balanced_accuracy", "balanced_accuracy_sample_sd", "accuracy", "accuracy_sample_sd", "macro_precision", "macro_recall", "absolute_drop", "retention_percent", "relative_loss_percent", "prediction_flip_rate", "harmful_flip_count", "beneficial_flip_count"),
        "family_summary": ("method", "corruption_family", "clean_macro_f1", "mild_macro_f1", "moderate_macro_f1", "severe_macro_f1", "mean_degraded_macro_f1", "severe_retention", "severe_retention_percent", "mean_degraded_retention", "mean_degraded_retention_percent"),
        "severe_summary": ("method", "corruption_family", "macro_f1", "macro_f1_sample_sd", "retention", "retention_percent", "prediction_flip_rate", "harmful_flip_count", "beneficial_flip_count"),
        "overall_summary": ("method", "clean_macro_f1", "mean_degraded_condition_macro_f1", "mean_degraded_condition_retention", "mean_degraded_condition_retention_percent"),
        "per_class_robustness": ("method", "condition_id", "corruption_family", "severity", "class", "support", "f1_mean", "f1_sample_sd", "clean_f1_mean", "class_f1_drop"),
        "prediction_flip_rates": ("method", "condition_id", "corruption_family", "severity", "prediction_flip_rate", "prediction_flip_rate_sample_sd", "prediction_flip_count", "harmful_flip_count", "beneficial_flip_count"),
        "error_transitions_per_seed": ("method", "seed", "condition_id", "corruption_family", "severity", "prediction_flip_count", "prediction_flip_rate", "harmful_flip_count", "beneficial_flip_count", "correct_to_correct", "correct_to_incorrect", "incorrect_to_correct", "incorrect_to_incorrect"),
        "error_transitions": ("method", "condition_id", "corruption_family", "severity", "correct_to_correct_mean", "correct_to_correct_sample_sd", "correct_to_incorrect_mean", "correct_to_incorrect_sample_sd", "incorrect_to_correct_mean", "incorrect_to_correct_sample_sd", "incorrect_to_incorrect_mean", "incorrect_to_incorrect_sample_sd"),
        "instance_robustness": ("method", "seed", "instance_id", "source_image_id", "true_label", "corruption_family", "severity", "clean_prediction", "corrupted_prediction", "clean_correct", "corrupted_correct", "prediction_changed", "harmful_flip", "beneficial_flip", *(f"logit_{label}" for label in LABELS)),
    }
    for name, columns in fields.items():
        write_csv(aggregate_root / f"{name}.csv", list(aggregates[name]), columns)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((str(entry["method"]), str(entry["condition_id"])), []).append(entry)
    matrix_root = aggregate_root / "confusion_matrices"
    for (method, condition), values in grouped.items():
        matrices = np.asarray([entry["result"]["metrics"]["confusion_matrix_row_normalized"] for entry in values])
        write_matrix_csv(matrix_root / f"{method}_{condition}_mean_normalized.csv", matrices.mean(axis=0).tolist(), LABELS)


def _evaluate_all(config: ResolvedConfig, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config.as_dict()
    entries, traditional, cnn_models, clean_reports = _clean_gate_with_models(config, root)
    # The scientific dataset is not generated until all eight frozen clean
    # identities have reproduced their historical predictions and metrics.
    generate_corruption_dataset(config, root)
    validate_corruption_dataset(config, root)
    rows = clean_rows(config, root)
    device = torch.device(data["runtime"]["device"])
    for spec in condition_specs(data, include_clean=False):
        condition_id = str(spec["condition_id"])
        paths = _image_paths(data, root, rows, condition_id)
        for method in ("hog", "lbp"):
            model, feature_config = traditional[method]
            result = evaluate_traditional_condition(method, model, feature_config, rows, paths)
            entries.append({"method": method, "seed": None, "condition_id": condition_id, "result": result})
        for method in ("resnet18", "mobilenet_v3_small"):
            for seed in (17, 29, 43):
                result = evaluate_cnn_condition(cnn_models[(method, seed)], rows, paths, device=device, batch_size=int(data["evaluation"]["batch_size"]))
                entries.append({"method": method, "seed": seed, "condition_id": condition_id, "result": result})
    return entries, clean_reports


def _one_complete_pass(config: ResolvedConfig, root: Path, upstream: Mapping[str, Any]) -> dict[str, Any]:
    data = config.as_dict()
    summary_root = root / data["phase8"]["summary_root"]
    _safe_clean_summary(summary_root, root)
    entries, clean_reports = _evaluate_all(config, root)
    specs = condition_specs(data, include_clean=True)
    aggregates = aggregate_evaluations(entries, specs)
    _write_evaluation_tree(summary_root, entries)
    _write_aggregates(summary_root, aggregates, entries)
    write_csv(summary_root / "conditions.csv", specs, ("condition_id", "corruption_family", "severity", "parameter", "transformation_config_hash"))
    atomic_write_text(summary_root / "clean_reproduction" / "status.json", json_text({"status": "PASS", "checks": clean_reports}))
    atomic_write_text(summary_root / "resolved_config.yaml", config.to_yaml())
    dataset_summary = validate_corruption_dataset(config, root)
    figure_paths = [
        Path(path).resolve().relative_to(root).as_posix()
        for path in create_main_figures(aggregates, entries, root / data["phase8"]["figures_root"])
    ]
    manifest = {
        "status": "completed",
        "phase": 8,
        "result_id": data["phase8"]["result_id"],
        "completed_utc": format_utc(utc_now()),
        "upstream_gate": dict(upstream),
        "base_dataset_fingerprint": data["dataset"]["base_fingerprint"],
        "corruption_config_fingerprint": calculate_config_hash(data, length=64),
        "robustness_dataset_fingerprint": dataset_summary["robustness_dataset_fingerprint"],
        "pillow_environment": pillow_environment(),
        "clean_reproduction": clean_reports,
        "unique_conditions": len(specs),
        "degraded_conditions": len(specs) - 1,
        "corrupted_image_count": dataset_summary["corrupted_image_count"],
        "evaluation_entries": len(entries),
        "model_training_count": 0,
        "svm_or_scaler_refit_count": 0,
        "figures": figure_paths,
        "environment": capture_environment(data["runtime"]["device"], root),
        "phase9_started": False,
    }
    atomic_write_text(summary_root / "manifest.json", json_text(manifest))
    return {"manifest": manifest, "aggregates": aggregates, "entries": entries}


def _scientific_hashes(path: Path) -> dict[str, str]:
    excluded = {"manifest.json", "reproducibility.json"}
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name not in excluded
    }


def validate_robustness_results(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    data = config.as_dict()
    dataset = validate_corruption_dataset(config, repository)
    summary_root = repository / data["phase8"]["summary_root"]
    manifest = json.loads((summary_root / "manifest.json").read_text(encoding="utf-8"))
    reproduction = json.loads((summary_root / "reproducibility.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or reproduction.get("status") != "PASS":
        raise RobustnessRunError("Phase 8 manifest or full reproduction did not pass")
    expected = {
        "robustness_summary.csv": 52,
        "per_seed_results.csv": 104,
        "family_summary.csv": 16,
        "severe_summary.csv": 16,
        "overall_summary.csv": 4,
        "per_class_robustness.csv": 312,
        "prediction_flip_rates.csv": 52,
        "instance_robustness.csv": 15552,
    }
    for filename, count in expected.items():
        rows = read_csv(summary_root / "aggregate" / filename)
        if len(rows) != count:
            raise RobustnessRunError(f"{filename} has {len(rows)} rows; expected {count}")
    clean_status = json.loads((summary_root / "clean_reproduction" / "status.json").read_text(encoding="utf-8"))
    if clean_status.get("status") != "PASS" or len(clean_status.get("checks", [])) != 8:
        raise RobustnessRunError("clean reproduction record is incomplete")
    if len(list((repository / data["phase8"]["figures_root"]).glob("*.png"))) < 10:
        raise RobustnessRunError("required main Phase 8 figures are missing")
    return {
        "status": "PASS",
        "result_id": data["phase8"]["result_id"],
        "base_dataset_fingerprint": data["dataset"]["base_fingerprint"],
        "robustness_dataset_fingerprint": dataset["robustness_dataset_fingerprint"],
        "corrupted_image_count": dataset["corrupted_image_count"],
        "clean_reproduction_checks": 8,
        "full_reproduction": reproduction["status"],
    }


def run_robustness(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    data = config.as_dict()
    upstream = verify_upstream_gates(config, repository, require_clean_git=True)
    first = _one_complete_pass(config, repository, upstream)
    summary_root = repository / data["phase8"]["summary_root"]
    first_hashes = _scientific_hashes(summary_root)
    first_dataset = validate_corruption_dataset(config, repository)
    first_png_hashes = (repository / data["dataset"]["output_root"] / "corruption_checksum_manifest.csv").read_text(encoding="utf-8")
    second = _one_complete_pass(config, repository, upstream)
    second_hashes = _scientific_hashes(summary_root)
    second_dataset = validate_corruption_dataset(config, repository)
    second_png_hashes = (repository / data["dataset"]["output_root"] / "corruption_checksum_manifest.csv").read_text(encoding="utf-8")
    checks = {
        "all_1944_png_hashes_identical": first_png_hashes == second_png_hashes,
        "robustness_dataset_fingerprint_identical": first_dataset["robustness_dataset_fingerprint"] == second_dataset["robustness_dataset_fingerprint"],
        "scientific_output_file_set_identical": set(first_hashes) == set(second_hashes),
        "all_predictions_metrics_aggregates_identical": first_hashes == second_hashes,
    }
    if not all(checks.values()):
        raise RobustnessRunError(f"full Phase 8 deterministic regeneration failed: {checks}")
    record = {
        "status": "PASS", "passes": 2, "timing_values_excluded": True,
        "corrupted_png_count": 1944, "scientific_file_count": len(second_hashes), "checks": checks,
    }
    atomic_write_text(summary_root / "reproducibility.json", json_text(record))
    manifest = second["manifest"]
    manifest["full_reproducibility"] = record
    atomic_write_text(summary_root / "manifest.json", json_text(manifest))
    return validate_robustness_results(config, repository)
