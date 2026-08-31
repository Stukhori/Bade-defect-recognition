"""Canonical Phase 9A gate, generation, validation, and two-pass reproduction."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from windblade.config import ResolvedConfig, calculate_config_hash, load_config
from windblade.data.processed import LABELS, read_csv, sha256_file, validate_processed_dataset
from windblade.data_efficiency import validate_data_efficiency_results
from windblade.error_analysis.core import (
    CNN_METHODS, EVENTS, LOGIT_FIELDS, METHODS, SEEDS, ErrorAnalysisError,
    build_error_manifest, class_difficulty, condition_rows, confusion_tables,
    corruption_paths, cross_method_agreement, cross_seed_stability, csv_write,
    event_category, fingerprint_mapping, geometry_tables, json_write,
    load_prediction_sets, natural_instance_key, select_exemplars, test_geometry,
    transition_tables, tree_hashes,
)
from windblade.error_analysis.gradcam import generate_gradcams, validate_target_identity
from windblade.error_analysis.plots import create_figures
from windblade.error_analysis.review import create_review_packet, pass_b_caption_mismatches
from windblade.mobilenet_experiment import validate_mobilenet_results
from windblade.resnet_experiment import validate_resnet18_results
from windblade.robustness.runner import validate_robustness_results
from windblade.traditional import validate_traditional_results
from windblade_review.schema import load_pass_schema
from windblade_review.store import ReviewDataError, ReviewStore


ERROR_MANIFEST_FIELDS = (
    "instance_id", "source_image_id", "true_class_id", "true_label", "crop_identifier", "crop_path",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax",
    "crop_side", "defect_occupancy", "occupancy_bin", "boundary_shifted", "max_side_clipped",
    "method", "seed", "corruption_family", "severity", "condition_id", "corruption_parameter",
    "predicted_class_id", "predicted_label", "correct", "clean_predicted_class_id", "clean_predicted_label",
    "clean_correct", "prediction_changed_from_clean", "event_category", "score_type", "predicted_softmax", *LOGIT_FIELDS,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False)
    if result.returncode:
        raise ErrorAnalysisError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _phase8_correction_audit(root: Path, commit: str) -> dict[str, Any]:
    names = [line for line in _git(root, "diff", "--name-only", f"{commit}^", commit).splitlines() if line]
    diff = _git(root, "diff", "--unified=0", f"{commit}^", commit)
    if names != ["docs/phase8_robustness.md"]:
        raise ErrorAnalysisError("commit 681ed81 changed files beyond the Phase 8 report")
    prohibited = ("predictions.csv", "logit_", "corrupted_image", "macro_f1\"")
    if any(token in diff for token in prohibited):
        raise ErrorAnalysisError("Phase 8 per-class reporting correction appears to alter scientific artifacts")
    return {
        "commit": commit, "parent": _git(root, "rev-parse", f"{commit}^"), "changed_files": names,
        "changed_lines": {"insertions": 13, "deletions": 13},
        "correction": "documentation table now labels CNN per-class values as mean plus-or-minus sample SD and displays the already-generated SD values",
        "predictions_unchanged": True, "cnn_logits_unchanged": True, "corrupted_image_hashes_unchanged": True,
        "aggregate_macro_f1_unchanged": True, "labels_and_checkpoints_unchanged": True, "presentation_only": True,
    }


def _checkpoint_records(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    records = []
    for method in CNN_METHODS:
        for seed in SEEDS:
            path = root / config["inputs"]["checkpoints"][method] / f"seed_{seed}" / "best_state_dict.pt"
            metadata = path.with_suffix(".json")
            if not path.is_file() or not metadata.is_file():
                raise ErrorAnalysisError(f"missing frozen checkpoint: {method}/seed{seed}")
            record = json.loads(metadata.read_text(encoding="utf-8"))
            records.append({"method": method, "seed": seed, "path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "checkpoint_fingerprint": record["checkpoint_fingerprint"], "processed_dataset_fingerprint": record["processed_dataset_fingerprint"]})
    return records


def preflight(config: ResolvedConfig, root: str | Path, *, require_clean: bool) -> dict[str, Any]:
    repository = Path(root).resolve(); data = config.as_dict()
    phase3 = validate_processed_dataset(load_config(repository / "configs/crop_dataset.yaml"), repository)
    phase4 = validate_traditional_results(load_config(repository / "configs/traditional_baselines.yaml"), repository)
    phase5 = validate_resnet18_results(load_config(repository / "configs/resnet18_baseline.yaml"), repository)
    phase6 = validate_mobilenet_results(load_config(repository / "configs/mobilenet_v3_small_baseline.yaml"), repository)
    phase7 = validate_data_efficiency_results(load_config(repository / "configs/data_efficiency.yaml"), repository)
    phase8 = validate_robustness_results(load_config(repository / data["inputs"]["phase8_config"]), repository)
    statuses = {f"phase{index}": result.get("status") for index, result in zip(range(3, 9), (phase3, phase4, phase5, phase6, phase7, phase8), strict=True)}
    if set(statuses.values()) != {"PASS"}:
        raise ErrorAnalysisError(f"frozen Phase 3–8 validator failure: {statuses}")
    phase8_manifest = json.loads((repository / data["inputs"]["phase8_summary_root"] / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "base_dataset_fingerprint": data["dataset"]["processed_fingerprint"],
        "corruption_config_fingerprint": data["inputs"]["phase8_corruption_config_fingerprint"],
        "robustness_dataset_fingerprint": data["inputs"]["phase8_robustness_dataset_fingerprint"],
        "result_id": data["inputs"]["expected_phase8_version"],
    }
    for key, value in expected.items():
        if phase8_manifest.get(key) != value: raise ErrorAnalysisError(f"Phase 8 input mismatch: {key}")
    predictions, prediction_hashes = load_prediction_sets(data, repository)
    checkpoints = _checkpoint_records(data, repository)
    if any(row["processed_dataset_fingerprint"] != data["dataset"]["processed_fingerprint"] for row in checkpoints):
        raise ErrorAnalysisError("checkpoint dataset lineage changed")
    geometry = test_geometry(data, repository)
    corruptions = corruption_paths(data, repository)
    git_record: dict[str, Any] = {"branch": _git(repository, "branch", "--show-current"), "head": _git(repository, "rev-parse", "HEAD"), "origin_main": _git(repository, "rev-parse", "origin/main"), "clean_required": require_clean}
    if git_record["branch"] != "main": raise ErrorAnalysisError("Phase 9A requires local main")
    if _git(repository, "merge-base", "--is-ancestor", data["inputs"]["expected_phase8_commit"], "HEAD"):
        raise ErrorAnalysisError("expected Phase 8 handoff is not an ancestor of HEAD")
    if require_clean and _git(repository, "status", "--porcelain", "--untracked-files=normal"):
        raise ErrorAnalysisError("Phase 9A scientific generation requires a clean worktree")
    git_record["clean"] = not bool(_git(repository, "status", "--porcelain", "--untracked-files=normal"))
    input_hashes = dict(prediction_hashes)
    for row in checkpoints: input_hashes[str(row["path"])] = str(row["sha256"])
    for relative in (data["dataset"]["manifest"], data["inputs"]["phase8_corruption_manifest"], data["inputs"]["phase8_config"]):
        input_hashes[str(relative)] = sha256_file(repository / relative)
    return {
        "status": "PASS", "validators": statuses, "git": git_record,
        "processed_dataset_fingerprint": phase3["processed_dataset_fingerprint"],
        "phase8_corruption_config_fingerprint": phase8_manifest["corruption_config_fingerprint"],
        "phase8_robustness_dataset_fingerprint": phase8_manifest["robustness_dataset_fingerprint"],
        "prediction_sets": len(predictions), "prediction_rows": sum(len(value) for value in predictions.values()),
        "cnn_logit_sets": sum(key[0] in CNN_METHODS for key in predictions), "traditional_prediction_sets": sum(key[0] not in CNN_METHODS for key in predictions),
        "test_instances": len(geometry), "test_sources": len({row["source_image_id"] for row in geometry.values()}),
        "corrupted_sample_conditions": len(corruptions), "checkpoints": checkpoints,
        "input_file_hashes": input_hashes, "input_fingerprint": fingerprint_mapping(input_hashes),
        "phase8_reporting_correction": _phase8_correction_audit(repository, data["inputs"]["expected_phase8_commit"]),
        "model_training_count": 0, "svm_or_scaler_refit_count": 0,
    }


def apparatus_check(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    result = preflight(config, root, require_clean=False)
    result.update({"analysis_config_fingerprint": calculate_config_hash(config.as_dict(), length=64), "scientific_outputs_generated": 0, "status": "PASS"})
    return result


def _safe_clear(path: Path, repository: Path, expected_name: str, allowed_parent: Path) -> None:
    resolved, allowed = path.resolve(), allowed_parent.resolve()
    if resolved.name != expected_name or allowed not in resolved.parents:
        raise ErrorAnalysisError(f"unsafe Phase 9A output root: {resolved}")
    if resolved.exists(): shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=False)


def _candidate_index(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    body = "<p>Complete deterministic candidate inventory. This index contains metadata only and provides no automated visual interpretation.</p><table><tr>" + "".join(f"<th>{key}</th>" for key in candidates[0]) + "</tr>"
    for row in candidates: body += "<tr>" + "".join(f"<td>{str(row[key])}</td>" for key in row) + "</tr>"
    body += "</table>"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<!doctype html><meta charset=\"utf-8\"><title>Phase 9A candidate inventory</title><h1>Phase 9A candidate inventory</h1>{body}\n", encoding="utf-8")


def _generate_once(config: ResolvedConfig, repository: Path, preflight_record: Mapping[str, Any]) -> dict[str, Any]:
    data = config.as_dict(); summary_root = repository / data["outputs"]["summary_root"]; figures_root = repository / data["outputs"]["figures_root"]
    error_rows, prediction_hashes = build_error_manifest(data, repository)
    confusion, dominant = confusion_tables(error_rows)
    transitions_seed, transitions = transition_tables(error_rows)
    seed_samples, seed_distribution = cross_seed_stability(error_rows)
    agreement = cross_method_agreement(error_rows)
    difficulty = class_difficulty(error_rows)
    geometry_rates, geometry_continuous = geometry_tables(error_rows)
    candidates, selected, selection_summary = select_exemplars(error_rows, data)
    csv_write(summary_root / "error_manifest.csv", error_rows, ERROR_MANIFEST_FIELDS)
    tables = {
        "confusion_cells.csv": confusion, "dominant_confusions.csv": dominant,
        "error_transitions_per_seed.csv": transitions_seed, "error_transitions.csv": transitions,
        "cross_seed_samples.csv": seed_samples, "cross_seed_distributions.csv": seed_distribution,
        "cross_method_agreement.csv": agreement, "class_difficulty.csv": difficulty,
        "geometry_associations.csv": geometry_rates, "geometry_continuous.csv": geometry_continuous,
        "candidate_inventory.csv": candidates, "selected_exemplars.csv": selected,
    }
    for name, values in tables.items(): csv_write(summary_root / "tables" / name, values)
    _candidate_index(summary_root / "candidate_gallery" / "index.html", candidates)
    phase8_config = load_config(repository / data["inputs"]["phase8_config"]).as_dict()
    geometry = test_geometry(data, repository); corrupted = corruption_paths(data, repository)
    gradcam, model_checks = generate_gradcams(data, phase8_config, repository, selected, error_rows, geometry, corrupted, figures_root)
    csv_write(summary_root / "gradcam" / "gradcam_manifest.csv", gradcam)
    json_write(summary_root / "gradcam" / "model_integrity.json", {"status": "PASS", "checks": model_checks, "maps_independently_normalized": True, "cross_map_color_comparison_allowed": False})
    review = create_review_packet(data, repository, selected, geometry, corrupted, gradcam, figures_root, summary_root)
    figure_paths = [Path(path).relative_to(repository).as_posix() for path in create_figures(confusion, transitions, difficulty, geometry_rates, figures_root)]
    json_write(summary_root / "preflight.json", preflight_record)
    json_write(summary_root / "selection_summary.json", selection_summary)
    json_write(summary_root / "resolved_config.json", data)
    manifest = {
        "status": "PHASE 9A COMPLETE — AWAITING HUMAN REVIEW", "phase": "9A", "result_id": data["outputs"]["result_id"],
        "schema_version": data["outputs"]["schema_version"], "analysis_config_fingerprint": calculate_config_hash(data, length=64),
        "input_fingerprint": preflight_record["input_fingerprint"], "processed_dataset_fingerprint": data["dataset"]["processed_fingerprint"],
        "phase8_corruption_config_fingerprint": data["inputs"]["phase8_corruption_config_fingerprint"],
        "phase8_robustness_dataset_fingerprint": data["inputs"]["phase8_robustness_dataset_fingerprint"],
        "error_manifest_rows": len(error_rows), "prediction_sets": 104, "selected_exemplars": len(selected),
        "gradcam_records": len(gradcam), "review_packet": review, "quantitative_figures": figure_paths,
        "model_training_count": 0, "svm_or_scaler_refit_count": 0, "automated_visual_interpretations": 0,
        "human_review_status": "awaiting", "phase9_complete": False, "phase10_started": False,
        "frozen_prediction_hashes": prediction_hashes,
    }
    json_write(summary_root / "manifest.json", manifest)
    return manifest


def _generated_hashes(repository: Path, summary_root: Path, figures_root: Path) -> dict[str, str]:
    result = {}
    for base in (summary_root, figures_root):
        for relative, digest in tree_hashes(base, excluded=("reproducibility.json",)).items():
            result[f"{base.relative_to(repository).as_posix()}/{relative}"] = digest
    return result


def _completed_review_data_exist(summary_root: Path) -> bool:
    for relative in (
        "human_review_packet/pass_a/pass_a_review_form.csv",
        "human_review_packet/pass_b/pass_b_review_form.csv",
    ):
        path = summary_root / relative
        if path.is_file() and any(
            any(str(value).strip() for key, value in row.items() if key != "review_id")
            for row in read_csv(path)
        ):
            return True
    return False


def run_error_analysis(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve(); data = config.as_dict()
    gate = preflight(config, repository, require_clean=True)
    summary_root, figures_root = repository / data["outputs"]["summary_root"], repository / data["outputs"]["figures_root"]
    if _completed_review_data_exist(summary_root):
        raise ErrorAnalysisError(
            "refusing Phase 9A regeneration because completed human-review inputs exist"
        )
    _safe_clear(summary_root, repository, "phase9_error_analysis_v1", repository / "experiments" / "summaries")
    _safe_clear(figures_root, repository, "phase9", repository / "figures")
    _generate_once(config, repository, gate)
    first = _generated_hashes(repository, summary_root, figures_root)
    _safe_clear(summary_root, repository, "phase9_error_analysis_v1", repository / "experiments" / "summaries")
    _safe_clear(figures_root, repository, "phase9", repository / "figures")
    manifest = _generate_once(config, repository, gate)
    second = _generated_hashes(repository, summary_root, figures_root)
    if first != second:
        missing = sorted(set(first) ^ set(second)); changed = sorted(path for path in set(first) & set(second) if first[path] != second[path])
        raise ErrorAnalysisError(f"Phase 9A two-pass mismatch; membership={missing[:5]}, changed={changed[:5]}")
    reproduction = {"status": "PASS", "passes": 2, "file_count": len(second), "machine_readable_manifests_equal": True, "numerical_tables_equal": True, "selected_exemplar_ids_equal": True, "gradcam_canonical_hashes_equal": True, "rendered_figure_hashes_equal": True, "output_inventory_equal": True, "output_fingerprint": fingerprint_mapping(second), "file_hashes": second}
    json_write(summary_root / "reproducibility.json", reproduction)
    validation = validate_error_analysis(config, repository)
    return {**manifest, "reproducibility": reproduction["status"], "output_fingerprint": reproduction["output_fingerprint"], "validation": validation["status"]}


def _review_form_status(
    repository: Path,
    summary_root: Path,
    expected_ids: tuple[str, ...],
) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for pass_name, total in (("pass_a", 300), ("pass_b", 240)):
        schema = load_pass_schema(repository / "configs/error_analysis.yaml", pass_name)
        filename = f"{pass_name}_review_form.csv"
        store = ReviewStore(
            summary_root / "human_review_packet" / pass_name / filename,
            schema,
            expected_ids,
        )
        try:
            snapshot = store.load()
        except ReviewDataError as exc:
            raise ErrorAnalysisError(f"invalid {pass_name} human-review form: {exc}") from exc
        if snapshot.total_required != total:
            raise ErrorAnalysisError(f"unexpected {pass_name} required-answer count")
        statuses[pass_name] = {
            "answered_required": snapshot.answered_required,
            "total_required": snapshot.total_required,
            "complete": snapshot.complete,
            "sha256": store.sha256(),
        }
    if statuses["pass_a"]["answered_required"] not in {0, 300}:
        raise ErrorAnalysisError("Pass A must be either pristine or complete at this validation gate")
    if statuses["pass_b"]["answered_required"] not in {0, 240}:
        raise ErrorAnalysisError("Pass B must be either pristine or complete at this validation gate")
    if statuses["pass_b"]["complete"] and not statuses["pass_a"]["complete"]:
        raise ErrorAnalysisError("Pass B cannot be complete unless Pass A is complete")
    return statuses


def _forbidden_paths(root: Path) -> list[str]:
    candidates = [root / "figures/phase10", root / "docs/phase10", root / "src/windblade/app", root / "scripts/run_yolo.py", root / "configs/yolo.yaml"]
    return [path.relative_to(root).as_posix() for path in candidates if path.exists()]


def validate_error_analysis(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve(); data = config.as_dict(); summary_root = repository / data["outputs"]["summary_root"]
    if not summary_root.is_dir(): raise ErrorAnalysisError("Phase 9A output root is absent")
    manifest = json.loads((summary_root / "manifest.json").read_text(encoding="utf-8"))
    reproduction = json.loads((summary_root / "reproducibility.json").read_text(encoding="utf-8")) if (summary_root / "reproducibility.json").is_file() else None
    expected_rows, _ = build_error_manifest(data, repository)
    observed = read_csv(summary_root / "error_manifest.csv")
    if len(observed) != len(expected_rows) or len(observed) != 16848: raise ErrorAnalysisError("error manifest completeness failure")
    expected_index = {(str(row["method"]), str(row["seed"]), str(row["condition_id"]), str(row["instance_id"])): row for row in expected_rows}
    for row in observed:
        key = (row["method"], row["seed"], row["condition_id"], row["instance_id"])
        if key not in expected_index: raise ErrorAnalysisError(f"unknown Phase 9A prediction identity: {key}")
        expected = expected_index[key]
        for field in ("source_image_id", "true_label", "predicted_label", "clean_predicted_label", "event_category"):
            if row[field] != str(expected[field]): raise ErrorAnalysisError(f"frozen prediction mismatch: {key}/{field}")
        if row["event_category"] not in EVENTS: raise ErrorAnalysisError("unknown event category")
    confusion = read_csv(summary_root / "tables/confusion_cells.csv")
    matrix_groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in confusion: matrix_groups[(row["method"], row["seed"], row["condition_id"])].append(row)
    if len(matrix_groups) != 104 or any(sum(int(row["count"]) for row in values) != 162 for values in matrix_groups.values()):
        raise ErrorAnalysisError("confusion matrices do not reconcile with supports")
    transition_rows = read_csv(summary_root / "tables/error_transitions_per_seed.csv")
    if len(transition_rows) != 96:
        raise ErrorAnalysisError("transition condition count mismatch")
    if any(sum(int(float(row[event])) for event in EVENTS[1:]) != int(row["denominator"]) for row in transition_rows):
        raise ErrorAnalysisError("transition counts do not reconcile")
    _, selected_expected, selection_summary = select_exemplars(expected_rows, data)
    selected_observed = read_csv(summary_root / "tables/selected_exemplars.csv")
    expected_keys = [(str(row["review_id"]), str(row["instance_id"]), str(row["condition_id"]), str(row["method"])) for row in selected_expected]
    observed_keys = [(row["review_id"], row["instance_id"], row["condition_id"], row["method"]) for row in selected_observed]
    if expected_keys != observed_keys or len({row["review_id"] for row in selected_observed}) != len(selected_observed):
        raise ErrorAnalysisError("deterministic exemplar selection mismatch")
    mapping = read_csv(summary_root / "human_review_packet/id_mapping/review_id_mapping.csv")
    if len(mapping) != len(selected_observed) or {row["review_id"] for row in mapping} != {row["review_id"] for row in selected_observed}:
        raise ErrorAnalysisError("human-review mapping mismatch")
    gradcam = read_csv(summary_root / "gradcam/gradcam_manifest.csv")
    if not gradcam or any(row["finite"] != "True" or len(row["array_sha256"]) != 64 for row in gradcam): raise ErrorAnalysisError("invalid Grad-CAM evidence")
    gradcam_groups: dict[tuple[str, str, str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in gradcam:
        key = (row["method"], row["seed"], row["input_condition_id"], row["instance_id"])
        if key not in expected_index:
            raise ErrorAnalysisError(f"unknown Grad-CAM prediction identity: {key}")
        expected = expected_index[key]
        for field in ("true_label", "predicted_label"):
            if row[field] != str(expected[field]):
                raise ErrorAnalysisError(f"Grad-CAM frozen metadata mismatch: {key}/{field}")
        validate_target_identity(
            row["target_role"], int(row["target_class_id"]), row["target_label"], expected
        )
        expected_token = f'{row["target_role"]}_{row["target_label"]}'
        if expected_token not in Path(row["heatmap_path"]).stem or expected_token not in Path(row["overlay_path"]).stem:
            raise ErrorAnalysisError(f"Grad-CAM target filename mismatch: {key}/{row['target_role']}")
        group_key = (
            row["review_id"], row["method"], row["seed"], row["input_condition_id"],
            row["instance_id"], row["input_state"],
        )
        gradcam_groups[group_key].append(row)
    for group_key, rows in gradcam_groups.items():
        key = (group_key[1], group_key[2], group_key[3], group_key[4])
        expected = expected_index[key]
        roles = [row["target_role"] for row in rows]
        expected_roles = ["true_class"]
        if str(expected["predicted_label"]) != str(expected["true_label"]):
            expected_roles.append("predicted_class")
        if sorted(roles) != sorted(expected_roles):
            raise ErrorAnalysisError(f"Grad-CAM target-role coverage mismatch: {group_key}")
    if any(not (repository / row[path_field]).is_file() for row in gradcam for path_field in ("input_path", "annotation_path", "heatmap_path", "overlay_path")):
        raise ErrorAnalysisError("missing Grad-CAM render")
    pass_b_path = summary_root / "human_review_packet/pass_b/index.html"
    caption_mismatches = pass_b_caption_mismatches(
        pass_b_path.read_text(encoding="utf-8"),
        gradcam,
        repository,
        summary_root / "human_review_packet",
    )
    if caption_mismatches:
        raise ErrorAnalysisError(
            f"Pass B Grad-CAM caption mismatch: {caption_mismatches[0]}"
        )
    review_ids = tuple(row["review_id"] for row in selected_observed)
    form_status = _review_form_status(repository, summary_root, review_ids)
    integrity = json.loads((summary_root / "gradcam/model_integrity.json").read_text(encoding="utf-8"))
    if any(not row["parameters_unchanged"] or row["state_dict_before"] != row["state_dict_after"] for row in integrity["checks"]):
        raise ErrorAnalysisError("Grad-CAM checkpoint-integrity failure")
    current_checkpoints = _checkpoint_records(data, repository)
    preflight_record = json.loads((summary_root / "preflight.json").read_text(encoding="utf-8"))
    if [(row["method"], row["seed"], row["sha256"]) for row in current_checkpoints] != [(row["method"], row["seed"], row["sha256"]) for row in preflight_record["checkpoints"]]:
        raise ErrorAnalysisError("checkpoint hash changed")
    if manifest["input_fingerprint"] != preflight_record["input_fingerprint"]: raise ErrorAnalysisError("input fingerprint changed")
    if reproduction is not None and reproduction.get("status") != "PASS": raise ErrorAnalysisError("two-pass reproduction did not pass")
    forbidden = _forbidden_paths(repository)
    if forbidden: raise ErrorAnalysisError(f"forbidden Phase 10/app paths exist: {forbidden}")
    return {"status": "PASS", "error_manifest_rows": len(observed), "prediction_sets": len(matrix_groups), "transition_rows": len(transition_rows), "selected_exemplars": len(selected_observed), "selection_shortfalls": selection_summary["shortfalls"], "gradcam_records": len(gradcam), "gradcam_target_identities": "PASS", "pass_b_captions": "PASS", "review_forms_blank": all(status["answered_required"] == 0 for status in form_status.values()), "review_forms_complete": all(status["complete"] for status in form_status.values()), "review_forms": form_status, "checkpoints_unchanged": True, "predictions_unchanged": True, "input_fingerprints_unchanged": True, "two_pass_reproduction": reproduction["status"] if reproduction else "PENDING", "phase10_or_app_paths": 0}
