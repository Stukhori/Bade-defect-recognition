"""Deterministic Phase 9B human-review ingestion and descriptive synthesis."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from windblade.config import ResolvedConfig, calculate_config_hash
from windblade.data.processed import read_csv, sha256_file
from windblade.error_analysis.core import (
    ErrorAnalysisError,
    bool_value,
    csv_write,
    fingerprint_mapping,
    json_write,
    tree_hashes,
)
from windblade_review.packet import load_review_packet
from windblade_review.schema import PassSchema, load_pass_schema
from windblade_review.store import ReviewDataError, ReviewStore


REVIEWER_ATTESTATION = (
    "I completed and saved every Pass A judgment before opening or completing Pass B. "
    "Both forms contain my own visual judgments. No review fields were filled automatically."
)
FORM_RELATIVES = {
    "pass_a": "human_review_packet/pass_a/pass_a_review_form.csv",
    "pass_b": "human_review_packet/pass_b/pass_b_review_form.csv",
}
CASE_GROUP_FIELDS = (
    "method",
    "corruption_family",
    "severity",
    "condition_group",
    "selection_event",
    "analysis_group",
    "eligibility_rule",
    "true_label",
    "selected_correctness",
)
PREDICTION_GROUP_FIELDS = (
    "method",
    "corruption_family",
    "severity",
    "selection_event",
    "analysis_group",
    "true_label",
    "predicted_label",
    "correct",
    "event_category",
    "confusion_pair",
)
SELECTION_GROUPS = {
    "clean_consensus_error": "clean_consensus_error",
    "severe_harmful_flip": "harmful_flip",
    "severe_stable_wrong": "stable_wrong",
    "severe_stable_correct": "stable_correct",
    "beneficial_flip": "beneficial_flip",
    "seed_disagreement": "seed_disagreement",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if result.returncode:
        raise ErrorAnalysisError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_attestation(phase9b: Mapping[str, Any]) -> None:
    """Require the exact reviewer-authorship statement and corrected packet confirmation."""

    if phase9b.get("reviewer_attestation") != REVIEWER_ATTESTATION:
        raise ErrorAnalysisError("the exact Phase 9B reviewer attestation is absent")
    if phase9b.get("reviewer_attestation_confirmed") is not True:
        raise ErrorAnalysisError("the Phase 9B reviewer attestation is not confirmed")
    if phase9b.get("reviewer_confirmed_corrected_pass_b") is not True:
        raise ErrorAnalysisError("corrected Pass B use is not reviewer-confirmed")


def _form_record(
    repository: Path,
    summary_root: Path,
    pass_name: str,
    expected_ids: tuple[str, ...],
) -> tuple[tuple[dict[str, str], ...], dict[str, Any], bytes, PassSchema]:
    schema = load_pass_schema(repository / "configs/error_analysis.yaml", pass_name)
    path = summary_root / FORM_RELATIVES[pass_name]
    before = path.read_bytes()
    store = ReviewStore(path, schema, expected_ids)
    try:
        snapshot = store.load(require_complete=True)
    except (OSError, ReviewDataError) as exc:
        raise ErrorAnalysisError(f"invalid completed {pass_name} review form: {exc}") from exc
    expected_total = 300 if pass_name == "pass_a" else 240
    if snapshot.total_required != expected_total or snapshot.answered_required != expected_total:
        raise ErrorAnalysisError(f"{pass_name} required-answer count is not {expected_total}")
    choice_counts = {
        field.name: dict(sorted(Counter(row[field.name] for row in snapshot.rows).items()))
        for field in schema.fields
        if field.required
    }
    try:
        recorded_path = path.relative_to(repository).as_posix()
    except ValueError:
        recorded_path = path.as_posix()
    record = {
        "path": recorded_path,
        "sha256": _sha256_bytes(before),
        "rows": len(snapshot.rows),
        "required_fields": len(schema.required_fields),
        "answered_required": snapshot.answered_required,
        "total_required": snapshot.total_required,
        "complete_cases": snapshot.completed_cases,
        "complete": snapshot.complete,
        "allowed_choices_valid": True,
        "review_id_sequence": list(expected_ids),
        "nonempty_notes": sum(bool(row[schema.notes_field].strip()) for row in snapshot.rows),
        "choice_counts": choice_counts,
    }
    if path.read_bytes() != before:
        raise ErrorAnalysisError(f"{pass_name} form changed while it was being validated")
    return snapshot.rows, record, before, schema


def validate_mapping_rows(
    mapping: Sequence[Mapping[str, str]],
    selected: Sequence[Mapping[str, str]],
    expected_ids: tuple[str, ...],
) -> None:
    """Require a one-to-one, ordered mapping to the frozen selected cases."""

    observed_ids = tuple(row.get("review_id", "") for row in mapping)
    selected_ids = tuple(row.get("review_id", "") for row in selected)
    if observed_ids != expected_ids or selected_ids != expected_ids:
        raise ErrorAnalysisError("review IDs or their frozen order changed")
    if len(set(observed_ids)) != len(observed_ids):
        raise ErrorAnalysisError("duplicate review IDs are forbidden")
    fields = (
        "review_id",
        "instance_id",
        "source_image_id",
        "method",
        "condition_id",
        "selection_event",
        "eligibility_rule",
        "true_label",
    )
    for mapped, chosen in zip(mapping, selected, strict=True):
        for field in fields:
            if str(mapped.get(field, "")) != str(chosen.get(field, "")):
                raise ErrorAnalysisError(
                    f"review mapping differs from frozen selection: {mapped.get('review_id')}/{field}"
                )


def _validate_blinding(repository: Path, summary_root: Path) -> dict[str, Any]:
    packet_root = summary_root / "human_review_packet"
    packet = load_review_packet(repository, packet_root, "pass_a")
    expected_assets = {
        "clean.png",
        "clean_annotation.png",
        "degraded.png",
        "degraded_annotation.png",
    }
    for case in packet.cases:
        if case.metadata != f"Dataset true label: {case.true_label}":
            raise ErrorAnalysisError(f"Pass A exposes unexpected metadata: {case.review_id}")
        if any(asset.path.name not in expected_assets for asset in case.assets):
            raise ErrorAnalysisError(f"Pass A exposes an unexpected asset: {case.review_id}")
        if any("gradcam" in asset.path.as_posix().lower() for asset in case.assets):
            raise ErrorAnalysisError(f"Pass A exposes Grad-CAM: {case.review_id}")
    return {
        "status": "PASS",
        "cases": len(packet.cases),
        "case_metadata_visible": ["review_id", "dataset_true_label"],
        "model_identity_hidden": True,
        "predictions_hidden": True,
        "correctness_hidden": True,
        "event_type_hidden": True,
        "gradcam_hidden": True,
        "mapping_hidden": True,
        "pass_b_opened_after_pass_a_by_reviewer_attestation": True,
    }


def _zip_entry_hash(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_zip_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ErrorAnalysisError(f"unsafe corrected Pass B archive path: {value}")
    return path.as_posix()


def validate_corrected_packet(
    repository: Path,
    phase9b: Mapping[str, Any],
    expected_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Bind the completed form to the corrected, blank Pass B packet."""

    expected = phase9b["corrected_pass_b"]
    archive_path = repository / str(expected["archive"])
    if not archive_path.is_file():
        raise ErrorAnalysisError("corrected Pass B archive is missing")
    archive_hash = sha256_file(archive_path)
    if archive_hash != str(expected["archive_sha256"]):
        raise ErrorAnalysisError("corrected Pass B archive hash changed")
    form_suffix = "human_review_packet/pass_b/pass_b_review_form.csv"
    index_suffix = "human_review_packet/pass_b/index.html"
    blank_form_bytes: bytes | None = None
    index_hash = ""
    compared = 0
    missing: list[str] = []
    mismatches: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        for info in infos:
            normalized = _normalized_zip_path(info.filename)
            if info.is_dir() or normalized.endswith("/"):
                continue
            digest = _zip_entry_hash(archive, info)
            if normalized.endswith(form_suffix):
                blank_form_bytes = archive.read(info)
                if digest != str(expected["blank_form_sha256"]):
                    raise ErrorAnalysisError("corrected packet blank form hash changed")
                continue
            if normalized.endswith(index_suffix):
                index_hash = digest
            repository_path = repository / normalized
            if not repository_path.is_file():
                missing.append(normalized)
            elif sha256_file(repository_path) != digest:
                mismatches.append(normalized)
            compared += 1
    if blank_form_bytes is None or not index_hash:
        raise ErrorAnalysisError("corrected Pass B archive lacks its canonical form or index")
    if index_hash != str(expected["index_sha256"]):
        raise ErrorAnalysisError("corrected Pass B index hash changed")
    if missing or mismatches:
        raise ErrorAnalysisError(
            f"corrected Pass B packet differs from repository: missing={missing[:3]}, changed={mismatches[:3]}"
        )
    reader = csv.DictReader(io.StringIO(blank_form_bytes.decode("utf-8")))
    blank_rows = list(reader)
    if tuple(row["review_id"] for row in blank_rows) != expected_ids:
        raise ErrorAnalysisError("corrected packet review IDs differ from completed Pass B")
    if any(value for row in blank_rows for key, value in row.items() if key != "review_id"):
        raise ErrorAnalysisError("corrected packet source form is not blank")
    return {
        "status": "PASS",
        "archive": str(expected["archive"]),
        "archive_sha256": archive_hash,
        "archive_file_count": compared + 1,
        "nonform_files_compared": compared,
        "nonform_missing": 0,
        "nonform_hash_mismatches": 0,
        "blank_form_sha256": str(expected["blank_form_sha256"]),
        "corrected_index_sha256": index_hash,
        "review_id_sequence_matches": True,
        "reviewer_confirmed_corrected_packet": True,
        "superseded_packet_used": False,
    }


def _validate_frozen_assets(repository: Path, summary_root: Path) -> dict[str, Any]:
    preflight = json.loads((summary_root / "preflight.json").read_text(encoding="utf-8"))
    phase9a_reproduction = json.loads(
        (summary_root / "reproducibility.json").read_text(encoding="utf-8")
    )
    input_mismatches: list[str] = []
    for relative, expected_hash in preflight["input_file_hashes"].items():
        path = repository / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            input_mismatches.append(relative)
    excluded = set(FORM_RELATIVES.values())
    phase9a_mismatches: list[str] = []
    checked = 0
    for relative, expected_hash in phase9a_reproduction["file_hashes"].items():
        summary_relative = relative.removeprefix(
            "experiments/summaries/phase9_error_analysis_v1/"
        )
        if summary_relative in excluded:
            continue
        path = repository / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            phase9a_mismatches.append(relative)
        checked += 1
    if input_mismatches or phase9a_mismatches:
        raise ErrorAnalysisError(
            "frozen Phase 3-9A assets changed: "
            f"inputs={input_mismatches[:3]}, phase9a={phase9a_mismatches[:3]}"
        )
    manifest = json.loads((summary_root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "status": "PASS",
        "frozen_input_files_checked": len(preflight["input_file_hashes"]),
        "frozen_input_mismatches": 0,
        "phase9a_nonform_files_checked": checked,
        "phase9a_nonform_mismatches": 0,
        "completed_forms_excluded_from_phase9a_blank_form_hash_check": True,
        "phase9a_input_fingerprint": preflight["input_fingerprint"],
        "phase9a_analysis_config_fingerprint": manifest["analysis_config_fingerprint"],
        "phase9a_corrected_output_fingerprint": phase9a_reproduction["output_fingerprint"],
        "checkpoints_unchanged": True,
        "predictions_and_logits_unchanged": True,
        "datasets_and_mappings_unchanged": True,
        "gradcam_records_and_renders_unchanged": True,
    }


def collect_phase9b_inputs(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    """Validate and load immutable human inputs plus their frozen join metadata."""

    repository = Path(root).resolve()
    data = config.as_dict()
    phase9b = data.get("phase9b")
    if not isinstance(phase9b, dict) or phase9b.get("version") != "phase9b_human_review_synthesis_v1":
        raise ErrorAnalysisError("missing or invalid Phase 9B configuration")
    validate_attestation(phase9b)
    if _git(repository, "branch", "--show-current") != "main":
        raise ErrorAnalysisError("Phase 9B requires local main")
    source_commit = str(phase9b["source_commit"])
    if _git(repository, "merge-base", "--is-ancestor", source_commit, "HEAD"):
        raise ErrorAnalysisError("the recorded Phase 9B source commit is not an ancestor of HEAD")

    summary_root = repository / data["outputs"]["summary_root"]
    selected = read_csv(summary_root / "tables/selected_exemplars.csv")
    expected_ids = tuple(row["review_id"] for row in selected)
    if expected_ids != tuple(f"P9A-{index:03d}" for index in range(1, 61)):
        raise ErrorAnalysisError("frozen selected review-ID sequence changed")
    mapping_path = summary_root / "human_review_packet/id_mapping/review_id_mapping.csv"
    mapping = read_csv(mapping_path)
    validate_mapping_rows(mapping, selected, expected_ids)
    mapping_hash = sha256_file(mapping_path)
    if mapping_hash != str(phase9b["corrected_pass_b"]["mapping_sha256"]):
        raise ErrorAnalysisError("review-ID mapping hash changed")

    pass_a, pass_a_record, pass_a_bytes, pass_a_schema = _form_record(
        repository, summary_root, "pass_a", expected_ids
    )
    pass_b, pass_b_record, pass_b_bytes, pass_b_schema = _form_record(
        repository, summary_root, "pass_b", expected_ids
    )
    blinding = _validate_blinding(repository, summary_root)
    corrected_packet = validate_corrected_packet(repository, phase9b, expected_ids)
    frozen_assets = _validate_frozen_assets(repository, summary_root)
    error_rows = read_csv(summary_root / "error_manifest.csv")
    return {
        "repository": repository,
        "summary_root": summary_root,
        "phase9b_config": phase9b,
        "selected": selected,
        "mapping": mapping,
        "expected_ids": expected_ids,
        "pass_a": pass_a,
        "pass_b": pass_b,
        "pass_a_schema": pass_a_schema,
        "pass_b_schema": pass_b_schema,
        "pass_a_bytes": pass_a_bytes,
        "pass_b_bytes": pass_b_bytes,
        "error_rows": error_rows,
        "form_records": {"pass_a": pass_a_record, "pass_b": pass_b_record},
        "mapping_record": {
            "status": "PASS",
            "path": mapping_path.relative_to(repository).as_posix(),
            "sha256": mapping_hash,
            "rows": len(mapping),
            "one_to_one": True,
            "ordered_review_ids_match": True,
            "selected_case_metadata_match": True,
            "opened_read_only": True,
        },
        "blinding_record": blinding,
        "corrected_packet_record": corrected_packet,
        "frozen_assets_record": frozen_assets,
    }


def _selected_predictions(
    selected: Mapping[str, str],
    error_index: Mapping[tuple[str, str, str, str], Mapping[str, str]],
) -> list[Mapping[str, str]]:
    rows: list[Mapping[str, str]] = []
    for seed in (17, 29, 43):
        key = (
            selected["method"],
            str(seed),
            selected["condition_id"],
            selected["instance_id"],
        )
        if key not in error_index:
            raise ErrorAnalysisError(f"selected review case lacks frozen prediction row: {key}")
        rows.append(error_index[key])
    return rows


def build_joined_review_tables(inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join case-level judgments to frozen case and per-seed prediction metadata."""

    error_index = {
        (row["method"], row["seed"], row["condition_id"], row["instance_id"]): row
        for row in inputs["error_rows"]
    }
    pass_a_by_id = {row["review_id"]: row for row in inputs["pass_a"]}
    pass_b_by_id = {row["review_id"]: row for row in inputs["pass_b"]}
    mapping_by_id = {row["review_id"]: row for row in inputs["mapping"]}
    case_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for selected in inputs["selected"]:
        review_id = selected["review_id"]
        mapped = mapping_by_id[review_id]
        predictions = _selected_predictions(selected, error_index)
        correct_values = [bool_value(row["correct"]) for row in predictions]
        selected_correctness = (
            "all_correct"
            if all(correct_values)
            else "all_incorrect"
            if not any(correct_values)
            else "mixed"
        )
        analysis_group = SELECTION_GROUPS.get(
            selected["selection_event"], selected["selection_event"]
        )
        base: dict[str, Any] = {
            "review_id": review_id,
            "instance_id": selected["instance_id"],
            "source_image_id": selected["source_image_id"],
            "method": selected["method"],
            "condition_id": selected["condition_id"],
            "condition_group": "clean" if selected["condition_id"] == "clean" else "degraded",
            "corruption_family": selected["corruption_family"],
            "severity": selected["severity"],
            "selection_event": selected["selection_event"],
            "analysis_group": analysis_group,
            "eligibility_rule": selected["eligibility_rule"],
            "true_label": selected["true_label"],
            "satisfying_seed_count": selected["satisfying_seed_count"],
            "selected_correctness": selected_correctness,
            "selected_prediction_labels": "|".join(
                f'{row["seed"]}:{row["predicted_label"]}' for row in predictions
            ),
            "selected_event_categories": "|".join(
                sorted({row["event_category"] for row in predictions})
            ),
            "clean_asset": mapped["clean_asset"],
            "degraded_asset": mapped["degraded_asset"],
        }
        for field in inputs["pass_a_schema"].fields:
            base[f"pass_a_{field.name}"] = pass_a_by_id[review_id][field.name]
        for field in inputs["pass_b_schema"].fields:
            base[f"pass_b_{field.name}"] = pass_b_by_id[review_id][field.name]
        case_rows.append(base)
        for prediction in predictions:
            prediction_rows.append(
                {
                    **base,
                    "seed": prediction["seed"],
                    "predicted_class_id": prediction["predicted_class_id"],
                    "predicted_label": prediction["predicted_label"],
                    "correct": str(bool_value(prediction["correct"])).lower(),
                    "event_category": prediction["event_category"],
                    "prediction_changed_from_clean": str(
                        bool_value(prediction["prediction_changed_from_clean"])
                    ).lower(),
                    "confusion_pair": f'{prediction["true_label"]}->{prediction["predicted_label"]}',
                }
            )
    if len(case_rows) != 60 or len(prediction_rows) != 180:
        raise ErrorAnalysisError("joined Phase 9B row counts are not 60 cases and 180 predictions")
    return case_rows, prediction_rows


def _response_specs(inputs: Mapping[str, Any]) -> list[tuple[str, PassSchema]]:
    return [
        ("pass_a", inputs["pass_a_schema"]),
        ("pass_b", inputs["pass_b_schema"]),
    ]


def build_response_summary(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pass_name, schema in _response_specs(inputs):
        values = inputs[pass_name]
        for field in schema.fields:
            if field.required:
                counts = Counter(row[field.name] for row in values)
                responses = field.choices
            else:
                counts = Counter(
                    "nonempty" if row[field.name].strip() else "empty" for row in values
                )
                responses = ("nonempty", "empty")
            for response in responses:
                count = counts[response]
                rows.append(
                    {
                        "pass_name": pass_name,
                        "field": field.name,
                        "required": str(field.required).lower(),
                        "response": response,
                        "count": count,
                        "denominator": len(values),
                        "percentage": f"{100.0 * count / len(values):.6f}",
                    }
                )
    return rows


def _response_choices(inputs: Mapping[str, Any]) -> list[tuple[str, str, tuple[str, ...]]]:
    result: list[tuple[str, str, tuple[str, ...]]] = []
    for pass_name, schema in _response_specs(inputs):
        for field in schema.fields:
            if field.required:
                result.append((pass_name, f"{pass_name}_{field.name}", field.choices))
    return result


def build_crosstabs(
    inputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    unit: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pass_name, response_field, choices in _response_choices(inputs):
        for group_field in group_fields:
            group_values = sorted({str(row[group_field]) for row in rows})
            for group_value in group_values:
                members = [row for row in rows if str(row[group_field]) == group_value]
                counts = Counter(str(row[response_field]) for row in members)
                for choice in choices:
                    count = counts[choice]
                    output.append(
                        {
                            "unit": unit,
                            "pass_name": pass_name,
                            "response_field": response_field.removeprefix(f"{pass_name}_"),
                            "response_value": choice,
                            "group_dimension": group_field,
                            "group_value": group_value,
                            "count": count,
                            "denominator": len(members),
                            "percentage": f"{100.0 * count / len(members):.6f}",
                        }
                    )
    return output


def _save_figure(figure: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "windblade-phase9b"},
    )
    plt.close(figure)
    return path.name


def _response_distribution_figure(
    summary: Sequence[Mapping[str, Any]], pass_name: str, path: Path
) -> str:
    fields = list(dict.fromkeys(row["field"] for row in summary if row["pass_name"] == pass_name and row["required"] == "true"))
    figure, axes = plt.subplots(len(fields), 1, figsize=(10, 2.2 * len(fields)))
    axes_list = list(axes) if hasattr(axes, "__len__") else [axes]
    for axis, field in zip(axes_list, fields, strict=True):
        values = [row for row in summary if row["pass_name"] == pass_name and row["field"] == field and row["required"] == "true"]
        labels = [str(row["response"]) for row in values]
        percentages = [float(row["percentage"]) for row in values]
        axis.barh(labels, percentages, color="#3b82f6" if pass_name == "pass_a" else "#7c3aed")
        axis.set_xlim(0, 100)
        axis.set_xlabel("Percent of 60 reviewed cases")
        axis.set_title(field.replace("_", " "))
        for index, percentage in enumerate(percentages):
            axis.text(min(percentage + 1, 96), index, f"{percentage:.1f}%", va="center", fontsize=8)
    figure.suptitle(f"{pass_name.replace('_', ' ').title()} response distributions")
    figure.tight_layout()
    return _save_figure(figure, path)


def _stacked_figure(
    rows: Sequence[Mapping[str, Any]],
    response_field: str,
    group_field: str,
    choices: Sequence[str],
    title: str,
    path: Path,
) -> str:
    groups = sorted({str(row[group_field]) for row in rows})
    figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(groups)), 5))
    bottoms = [0.0] * len(groups)
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#64748b", "#059669"]
    for choice, color in zip(choices, colors, strict=False):
        percentages: list[float] = []
        for group in groups:
            members = [row for row in rows if str(row[group_field]) == group]
            count = sum(str(row[response_field]) == choice for row in members)
            percentages.append(100.0 * count / len(members))
        axis.bar(groups, percentages, bottom=bottoms, label=choice, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, percentages, strict=True)]
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percent within group")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    return _save_figure(figure, path)


def generate_phase9b_figures(
    inputs: Mapping[str, Any],
    case_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    response_summary: Sequence[Mapping[str, Any]],
    figures_root: Path,
) -> list[str]:
    figures_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        _response_distribution_figure(
            response_summary, "pass_a", figures_root / "pass_a_response_distributions.png"
        ),
        _response_distribution_figure(
            response_summary, "pass_b", figures_root / "pass_b_response_distributions.png"
        ),
        _stacked_figure(
            case_rows,
            "pass_b_activation_primarily_inside_annotation",
            "analysis_group",
            inputs["pass_b_schema"].definition("activation_primarily_inside_annotation").choices,
            "Activation location by selected Phase 9 group",
            figures_root / "activation_location_by_group.png",
        ),
        _stacked_figure(
            case_rows,
            "pass_b_activation_concentrated_on_degradation_artifact",
            "corruption_family",
            inputs["pass_b_schema"].definition("activation_concentrated_on_degradation_artifact").choices,
            "Degradation-artifact activation by corruption family",
            figures_root / "artifact_activation_by_corruption.png",
        ),
        _stacked_figure(
            case_rows,
            "pass_b_pattern_consistent_across_cnn_seeds",
            "analysis_group",
            inputs["pass_b_schema"].definition("pattern_consistent_across_cnn_seeds").choices,
            "Seed-pattern consistency by selected Phase 9 group",
            figures_root / "seed_consistency_by_group.png",
        ),
    ]
    if len(prediction_rows) != 180:
        raise ErrorAnalysisError("prediction-level figure input changed")
    return outputs


def _runtime_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("matplotlib", "numpy", "Pillow", "PyYAML", "scikit-learn", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _summary_record(
    inputs: Mapping[str, Any], response_summary: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    distributions: dict[str, dict[str, dict[str, Any]]] = {}
    for row in response_summary:
        distributions.setdefault(str(row["pass_name"]), {}).setdefault(
            str(row["field"]), {}
        )[str(row["response"])] = {
            "count": int(row["count"]),
            "denominator": int(row["denominator"]),
            "percentage": str(row["percentage"]),
        }
    return {
        "status": "PASS",
        "reviewer_count": 1,
        "reviewed_cases": 60,
        "pass_a_required_answers": 300,
        "pass_b_required_answers": 240,
        "pass_a_nonempty_notes": inputs["form_records"]["pass_a"]["nonempty_notes"],
        "pass_b_nonempty_notes": inputs["form_records"]["pass_b"]["nonempty_notes"],
        "response_distributions": distributions,
        "analysis_type": "single-reviewer post-hoc descriptive synthesis",
        "inter_rater_agreement_calculated": False,
        "hypothesis_tests_calculated": False,
        "notes_systematically_coded": False,
        "gradcam_causal_claims_allowed": False,
        "cross_map_color_comparison_allowed": False,
        "real_flight_generalization_allowed": False,
    }


def _provenance_record(inputs: Mapping[str, Any]) -> dict[str, Any]:
    repository: Path = inputs["repository"]
    phase9b = inputs["phase9b_config"]
    return {
        "status": "PASS",
        "version": phase9b["version"],
        "incorporated_utc": phase9b["incorporated_utc"],
        "source_commit": phase9b["source_commit"],
        "apparatus_commit": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "reviewer_attestation": phase9b["reviewer_attestation"],
        "reviewer_attestation_confirmed": True,
        "forms": inputs["form_records"],
        "mapping": inputs["mapping_record"],
        "blinding": inputs["blinding_record"],
        "corrected_pass_b": inputs["corrected_packet_record"],
        "frozen_assets": inputs["frozen_assets_record"],
        "phase9b_config_fingerprint": calculate_config_hash(phase9b, length=64),
        "human_inputs_rewritten": False,
        "ai_comparison_workbook_used": False,
        "automated_visual_interpretations": 0,
        "training_runs": 0,
        "fine_tuning_runs": 0,
        "calibration_runs": 0,
        "svm_or_scaler_refits": 0,
        "downloads_or_external_research": 0,
        "phase10_started": False,
    }


def _write_phase9b_once(
    inputs: Mapping[str, Any], derived_root: Path, figures_root: Path
) -> dict[str, Any]:
    derived_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    case_rows, prediction_rows = build_joined_review_tables(inputs)
    response_summary = build_response_summary(inputs)
    case_crosstabs = build_crosstabs(inputs, case_rows, CASE_GROUP_FIELDS, "review_case")
    prediction_crosstabs = build_crosstabs(
        inputs, prediction_rows, PREDICTION_GROUP_FIELDS, "case_seed_prediction"
    )
    tables_root = derived_root / "tables"
    csv_write(tables_root / "joined_review_cases.csv", case_rows)
    csv_write(tables_root / "joined_review_predictions.csv", prediction_rows)
    csv_write(tables_root / "response_summary.csv", response_summary)
    csv_write(tables_root / "case_crosstabs.csv", case_crosstabs)
    csv_write(tables_root / "prediction_crosstabs.csv", prediction_crosstabs)
    figures = generate_phase9b_figures(
        inputs, case_rows, prediction_rows, response_summary, figures_root
    )
    provenance = _provenance_record(inputs)
    validation = {
        "status": "PASS",
        "forms": inputs["form_records"],
        "mapping": inputs["mapping_record"],
        "blinding": inputs["blinding_record"],
        "corrected_pass_b": inputs["corrected_packet_record"],
        "frozen_assets": inputs["frozen_assets_record"],
        "joined_case_rows": len(case_rows),
        "joined_prediction_rows": len(prediction_rows),
        "response_summary_rows": len(response_summary),
        "case_crosstab_rows": len(case_crosstabs),
        "prediction_crosstab_rows": len(prediction_crosstabs),
        "human_inputs_rewritten": False,
    }
    summary = _summary_record(inputs, response_summary)
    manifest = {
        "status": "PHASE 9 COMPLETE - VALIDATED AND FROZEN",
        "phase": "9B",
        "version": inputs["phase9b_config"]["version"],
        "phase9_complete": True,
        "phase9_frozen": True,
        "phase10_started": False,
        "reviewed_cases": 60,
        "joined_prediction_rows": 180,
        "human_reviewer_count": 1,
        "derived_tables": [
            "tables/joined_review_cases.csv",
            "tables/joined_review_predictions.csv",
            "tables/response_summary.csv",
            "tables/case_crosstabs.csv",
            "tables/prediction_crosstabs.csv",
        ],
        "figures": figures,
        "model_training_count": 0,
        "svm_or_scaler_refit_count": 0,
        "automated_visual_interpretations": 0,
    }
    json_write(derived_root / "provenance.json", provenance)
    json_write(derived_root / "validation.json", validation)
    json_write(derived_root / "summary.json", summary)
    json_write(derived_root / "runtime.json", _runtime_record())
    json_write(derived_root / "manifest.json", manifest)
    return manifest


def _inventory(
    derived_root: Path,
    figures_root: Path,
    derived_prefix: str,
    figures_prefix: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative, digest in tree_hashes(derived_root, excluded=("reproducibility.json",)).items():
        result[f"{derived_prefix.rstrip('/')}/{relative}"] = digest
    for relative, digest in tree_hashes(figures_root).items():
        result[f"{figures_prefix.rstrip('/')}/{relative}"] = digest
    return result


def _safe_reset(path: Path, repository: Path, expected_name: str, allowed_parent: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_parent.resolve()
    if resolved.name != expected_name or allowed not in resolved.parents:
        raise ErrorAnalysisError(f"unsafe Phase 9B derived-output root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=False)


def _assert_forms_unchanged(inputs: Mapping[str, Any]) -> None:
    summary_root: Path = inputs["summary_root"]
    for pass_name, before in (
        ("pass_a", inputs["pass_a_bytes"]),
        ("pass_b", inputs["pass_b_bytes"]),
    ):
        path = summary_root / FORM_RELATIVES[pass_name]
        if path.read_bytes() != before:
            raise ErrorAnalysisError(f"immutable {pass_name} form changed during Phase 9B")


def run_phase9b(config: ResolvedConfig, root: str | Path) -> dict[str, Any]:
    """Generate Phase 9B twice in separate scratch locations, then publish derived files."""

    repository = Path(root).resolve()
    inputs = collect_phase9b_inputs(config, repository)
    phase9b = inputs["phase9b_config"]
    derived_relative = str(phase9b["outputs"]["derived_root"])
    figures_relative = str(phase9b["outputs"]["figures_root"])
    canonical_derived = repository / derived_relative
    canonical_figures = repository / figures_relative
    scratch_root = repository / str(phase9b["outputs"]["reproduction_scratch_root"])
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pass_1_", dir=scratch_root) as first_name, tempfile.TemporaryDirectory(prefix="pass_2_", dir=scratch_root) as second_name:
        first_root = Path(first_name)
        second_root = Path(second_name)
        _write_phase9b_once(inputs, first_root / "derived", first_root / "figures")
        _assert_forms_unchanged(inputs)
        first = _inventory(
            first_root / "derived",
            first_root / "figures",
            derived_relative,
            figures_relative,
        )
        _write_phase9b_once(inputs, second_root / "derived", second_root / "figures")
        _assert_forms_unchanged(inputs)
        second = _inventory(
            second_root / "derived",
            second_root / "figures",
            derived_relative,
            figures_relative,
        )
        if first != second:
            membership = sorted(set(first) ^ set(second))
            changed = sorted(
                path for path in set(first) & set(second) if first[path] != second[path]
            )
            raise ErrorAnalysisError(
                f"Phase 9B two-pass mismatch: membership={membership[:5]}, changed={changed[:5]}"
            )
        _safe_reset(
            canonical_derived,
            repository,
            "phase9b",
            repository / "experiments/summaries/phase9_error_analysis_v1",
        )
        _safe_reset(
            canonical_figures,
            repository,
            "human_review",
            repository / "figures/phase9",
        )
        shutil.copytree(first_root / "derived", canonical_derived, dirs_exist_ok=True)
        shutil.copytree(first_root / "figures", canonical_figures, dirs_exist_ok=True)
    if scratch_root.is_dir() and not any(scratch_root.iterdir()):
        scratch_root.rmdir()
    canonical_inventory = _inventory(
        canonical_derived,
        canonical_figures,
        derived_relative,
        figures_relative,
    )
    if canonical_inventory != first:
        raise ErrorAnalysisError("published Phase 9B outputs differ from reproduced outputs")
    reproduction = {
        "status": "PASS",
        "passes": 2,
        "separate_temporary_locations": True,
        "validation_records_equal": True,
        "joined_tables_equal": True,
        "response_summaries_equal": True,
        "cross_tabulations_equal": True,
        "figures_equal": True,
        "output_inventory_equal": True,
        "file_count": len(canonical_inventory),
        "file_hashes": canonical_inventory,
        "phase9b_derived_output_fingerprint": fingerprint_mapping(canonical_inventory),
        "completed_form_hashes": {
            pass_name: record["sha256"]
            for pass_name, record in inputs["form_records"].items()
        },
        "mapping_sha256": inputs["mapping_record"]["sha256"],
        "phase9a_input_fingerprint": inputs["frozen_assets_record"]["phase9a_input_fingerprint"],
        "phase9b_config_fingerprint": calculate_config_hash(phase9b, length=64),
    }
    json_write(canonical_derived / "reproducibility.json", reproduction)
    _assert_forms_unchanged(inputs)
    validation = validate_phase9b(config, repository, validate_phase9a=True)
    manifest = json.loads((canonical_derived / "manifest.json").read_text(encoding="utf-8"))
    return {
        **manifest,
        "output_file_count": len(canonical_inventory),
        "output_fingerprint": reproduction["phase9b_derived_output_fingerprint"],
        "two_pass_reproduction": "PASS",
        "validation": validation["status"],
    }


def validate_phase9b(
    config: ResolvedConfig,
    root: str | Path,
    *,
    validate_phase9a: bool = True,
) -> dict[str, Any]:
    """Validate completed inputs, deterministic outputs, and Phase 9 freeze gates."""

    repository = Path(root).resolve()
    inputs = collect_phase9b_inputs(config, repository)
    phase9b = inputs["phase9b_config"]
    derived_root = repository / str(phase9b["outputs"]["derived_root"])
    figures_root = repository / str(phase9b["outputs"]["figures_root"])
    if not derived_root.is_dir() or not figures_root.is_dir():
        raise ErrorAnalysisError("Phase 9B derived outputs are absent")
    reproduction = json.loads(
        (derived_root / "reproducibility.json").read_text(encoding="utf-8")
    )
    if reproduction.get("status") != "PASS" or reproduction.get("passes") != 2:
        raise ErrorAnalysisError("Phase 9B two-pass reproduction did not pass")
    observed = _inventory(
        derived_root,
        figures_root,
        str(phase9b["outputs"]["derived_root"]),
        str(phase9b["outputs"]["figures_root"]),
    )
    if observed != reproduction.get("file_hashes"):
        raise ErrorAnalysisError("Phase 9B output inventory or hashes changed")
    if fingerprint_mapping(observed) != reproduction.get("phase9b_derived_output_fingerprint"):
        raise ErrorAnalysisError("Phase 9B output fingerprint changed")
    for pass_name, record in inputs["form_records"].items():
        if reproduction["completed_form_hashes"].get(pass_name) != record["sha256"]:
            raise ErrorAnalysisError(f"completed {pass_name} hash differs from reproduction record")
    cases = read_csv(derived_root / "tables/joined_review_cases.csv")
    predictions = read_csv(derived_root / "tables/joined_review_predictions.csv")
    responses = read_csv(derived_root / "tables/response_summary.csv")
    if len(cases) != 60 or len(predictions) != 180:
        raise ErrorAnalysisError("Phase 9B joined table row count changed")
    required_fields = sum(
        len(schema.required_fields) for _, schema in _response_specs(inputs)
    )
    for field_rows in (
        [row for row in responses if row["required"] == "true" and row["field"] == field]
        for _, schema in _response_specs(inputs)
        for field in schema.required_fields
    ):
        if sum(int(row["count"]) for row in field_rows) != 60:
            raise ErrorAnalysisError("Phase 9B response summary does not reconcile to 60 cases")
    if required_fields != 9:
        raise ErrorAnalysisError("frozen required review-field count changed")
    manifest = json.loads((derived_root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("phase9_complete") or not manifest.get("phase9_frozen"):
        raise ErrorAnalysisError("Phase 9B manifest does not freeze Phase 9")
    if manifest.get("phase10_started"):
        raise ErrorAnalysisError("Phase 10 was started")
    if len(manifest.get("figures", [])) != 5:
        raise ErrorAnalysisError("Phase 9B figure inventory changed")
    forbidden = [
        repository / "figures/phase11",
        repository / "figures/phase12",
        repository / "docs/phase11",
        repository / "docs/phase12",
        repository / "scripts/run_yolo.py",
        repository / "configs/yolo.yaml",
    ]
    if any(path.exists() for path in forbidden):
        raise ErrorAnalysisError("forbidden optional Phase 11/12 output exists")
    phase9a_status = "NOT_RUN"
    if validate_phase9a:
        from windblade.error_analysis.runner import validate_error_analysis

        phase9a_status = validate_error_analysis(config, repository)["status"]
    _assert_forms_unchanged(inputs)
    return {
        "status": "PASS",
        "phase9a_validator": phase9a_status,
        "pass_a_required_answers": inputs["form_records"]["pass_a"]["answered_required"],
        "pass_b_required_answers": inputs["form_records"]["pass_b"]["answered_required"],
        "mapping_integrity": "PASS",
        "corrected_pass_b_provenance": "PASS",
        "blinding_integrity": "PASS",
        "joined_case_rows": len(cases),
        "joined_prediction_rows": len(predictions),
        "response_summary_fields": required_fields,
        "two_pass_reproduction": "PASS",
        "output_file_count": len(observed),
        "output_fingerprint": reproduction["phase9b_derived_output_fingerprint"],
        "human_inputs_unchanged": True,
        "frozen_assets_unchanged": True,
        "phase9_complete": True,
        "phase9_frozen": True,
        "phase10_started": False,
    }
