"""Deterministic Phase 9A input loading, event construction, and tabulation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from windblade.data.processed import LABELS, read_csv, sha256_file
from windblade.evaluation.metrics import classification_metrics


class ErrorAnalysisError(RuntimeError):
    """Raised when a frozen Phase 9A invariant is violated."""


METHODS = ("hog", "lbp", "resnet18", "mobilenet_v3_small")
CNN_METHODS = ("resnet18", "mobilenet_v3_small")
SEEDS = (17, 29, 43)
EVENTS = ("clean_only", "stable_correct", "harmful_flip", "beneficial_flip", "changed_wrong", "stable_wrong")
LOGIT_FIELDS = tuple(f"logit_{label}" for label in LABELS)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() not in {"true", "false"}:
        raise ErrorAnalysisError(f"invalid Boolean value: {value!r}")
    return str(value).lower() == "true"


def natural_instance_key(value: str) -> tuple[int, int, str]:
    source, _, object_index = value.partition("_")
    if source.isdigit() and object_index.isdigit():
        return int(source), int(object_index), value
    return 2**31 - 1, 2**31 - 1, value


def sample_sd(values: Sequence[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def event_category(clean_correct: bool, degraded_correct: bool, clean_prediction: str, degraded_prediction: str, condition: str) -> str:
    if condition == "clean":
        return "clean_only"
    if clean_correct and degraded_correct:
        return "stable_correct"
    if clean_correct and not degraded_correct:
        return "harmful_flip"
    if not clean_correct and degraded_correct:
        return "beneficial_flip"
    return "changed_wrong" if clean_prediction != degraded_prediction else "stable_wrong"


def occupancy_bin(value: float, bins: Sequence[Mapping[str, Any]]) -> str:
    for spec in bins:
        lower, upper = spec.get("lower"), spec.get("upper")
        if (lower is None or value >= float(lower)) and (upper is None or value < float(upper)):
            return str(spec["id"])
    raise ErrorAnalysisError(f"occupancy is outside frozen bins: {value}")


def condition_rows(config: Mapping[str, Any], root: Path) -> list[dict[str, str]]:
    path = root / config["inputs"]["phase8_summary_root"] / "conditions.csv"
    rows = read_csv(path)
    expected = list(config["evaluation"]["conditions"])
    if [row["condition_id"] for row in rows] != expected:
        raise ErrorAnalysisError("Phase 8 condition order or membership changed")
    return rows


def test_geometry(config: Mapping[str, Any], root: Path) -> dict[str, dict[str, str]]:
    rows = [row for row in read_csv(root / config["dataset"]["manifest"]) if row["split"] == "test"]
    if len(rows) != int(config["dataset"]["expected_test_instances"]):
        raise ErrorAnalysisError("frozen test instance count changed")
    if len({row["source_image_id"] for row in rows}) != int(config["dataset"]["expected_test_sources"]):
        raise ErrorAnalysisError("frozen test source count changed")
    identities = {row["instance_id"]: row for row in rows}
    if len(identities) != len(rows):
        raise ErrorAnalysisError("duplicate Phase 3 test sample ID")
    return identities


def prediction_path(config: Mapping[str, Any], root: Path, method: str, seed: int | None, condition: str) -> Path:
    replicate = "deterministic" if seed is None else f"seed_{seed}"
    return root / config["inputs"]["phase8_summary_root"] / method / replicate / condition / "predictions.csv"


def load_prediction_sets(config: Mapping[str, Any], root: Path) -> tuple[dict[tuple[str, int | None, str], dict[str, dict[str, str]]], dict[str, str]]:
    geometry = test_geometry(config, root)
    expected_ids = set(geometry)
    sets: dict[tuple[str, int | None, str], dict[str, dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        seeds: tuple[int | None, ...] = SEEDS if method in CNN_METHODS else (None,)
        for seed in seeds:
            for condition in config["evaluation"]["conditions"]:
                path = prediction_path(config, root, method, seed, condition)
                if not path.is_file():
                    raise ErrorAnalysisError(f"missing frozen prediction file: {path}")
                rows = read_csv(path)
                indexed = {row["instance_id"]: row for row in rows}
                if len(rows) != 162 or set(indexed) != expected_ids:
                    raise ErrorAnalysisError(f"prediction membership mismatch: {method}/{seed}/{condition}")
                expect_logits = method in CNN_METHODS
                if expect_logits and any(field not in rows[0] for field in LOGIT_FIELDS):
                    raise ErrorAnalysisError(f"CNN logits unavailable: {method}/{seed}/{condition}")
                for instance_id, row in indexed.items():
                    meta = geometry[instance_id]
                    if row["source_image_id"] != meta["source_image_id"] or row["true_label"] != meta["canonical_label"]:
                        raise ErrorAnalysisError(f"prediction identity mismatch: {instance_id}")
                    if int(row["true_class_id"]) != int(meta["class_id"]):
                        raise ErrorAnalysisError(f"prediction class ID mismatch: {instance_id}")
                key = (method, seed, str(condition))
                sets[key] = indexed
                hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    if len(sets) != 104:
        raise ErrorAnalysisError(f"expected 104 prediction sets; received {len(sets)}")
    return sets, hashes


def corruption_paths(config: Mapping[str, Any], root: Path) -> dict[tuple[str, str], str]:
    rows = read_csv(root / config["inputs"]["phase8_corruption_manifest"])
    result = {(row["instance_id"], f"{row['corruption_family']}_{row['severity']}"): row["corrupted_image_path"] for row in rows}
    if len(result) != 1944:
        raise ErrorAnalysisError("Phase 8 corruption manifest membership changed")
    return result


def build_error_manifest(config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    geometry = test_geometry(config, root)
    predictions, prediction_hashes = load_prediction_sets(config, root)
    conditions = {row["condition_id"]: row for row in condition_rows(config, root)}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        seeds: tuple[int | None, ...] = SEEDS if method in CNN_METHODS else (None,)
        for seed in seeds:
            clean = predictions[(method, seed, "clean")]
            for condition_id in config["evaluation"]["conditions"]:
                condition = conditions[condition_id]
                current = predictions[(method, seed, condition_id)]
                for instance_id in sorted(geometry, key=natural_instance_key):
                    meta, pred, clean_pred = geometry[instance_id], current[instance_id], clean[instance_id]
                    correct, clean_correct = bool_value(pred["correct"]), bool_value(clean_pred["correct"])
                    event = event_category(clean_correct, correct, clean_pred["predicted_label"], pred["predicted_label"], condition_id)
                    item: dict[str, Any] = {
                        "instance_id": instance_id,
                        "source_image_id": meta["source_image_id"],
                        "true_class_id": int(meta["class_id"]),
                        "true_label": meta["canonical_label"],
                        "crop_identifier": meta["processed_image_sha256"],
                        "crop_path": f"{config['dataset']['processed_root']}/{meta['output_relative_path']}",
                        "bbox_xmin": int(meta["bbox_xmin"]), "bbox_ymin": int(meta["bbox_ymin"]),
                        "bbox_xmax": int(meta["bbox_xmax"]), "bbox_ymax": int(meta["bbox_ymax"]),
                        "crop_xmin": int(meta["crop_xmin"]), "crop_ymin": int(meta["crop_ymin"]),
                        "crop_xmax": int(meta["crop_xmax"]), "crop_ymax": int(meta["crop_ymax"]),
                        "crop_side": int(meta["crop_side"]), "defect_occupancy": float(meta["defect_occupancy"]),
                        "occupancy_bin": occupancy_bin(float(meta["defect_occupancy"]), config["geometry"]["occupancy_bins"]),
                        "boundary_shifted": bool_value(meta["boundary_shifted"]),
                        "max_side_clipped": bool_value(meta["max_side_clipped"]),
                        "method": method, "seed": seed if seed is not None else "not_applicable",
                        "corruption_family": condition["corruption_family"], "severity": condition["severity"],
                        "condition_id": condition_id, "corruption_parameter": condition["parameter"],
                        "predicted_class_id": int(pred["predicted_class_id"]), "predicted_label": pred["predicted_label"],
                        "correct": correct, "clean_predicted_class_id": int(clean_pred["predicted_class_id"]),
                        "clean_predicted_label": clean_pred["predicted_label"], "clean_correct": clean_correct,
                        "prediction_changed_from_clean": pred["predicted_label"] != clean_pred["predicted_label"],
                        "event_category": event,
                        "score_type": "cnn_logits_and_uncalibrated_softmax" if method in CNN_METHODS else "not_recorded_for_svm",
                        "predicted_softmax": "",
                    }
                    if method in CNN_METHODS:
                        logits = np.asarray([float(pred[field]) for field in LOGIT_FIELDS], dtype=np.float64)
                        exp = np.exp(logits - logits.max())
                        item["predicted_softmax"] = float(exp[int(pred["predicted_class_id"])] / exp.sum())
                        item.update({field: float(pred[field]) for field in LOGIT_FIELDS})
                    else:
                        item.update({field: "" for field in LOGIT_FIELDS})
                    rows.append(item)
    expected = 162 * 13 * (2 + 2 * 3)
    if len(rows) != expected:
        raise ErrorAnalysisError(f"error manifest row count mismatch: {len(rows)} != {expected}")
    return rows, prediction_hashes


def confusion_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    dominant: list[dict[str, Any]] = []
    grouped: dict[tuple[str, Any, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), row["seed"], str(row["condition_id"]))].append(row)
    for (method, seed, condition), values in grouped.items():
        true = np.asarray([int(row["true_class_id"]) for row in values])
        predicted = np.asarray([int(row["predicted_class_id"]) for row in values])
        metrics = classification_metrics(true, predicted, LABELS)
        counts = np.asarray(metrics["confusion_matrix_counts"], dtype=int)
        normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=float)
        for true_id, true_label in enumerate(LABELS):
            support = int(counts[true_id].sum())
            incorrect = counts[true_id].copy()
            incorrect[true_id] = 0
            maximum = int(incorrect.max())
            destinations = [LABELS[index] for index, value in enumerate(incorrect) if value == maximum and maximum > 0]
            dominant.append({"method": method, "seed": seed, "condition_id": condition, "true_label": true_label, "support": support, "dominant_incorrect_destination": "|".join(destinations) if destinations else "none", "dominant_incorrect_count": maximum})
            for predicted_id, predicted_label in enumerate(LABELS):
                output.append({"method": method, "seed": seed, "condition_id": condition, "true_class_id": true_id, "true_label": true_label, "predicted_class_id": predicted_id, "predicted_label": predicted_label, "count": int(counts[true_id, predicted_id]), "row_normalized": float(normalized[true_id, predicted_id]), "true_class_support": support})
    return output, dominant


def transition_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    grouped: dict[tuple[str, Any, str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["condition_id"] != "clean":
            grouped[(str(row["method"]), row["seed"], str(row["condition_id"]), str(row["corruption_family"]), str(row["severity"]))][str(row["event_category"])] += 1
    for (method, seed, condition, family, severity), counts in grouped.items():
        total = sum(counts.values())
        flips = counts["harmful_flip"] + counts["beneficial_flip"] + counts["changed_wrong"]
        per_seed.append({"method": method, "seed": seed, "condition_id": condition, "corruption_family": family, "severity": severity, **{event: counts[event] for event in EVENTS[1:]}, "total_prediction_flips": flips, "denominator": total, "prediction_flip_rate": flips / total})
    aggregates: list[dict[str, Any]] = []
    by_condition: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_condition[(str(row["method"]), str(row["condition_id"]), str(row["corruption_family"]), str(row["severity"]))].append(row)
    numeric = (*EVENTS[1:], "total_prediction_flips", "prediction_flip_rate")
    for key, values in by_condition.items():
        item: dict[str, Any] = {"method": key[0], "condition_id": key[1], "corruption_family": key[2], "severity": key[3], "replicate_count": len(values), "denominator_per_replicate": 162}
        for field in numeric:
            numbers = [float(row[field]) for row in values]
            item[f"{field}_mean"] = mean(numbers)
            item[f"{field}_sample_sd"] = sample_sd(numbers)
        aggregates.append(item)
    return per_seed, aggregates


def cross_seed_stability(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] in CNN_METHODS:
            grouped[(str(row["method"]), str(row["condition_id"]), str(row["instance_id"]))].append(row)
    samples: list[dict[str, Any]] = []
    for (method, condition, instance_id), values in grouped.items():
        if len(values) != 3:
            raise ErrorAnalysisError("CNN seed group does not contain exactly three rows")
        correct_count = sum(bool(row["correct"]) for row in values)
        harmful_count = sum(row["event_category"] == "harmful_flip" for row in values)
        predictions = [str(row["predicted_label"]) for row in values]
        samples.append({"method": method, "condition_id": condition, "instance_id": instance_id, "source_image_id": values[0]["source_image_id"], "true_label": values[0]["true_label"], "correct_seed_count": correct_count, "harmful_flip_seed_count": harmful_count, "all_seeds_same_label": len(set(predictions)) == 1, "distinct_predicted_labels": len(set(predictions)), "unanimous_failure": correct_count == 0, "majority_only_failure": correct_count == 1, "seed_predictions": "|".join(f"{row['seed']}:{row['predicted_label']}" for row in sorted(values, key=lambda item: int(item["seed"])))})
    distributions: list[dict[str, Any]] = []
    dist_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        dist_groups[(str(row["method"]), str(row["condition_id"]))].append(row)
    for (method, condition), values in dist_groups.items():
        for correct_count in range(4):
            distributions.append({"method": method, "condition_id": condition, "measure": "correct_seed_count", "value": correct_count, "count": sum(int(row["correct_seed_count"]) == correct_count for row in values), "denominator": len(values)})
        for label, predicate in (("unanimous_failure", lambda row: bool(row["unanimous_failure"])), ("majority_only_failure", lambda row: bool(row["majority_only_failure"])), ("seed_disagreement", lambda row: not bool(row["all_seeds_same_label"]))):
            distributions.append({"method": method, "condition_id": condition, "measure": label, "value": True, "count": sum(predicate(row) for row in values), "denominator": len(values)})
    return samples, distributions


def _cnn_status(seed_rows: Sequence[Mapping[str, Any]], predicate: str, rule: str) -> bool:
    count = sum(bool(row[predicate]) for row in seed_rows)
    return count == 3 if rule == "strict" else count >= 2


def cross_method_agreement(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, Any, str, str], Mapping[str, Any]] = {(str(row["method"]), row["seed"], str(row["condition_id"]), str(row["instance_id"])): row for row in rows}
    instances = sorted({str(row["instance_id"]) for row in rows}, key=natural_instance_key)
    output: list[dict[str, Any]] = []
    for rule in ("strict", "majority"):
        for instance_id in instances:
            hog = indexed[("hog", "not_applicable", "clean", instance_id)]
            lbp = indexed[("lbp", "not_applicable", "clean", instance_id)]
            res = [indexed[("resnet18", seed, "clean", instance_id)] for seed in SEEDS]
            mobile = [indexed[("mobilenet_v3_small", seed, "clean", instance_id)] for seed in SEEDS]
            correct = {"hog": bool(hog["correct"]), "lbp": bool(lbp["correct"]), "resnet18": _cnn_status(res, "correct", rule), "mobilenet_v3_small": _cnn_status(mobile, "correct", rule)}
            categories = []
            if not any(correct.values()): categories.append("misclassified_by_all_four")
            if correct["resnet18"] and correct["mobilenet_v3_small"] and not correct["hog"] and not correct["lbp"]: categories.append("both_cnns_correct_both_handcrafted_wrong")
            if not correct["resnet18"] and not correct["mobilenet_v3_small"]: categories.append("missed_by_both_cnns")
            if sum(not value for value in correct.values()) == 1: categories.append("model_specific_failure")
            for category in categories:
                output.append({"agreement_rule": rule, "scope": "clean", "condition_id": "clean", "instance_id": instance_id, "true_label": hog["true_label"], "category": category, **{f"{method}_meets_correct_rule": correct[method] for method in METHODS}})
        for family in ("gaussian_blur", "resolution", "brightness", "jpeg"):
            condition = f"{family}_severe"
            for instance_id in instances:
                hog = indexed[("hog", "not_applicable", condition, instance_id)]
                lbp = indexed[("lbp", "not_applicable", condition, instance_id)]
                res = [indexed[("resnet18", seed, condition, instance_id)] for seed in SEEDS]
                mobile = [indexed[("mobilenet_v3_small", seed, condition, instance_id)] for seed in SEEDS]
                harmful = {"hog": hog["event_category"] == "harmful_flip", "lbp": lbp["event_category"] == "harmful_flip", "resnet18": sum(row["event_category"] == "harmful_flip" for row in res) == (3 if rule == "strict" else 2) if rule == "strict" else sum(row["event_category"] == "harmful_flip" for row in res) >= 2, "mobilenet_v3_small": sum(row["event_category"] == "harmful_flip" for row in mobile) == (3 if rule == "strict" else 2) if rule == "strict" else sum(row["event_category"] == "harmful_flip" for row in mobile) >= 2}
                if sum(harmful.values()) >= 2:
                    output.append({"agreement_rule": rule, "scope": "severe_harmful_flip", "condition_id": condition, "instance_id": instance_id, "true_label": hog["true_label"], "category": "shared_severe_harmful_flip", **{f"{method}_harmful": harmful[method] for method in METHODS}})
    return output


def geometry_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rates: list[dict[str, Any]] = []
    continuous: list[dict[str, Any]] = []
    for row in rows:
        continuous.append({key: row[key] for key in ("instance_id", "method", "seed", "condition_id", "true_label", "crop_side", "defect_occupancy", "boundary_shifted", "max_side_clipped", "correct", "event_category")})
    for variable in ("occupancy_bin", "boundary_shifted", "max_side_clipped"):
        grouped: dict[tuple[str, Any, str, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["method"]), row["seed"], str(row["condition_id"]), row[variable])].append(row)
        for (method, seed, condition, value), values in grouped.items():
            rates.append({"method": method, "seed": seed, "condition_id": condition, "geometry_variable": variable, "geometry_value": value, "sample_count": len(values), "error_count": sum(not bool(row["correct"]) for row in values), "error_rate": mean([not bool(row["correct"]) for row in values]), "harmful_flip_count": sum(row["event_category"] == "harmful_flip" for row in values), "harmful_flip_rate": mean([row["event_category"] == "harmful_flip" for row in values]) if condition != "clean" else "not_applicable"})
    return rates, continuous


def class_difficulty(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        seeds = list(SEEDS) if method in CNN_METHODS else ["not_applicable"]
        for label in LABELS:
            support = sum(row["true_label"] == label for row in method_rows if row["condition_id"] == "clean" and row["seed"] == seeds[0])
            clean_f1, clean_recall, degraded_f1, degraded_recall, severe_f1, severe_recall = [], [], [], [], [], []
            harmful_numerator = harmful_denominator = 0
            dominant_counts: Counter[str] = Counter()
            for seed in seeds:
                seed_rows = [row for row in method_rows if row["seed"] == seed]
                for condition in sorted({str(row["condition_id"]) for row in seed_rows}):
                    values = [row for row in seed_rows if row["condition_id"] == condition]
                    metrics = classification_metrics([int(row["true_class_id"]) for row in values], [int(row["predicted_class_id"]) for row in values], LABELS)["per_class"][label]
                    if condition == "clean": clean_f1.append(metrics["f1"]); clean_recall.append(metrics["recall"])
                    else:
                        degraded_f1.append(metrics["f1"]); degraded_recall.append(metrics["recall"])
                        class_values = [row for row in values if row["true_label"] == label]
                        harmful_numerator += sum(row["event_category"] == "harmful_flip" for row in class_values)
                        harmful_denominator += len(class_values)
                        for row in class_values:
                            if not row["correct"]: dominant_counts[str(row["predicted_label"])] += 1
                        if row_severity(condition) == "severe": severe_f1.append(metrics["f1"]); severe_recall.append(metrics["recall"])
            top = max(dominant_counts.values(), default=0)
            dominant = "|".join(sorted(label_name for label_name, count in dominant_counts.items() if count == top and top > 0)) or "none"
            seed_stability = "not_applicable"
            if method in CNN_METHODS:
                groups: dict[tuple[str, str], set[str]] = defaultdict(set)
                for row in method_rows:
                    if row["true_label"] == label: groups[(str(row["condition_id"]), str(row["instance_id"]))].add(str(row["predicted_label"]))
                seed_stability = mean([len(predictions) == 1 for predictions in groups.values()])
            result.append({"method": method, "class": label, "support": support, "clean_f1_mean": mean(clean_f1), "clean_f1_sample_sd": sample_sd(clean_f1), "clean_recall_mean": mean(clean_recall), "clean_recall_sample_sd": sample_sd(clean_recall), "mean_degraded_f1": mean(degraded_f1), "mean_degraded_recall": mean(degraded_recall), "mean_severe_f1": mean(severe_f1), "mean_severe_recall": mean(severe_recall), "harmful_flip_count": harmful_numerator, "harmful_flip_denominator": harmful_denominator, "harmful_flip_frequency": harmful_numerator / harmful_denominator, "dominant_degraded_confusion_destinations": dominant, "seed_prediction_unanimity_rate": seed_stability})
    return result


def row_severity(condition: str) -> str:
    return "clean" if condition == "clean" else condition.rsplit("_", 1)[1]


def select_exemplars(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cnn_rows = [row for row in rows if row["method"] in CNN_METHODS]
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cnn_rows:
        groups[(str(row["method"]), str(row["condition_id"]), str(row["instance_id"]))].append(row)
    candidates: list[dict[str, Any]] = []
    for (method, condition, instance_id), values in groups.items():
        if len(values) != 3:
            raise ErrorAnalysisError("candidate seed group is incomplete")
        family, severity = str(values[0]["corruption_family"]), str(values[0]["severity"])
        correct_count = sum(bool(row["correct"]) for row in values)
        event_counts = Counter(str(row["event_category"]) for row in values)
        event_types: list[str] = []
        if condition == "clean" and correct_count == 0: event_types.append("clean_consensus_error")
        if severity == "severe" and event_counts["harmful_flip"] >= 2: event_types.append("severe_harmful_flip")
        if severity == "severe" and event_counts["stable_wrong"] >= 2: event_types.append("severe_stable_wrong")
        if severity == "severe" and event_counts["stable_correct"] >= 2: event_types.append("severe_stable_correct")
        if condition != "clean" and event_counts["beneficial_flip"] >= 2: event_types.append("beneficial_flip")
        if len({str(row["predicted_label"]) for row in values}) > 1: event_types.append("seed_disagreement")
        for event in event_types:
            if event == "seed_disagreement":
                rule = "seed_disagreement"
                satisfying = 3
            elif event == "clean_consensus_error":
                rule, satisfying = "consensus", 3
            else:
                base = {"severe_harmful_flip": "harmful_flip", "severe_stable_wrong": "stable_wrong", "severe_stable_correct": "stable_correct", "beneficial_flip": "beneficial_flip"}[event]
                satisfying = event_counts[base]
                rule = "consensus" if satisfying == 3 else "majority"
            candidates.append({"selection_event": event, "method": method, "condition_id": condition, "corruption_family": family, "severity": severity, "instance_id": instance_id, "source_image_id": values[0]["source_image_id"], "true_label": values[0]["true_label"], "eligibility_rule": rule, "satisfying_seed_count": satisfying, "seed_predictions": "|".join(f"{row['seed']}:{row['predicted_label']}:{row['event_category']}" for row in sorted(values, key=lambda item: int(item["seed"])))})
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    class_order = list(config["selection"]["class_order"])
    priority = {name: index for index, name in enumerate(config["selection"]["eligibility_priority"])}
    shortfalls: dict[str, int] = {}
    for event, quota_value in config["selection"]["quotas"].items():
        quota = int(quota_value)
        eligible = [row for row in candidates if row["selection_event"] == event]
        eligible.sort(key=lambda row: (priority.get(str(row["eligibility_rule"]), 99), METHODS.index(str(row["method"])), str(row["corruption_family"]), class_order.index(str(row["true_label"])), natural_instance_key(str(row["instance_id"])), str(row["condition_id"])))
        by_class: dict[str, list[dict[str, Any]]] = {label: [] for label in class_order}
        for row in eligible: by_class[str(row["true_label"])].append(row)
        while len([row for row in selected if row["selection_event"] == event]) < quota:
            progress = False
            for label in class_order:
                while by_class[label]:
                    candidate = by_class[label].pop(0)
                    key = (str(candidate["instance_id"]), str(candidate["condition_id"]))
                    if key not in used:
                        selected.append(dict(candidate)); used.add(key); progress = True; break
                if len([row for row in selected if row["selection_event"] == event]) >= quota: break
            if not progress: break
        actual = sum(row["selection_event"] == event for row in selected)
        if actual < quota: shortfalls[event] = quota - actual
    selected.sort(key=lambda row: (list(config["selection"]["quotas"]).index(str(row["selection_event"])), natural_instance_key(str(row["instance_id"])), str(row["condition_id"]), str(row["method"])))
    for index, row in enumerate(selected, start=1): row["review_id"] = f"P9A-{index:03d}"
    return candidates, selected, {"target": int(config["selection"]["target_distinct_cases"]), "selected": len(selected), "shortfalls": shortfalls, "unique_sample_conditions": len(used), "algorithm": "class-balanced deterministic round robin with consensus before majority"}


def csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def tree_hashes(root: Path, *, excluded: Iterable[str] = ()) -> dict[str, str]:
    excluded_set = set(excluded)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in excluded_set}


def fingerprint_mapping(mapping: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(mapping.items())), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
