"""Deterministic nested, source-grouped WTBD training subsets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

from windblade.config import ResolvedConfig
from windblade.data.processed import LABELS, ProcessedDatasetError, csv_text, json_text, read_csv
from windblade.utils import atomic_write_text


SUBSET_FIELDS = ("source_image_id", "instance_id", "canonical_label", "class_id")


def _natural_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _group_training_rows(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, tuple[int, ...]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            raise ProcessedDatasetError("training subset constructor received non-training data")
        if row.get("canonical_label") not in labels:
            raise ProcessedDatasetError("training subset constructor received an unknown class")
        groups[str(row["source_image_id"])].append(row)
    vectors: dict[str, tuple[int, ...]] = {}
    for source_id, group in groups.items():
        counts = Counter(str(row["canonical_label"]) for row in group)
        vectors[source_id] = tuple(counts[label] for label in labels)
    return dict(groups), vectors


def select_nested_sources(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[int],
    seed: int,
    labels: Sequence[str] = LABELS,
) -> dict[int, tuple[str, ...]]:
    """Greedily approximate full-training class shares with nested source sets.

    At each target size, the existing selection is extended. Candidate source
    groups are scored by normalized squared deviation from the ideal class-count
    vector for that target. A seeded permutation provides deterministic ties and
    valid variation between scientific seeds.
    """

    groups, vectors = _group_training_rows(training_rows, labels)
    source_ids = sorted(groups, key=_natural_id)
    if not source_ids:
        raise ProcessedDatasetError("cannot construct subsets from an empty training set")
    ordered_targets = list(targets)
    if ordered_targets != sorted(set(ordered_targets)):
        raise ProcessedDatasetError("subset targets must be strictly increasing")
    if ordered_targets[-1] != len(source_ids) or ordered_targets[0] <= 0:
        raise ProcessedDatasetError("subset targets must end at the complete training source set")
    full_vector = tuple(sum(vectors[source][index] for source in source_ids) for index in range(len(labels)))
    if any(value == 0 for value in full_vector):
        raise ProcessedDatasetError("full training data omit a required class")

    tie_order = source_ids.copy()
    random.Random(seed).shuffle(tie_order)
    tie_rank = {source: index for index, source in enumerate(tie_order)}
    selected: set[str] = set()
    current = [0] * len(labels)
    snapshots: dict[int, tuple[str, ...]] = {}
    for target in ordered_targets:
        ideal = [value * target / len(source_ids) for value in full_vector]
        while len(selected) < target:
            best_source: str | None = None
            best_key: tuple[float, int] | None = None
            for source in source_ids:
                if source in selected:
                    continue
                candidate = [current[index] + vectors[source][index] for index in range(len(labels))]
                score = sum(
                    ((candidate[index] - ideal[index]) / max(ideal[index], 1.0)) ** 2
                    for index in range(len(labels))
                )
                # Before every class is represented, strongly prefer candidates
                # that fill a currently missing class. This is still based only
                # on the full training labels and keeps the objective transparent.
                missing_after = sum(candidate[index] == 0 for index in range(len(labels)))
                key = (missing_after * 1_000_000.0 + score, tie_rank[source])
                if best_key is None or key < best_key:
                    best_key, best_source = key, source
            if best_source is None:
                raise ProcessedDatasetError("greedy subset construction ran out of source groups")
            selected.add(best_source)
            current = [current[index] + vectors[best_source][index] for index in range(len(labels))]
        snapshot = tuple(sorted(selected, key=_natural_id))
        selected_classes = {
            str(row["canonical_label"]) for source in snapshot for row in groups[source]
        }
        if selected_classes != set(labels):
            raise ProcessedDatasetError(f"seed {seed} target {target} omits a required class")
        snapshots[target] = snapshot
    return snapshots


def subset_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_nested_subsets(
    snapshots: Mapping[int, Sequence[str]], full_sources: Sequence[str], targets: Sequence[int]
) -> None:
    prior: set[str] = set()
    for target in targets:
        current = set(snapshots[target])
        if len(current) != target:
            raise ProcessedDatasetError(f"subset target {target} has {len(current)} source images")
        if not prior < current and prior:
            raise ProcessedDatasetError("training source subsets are not strictly nested")
        prior = current
    if prior != set(full_sources):
        raise ProcessedDatasetError("100% subset does not equal the full training source set")


def build_training_subsets(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Rebuild all three scientific nested training-subset families."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    phase3 = data["crop_dataset"]
    output_root = (root / phase3["output_root"]).resolve()
    manifest_rows = read_csv(output_root / "manifest.csv")
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    groups, _ = _group_training_rows(train_rows, LABELS)
    full_sources = sorted(groups, key=_natural_id)
    fraction_targets = [(str(fraction), int(target)) for fraction, target in data["training_subsets"]["fractions"].items()]
    fraction_targets.sort(key=lambda item: float(item[0]))
    targets = [target for _, target in fraction_targets]
    if targets != [128, 255, 383, 510] or len(full_sources) != 510:
        raise ProcessedDatasetError("frozen training source targets or full source count changed")

    split_root = (root / phase3["split_root"]).resolve()
    expected_parent = (root / "data" / "splits" / phase3["version"]).resolve()
    if split_root != expected_parent:
        raise ProcessedDatasetError("refusing to write subsets outside the versioned split root")
    split_root.mkdir(parents=True, exist_ok=True)

    full_counts = Counter(row["canonical_label"] for row in train_rows)
    full_total = len(train_rows)
    summary_seeds: dict[str, Any] = {}
    for seed in [int(value) for value in data["training_subsets"]["seeds"]]:
        snapshots = select_nested_sources(train_rows, targets=targets, seed=seed)
        validate_nested_subsets(snapshots, full_sources, targets)
        seed_root = split_root / f"seed_{seed}"
        if seed_root.exists():
            shutil.rmtree(seed_root)
        seed_root.mkdir(parents=True)
        fraction_summaries: dict[str, Any] = {}
        for fraction, target in fraction_targets:
            selected = set(snapshots[target])
            selected_rows = sorted(
                (row for row in train_rows if row["source_image_id"] in selected),
                key=lambda row: (_natural_id(row["source_image_id"]), int(row["object_index"])),
            )
            manifest = [
                {
                    "source_image_id": row["source_image_id"],
                    "instance_id": row["instance_id"],
                    "canonical_label": row["canonical_label"],
                    "class_id": row["class_id"],
                }
                for row in selected_rows
            ]
            content = csv_text(manifest, SUBSET_FIELDS)
            suffix = f"{round(float(fraction) * 100):03d}"
            output_path = seed_root / f"train_{suffix}.csv"
            atomic_write_text(output_path, content)
            counts = Counter(row["canonical_label"] for row in selected_rows)
            if set(counts) != set(LABELS):
                raise ProcessedDatasetError(f"seed {seed} fraction {fraction} omits a class")
            if {row["source_image_id"] for row in selected_rows} != selected:
                raise ProcessedDatasetError("a selected source image has no selected instances")
            # Every selected source contributes every one of its training objects.
            for source in selected:
                expected_ids = {str(row["instance_id"]) for row in groups[source]}
                actual_ids = {
                    str(row["instance_id"])
                    for row in selected_rows
                    if str(row["source_image_id"]) == source
                }
                if actual_ids != expected_ids:
                    raise ProcessedDatasetError(f"source image {source} was partially selected")
            per_class = {}
            for label in LABELS:
                subset_share = counts[label] / len(selected_rows)
                full_share = full_counts[label] / full_total
                per_class[label] = {
                    "instance_count": counts[label],
                    "subset_share": subset_share,
                    "full_training_share": full_share,
                    "share_deviation": subset_share - full_share,
                }
            fraction_summaries[fraction] = {
                "target_source_image_count": target,
                "actual_source_image_count": len(selected),
                "instance_count": len(selected_rows),
                "instance_fraction_of_full_training": len(selected_rows) / full_total,
                "per_class": per_class,
                "distribution_deviation_l1": sum(
                    abs(per_class[label]["share_deviation"]) for label in LABELS
                ),
                "max_absolute_share_deviation": max(
                    abs(per_class[label]["share_deviation"]) for label in LABELS
                ),
                "manifest_relative_path": output_path.relative_to(root).as_posix(),
                "sha256": subset_fingerprint(content),
            }
        summary_seeds[str(seed)] = fraction_summaries

    summary = {
        "schema_version": "1.0",
        "dataset_version": phase3["version"],
        "selection_unit": "source_image",
        "algorithm": "nested_greedy_normalized_class_count_deviation_v1",
        "uses_validation_or_test": False,
        "full_training_source_images": len(full_sources),
        "full_training_instances": full_total,
        "full_training_class_counts": {label: full_counts[label] for label in LABELS},
        "seeds": summary_seeds,
        "nesting_validated": True,
        "all_subsets_contain_all_classes": True,
    }
    atomic_write_text((root / phase3["subset_summary_file"]).resolve(), json_text(summary))
    return summary


def validate_training_subsets(config: ResolvedConfig, repository_root: str | Path) -> dict[str, Any]:
    """Validate existing subset files, hashes, source grouping, and nesting."""

    root = Path(repository_root).resolve()
    data = config.as_dict()
    phase3 = data["crop_dataset"]
    summary_path = (root / phase3["subset_summary_file"]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_rows = read_csv((root / phase3["output_root"] / "manifest.csv").resolve())
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    groups, _ = _group_training_rows(train_rows, LABELS)
    full_sources = sorted(groups, key=_natural_id)
    targets = [128, 255, 383, 510]
    for seed, fractions in summary["seeds"].items():
        snapshots: dict[int, tuple[str, ...]] = {}
        for fraction, target in zip(("0.25", "0.50", "0.75", "1.00"), targets, strict=True):
            details = fractions[fraction]
            path = root / details["manifest_relative_path"]
            content = path.read_text(encoding="utf-8")
            if subset_fingerprint(content) != details["sha256"]:
                raise ProcessedDatasetError(f"subset fingerprint mismatch: seed {seed}, {fraction}")
            rows = read_csv(path)
            sources = tuple(sorted({row["source_image_id"] for row in rows}, key=_natural_id))
            snapshots[target] = sources
            for source in sources:
                expected = {str(row["instance_id"]) for row in groups[source]}
                actual = {row["instance_id"] for row in rows if row["source_image_id"] == source}
                if expected != actual:
                    raise ProcessedDatasetError(f"subset partially selects source {source}")
            if {row["canonical_label"] for row in rows} != set(LABELS):
                raise ProcessedDatasetError("subset omits a required class")
        validate_nested_subsets(snapshots, full_sources, targets)
    return summary
