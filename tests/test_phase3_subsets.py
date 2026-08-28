from __future__ import annotations

from collections import Counter

import pytest

from windblade.data.processed import LABELS, ProcessedDatasetError
from windblade.data.subsets import select_nested_sources, subset_fingerprint, validate_nested_subsets


def _rows(source_count: int = 24):
    rows = []
    for source in range(source_count):
        labels = [LABELS[source % len(LABELS)]]
        if source % 4 == 0:
            labels.append(LABELS[(source + 1) % len(LABELS)])
        for object_index, label in enumerate(labels):
            rows.append(
                {
                    "source_image_id": str(source),
                    "instance_id": f"{source}_{object_index}",
                    "object_index": str(object_index),
                    "canonical_label": label,
                    "class_id": str(LABELS.index(label)),
                    "split": "train",
                }
            )
    return rows


def test_nested_grouped_selection_is_complete_and_deterministic():
    rows = _rows()
    targets = [12, 18, 21, 24]
    first = select_nested_sources(rows, targets=targets, seed=17)
    second = select_nested_sources(rows, targets=targets, seed=17)
    assert first == second
    validate_nested_subsets(first, [str(value) for value in range(24)], targets)
    for target in targets:
        selected = set(first[target])
        selected_rows = [row for row in rows if row["source_image_id"] in selected]
        assert {row["canonical_label"] for row in selected_rows} == set(LABELS)
        for source in selected:
            assert sum(row["source_image_id"] == source for row in selected_rows) == sum(
                row["source_image_id"] == source for row in rows
            )


def test_different_seeds_may_select_different_valid_sources():
    rows = _rows(30)
    targets = [12, 20, 25, 30]
    assert select_nested_sources(rows, targets=targets, seed=17)[12] != select_nested_sources(
        rows, targets=targets, seed=29
    )[12]


def test_group_objective_counts_every_instance():
    rows = _rows()
    snapshots = select_nested_sources(rows, targets=[12, 18, 21, 24], seed=43)
    selected = set(snapshots[12])
    counts = Counter(row["canonical_label"] for row in rows if row["source_image_id"] in selected)
    assert sum(counts.values()) == sum(
        1 for row in rows if row["source_image_id"] in selected
    )


def test_nontraining_input_fails():
    rows = _rows()
    rows[0]["split"] = "validation"
    with pytest.raises(ProcessedDatasetError):
        select_nested_sources(rows, targets=[12, 18, 21, 24], seed=17)


def test_bad_targets_fail():
    with pytest.raises(ProcessedDatasetError):
        select_nested_sources(_rows(), targets=[12, 18, 21, 23], seed=17)


def test_subset_fingerprint_changes_with_membership():
    original = "source_image_id,instance_id\n1,1_0\n"
    assert subset_fingerprint(original) == subset_fingerprint(original)
    assert subset_fingerprint(original) != subset_fingerprint(original.replace("1_0", "2_0"))
