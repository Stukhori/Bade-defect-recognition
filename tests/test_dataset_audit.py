from __future__ import annotations

import csv
from pathlib import Path

import pytest

from windblade.data.audit import (
    SplitFormatError,
    calculate_dataset_fingerprint,
    canonicalize_label,
    pair_image_and_annotation_ids,
    parse_official_split,
    validate_split_records,
    write_deterministic_csv,
)
from windblade.results import read_json, write_json


def test_pairing_reports_both_missing_directions_and_duplicates() -> None:
    result = pair_image_and_annotation_ids(
        ["0.jpg", "1.jpg", "1.jpeg", "2.jpg"],
        ["0.xml", "2.xml", "3.xml"],
    )

    assert result["matched_ids"] == ["0", "2"]
    assert result["images_without_xml"] == ["1"]
    assert result["xml_without_images"] == ["3"]
    assert result["duplicate_image_ids"] == {"1": ["1.jpeg", "1.jpg"]}


def test_raw_label_mapping_is_case_only_and_deterministic() -> None:
    assert canonicalize_label("Surface_injure") == "surface_injure"
    assert canonicalize_label("surface_injure") == "surface_injure"
    assert canonicalize_label("surface injury") is None
    assert canonicalize_label(" surface_injure") is None


def write_split(path: Path, body: str) -> None:
    path.write_text("ImageID,Subset\n" + body, encoding="utf-8")


def test_valid_disjoint_split_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    write_split(path, "0.jpg,train\n1.jpg,val\n2.jpg,test\n")
    records = parse_official_split(path)
    validation = validate_split_records(records, {"0", "1", "2"})

    assert [row["split"] for row in records] == ["train", "validation", "test"]
    assert validation["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert validation["overlap_ids"] == []
    assert validation["omitted_ids"] == []


def test_split_overlap_duplicate_omitted_and_unknown_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    write_split(path, "0.jpg,train\n0.jpg,test\n9.jpg,val\n")
    records = parse_official_split(path)
    validation = validate_split_records(records, {"0", "1"})

    assert validation["overlap_ids"] == ["0"]
    assert validation["duplicate_ids"] == ["0"]
    assert validation["omitted_ids"] == ["1"]
    assert validation["unknown_ids"] == ["9"]


def test_unrecognized_split_syntax_fails(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    write_split(path, "0.jpg,development\n")
    with pytest.raises(SplitFormatError, match="unknown split"):
        parse_official_split(path)


def test_dataset_fingerprint_is_order_independent_and_sensitive() -> None:
    first = [
        {"relative_path": "b.xml", "sha256": "bbb"},
        {"relative_path": "a.jpg", "sha256": "aaa"},
    ]
    reordered = list(reversed(first))
    changed = [dict(first[0]), {"relative_path": "a.jpg", "sha256": "changed"}]

    assert calculate_dataset_fingerprint(first) == calculate_dataset_fingerprint(reordered)
    assert calculate_dataset_fingerprint(first) != calculate_dataset_fingerprint(changed)


def test_csv_serialization_preserves_requested_deterministic_order(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    rows = [{"id": "0", "value": 1}, {"id": "1", "value": None}]
    write_deterministic_csv(path, rows, ["id", "value"])

    with path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"id": "0", "value": "1"},
            {"id": "1", "value": ""},
        ]
    assert path.read_text(encoding="utf-8").splitlines()[0] == "id,value"


def test_audit_summary_json_writes_and_reads_without_manual_transcription(tmp_path: Path) -> None:
    path = tmp_path / "audit_summary.json"
    summary = {
        "schema_version": "1.0",
        "dataset": "synthetic-fixture",
        "audit": {"status": "pass", "critical_errors": [], "warnings": []},
    }

    write_json(path, summary)

    assert read_json(path) == summary
