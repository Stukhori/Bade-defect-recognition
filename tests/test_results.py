from __future__ import annotations

from pathlib import Path

import pytest

from windblade.results import read_json, write_json


def test_results_json_round_trip_preserves_scalar_types(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    record = {
        "integer": 7,
        "floating": 2.5,
        "text": "synthetic",
        "flag": True,
        "missing": None,
        "nested": {"values": [1, "two", False, None]},
    }

    write_json(path, record)

    assert read_json(path) == record


def test_non_finite_numeric_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
