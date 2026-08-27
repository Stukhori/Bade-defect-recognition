"""Deterministic, machine-readable result serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from windblade.utils import atomic_write_text


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, record: Mapping[str, Any]) -> None:
    """Write a mapping as stable, human-readable JSON without non-finite values."""

    if not isinstance(record, Mapping):
        raise TypeError("JSON result record must be a mapping")
    content = json.dumps(
        record,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    atomic_write_text(Path(path), content + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON record and require a mapping at the document root."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON record root must be an object")
    return value
