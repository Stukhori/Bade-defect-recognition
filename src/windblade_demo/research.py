"""Read-only access to frozen Phase 10 application summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PHASE10_FINGERPRINT = "6064922c936a05c33c38068ba86fa68c6b9b7f931d28df4e37a5e880edd5dbf0"
TABLES = (
    "clean_method_comparison.csv",
    "data_efficiency_summary.csv",
    "robustness_retention_summary.csv",
    "error_human_review_summary.csv",
)


class FrozenResearchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phase10(root: str | Path) -> dict[str, Any]:
    base = Path(root) / "experiments/summaries/phase10_final_synthesis_v1"
    repro_path = base / "reproducibility.json"
    summary_path = base / "summary.json"
    if not repro_path.is_file() or not summary_path.is_file():
        raise FrozenResearchError("The verified research-summary sources could not be loaded.")
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if repro.get("phase10_scientific_output_fingerprint") != PHASE10_FINGERPRINT:
        raise FrozenResearchError("The research-summary fingerprint does not match the application contract.")
    loaded: dict[str, Any] = {}
    for name in TABLES:
        path = base / "tables" / name
        expected = repro.get("inventory", {}).get(f"tables/{name}")
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise FrozenResearchError(f"A research-summary table failed verification: {name}")
        with path.open(newline="", encoding="utf-8") as handle:
            loaded[name.removesuffix(".csv")] = list(csv.DictReader(handle))
    return {
        "status": "PASS",
        "scientific_output_fingerprint": PHASE10_FINGERPRINT,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "tables": loaded,
        "source_directory": str(base),
    }
