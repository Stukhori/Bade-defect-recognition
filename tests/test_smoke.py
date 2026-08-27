from __future__ import annotations

import json
from pathlib import Path

from windblade.smoke import run_smoke_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "smoke.yaml"
REQUIRED_ARTIFACTS = {
    "resolved_config.yaml",
    "manifest.json",
    "results.json",
    "run.log",
}


def run_in_tmp(tmp_path: Path, seed: int):
    return run_smoke_experiment(
        SMOKE_CONFIG,
        REPOSITORY_ROOT,
        seed_override=seed,
        output_root_override=tmp_path / "outputs",
    )


def test_smoke_experiment_succeeds_and_writes_required_artifacts(tmp_path: Path) -> None:
    outcome = run_in_tmp(tmp_path, 42)
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((outcome.run_dir / "results.json").read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACTS <= {path.name for path in outcome.run_dir.iterdir()}
    assert manifest["status"] == "completed"
    assert results["synthetic_only"] is True
    assert results["payload"]["label"] == "synthetic_infrastructure_smoke_not_scientific"


def test_same_seed_produces_same_payload_and_hash_but_unique_ids(tmp_path: Path) -> None:
    first = run_in_tmp(tmp_path, 42)
    second = run_in_tmp(tmp_path, 42)

    assert first.experiment_id != second.experiment_id
    assert first.config_hash == second.config_hash
    assert first.payload == second.payload


def test_different_seed_changes_hash_and_seeded_payload(tmp_path: Path) -> None:
    first = run_in_tmp(tmp_path, 42)
    second = run_in_tmp(tmp_path, 43)

    assert first.config_hash != second.config_hash
    assert first.payload != second.payload
