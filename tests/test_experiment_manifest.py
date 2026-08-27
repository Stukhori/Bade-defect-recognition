from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from windblade.config import load_config
from windblade.environment import capture_environment
from windblade.experiment import ExperimentRun
from windblade.reproducibility import set_global_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "smoke.yaml"


def temporary_config(tmp_path: Path):
    return load_config(SMOKE_CONFIG).with_overrides(
        {"experiment.output_root": str(tmp_path / "experiment-output")}
    )


def test_completed_manifest_contains_mandatory_fields_and_utc(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    with ExperimentRun.create(config, repository_root=REPOSITORY_ROOT) as run:
        run.record_reproducibility(set_global_seed(42))
        run.write_results({"dummy": 1})

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    mandatory = {
        "schema_version",
        "experiment_id",
        "project",
        "phase",
        "experiment_name",
        "timestamp_utc",
        "config_hash",
        "seed",
        "dataset",
        "method",
        "runtime",
        "environment",
        "timing",
        "artifacts",
        "status",
    }
    assert mandatory <= manifest.keys()
    assert manifest["status"] == "completed"
    assert manifest["timestamp_utc"].endswith("Z")
    assert manifest["timing"]["started_utc"].endswith("Z")
    assert manifest["timing"]["finished_utc"].endswith("Z")
    assert re.fullmatch(r"[0-9]{8}T[0-9]{12}Z_smoke_[0-9a-f]{12}", run.experiment_id)


def test_failed_run_preserves_manifest_log_and_exception(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    with pytest.raises(RuntimeError, match="intentional test failure"):
        with ExperimentRun.create(config, repository_root=REPOSITORY_ROOT) as run:
            raise RuntimeError("intentional test failure")

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "RuntimeError",
        "message": "intentional test failure",
    }
    assert run.resolved_config_path.is_file()
    assert run.log_path.is_file()
    assert "intentional test failure" in run.log_path.read_text(encoding="utf-8")


def test_environment_capture_handles_unavailable_optional_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str):
        if name == "torch":
            raise ModuleNotFoundError("synthetic unavailable torch")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("windblade.environment.importlib.import_module", unavailable)
    environment = capture_environment("cpu", REPOSITORY_ROOT)

    assert environment["pytorch_available"] is False
    assert environment["cuda_available"] is None
    assert environment["python_version"]
    assert "git_commit" in environment
    assert "git_dirty" in environment


def test_each_experiment_receives_an_isolated_run_directory(tmp_path: Path) -> None:
    config = temporary_config(tmp_path)
    with ExperimentRun.create(config, repository_root=REPOSITORY_ROOT) as first:
        pass
    with ExperimentRun.create(config, repository_root=REPOSITORY_ROOT) as second:
        pass

    assert first.experiment_id != second.experiment_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.is_dir()
    assert second.run_dir.is_dir()
