"""Deterministic synthetic experiment used only to validate infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np

from windblade.config import ConfigError, load_config
from windblade.experiment import ExperimentRun
from windblade.reproducibility import set_global_seed
from windblade.results import read_json


@dataclass(frozen=True)
class SmokeOutcome:
    experiment_id: str
    run_dir: Path
    config_hash: str
    seed: int
    payload: dict[str, Any]


def _validate_smoke_scope(config) -> None:
    project = config.section("project")
    dataset = config.section("dataset")
    method = config.section("method")
    training = config.section("training")
    runtime = config.section("runtime")
    if project["phase"] != 1:
        raise ConfigError("smoke experiment requires project.phase = 1")
    if dataset["name"] != "synthetic":
        raise ConfigError("smoke experiment accepts only dataset.name = synthetic")
    if method["family"] != "smoke":
        raise ConfigError("smoke experiment accepts only method.family = smoke")
    if training["epochs"] != 0:
        raise ConfigError("smoke experiment must not train; training.epochs must be 0")
    if runtime["device"] != "cpu":
        raise ConfigError("Phase 1 smoke experiment requires runtime.device = cpu")


def deterministic_synthetic_payload() -> dict[str, Any]:
    """Generate a tiny seeded payload with no scientific interpretation."""

    python_draws = [random.randrange(0, 10_000) for _ in range(4)]
    matrix = np.random.randint(0, 1_000, size=(4, 4), dtype=np.int64)
    return {
        "label": "synthetic_infrastructure_smoke_not_scientific",
        "python_draws": python_draws,
        "matrix": matrix.tolist(),
        "row_sums": matrix.sum(axis=1).tolist(),
        "total": int(matrix.sum()),
    }


def run_smoke_experiment(
    config_path: str | Path,
    repository_root: str | Path,
    *,
    seed_override: int | None = None,
    output_root_override: str | Path | None = None,
) -> SmokeOutcome:
    """Run the complete Phase 1 synthetic infrastructure exercise."""

    config = load_config(config_path)
    overrides: dict[str, Any] = {}
    if seed_override is not None:
        overrides["experiment.seed"] = seed_override
    if output_root_override is not None:
        overrides["experiment.output_root"] = str(Path(output_root_override).resolve())
    if overrides:
        config = config.with_overrides(overrides)
    _validate_smoke_scope(config)

    seed = config.section("experiment")["seed"]
    with ExperimentRun.create(
        config,
        repository_root=repository_root,
        actual_device="cpu",
    ) as run:
        settings = set_global_seed(seed)
        run.record_reproducibility(settings)
        run.logger.info("Synthetic deterministic computation started")
        payload = deterministic_synthetic_payload()
        run.logger.info("Synthetic deterministic computation completed")
        run.write_results(payload)

    completed = read_json(run.manifest_path)
    if completed["status"] != "completed":
        raise RuntimeError("smoke experiment did not finish with completed status")
    return SmokeOutcome(
        experiment_id=run.experiment_id,
        run_dir=run.run_dir,
        config_hash=config.config_hash,
        seed=seed,
        payload=payload,
    )
