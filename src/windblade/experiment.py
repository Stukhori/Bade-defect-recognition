"""Experiment identity, run-directory, manifest, and lifecycle handling."""

from __future__ import annotations

from pathlib import Path
import time
from types import TracebackType
from typing import Any, Mapping

from windblade.config import ResolvedConfig, save_resolved_config
from windblade.environment import capture_environment
from windblade.logging_utils import close_run_logger, configure_run_logger
from windblade.results import write_json
from windblade.utils import compact_utc, format_utc, sanitize_path_component, utc_now


MANIFEST_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"


def generate_experiment_id(
    experiment_name: str,
    config_hash: str,
    timestamp=None,
) -> str:
    """Build a readable experiment ID from UTC time, name, and config hash."""

    created = utc_now() if timestamp is None else timestamp
    safe_name = sanitize_path_component(experiment_name)
    return f"{compact_utc(created)}_{safe_name}_{config_hash}"


class ExperimentRun:
    """Own an isolated run directory and preserve completion or failure state."""

    def __init__(
        self,
        config: ResolvedConfig,
        repository_root: str | Path,
        actual_device: str,
    ) -> None:
        self.config = config
        self.repository_root = Path(repository_root).resolve()
        experiment = config.section("experiment")
        project = config.section("project")
        dataset = config.section("dataset")
        method = config.section("method")
        runtime = config.section("runtime")

        output_root = Path(experiment["output_root"])
        if not output_root.is_absolute():
            output_root = self.repository_root / output_root
        results_root = output_root / "results"
        results_root.mkdir(parents=True, exist_ok=True)

        self.started_at = utc_now()
        self._started_counter = time.perf_counter()
        self.experiment_id = generate_experiment_id(
            experiment["name"], config.config_hash, self.started_at
        )
        self.run_dir = results_root / self.experiment_id
        self.run_dir.mkdir(parents=False, exist_ok=False)

        self.resolved_config_path = self.run_dir / "resolved_config.yaml"
        self.manifest_path = self.run_dir / "manifest.json"
        self.results_path = self.run_dir / "results.json"
        self.log_path = self.run_dir / "run.log"
        self.logger = configure_run_logger(self.experiment_id, self.log_path)

        save_resolved_config(config, self.resolved_config_path)
        self.logger.info("Experiment %s started", self.experiment_id)
        self.logger.info("Configuration loaded and resolved: %s", config.config_hash)
        environment = capture_environment(
            selected_device=actual_device,
            repository_root=self.repository_root,
        )
        self.logger.info("Environment and Git provenance captured")

        self.manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "project": project["name"],
            "phase": project["phase"],
            "experiment_name": experiment["name"],
            "timestamp_utc": format_utc(self.started_at),
            "config_hash": config.config_hash,
            "seed": experiment["seed"],
            "dataset": dataset,
            "method": method,
            "runtime": {
                "requested_device": runtime["device"],
                "actual_device": actual_device,
            },
            "environment": environment,
            "reproducibility": None,
            "timing": {
                "started_utc": format_utc(self.started_at),
                "finished_utc": None,
                "elapsed_seconds": None,
            },
            "artifacts": {
                "resolved_config": "resolved_config.yaml",
                "manifest": "manifest.json",
                "log": "run.log",
            },
            "status": "running",
            "failure": None,
        }
        self._write_manifest()

    @classmethod
    def create(
        cls,
        config: ResolvedConfig,
        repository_root: str | Path | None = None,
        actual_device: str | None = None,
    ) -> "ExperimentRun":
        root = Path.cwd() if repository_root is None else Path(repository_root)
        requested = config.section("runtime")["device"]
        return cls(config, root, requested if actual_device is None else actual_device)

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self.manifest)

    def record_reproducibility(self, settings: Mapping[str, Any]) -> None:
        self.manifest["reproducibility"] = dict(settings)
        self._write_manifest()
        self.logger.info("Global seed initialized: %s", self.manifest["seed"])

    def write_results(
        self,
        payload: Mapping[str, Any],
        *,
        synthetic_only: bool,
    ) -> dict[str, Any]:
        if not isinstance(synthetic_only, bool):
            raise TypeError("synthetic_only must be a boolean")
        record = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "config_hash": self.config.config_hash,
            "seed": self.manifest["seed"],
            "synthetic_only": synthetic_only,
            "payload": dict(payload),
        }
        write_json(self.results_path, record)
        self.manifest["artifacts"]["results"] = "results.json"
        self._write_manifest()
        self.logger.info("Synthetic result payload serialized")
        return record

    def _finish(self, status: str, failure: dict[str, str] | None = None) -> None:
        finished = utc_now()
        self.manifest["status"] = status
        self.manifest["failure"] = failure
        self.manifest["timing"]["finished_utc"] = format_utc(finished)
        self.manifest["timing"]["elapsed_seconds"] = round(
            time.perf_counter() - self._started_counter, 6
        )
        self._write_manifest()
        self.logger.info(
            "Experiment %s %s in %.6f seconds",
            self.experiment_id,
            status,
            self.manifest["timing"]["elapsed_seconds"],
        )
        close_run_logger(self.logger)

    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            self._finish("completed")
            return False

        failure = {
            "type": exc_type.__name__,
            "message": str(exc_value) if exc_value is not None else "unknown failure",
        }
        self.logger.error(
            "Experiment %s failed: %s: %s",
            self.experiment_id,
            failure["type"],
            failure["message"],
            exc_info=(exc_type, exc_value, traceback),
        )
        self._finish("failed", failure)
        return False
