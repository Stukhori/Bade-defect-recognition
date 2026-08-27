"""Small YAML configuration loader with validation and stable hashing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from windblade.utils import atomic_write_text


class ConfigError(ValueError):
    """Raised when a configuration cannot be parsed or validated."""


REQUIRED_FIELDS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "project": {"name": str, "phase": int},
    "experiment": {"name": str, "seed": int, "output_root": str},
    "dataset": {"name": str, "version": str, "split_id": str},
    "method": {"family": str, "name": str},
    "training": {"train_fraction": (int, float), "epochs": int},
    "evaluation": {"primary_metric": str},
    "runtime": {"device": str},
}


def _is_instance(value: Any, expected: type | tuple[type, ...]) -> bool:
    if isinstance(value, bool) and (expected is int or expected == (int, float)):
        return False
    return isinstance(value, expected)


def _validate_string_keys(value: Any, location: str = "configuration") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{location} contains a non-string key: {key!r}")
            _validate_string_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_string_keys(child, f"{location}[{index}]")


def validate_config(data: Mapping[str, Any]) -> None:
    """Validate the small Phase 1 configuration schema."""

    _validate_string_keys(data)
    for section, fields in REQUIRED_FIELDS.items():
        if section not in data:
            raise ConfigError(f"missing required top-level field: {section}")
        section_value = data[section]
        if not isinstance(section_value, Mapping):
            raise ConfigError(f"field '{section}' must be a mapping")
        for field, expected_type in fields.items():
            if field not in section_value:
                raise ConfigError(f"missing required field: {section}.{field}")
            value = section_value[field]
            if not _is_instance(value, expected_type):
                expected_name = (
                    " or ".join(item.__name__ for item in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                raise ConfigError(
                    f"field '{section}.{field}' must be {expected_name}; "
                    f"received {type(value).__name__}"
                )

    for section, field in (
        ("project", "name"),
        ("experiment", "name"),
        ("experiment", "output_root"),
        ("dataset", "name"),
        ("dataset", "version"),
        ("dataset", "split_id"),
        ("method", "family"),
        ("method", "name"),
        ("evaluation", "primary_metric"),
        ("runtime", "device"),
    ):
        if not data[section][field].strip():
            raise ConfigError(f"field '{section}.{field}' must not be empty")

    if data["project"]["phase"] < 0:
        raise ConfigError("field 'project.phase' must be non-negative")
    seed = data["experiment"]["seed"]
    if seed < 0 or seed > 2**32 - 1:
        raise ConfigError("field 'experiment.seed' must be between 0 and 2^32 - 1")
    fraction = float(data["training"]["train_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ConfigError("field 'training.train_fraction' must be in (0, 1]")
    if data["training"]["epochs"] < 0:
        raise ConfigError("field 'training.epochs' must be non-negative")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_mapping(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ConfigError(f"configuration inheritance cycle detected at: {resolved_path}")
    if not resolved_path.is_file():
        raise ConfigError(f"configuration file does not exist: {resolved_path}")
    try:
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {resolved_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {resolved_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration root must be a mapping: {resolved_path}")

    current = dict(loaded)
    parent_reference = current.pop("extends", None)
    if parent_reference is None:
        return deepcopy(current)
    if not isinstance(parent_reference, str) or not parent_reference.strip():
        raise ConfigError("field 'extends' must be a non-empty relative path string")
    parent_path = (resolved_path.parent / parent_reference).resolve()
    parent = _load_yaml_mapping(parent_path, seen | {resolved_path})
    return _deep_merge(parent, current)


def canonical_config_json(data: Mapping[str, Any]) -> str:
    """Return a deterministic JSON representation used for hashing."""

    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"configuration is not deterministically serializable: {exc}") from exc


def calculate_config_hash(data: Mapping[str, Any], length: int = 12) -> str:
    """Return a shortened SHA-256 fingerprint for a resolved configuration."""

    if length < 10 or length > 64:
        raise ValueError("configuration hash length must be between 10 and 64")
    return hashlib.sha256(canonical_config_json(data).encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated, resolved configuration with defensive-copy access."""

    _data: dict[str, Any]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        copied = deepcopy(self._data)
        validate_config(copied)
        canonical_config_json(copied)
        object.__setattr__(self, "_data", copied)

    @property
    def config_hash(self) -> str:
        return calculate_config_hash(self._data)

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def section(self, name: str) -> dict[str, Any]:
        value = self._data[name]
        return deepcopy(dict(value))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ResolvedConfig":
        updated = self.as_dict()
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            if not parts or any(not part for part in parts):
                raise ConfigError(f"invalid override path: {dotted_key!r}")
            target: dict[str, Any] = updated
            for part in parts[:-1]:
                child = target.get(part)
                if not isinstance(child, dict):
                    raise ConfigError(f"override parent is not a mapping: {dotted_key}")
                target = child
            target[parts[-1]] = value
        return ResolvedConfig(updated, self.source_path)

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self._data,
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        )


def load_config(path: str | Path) -> ResolvedConfig:
    """Load, resolve inheritance, and validate a YAML configuration."""

    source = Path(path).resolve()
    data = _load_yaml_mapping(source, frozenset())
    return ResolvedConfig(data, source)


def save_resolved_config(config: ResolvedConfig, path: str | Path) -> None:
    """Write the immutable per-run configuration snapshot."""

    atomic_write_text(Path(path), config.to_yaml())
