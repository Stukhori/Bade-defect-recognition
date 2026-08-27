from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from windblade.config import ConfigError, calculate_config_hash, load_config


def valid_mapping() -> dict:
    return {
        "project": {"name": "windblade", "phase": 1},
        "experiment": {"name": "smoke", "seed": 42, "output_root": "experiments"},
        "dataset": {
            "name": "synthetic",
            "version": "smoke-v1",
            "split_id": "synthetic-fixed",
        },
        "method": {"family": "smoke", "name": "deterministic_dummy"},
        "training": {"train_fraction": 1.0, "epochs": 0},
        "evaluation": {"primary_metric": "macro_f1"},
        "runtime": {"device": "cpu"},
    }


def write_yaml(path: Path, value: dict, *, sort_keys: bool = True) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=sort_keys), encoding="utf-8")


def test_valid_yaml_loads_and_resolves_parent(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    write_yaml(base, valid_mapping())
    child.write_text("extends: base.yaml\nexperiment:\n  name: child-smoke\n", encoding="utf-8")

    config = load_config(child)

    assert config.section("experiment")["name"] == "child-smoke"
    assert config.section("dataset")["name"] == "synthetic"
    assert "extends" not in config.as_dict()


def test_malformed_yaml_fails_with_useful_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("project: [unterminated", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_missing_required_field_fails(tmp_path: Path) -> None:
    value = valid_mapping()
    del value["runtime"]
    path = tmp_path / "missing.yaml"
    write_yaml(path, value)

    with pytest.raises(ConfigError, match="missing required top-level field: runtime"):
        load_config(path)


def test_config_hash_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    first_value = valid_mapping()
    second_value = dict(reversed(list(first_value.items())))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    write_yaml(first, first_value, sort_keys=False)
    write_yaml(second, second_value, sort_keys=False)

    first_config = load_config(first)
    second_config = load_config(second)

    assert first_config.config_hash == second_config.config_hash
    assert len(first_config.config_hash) == 12
    assert first_config.config_hash == calculate_config_hash(first_config.as_dict())


def test_seed_override_changes_resolved_hash(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    write_yaml(path, valid_mapping())
    original = load_config(path)
    changed = original.with_overrides({"experiment.seed": 43})

    assert original.config_hash != changed.config_hash
    assert changed.section("experiment")["seed"] == 43
