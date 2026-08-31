from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from windblade.config import load_config
from windblade.final_synthesis import core


ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/final_synthesis.yaml")


def test_statistical_plan_is_frozen_and_complete() -> None:
    result = core.validate_statistical_plan(config())
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_modified_upstream_fingerprint_is_rejected() -> None:
    changed = config().with_overrides({"upstream.phase9a_output_fingerprint": "0" * 64})
    with pytest.raises(core.FinalSynthesisError, match="phase9a_output"):
        core._validate_upstream_identity(changed, ROOT)


def test_missing_phase9_completion_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = core._json

    def fake(path: Path):
        value = original(path)
        if path.as_posix().endswith("phase9b/manifest.json"):
            value = dict(value)
            value["phase9_complete"] = False
        return value

    monkeypatch.setattr(core, "_json", fake)
    with pytest.raises(core.FinalSynthesisError, match="phase9_complete"):
        core._validate_upstream_identity(config(), ROOT)


def test_bootstrap_is_deterministic_paired_and_class_stratified() -> None:
    _, true, _ = core.load_clean_predictions(ROOT)
    first = core.bootstrap_indices(true, resamples=50, seed=20260831)
    second = core.bootstrap_indices(true, resamples=50, seed=20260831)
    assert np.array_equal(first, second)
    assert first.shape == (50, 162)
    expected = [27, 30, 33, 9, 14, 49]
    assert all([int(np.sum(true[row] == label)) for label in range(6)] == expected for row in first)


def test_common_indices_preserve_pairing() -> None:
    _, true, predictions = core.load_clean_predictions(ROOT)
    indices = core.bootstrap_indices(true, resamples=3, seed=7)
    boot = core._bootstrap_metrics(true, predictions, indices)
    assert set(boot) == set(core.METHODS)
    assert all(values.shape == (3, 23) for values in boot.values())


def test_deterministic_sd_is_na_and_cnn_seed_aggregation_is_retained() -> None:
    small = config().with_overrides({"statistical_plan.bootstrap.resamples": 20})
    result = core._clean_analysis(small, ROOT)
    rows = {row["method"]: row for row in result.comparison}
    assert rows["hog"]["macro_f1_seed_sd"] is None
    assert rows["lbp"]["seed_sd_definition"] == "N/A"
    assert rows["resnet18"]["cnn_seed_count"] == 3
    assert rows["mobilenet_v3_small"]["macro_f1_seed_sd"] is not None


def test_cnn_bootstrap_aggregation_is_mean_of_seed_metrics() -> None:
    true = np.arange(6, dtype=np.int64)
    perfect = true.copy()
    shifted = (true + 1) % 6
    predictions = {
        "hog": [perfect], "lbp": [perfect],
        "resnet18": [perfect, perfect, shifted],
        "mobilenet_v3_small": [perfect, shifted, shifted],
    }
    indices = np.asarray([np.arange(6)], dtype=np.int64)
    result = core._bootstrap_metrics(true, predictions, indices)
    expected = np.mean([core._metric_vector(true, perfect), core._metric_vector(true, perfect), core._metric_vector(true, shifted)], axis=0)
    assert np.allclose(result["resnet18"][0], expected)


def test_percentile_interval_is_exact_and_ordered() -> None:
    low, high = core._percentile(np.arange(1, 101, dtype=float), 0.95)
    assert low == pytest.approx(3.475)
    assert high == pytest.approx(97.525)
    assert low < high


def test_holm_adjustment_utility_is_correct_although_p_values_are_omitted() -> None:
    assert core.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert config().as_dict()["statistical_plan"]["p_values"] == "omitted"


def test_table_csv_json_consistency(tmp_path: Path) -> None:
    rows = [{"method": "hog", "value": 1.0, "sd": None}, {"method": "lbp", "value": 2.0, "sd": None}]
    core._write_table(tmp_path, "example", rows, ["method", "value", "sd"])
    csv_rows = core.read_csv(tmp_path / "tables/example.csv")
    payload = json.loads((tmp_path / "tables/example.json").read_text(encoding="utf-8"))
    assert csv_rows == [{field: core._cell(row.get(field)) for field in payload["fields"]} for row in payload["rows"]]


def test_phase11_is_allowed_downstream_but_phase12_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "figures/phase11").mkdir(parents=True)
    core._assert_no_optional_phase_paths(tmp_path)
    (tmp_path / "figures/phase12").mkdir(parents=True)
    with pytest.raises(core.FinalSynthesisError, match="Phase 12"):
        core._assert_no_optional_phase_paths(tmp_path)


def test_two_generation_hashes_tables_and_figures_are_exact(tmp_path: Path) -> None:
    small = config().with_overrides({"statistical_plan.bootstrap.resamples": 20})
    gate = {"status": "PASS", "validators": "unit-test fixture"}
    inventory = {"file_count": 1, "fingerprint": "fixture", "files": {"fixture": "fixture"}}
    first = core._generate_pass(small, ROOT, tmp_path / "first", gate, inventory)
    second = core._generate_pass(small, ROOT, tmp_path / "second", gate, inventory)
    assert first["inventory"] == second["inventory"]
    assert core._hash_mapping(first["inventory"]) == core._hash_mapping(second["inventory"])
    assert any(path.startswith("figures/") for path in first["inventory"])
    assert any(path.startswith("tables/") for path in first["inventory"])


def test_fingerprint_changes_when_inventory_changes() -> None:
    first = core._hash_mapping({"a": "1", "b": "2"})
    second = core._hash_mapping({"a": "1", "b": "3"})
    assert first != second
