from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import types
import venv
import zipfile

import pytest
import yaml

from windblade.detection import phase11b


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/detection_phase11b.yaml"


@pytest.fixture(scope="module")
def apparatus():
    return phase11b.load_apparatus(CONFIG_PATH)


def _execution_fixture(tmp_path: Path, apparatus, seed: int = 17) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = repo / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    data_root = (tmp_path / "trainval-data").resolve()
    dataset_root = data_root / "dataset"
    dataset_root.mkdir(parents=True)
    data_yaml = dataset_root / "trainval.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset_root), "names": {0: "defect"},
        "train": "images/train", "val": "images/validation",
    }, sort_keys=False), encoding="utf-8")
    artifacts = [{"path": "dataset/images/train/image.jpg", "sha256": "1" * 64}]
    materialization = {
        "scope": "trainval", "included_splits": ["train", "validation"],
        "artifact_fingerprint": phase11b.canonical_hash(artifacts), "artifacts": artifacts,
    }
    phase11b.atomic_json(data_root / "materialization.json", materialization)
    drive_root = (tmp_path / "drive").resolve()
    layout = phase11b.DriveLayout.from_root(drive_root)
    run_dir = layout.run(seed)
    weight = layout.provenance / "yolo11n.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"official initial weight fixture")
    record = {
        "apparatus_commit": "a" * 40,
        "sha256": phase11b.sha256_file(weight),
    }
    expected_state = phase11b._expected_run_state(
        seed, "config.yaml", phase11b.sha256_file(config_path), record, weight, materialization,
    )
    arguments = phase11b.training_arguments(apparatus, data_yaml, layout, seed)
    return {
        "repo": repo, "config_path": config_path, "data_root": data_root,
        "drive_root": drive_root, "layout": layout, "run_dir": run_dir,
        "weight": weight, "record_path": layout.provenance / "weight-record.json",
        "record": record, "state": expected_state, "arguments": arguments,
    }


def _write_existing_run(context: dict, epochs: int) -> None:
    run_dir = context["run_dir"]
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    phase11b.atomic_json(run_dir / "run_state.json", context["state"])
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump(context["arguments"], sort_keys=False), encoding="utf-8",
    )
    rows = ["epoch,metrics/mAP50-95(B)"] + [f"{epoch},0.5" for epoch in range(1, epochs + 1)]
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for index in range(epochs):
        (weights / f"epoch{index}.pt").write_bytes(f"epoch-{index}".encode())
    (weights / "last.pt").write_bytes(b"last checkpoint fixture")
    if epochs == 100:
        (weights / "best.pt").write_bytes(b"best checkpoint fixture")


def _patch_execution_checks(monkeypatch, context: dict) -> None:
    monkeypatch.setattr(phase11b, "validate_frozen_inputs", lambda config, repo: {"status": "PASS"})
    monkeypatch.setattr(
        phase11b, "validate_weight_record",
        lambda config, weight, record_path: dict(context["record"]),
    )
    monkeypatch.setattr(
        phase11b, "_git",
        lambda repo, *args, **kwargs: "b" * 40 if args == ("rev-parse", "HEAD") else "",
    )


def test_frozen_fingerprints_counts_and_run_matrix(apparatus) -> None:
    result = phase11b.validate_frozen_inputs(apparatus, ROOT)
    assert result["status"] == "PASS"
    assert result["images"] == 720
    assert result["boxes"] == 1065
    assert result["images_by_split"] == {"train": 510, "validation": 101, "test": 109}
    assert result["boxes_by_split"] == {"train": 757, "validation": 146, "test": 162}
    assert result["run_count"] == 3
    assert result["seeds"] == [17, 29, 43]
    assert result["dataset_fingerprint"] == phase11b.EXPECTED_PHASE11A["dataset_fingerprint"]
    assert result["split_fingerprint"] == phase11b.EXPECTED_PHASE11A["split_fingerprint"]


def test_one_based_inclusive_full_image_geometry_and_class_mapping() -> None:
    annotation = {"xmin": 1, "ymin": 1, "xmax": 10, "ymax": 20}
    image = {"width": 10, "height": 20}
    assert phase11b.format_class_agnostic_yolo_label(annotation, image) == (
        "0 0.500000000000 0.500000000000 1.000000000000 1.000000000000"
    )


def test_every_frozen_annotation_maps_to_class_zero(apparatus) -> None:
    rows = phase11b.read_csv(ROOT / apparatus["frozen_inputs"]["annotation_manifest"])
    assert {int(row["defect_class_id"]) for row in rows} == {0}
    assert {row["class_name"] for row in rows} == set(phase11b.CLASS_ORDER)


def test_drive_paths_and_seed_isolation(tmp_path: Path) -> None:
    layout = phase11b.DriveLayout.from_root(tmp_path.resolve())
    run_paths = {layout.run(seed) for seed in phase11b.SEEDS}
    assert len(run_paths) == 3
    assert all(path.parent == layout.runs for path in run_paths)
    with pytest.raises(phase11b.Phase11BError, match="undeclared seed"):
        layout.run(99)
    with pytest.raises(phase11b.Phase11BError, match="absolute"):
        phase11b.DriveLayout.from_root(Path("relative"))


def test_resume_requires_matching_seed_configuration_and_weight(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "seed_17"
    checkpoint = run / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"last")
    weight = tmp_path / "yolo11n.pt"
    weight.write_bytes(b"initial")
    state = {
        "seed": 17, "configuration_sha256": "a" * 64,
        "weight_path": str(weight), "weight_sha256": hashlib.sha256(b"initial").hexdigest(),
    }
    (run / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert phase11b.decide_resume(run, 17, "a" * 64) == checkpoint
    with pytest.raises(phase11b.Phase11BError, match="different seed or configuration"):
        phase11b.decide_resume(run, 29, "a" * 64)
    weight.write_bytes(b"changed")
    with pytest.raises(phase11b.Phase11BError, match="weight identity changed"):
        phase11b.decide_resume(run, 17, "a" * 64)


def test_partial_resume_state_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "seed_17"
    run.mkdir()
    (run / "run_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(phase11b.Phase11BError, match="partial run"):
        phase11b.decide_resume(run, 17, "a" * 64)


def test_completed_run_is_idempotent_without_state_rewrite_or_yolo(tmp_path: Path, apparatus, monkeypatch) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    _write_existing_run(context, 100)
    state_path = context["run_dir"] / "run_state.json"
    completed = {
        **context["state"], "status": "TRAINING_COMMAND_COMPLETED",
        "last_checkpoint_sha256": phase11b.sha256_file(context["run_dir"] / "weights/last.pt"),
        "applied_batch_size": 16, "applied_optimizer": "AdamW", "applied_amp": True,
    }
    phase11b.atomic_json(state_path, completed)
    original_bytes = state_path.read_bytes()
    _patch_execution_checks(monkeypatch, context)
    monkeypatch.setattr(
        phase11b, "_ultralytics_settings_off",
        lambda: pytest.fail("completed run must not initialize Ultralytics"),
    )
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments",
        lambda checkpoint: pytest.fail("completed run must not load last.pt"),
    )

    result = phase11b.train_seed(
        apparatus, context["config_path"], context["repo"], context["data_root"],
        context["drive_root"], context["weight"], context["record_path"], 17,
    )

    assert result == completed
    assert state_path.read_bytes() == original_bytes


def test_legacy_full_run_is_recovered_without_training_or_artifact_changes(
    tmp_path: Path, apparatus, monkeypatch,
) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    _write_existing_run(context, 100)
    protected = [context["run_dir"] / "results.csv", context["run_dir"] / "args.yaml"]
    protected.extend(sorted((context["run_dir"] / "weights").glob("*.pt")))
    before = {path: phase11b.sha256_file(path) for path in protected}
    _patch_execution_checks(monkeypatch, context)
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments", lambda checkpoint: dict(context["arguments"]),
    )
    monkeypatch.setattr(
        phase11b, "_ultralytics_settings_off",
        lambda: pytest.fail("legacy completion recovery must not initialize Ultralytics"),
    )

    result = phase11b.train_seed(
        apparatus, context["config_path"], context["repo"], context["data_root"],
        context["drive_root"], context["weight"], context["record_path"], 17,
    )

    assert result["status"] == "TRAINING_COMMAND_COMPLETED"
    assert result["last_checkpoint_sha256"] == phase11b.sha256_file(context["run_dir"] / "weights/last.pt")
    assert result["applied_batch_size"] == 16
    assert result["applied_optimizer"] == "AdamW"
    assert result["applied_amp"] is True
    assert result["completion_recovery"] == {
        "mode": "VERIFIED_PRE_EXISTING_ARTIFACTS",
        "training_invoked": False,
        "repository_commit": "b" * 40,
    }
    assert {path: phase11b.sha256_file(path) for path in protected} == before


@pytest.mark.parametrize("observed", [0, "0"])
def test_device_zero_accepts_only_supported_equivalent_representations(tmp_path: Path, apparatus, observed) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    arguments = {**context["arguments"], "device": observed}
    result = phase11b._critical_training_values(
        arguments, apparatus, context["data_root"] / "dataset/trainval.yaml",
        context["layout"], 17, "fixture",
    )
    assert result["device"] == observed


@pytest.mark.parametrize("observed", [False, 0.0, 1, "1", "cuda:0", "0,1", "", None, "cpu", " 0 "])
def test_device_zero_rejects_other_or_malformed_representations(tmp_path: Path, apparatus, observed) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    arguments = {**context["arguments"], "device": observed}
    with pytest.raises(phase11b.Phase11BError, match="device mismatch"):
        phase11b._critical_training_values(
            arguments, apparatus, context["data_root"] / "dataset/trainval.yaml",
            context["layout"], 17, "fixture",
        )


@pytest.mark.parametrize("yaml_device, checkpoint_device", [("0", 0), (0, "0")])
def test_device_zero_normalization_applies_to_args_and_checkpoint_without_training(
    tmp_path: Path, apparatus, monkeypatch, yaml_device, checkpoint_device,
) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    context["arguments"]["device"] = yaml_device
    _write_existing_run(context, 100)
    checkpoint_arguments = {**context["arguments"], "device": checkpoint_device}
    _patch_execution_checks(monkeypatch, context)
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments", lambda checkpoint: checkpoint_arguments,
    )
    monkeypatch.setattr(
        phase11b, "_ultralytics_settings_off",
        lambda: pytest.fail("device normalization recovery must not initialize Ultralytics"),
    )

    result = phase11b.train_seed(
        apparatus, context["config_path"], context["repo"], context["data_root"],
        context["drive_root"], context["weight"], context["record_path"], 17,
    )

    assert result["status"] == "TRAINING_COMMAND_COMPLETED"
    assert result["completion_recovery"]["training_invoked"] is False


@pytest.mark.parametrize("key, observed", [("batch", "16"), ("optimizer", "adamw"), ("amp", False)])
def test_device_normalization_does_not_relax_other_controls(
    tmp_path: Path, apparatus, key: str, observed,
) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    arguments = {**context["arguments"], key: observed}
    with pytest.raises(phase11b.Phase11BError, match=rf"{key} mismatch"):
        phase11b._critical_training_values(
            arguments, apparatus, context["data_root"] / "dataset/trainval.yaml",
            context["layout"], 17, "fixture",
        )


def test_partial_seed_29_run_remains_resumable_from_last_checkpoint(
    tmp_path: Path, apparatus, monkeypatch,
) -> None:
    context = _execution_fixture(tmp_path, apparatus, seed=29)
    _write_existing_run(context, 30)
    state_path = context["run_dir"] / "run_state.json"
    original_bytes = state_path.read_bytes()
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments", lambda checkpoint: dict(context["arguments"]),
    )

    recovered = phase11b.recover_legacy_completed_run(
        apparatus, context["repo"], context["data_root"], context["layout"],
        context["run_dir"], 29, context["state"], context["state"],
    )

    assert recovered is None
    assert phase11b.decide_resume(
        context["run_dir"], 29, context["state"]["configuration_sha256"],
    ) == context["run_dir"] / "weights/last.pt"
    assert state_path.read_bytes() == original_bytes


def test_fresh_seed_43_starts_once_from_official_initial_weight(
    tmp_path: Path, apparatus, monkeypatch,
) -> None:
    context = _execution_fixture(tmp_path, apparatus, seed=43)
    _patch_execution_checks(monkeypatch, context)
    calls: list[tuple] = []

    class FakeYOLO:
        def __init__(self, source: str):
            calls.append(("init", source))
            self.trainer = types.SimpleNamespace(
                args=types.SimpleNamespace(batch=16), optimizer=type("AdamW", (), {})(), amp=True,
            )

        def train(self, **arguments):
            calls.append(("train", arguments))
            weights = context["run_dir"] / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "last.pt").write_bytes(b"fake completed last checkpoint")

    fake_torch = types.ModuleType("torch")
    fake_torch.use_deterministic_algorithms = lambda *args, **kwargs: None
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=True, deterministic=False),
    )
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.settings = {}
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    result = phase11b.train_seed(
        apparatus, context["config_path"], context["repo"], context["data_root"],
        context["drive_root"], context["weight"], context["record_path"], 43,
    )

    assert calls[0] == ("init", str(context["weight"]))
    assert calls[1][0] == "train"
    assert calls[1][1]["seed"] == 43
    assert len(calls) == 2
    assert result["status"] == "TRAINING_COMMAND_COMPLETED"


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("missing_results", "lacks results.csv"),
        ("noncontiguous_results", "noncontiguous or ambiguous"),
        ("missing_args", "missing existing run args.yaml"),
        ("mismatched_args", "batch mismatch"),
        ("mismatched_checkpoint_args", "last.pt frozen training control batch mismatch"),
        ("missing_periodic", "expected periodic checkpoint"),
        ("missing_best", "lacks best.pt"),
        ("corrupt_last", "not a valid PyTorch archive"),
    ],
)
def test_legacy_completion_evidence_fails_closed(
    tmp_path: Path, apparatus, monkeypatch, corruption: str, message: str,
) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    _write_existing_run(context, 100)
    if corruption == "missing_results":
        (context["run_dir"] / "results.csv").unlink()
    elif corruption == "noncontiguous_results":
        (context["run_dir"] / "results.csv").write_text(
            "epoch,metrics/mAP50-95(B)\n1,0.5\n3,0.5\n", encoding="utf-8",
        )
    elif corruption == "missing_args":
        (context["run_dir"] / "args.yaml").unlink()
    elif corruption == "mismatched_args":
        changed = {**context["arguments"], "batch": 8}
        (context["run_dir"] / "args.yaml").write_text(yaml.safe_dump(changed), encoding="utf-8")
    elif corruption == "missing_periodic":
        (context["run_dir"] / "weights/epoch50.pt").unlink()
    elif corruption == "missing_best":
        (context["run_dir"] / "weights/best.pt").unlink()
    if corruption != "corrupt_last":
        checkpoint_arguments = dict(context["arguments"])
        if corruption == "mismatched_checkpoint_args":
            checkpoint_arguments["batch"] = 8
        monkeypatch.setattr(
            phase11b, "_inspect_checkpoint_training_arguments",
            lambda checkpoint: checkpoint_arguments,
        )

    with pytest.raises(phase11b.Phase11BError, match=message):
        phase11b.recover_legacy_completed_run(
            apparatus, context["repo"], context["data_root"], context["layout"],
            context["run_dir"], 17, context["state"], context["state"],
        )


def test_checkpoint_metadata_inspection_uses_restricted_pickle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    arguments = {"seed": 17, "data": "/content/trainval.yaml"}
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"model": object(), "train_args": arguments}))
    assert phase11b._inspect_checkpoint_training_arguments(checkpoint) == arguments


@pytest.mark.parametrize(
    "field, value",
    [
        ("seed", 29),
        ("configuration_sha256", "f" * 64),
        ("apparatus_commit", "f" * 40),
        ("weight_sha256", "f" * 64),
        ("materialization_fingerprint", "f" * 64),
    ],
)
def test_legacy_recovery_rejects_wrong_provenance_identities(
    tmp_path: Path, apparatus, monkeypatch, field: str, value,
) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    _write_existing_run(context, 100)
    changed = {**context["state"], field: value}
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments", lambda checkpoint: dict(context["arguments"]),
    )
    with pytest.raises(phase11b.Phase11BError, match="identity mismatch"):
        phase11b.recover_legacy_completed_run(
            apparatus, context["repo"], context["data_root"], context["layout"],
            context["run_dir"], 17, changed, context["state"],
        )


def test_recovery_rejects_test_materialization_and_test_training_path(tmp_path: Path, apparatus, monkeypatch) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    materialization_path = context["data_root"] / "materialization.json"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    materialization["artifacts"] = [{"path": "dataset/images/test/image.jpg", "sha256": "1" * 64}]
    materialization["artifact_fingerprint"] = phase11b.canonical_hash(materialization["artifacts"])
    with pytest.raises(phase11b.Phase11BError, match="test materialization"):
        phase11b._validate_trainval_materialization(materialization, context["data_root"])

    _write_existing_run(context, 100)
    changed_args = {**context["arguments"], "data": str(context["data_root"] / "dataset/test.yaml")}
    (context["run_dir"] / "args.yaml").write_text(yaml.safe_dump(changed_args), encoding="utf-8")
    monkeypatch.setattr(
        phase11b, "_inspect_checkpoint_training_arguments", lambda checkpoint: dict(changed_args),
    )
    with pytest.raises(phase11b.Phase11BError, match="forbidden test reference"):
        phase11b.recover_legacy_completed_run(
            apparatus, context["repo"], context["data_root"], context["layout"],
            context["run_dir"], 17, context["state"], context["state"],
        )


def test_invalid_completed_state_cannot_be_erased_by_resume(tmp_path: Path, apparatus, monkeypatch) -> None:
    context = _execution_fixture(tmp_path, apparatus)
    _write_existing_run(context, 100)
    state_path = context["run_dir"] / "run_state.json"
    completed = {
        **context["state"], "status": "TRAINING_COMMAND_COMPLETED",
        "last_checkpoint_sha256": "0" * 64,
    }
    phase11b.atomic_json(state_path, completed)
    original_bytes = state_path.read_bytes()
    _patch_execution_checks(monkeypatch, context)
    monkeypatch.setattr(
        phase11b, "_ultralytics_settings_off",
        lambda: pytest.fail("invalid completed state must fail before Ultralytics"),
    )
    with pytest.raises(phase11b.Phase11BError, match="last checkpoint identity mismatch"):
        phase11b.train_seed(
            apparatus, context["config_path"], context["repo"], context["data_root"],
            context["drive_root"], context["weight"], context["record_path"], 17,
        )
    assert state_path.read_bytes() == original_bytes


def test_selection_still_requires_completed_state_and_matching_last_sha(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed_17"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"last")
    state = {
        "seed": 17, "configuration_sha256": "a" * 64,
        "status": "TRAINING_COMMAND_COMPLETED", "last_checkpoint_sha256": phase11b.sha256_file(last),
    }
    phase11b.atomic_json(run_dir / "run_state.json", state)
    assert phase11b.validate_completed_run(run_dir, 17, "a" * 64) == state
    last.write_bytes(b"changed")
    with pytest.raises(phase11b.Phase11BError, match="last checkpoint identity mismatch"):
        phase11b.validate_completed_run(run_dir, 17, "a" * 64)


def test_ambiguous_existing_run_cannot_be_treated_as_fresh(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed_43"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
    with pytest.raises(phase11b.Phase11BError, match="ambiguous"):
        phase11b.decide_resume(run_dir, 43, "a" * 64)


def test_checkpoint_selection_is_validation_only_with_earliest_tie_break() -> None:
    candidates = [
        {"epoch": 4, "validation_map_50_95": 0.51, "sha256": "a"},
        {"epoch": 2, "validation_map_50_95": 0.51, "sha256": "b"},
        {"epoch": 1, "validation_map_50_95": 0.49, "sha256": "c"},
    ]
    assert phase11b.choose_checkpoint(candidates)["epoch"] == 2


def test_threshold_selection_and_lower_threshold_tie_break() -> None:
    rows = [
        {"threshold": 0.3, "f1": 0.8},
        {"threshold": 0.2, "f1": 0.8},
        {"threshold": 0.1, "f1": 0.7},
    ]
    assert phase11b.choose_threshold(rows)["threshold"] == 0.2
    assert phase11b.threshold_grid({"start": 0.01, "stop": 0.03, "step": 0.01}) == [0.01, 0.02, 0.03]


def test_threshold_metrics_are_class_agnostic() -> None:
    truth = {"image": [[1, 1, 10, 10]]}
    predictions = {"image": [{"box": [1, 1, 10, 10], "score": 0.8}, {"box": [20, 20, 30, 30], "score": 0.7}]}
    row = phase11b.threshold_metrics(truth, predictions, [0.75], 0.5)[0]
    assert row == {
        "threshold": 0.75, "true_positive": 1, "false_positive": 0, "false_negative": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


def test_artifact_hashing_and_bundle_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    left, right = tmp_path / "left.zip", tmp_path / "right.zip"
    left_hash = phase11b.write_deterministic_zip(left, {"b.txt": second, "a.txt": first})
    right_hash = phase11b.write_deterministic_zip(right, {"a.txt": first, "b.txt": second})
    assert left_hash == right_hash
    assert left.read_bytes() == right.read_bytes()


def test_weight_record_hash_is_enforced(tmp_path: Path, apparatus) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    marker = repo / "marker"
    marker.write_text("apparatus", encoding="utf-8")
    subprocess.run(["git", "add", "marker"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "apparatus"], cwd=repo, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    weight = tmp_path / "yolo11n.pt"
    weight.write_bytes(b"weight fixture")
    record_path = tmp_path / "record.json"
    record = phase11b.create_weight_record(apparatus, repo, weight, commit, record_path)
    assert record["sha256"] == hashlib.sha256(b"weight fixture").hexdigest()
    phase11b.validate_weight_record(apparatus, weight, record_path)
    weight.write_bytes(b"tampered")
    with pytest.raises(phase11b.Phase11BError, match="do not match"):
        phase11b.validate_weight_record(apparatus, weight, record_path)


def test_test_firewall_rejects_uncommitted_receipt(tmp_path: Path, apparatus) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    config_path = repo / "config.yaml"
    config_path.write_text("apparatus: fixture\n", encoding="utf-8")
    receipt = repo / apparatus["firewall"]["committed_selection_receipt"]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(phase11b.Phase11BError):
        phase11b.validate_receipt_firewall(apparatus, config_path, repo, tmp_path.resolve())


def test_test_firewall_accepts_only_consistent_committed_hashes(tmp_path: Path, apparatus) -> None:
    repo = tmp_path / "repo"
    drive = (tmp_path / "drive").resolve()
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    config_path = repo / "config.yaml"
    config_path.write_text("apparatus: fixture\n", encoding="utf-8")
    checkpoints = []
    candidate_registry = {}
    for seed in phase11b.SEEDS:
        checkpoint = drive / "runs" / f"seed_{seed}" / "weights" / "epoch1.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        item = {
            "epoch": 2, "validation_map_50_95": 0.5,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "size_bytes": checkpoint.stat().st_size,
        }
        candidate_registry[str(seed)] = [{**item, "path": str(checkpoint)}]
        checkpoints.append({**item, "seed": seed, "drive_relative_path": checkpoint.relative_to(drive).as_posix()})
    candidate_path = drive / "selection/validation_checkpoint_candidates.json"
    threshold_path = drive / "selection/validation_threshold_candidates.json"
    phase11b.atomic_json(candidate_path, candidate_registry)
    threshold_rows = [{"threshold": 0.2, "f1": 0.8}]
    phase11b.atomic_json(threshold_path, threshold_rows)
    threshold = {
        "value": 0.2, "validation_f1": 0.8, "policy": apparatus["scientific_protocol"]["threshold_selection"],
        "match_iou": 0.5, "candidate_artifact_sha256": phase11b.sha256_file(threshold_path),
    }
    receipt = {
        "status": "FROZEN_BEFORE_TEST",
        "configuration": {"path": "config.yaml", "sha256": phase11b.sha256_file(config_path)},
        "checkpoints": checkpoints, "threshold": threshold, "nms": apparatus["nms"],
        "validation_artifacts": {
            "checkpoint_candidates_sha256": phase11b.sha256_file(candidate_path),
            "threshold_candidates_sha256": phase11b.sha256_file(threshold_path),
        },
    }
    receipt["frozen_hashes"] = {
        "configuration": receipt["configuration"]["sha256"],
        "checkpoints": phase11b.canonical_hash(checkpoints),
        "threshold": phase11b.canonical_hash(threshold),
        "nms": phase11b.canonical_hash(apparatus["nms"]),
    }
    receipt_path = repo / apparatus["firewall"]["committed_selection_receipt"]
    phase11b.atomic_json(receipt_path, receipt)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "freeze selection"], cwd=repo, check=True, capture_output=True)
    assert phase11b.validate_receipt_firewall(apparatus, config_path, repo, drive)["status"] == "FROZEN_BEFORE_TEST"
    receipt["nms"] = {**receipt["nms"], "iou_threshold": 0.5}
    phase11b.atomic_json(receipt_path, receipt)
    with pytest.raises(phase11b.Phase11BError, match="committed and clean"):
        phase11b.validate_receipt_firewall(apparatus, config_path, repo, drive)


def test_colab_notebook_has_no_outputs_and_no_final_command() -> None:
    notebook = json.loads((ROOT / "notebooks/phase11b_train_validate.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join(code_cells)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "final-test" not in code
    assert "materialize-trainval" in code
    guard_index = next(index for index, source in enumerate(code_cells) if "EXPECTED_PYTHON = (3, 11)" in source)
    drive_index = next(index for index, source in enumerate(code_cells) if "drive.mount" in source)
    checkout_index = next(index for index, source in enumerate(code_cells) if "git', 'checkout" in source)
    requirements_index = next(index for index, source in enumerate(code_cells) if "requirements-detection-colab.txt" in source)
    editable_index = next(index for index, source in enumerate(code_cells) if "--editable" in source)
    apparatus_index = next(index for index, source in enumerate(code_cells) if "apparatus-check" in source)
    assert guard_index < drive_index < checkout_index < requirements_index < editable_index < apparatus_index
    assert "sys.version_info[:2] != EXPECTED_PYTHON" in code_cells[guard_index]
    assert "Runtime version: 2025.07" in code_cells[guard_index]
    assert "--no-deps" in code_cells[editable_index]
    assert "sys.path" not in code


def test_clean_clone_editable_install_makes_windblade_importable(tmp_path: Path) -> None:
    clone = tmp_path / "clean-clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--local", "--no-hardlinks", str(ROOT), str(clone)],
        check=True, capture_output=True, text=True,
    )
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-deps", "--no-build-isolation", "--editable", str(clone),
        ],
        check=False, capture_output=True, text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    imported = subprocess.check_output(
        [str(python), "-I", "-c", "import pathlib, windblade; print(pathlib.Path(windblade.__file__).resolve())"],
        text=True,
    ).strip()
    assert Path(imported) == (clone / "src/windblade/__init__.py").resolve()


def test_dependencies_and_training_controls_are_explicitly_pinned(apparatus) -> None:
    requirements = (ROOT / apparatus["environment"]["requirements"]).read_text(encoding="utf-8").splitlines()
    packages = [line for line in requirements if line and not line.startswith(("#", "--"))]
    assert packages and all("==" in line for line in packages)
    training = apparatus["training"]
    assert training["batch_size"] == 16
    assert training["optimizer"] == "AdamW"
    assert training["early_stopping_monitor"] == "validation_map_50_95"
    assert training["early_stopping_min_delta"] == 0.0
    assert training["checkpoint_every_epochs"] == 1
    assert training["deterministic"] is True
    assert training["amp"] is True
    assert apparatus["environment"]["telemetry_disabled"] is True
    assert apparatus["environment"]["external_trackers"] is False


def test_repository_is_explicitly_agpl_licensed() -> None:
    assert "AGPL-3.0-only" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "AGPL-3.0-only" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
