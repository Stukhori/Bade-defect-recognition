from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import venv

import pytest

from windblade.detection import phase11b


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/detection_phase11b.yaml"


@pytest.fixture(scope="module")
def apparatus():
    return phase11b.load_apparatus(CONFIG_PATH)


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
