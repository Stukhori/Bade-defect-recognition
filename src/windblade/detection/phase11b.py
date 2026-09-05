"""Reproducible Phase 11B YOLO11n apparatus with a validation/test firewall.

The module deliberately imports Ultralytics only inside GPU execution paths so
the repository's upstream validators and CPU test suite do not acquire a new
runtime dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import urllib.request
import zipfile

import yaml

from windblade.config import calculate_config_hash, load_config
from windblade.detection.core import box_iou, sha256_file, voc_inclusive_to_yolo


class Phase11BError(RuntimeError):
    """Raised when an apparatus or firewall invariant is violated."""


EXPECTED_MANIFEST_SHA256 = {
    "image_manifest": "1192550c71fe7bc801c3b7d96b3551187a3d86c344f52d49e842901c303a3ce7",
    "annotation_manifest": "28b348b2ffb9cd6f491d2e4d6017095369bac2365ed8de20f38df28865b6d610",
    "split_manifest": "059c448813a6c27a32c57f97eaeb2cc235ebdeb8949d421809aef434e3e3e35e",
}
EXPECTED_PHASE11A = {
    "dataset_fingerprint": "ad4ab59c3e3c85c6cf0b85b148177bd6b79d24f372f49bdff0043609e6fefc97",
    "split_fingerprint": "264f8460f203074374c2c098c8fd5d2e55fb7ee1f281a8d505e2dfb0de9a2bc3",
    "config_fingerprint": "9f4a20ba4404c9a6072277a504c466a0756143b908e79c7168d2ccf91ff32057",
    "scientific_output_fingerprint": "3f46cbdc6c7a2e3cf6093ff177dd1948d113fa4c36fa9eb907d7c8621e800461",
}
EXPECTED_COUNTS = {
    "images": 720,
    "boxes": 1065,
    "images_by_split": {"train": 510, "validation": 101, "test": 109},
    "boxes_by_split": {"train": 757, "validation": 146, "test": 162},
}
CLASS_ORDER = ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]
SEEDS = (17, 29, 43)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def load_apparatus(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase11BError("apparatus configuration must be a mapping")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, encoding="utf-8", capture_output=True, check=False
    )
    if check and result.returncode:
        raise Phase11BError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def validate_frozen_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Verify every frozen Phase 11A identity and the declared task matrix."""

    frozen = config["frozen_inputs"]
    protocol = config["scientific_protocol"]
    if tuple(protocol["seeds"]) != SEEDS or protocol["run_count"] != 3:
        raise Phase11BError("the frozen run matrix must be exactly seeds 17/29/43 and three total runs")
    if protocol["primary_task"] != "class_agnostic_localization":
        raise Phase11BError("the primary task must remain class-agnostic localization")

    paths = {key: repo / frozen[key] for key in EXPECTED_MANIFEST_SHA256}
    for key, expected in EXPECTED_MANIFEST_SHA256.items():
        if not paths[key].is_file() or sha256_file(paths[key]) != expected:
            raise Phase11BError(f"frozen {key} identity mismatch")

    phase11a = load_config(repo / frozen["phase11a_config"])
    config_fingerprint = calculate_config_hash(phase11a.as_dict(), 64)
    reproduction = json.loads((repo / frozen["audit_root"] / "reproducibility.json").read_text(encoding="utf-8"))
    observed_fingerprints = {
        "dataset_fingerprint": reproduction["dataset_fingerprint"],
        "split_fingerprint": reproduction["split_fingerprint"],
        "config_fingerprint": config_fingerprint,
        "scientific_output_fingerprint": reproduction["scientific_output_fingerprint"],
    }
    if observed_fingerprints != EXPECTED_PHASE11A:
        raise Phase11BError(f"frozen Phase 11A fingerprint mismatch: {observed_fingerprints}")
    for key, expected in EXPECTED_PHASE11A.items():
        configured_key = "phase11a_" + key if key in {"config_fingerprint", "scientific_output_fingerprint"} else key
        if frozen[configured_key] != expected:
            raise Phase11BError(f"apparatus-declared {configured_key} mismatch")

    images = read_csv(paths["image_manifest"])
    annotations = read_csv(paths["annotation_manifest"])
    splits = read_csv(paths["split_manifest"])
    image_counts = dict(Counter(row["split"] for row in images))
    box_counts = dict(Counter(row["split"] for row in annotations))
    observed_counts = {
        "images": len(images), "boxes": len(annotations),
        "images_by_split": image_counts, "boxes_by_split": box_counts,
    }
    if observed_counts != EXPECTED_COUNTS or frozen["expected"] != EXPECTED_COUNTS:
        raise Phase11BError(f"frozen count mismatch: {observed_counts}")
    split_lookup = {row["source_image_id"]: row["split"] for row in splits}
    if len(split_lookup) != 720 or any(split_lookup[row["source_image_id"]] != row["split"] for row in images):
        raise Phase11BError("split manifest does not match the image manifest")

    class_mapping = json.loads((repo / frozen["class_mapping"]).read_text(encoding="utf-8"))
    expected_multiclass = {name: index for index, name in enumerate(CLASS_ORDER)}
    if class_mapping != {"class_agnostic": {"defect": 0}, "multiclass": expected_multiclass}:
        raise Phase11BError("frozen class mapping mismatch")
    for row in annotations:
        if int(row["class_id"]) != expected_multiclass[row["class_name"]] or int(row["defect_class_id"]) != 0:
            raise Phase11BError(f"invalid class mapping for {row['instance_id']}")
        image = next(item for item in images if item["source_image_id"] == row["source_image_id"])
        converted = voc_inclusive_to_yolo(
            tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")),
            int(image["width"]), int(image["height"]),
        )
        declared = tuple(float(row[key]) for key in ("yolo_x_center", "yolo_y_center", "yolo_width", "yolo_height"))
        if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12) for left, right in zip(converted, declared)):
            raise Phase11BError(f"one-based-inclusive conversion mismatch for {row['instance_id']}")
    return {"status": "PASS", **observed_counts, **observed_fingerprints, "run_count": 3, "seeds": list(SEEDS)}


def verify_archive(config: Mapping[str, Any], archive: Path) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    if not archive.is_file():
        raise Phase11BError(f"raw archive does not exist: {archive}")
    observed = {"filename": archive.name, "size_bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
    expected = {
        "filename": frozen["raw_archive_filename"],
        "size_bytes": frozen["raw_archive_bytes"],
        "sha256": frozen["raw_archive_sha256"],
    }
    if observed != expected:
        raise Phase11BError(f"raw archive identity mismatch: {observed}")
    return {"status": "PASS", **observed}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise Phase11BError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _manifest_rows(config: Mapping[str, Any], repo: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    frozen = config["frozen_inputs"]
    return read_csv(repo / frozen["image_manifest"]), read_csv(repo / frozen["annotation_manifest"])


def format_class_agnostic_yolo_label(annotation: Mapping[str, Any], image: Mapping[str, Any]) -> str:
    """Convert one frozen one-based-inclusive VOC box to a class-zero label."""

    values = voc_inclusive_to_yolo(
        tuple(float(annotation[key]) for key in ("xmin", "ymin", "xmax", "ymax")),
        int(image["width"]), int(image["height"]),
    )
    return "0 " + " ".join(f"{value:.12f}" for value in values)


def materialize_dataset(
    config: Mapping[str, Any], repo: Path, archive: Path, destination: Path, *, scope: str = "trainval"
) -> dict[str, Any]:
    """Materialize immutable YOLO labels; trainval deliberately excludes test."""

    if scope not in {"trainval", "test"}:
        raise Phase11BError("materialization scope must be trainval or test")
    validate_frozen_inputs(config, repo)
    verify_archive(config, archive)
    if destination.exists():
        manifest = destination / "materialization.json"
        if manifest.is_file():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            identity_matches = (
                existing.get("scope") == scope
                and existing.get("source_archive_sha256") == config["frozen_inputs"]["raw_archive_sha256"]
            )
            artifacts_match = all(
                (destination / item["path"]).is_file()
                and sha256_file(destination / item["path"]) == item["sha256"]
                for item in existing.get("artifacts", [])
            )
            if identity_matches and existing.get("artifacts") and artifacts_match:
                return existing
        raise Phase11BError(f"refusing to replace existing materialization: {destination}")

    images, annotations = _manifest_rows(config, repo)
    selected = {"train", "validation"} if scope == "trainval" else {"test"}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotations:
        grouped[row["source_image_id"]].append(row)
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    raw_root = temporary / "raw"
    dataset_root = temporary / "dataset"
    try:
        _safe_extract(archive, raw_root)
        source_images = raw_root / "WT blade defect dataset" / "JPEGImages"
        if not source_images.is_dir():
            raise Phase11BError("expected JPEGImages directory is absent from the verified archive")
        artifact_rows: list[dict[str, Any]] = []
        for image in sorted(images, key=lambda row: int(row["source_image_id"])):
            source = source_images / image["filename"]
            if not source.is_file() or sha256_file(source) != image["sha256"]:
                raise Phase11BError(f"image identity mismatch: {image['filename']}")
            if image["split"] not in selected:
                continue
            image_out = dataset_root / "images" / image["split"] / image["filename"]
            label_out = dataset_root / "labels" / image["split"] / f"{Path(image['filename']).stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, image_out)
            labels = []
            for annotation in sorted(grouped[image["source_image_id"]], key=lambda row: int(row["object_index"])):
                labels.append(format_class_agnostic_yolo_label(annotation, image))
            label_out.write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")
            artifact_rows.extend([
                {"path": image_out.relative_to(temporary).as_posix(), "sha256": sha256_file(image_out)},
                {"path": label_out.relative_to(temporary).as_posix(), "sha256": sha256_file(label_out)},
            ])
        yaml_value: dict[str, Any] = {"path": str((destination / "dataset").resolve()), "names": {0: "defect"}}
        if scope == "trainval":
            yaml_value.update({"train": "images/train", "val": "images/validation"})
        else:
            yaml_value.update({"test": "images/test"})
        data_yaml = dataset_root / ("trainval.yaml" if scope == "trainval" else "test.yaml")
        data_yaml.write_text(yaml.safe_dump(yaml_value, sort_keys=False), encoding="utf-8", newline="\n")
        included_images = [row for row in images if row["split"] in selected]
        included_boxes = [row for row in annotations if row["split"] in selected]
        result = {
            "status": "PASS", "scope": scope,
            "source_archive_sha256": config["frozen_inputs"]["raw_archive_sha256"],
            "included_splits": sorted(selected), "image_count": len(included_images), "box_count": len(included_boxes),
            "artifact_fingerprint": canonical_hash(sorted(artifact_rows, key=lambda row: row["path"])),
            "artifacts": sorted(artifact_rows, key=lambda row: row["path"]),
        }
        atomic_json(temporary / "materialization.json", result)
        temporary.replace(destination)
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


@dataclass(frozen=True)
class DriveLayout:
    root: Path
    runs: Path
    provenance: Path
    selection: Path
    bundles: Path

    @classmethod
    def from_root(cls, root: Path) -> "DriveLayout":
        if not root.is_absolute():
            raise Phase11BError("Google Drive root must be absolute")
        resolved = root.resolve()
        return cls(resolved, resolved / "runs", resolved / "provenance", resolved / "selection", resolved / "bundles")

    def run(self, seed: int) -> Path:
        if seed not in SEEDS:
            raise Phase11BError(f"undeclared seed: {seed}")
        return self.runs / f"seed_{seed}"


def decide_resume(run_dir: Path, seed: int, config_sha256: str) -> Path | None:
    state_path, checkpoint = run_dir / "run_state.json", run_dir / "weights" / "last.pt"
    if not checkpoint.exists() and not state_path.exists():
        return None
    if not checkpoint.is_file() or not state_path.is_file():
        raise Phase11BError("partial run state cannot be resumed safely")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("seed") != seed or state.get("configuration_sha256") != config_sha256:
        raise Phase11BError("resume checkpoint belongs to a different seed or configuration")
    if state.get("weight_sha256") and state["weight_sha256"] != sha256_file(Path(state["weight_path"])):
        raise Phase11BError("resume run's initial-weight identity changed")
    return checkpoint


def create_weight_record(
    config: Mapping[str, Any], repo: Path, weight: Path, apparatus_commit: str, output: Path
) -> dict[str, Any]:
    if len(apparatus_commit) != 40 or any(character not in "0123456789abcdef" for character in apparatus_commit):
        raise Phase11BError("apparatus commit must be a full lowercase Git SHA")
    _git(repo, "cat-file", "-e", f"{apparatus_commit}^{{commit}}")
    if _git(repo, "rev-parse", "HEAD") != apparatus_commit:
        raise Phase11BError("weight acquisition must run from the exact apparatus commit")
    detector = config["detector"]
    if weight.name != detector["pretrained_filename"]:
        raise Phase11BError("pretrained weight filename mismatch")
    record = {
        "schema_version": "1.0", "status": "RECORDED_BEFORE_TRAINING",
        "apparatus_commit": apparatus_commit,
        "official_source": detector["pretrained_official_url"],
        "license": detector["license"], "license_source": detector["license_source"],
        "filename": weight.name, "size_bytes": weight.stat().st_size, "sha256": sha256_file(weight),
        "training_started": False,
    }
    atomic_json(output, record)
    return record


def acquire_official_weight(
    config: Mapping[str, Any], repo: Path, destination: Path, apparatus_commit: str, record_output: Path,
) -> dict[str, Any]:
    """Acquire the pinned official filename and record its observed bytes once."""

    if destination.exists() or record_output.exists():
        raise Phase11BError("refusing to overwrite an existing weight or acquisition record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(config["detector"]["pretrained_official_url"], headers={"User-Agent": "windblade-phase11b/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
        return create_weight_record(config, repo, destination, apparatus_commit, record_output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def validate_weight_record(config: Mapping[str, Any], weight: Path, record_path: Path) -> dict[str, Any]:
    if not record_path.is_file():
        raise Phase11BError("weight-acquisition record is required before training")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    detector = config["detector"]
    expected = {
        "official_source": detector["pretrained_official_url"], "license": detector["license"],
        "license_source": detector["license_source"], "filename": detector["pretrained_filename"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise Phase11BError("weight-acquisition provenance does not match the apparatus")
    if record.get("training_started") is not False or record.get("status") != "RECORDED_BEFORE_TRAINING":
        raise Phase11BError("weight record does not certify pre-training acquisition")
    if not weight.is_file() or record.get("size_bytes") != weight.stat().st_size or record.get("sha256") != sha256_file(weight):
        raise Phase11BError("pretrained weight bytes do not match their acquisition record")
    return record


def disable_external_services() -> None:
    values = {
        "WANDB_DISABLED": "true", "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED",
        "CLEARML_OFFLINE_MODE": "1", "NEPTUNE_MODE": "offline", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "ULTRALYTICS_HUB": "false", "DO_NOT_TRACK": "1",
    }
    os.environ.update(values)


def environment_preflight(config: Mapping[str, Any], repo: Path, archive: Path, drive_root: Path) -> dict[str, Any]:
    validate_frozen_inputs(config, repo)
    archive_result = verify_archive(config, archive)
    layout = DriveLayout.from_root(drive_root)
    if sys.version_info[:2] != (3, 11):
        raise Phase11BError(f"Python 3.11 is required; observed {sys.version.split()[0]}")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics")}
    expected = {"torch": config["environment"]["torch"], "torchvision": config["environment"]["torchvision"], "ultralytics": config["detector"]["version"]}
    if any(versions[name].split("+")[0] != expected[name] for name in expected):
        raise Phase11BError(f"dependency version mismatch: expected {expected}, observed {versions}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise Phase11BError("CUDA GPU is required")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram < float(config["environment"]["minimum_vram_gib"]):
        raise Phase11BError(f"at least 8 GiB GPU memory is required; observed {vram:.2f} GiB")
    for path in (layout.root, layout.runs, layout.provenance, layout.selection, layout.bundles):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "status": "PASS", "python": sys.version.split()[0], "dependencies": versions,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "vram_gib": round(vram, 3),
        "archive": archive_result, "drive_root": str(layout.root), "telemetry_disabled": True,
        "repository_commit": _git(repo, "rev-parse", "HEAD"),
        "requirements_sha256": sha256_file(repo / config["environment"]["requirements"]),
    }


def _ultralytics_settings_off() -> None:
    disable_external_services()
    from ultralytics import settings
    disabled = {key: False for key in ("sync", "wandb", "mlflow", "clearml", "comet", "dvc", "hub", "neptune", "raytune", "tensorboard") if key in settings}
    if disabled:
        settings.update(disabled)


def training_arguments(config: Mapping[str, Any], data_yaml: Path, layout: DriveLayout, seed: int) -> dict[str, Any]:
    train, aug = config["training"], config["training"]["augmentations"]
    return {
        "data": str(data_yaml), "epochs": train["epochs"], "patience": train["patience"],
        "imgsz": train["image_size"], "batch": train["batch_size"], "optimizer": train["optimizer"],
        "lr0": train["initial_learning_rate"], "lrf": train["final_learning_rate_fraction"],
        "cos_lr": train["schedule"] == "cosine", "warmup_epochs": train["warmup_epochs"],
        "warmup_momentum": train["warmup_momentum"], "warmup_bias_lr": train["warmup_bias_learning_rate"],
        "momentum": train["momentum"], "weight_decay": train["weight_decay"], "amp": train["amp"],
        "workers": train["workers"], "save_period": train["checkpoint_every_epochs"],
        "deterministic": train["deterministic"], "single_cls": train["single_class"], "cache": train["cache"],
        "rect": train["rectangular_batches"], "multi_scale": train["multi_scale"],
        "hsv_h": aug["hsv_h"], "hsv_s": aug["hsv_s"], "hsv_v": aug["hsv_v"],
        "degrees": aug["degrees"], "translate": aug["translate"], "scale": aug["scale"],
        "shear": aug["shear"], "perspective": aug["perspective"], "flipud": aug["flip_up_down"],
        "fliplr": aug["flip_left_right"], "mosaic": aug["mosaic"], "mixup": aug["mixup"],
        "copy_paste": aug["copy_paste"], "close_mosaic": aug["close_mosaic_epochs"],
        "seed": seed, "device": 0, "project": str(layout.runs), "name": f"seed_{seed}",
        "exist_ok": True, "pretrained": True, "plots": True, "verbose": True,
    }


def train_seed(
    config: Mapping[str, Any], config_path: Path, repo: Path, data_root: Path,
    drive_root: Path, weight: Path, record_path: Path, seed: int,
) -> dict[str, Any]:
    validate_frozen_inputs(config, repo)
    record = validate_weight_record(config, weight, record_path)
    apparatus_commit = str(record.get("apparatus_commit", ""))
    _git(repo, "merge-base", "--is-ancestor", apparatus_commit, "HEAD")
    config_relative = config_path.relative_to(repo).as_posix()
    _git(repo, "diff", "--quiet", apparatus_commit, "--", config_relative)
    _git(repo, "diff", "--quiet", "HEAD", "--", config_relative)
    if seed not in SEEDS:
        raise Phase11BError(f"undeclared seed: {seed}")
    materialization = json.loads((data_root / "materialization.json").read_text(encoding="utf-8"))
    if materialization.get("scope") != "trainval" or materialization.get("included_splits") != ["train", "validation"]:
        raise Phase11BError("training accepts only the train/validation materialization")
    data_yaml = data_root / "dataset" / "trainval.yaml"
    layout = DriveLayout.from_root(drive_root)
    run_dir = layout.run(seed)
    config_sha = sha256_file(config_path)
    resume = decide_resume(run_dir, seed, config_sha)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "seed": seed, "configuration_path": config_relative,
        "configuration_sha256": config_sha, "apparatus_commit": record["apparatus_commit"],
        "weight_path": str(weight.resolve()), "weight_sha256": record["sha256"],
        "materialization_fingerprint": materialization["artifact_fingerprint"],
    }
    atomic_json(run_dir / "run_state.json", state)
    _ultralytics_settings_off()
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from ultralytics import YOLO
    if resume:
        model = YOLO(str(resume))
        model.train(resume=True)
    else:
        model = YOLO(str(weight))
        model.train(**training_arguments(config, data_yaml, layout, seed))
    trainer = model.trainer
    expected_batch = int(config["training"]["batch_size"])
    applied_batch = int(trainer.args.batch)
    applied_optimizer = type(trainer.optimizer).__name__
    applied_amp = bool(trainer.amp)
    if applied_batch != expected_batch or applied_optimizer != config["training"]["optimizer"] or not applied_amp:
        raise Phase11BError(
            "training runtime changed a frozen batch/optimizer/AMP control: "
            f"batch={applied_batch}, optimizer={applied_optimizer}, amp={applied_amp}"
        )
    state.update({
        "status": "TRAINING_COMMAND_COMPLETED",
        "last_checkpoint_sha256": sha256_file(run_dir / "weights" / "last.pt"),
        "applied_batch_size": applied_batch, "applied_optimizer": applied_optimizer, "applied_amp": applied_amp,
    })
    atomic_json(run_dir / "run_state.json", state)
    return state


def choose_checkpoint(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise Phase11BError("no validation checkpoint candidates")
    normalized = []
    for candidate in candidates:
        metric = float(candidate["validation_map_50_95"])
        epoch = int(candidate["epoch"])
        if not math.isfinite(metric) or not 0 <= metric <= 1:
            raise Phase11BError("invalid validation mAP@0.50:0.95")
        normalized.append((metric, -epoch, dict(candidate)))
    return max(normalized, key=lambda item: (item[0], item[1]))[2]


def checkpoint_candidates(run_dir: Path) -> list[dict[str, Any]]:
    results = run_dir / "results.csv"
    if not results.is_file():
        raise Phase11BError(f"missing validation history: {results}")
    rows = read_csv(results)
    metric_key = next((key for key in rows[0] if key.strip() == "metrics/mAP50-95(B)"), None) if rows else None
    if metric_key is None:
        raise Phase11BError("Ultralytics results lack validation mAP50-95(B)")
    candidates = []
    for epoch, row in enumerate(rows):
        checkpoint = run_dir / "weights" / f"epoch{epoch}.pt"
        if not checkpoint.is_file() and epoch == len(rows) - 1:
            checkpoint = run_dir / "weights" / "last.pt"
        if checkpoint.is_file():
            candidates.append({
                "epoch": epoch + 1, "validation_map_50_95": float(row[metric_key]),
                "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), "size_bytes": checkpoint.stat().st_size,
            })
    return candidates


def validate_completed_run(run_dir: Path, seed: int, configuration_sha256: str) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise Phase11BError(f"missing run state for seed {seed}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("seed") != seed
        or state.get("configuration_sha256") != configuration_sha256
        or state.get("status") != "TRAINING_COMMAND_COMPLETED"
    ):
        raise Phase11BError(f"seed {seed} is not a completed isolated run for this configuration")
    last = run_dir / "weights" / "last.pt"
    if not last.is_file() or state.get("last_checkpoint_sha256") != sha256_file(last):
        raise Phase11BError(f"seed {seed} last checkpoint identity mismatch")
    return state


def threshold_grid(specification: Mapping[str, float]) -> list[float]:
    start, stop, step = (Decimal(str(specification[key])) for key in ("start", "stop", "step"))
    if step <= 0 or start < 0 or stop > 1 or start > stop:
        raise Phase11BError("invalid threshold grid")
    values, current = [], start
    while current <= stop:
        values.append(float(current)); current += step
    return values


def threshold_metrics(
    truth_by_image: Mapping[str, Sequence[Sequence[float]]],
    predictions_by_image: Mapping[str, Sequence[Mapping[str, Any]]], thresholds: Sequence[float], iou: float,
) -> list[dict[str, float | int]]:
    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for image_id, truth in truth_by_image.items():
            unmatched = set(range(len(truth)))
            predictions = sorted(
                (row for row in predictions_by_image.get(image_id, []) if float(row["score"]) >= threshold),
                key=lambda row: (-float(row["score"]), tuple(row["box"])),
            )
            for prediction in predictions:
                options = [(box_iou(prediction["box"], truth[index]), -index, index) for index in unmatched]
                best = max(options, default=(0.0, 0, -1))
                if best[0] >= iou:
                    tp += 1; unmatched.remove(best[2])
                else:
                    fp += 1
            fn += len(unmatched)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1})
    return rows


def choose_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Phase11BError("no validation threshold candidates")
    return dict(max(rows, key=lambda row: (float(row["f1"]), -float(row["threshold"]))))


def write_deterministic_zip(output: Path, files: Mapping[str, Path]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for archive_name, source in sorted(files.items()):
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, source.read_bytes())
    return sha256_file(output)


def validate_receipt_firewall(config: Mapping[str, Any], config_path: Path, repo: Path, drive_root: Path) -> dict[str, Any]:
    receipt_relative = config["firewall"]["committed_selection_receipt"]
    receipt_path = repo / receipt_relative
    _git(repo, "ls-files", "--error-unmatch", receipt_relative)
    config_relative = config_path.relative_to(repo).as_posix()
    _git(repo, "ls-files", "--error-unmatch", config_relative)
    if _git(repo, "diff", "--name-only", "HEAD", "--", receipt_relative, config_relative):
        raise Phase11BError("selection receipt and apparatus configuration must be committed and clean")
    committed = _git(repo, "show", f"HEAD:{receipt_relative}") + "\n"
    if committed.encode("utf-8") != receipt_path.read_bytes():
        raise Phase11BError("working selection receipt differs from committed bytes")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_BEFORE_TEST" or receipt.get("configuration", {}).get("sha256") != sha256_file(config_path):
        raise Phase11BError("selection receipt does not freeze the committed configuration")
    if receipt.get("nms") != config["nms"]:
        raise Phase11BError("selection receipt NMS differs from the apparatus")
    checkpoints = receipt.get("checkpoints", [])
    if sorted(item.get("seed") for item in checkpoints) != list(SEEDS):
        raise Phase11BError("selection receipt must lock all three seed checkpoints")
    for item in checkpoints:
        path = DriveLayout.from_root(drive_root).root / item["drive_relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise Phase11BError(f"frozen checkpoint mismatch for seed {item['seed']}")
    layout = DriveLayout.from_root(drive_root)
    candidate_path = layout.selection / "validation_checkpoint_candidates.json"
    threshold_path = layout.selection / "validation_threshold_candidates.json"
    artifacts = receipt.get("validation_artifacts", {})
    if (
        not candidate_path.is_file()
        or not threshold_path.is_file()
        or sha256_file(candidate_path) != artifacts.get("checkpoint_candidates_sha256")
        or sha256_file(threshold_path) != artifacts.get("threshold_candidates_sha256")
        or sha256_file(threshold_path) != receipt.get("threshold", {}).get("candidate_artifact_sha256")
    ):
        raise Phase11BError("validation selection artifacts are absent or changed")
    candidate_registry = json.loads(candidate_path.read_text(encoding="utf-8"))
    for selected in checkpoints:
        expected_selection = choose_checkpoint(candidate_registry[str(selected["seed"])])
        if any(
            selected[key] != expected_selection[key]
            for key in ("epoch", "validation_map_50_95", "sha256", "size_bytes")
        ):
            raise Phase11BError(f"checkpoint selection policy mismatch for seed {selected['seed']}")
    threshold_rows = json.loads(threshold_path.read_text(encoding="utf-8"))
    expected_threshold = choose_threshold(threshold_rows)
    if (
        receipt["threshold"]["value"] != expected_threshold["threshold"]
        or receipt["threshold"]["validation_f1"] != expected_threshold["f1"]
    ):
        raise Phase11BError("threshold selection policy mismatch")
    hashes = receipt.get("frozen_hashes", {})
    expected = {
        "configuration": sha256_file(config_path),
        "checkpoints": canonical_hash(checkpoints),
        "threshold": canonical_hash(receipt["threshold"]),
        "nms": canonical_hash(receipt["nms"]),
    }
    if hashes != expected:
        raise Phase11BError("selection receipt hash firewall failed")
    return receipt


def _split_truth(
    config: Mapping[str, Any], repo: Path, split: str,
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    images, annotations = _manifest_rows(config, repo)
    filename_to_id = {row["filename"]: row["source_image_id"] for row in images if row["split"] == split}
    truth: dict[str, list[list[float]]] = defaultdict(list)
    for row in annotations:
        if row["split"] == split:
            truth[row["source_image_id"]].append([float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax")])
    return dict(truth), filename_to_id


def _predict_split(
    config: Mapping[str, Any], repo: Path, data_root: Path, checkpoint: Path, split: str, confidence: float,
) -> tuple[dict[str, list[list[float]]], dict[str, list[dict[str, Any]]]]:
    """Return ground truth and post-NMS predictions for one explicit split."""

    _ultralytics_settings_off()
    from ultralytics import YOLO
    truth, filename_to_id = _split_truth(config, repo, split)
    source = sorted((data_root / "dataset" / "images" / split).glob("*.jpg"), key=lambda path: int(path.stem))
    nms, training = config["nms"], config["training"]
    model = YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in source], imgsz=training["image_size"], batch=training["batch_size"],
        conf=confidence, iou=nms["iou_threshold"], max_det=nms["maximum_detections"],
        agnostic_nms=nms["class_agnostic"], device=0, workers=training["workers"], stream=True,
        save=False, save_txt=False, save_conf=False, verbose=False,
    )
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        image_id = filename_to_id[Path(result.path).name]
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for box, score in zip(xyxy, scores):
            predictions[image_id].append({"box": [float(value) for value in box], "score": float(score)})
    return truth, dict(predictions)


def _predict_validation(
    config: Mapping[str, Any], repo: Path, data_root: Path, checkpoint: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, list[dict[str, Any]]]]:
    return _predict_split(config, repo, data_root, checkpoint, "validation", float(config["nms"]["confidence_floor"]))


def select_validation_configuration(
    config: Mapping[str, Any], config_path: Path, repo: Path, data_root: Path,
    drive_root: Path, receipt_output: Path,
) -> dict[str, Any]:
    """Select one checkpoint per seed and one pooled validation threshold."""

    validate_frozen_inputs(config, repo)
    materialization = json.loads((data_root / "materialization.json").read_text(encoding="utf-8"))
    if materialization.get("scope") != "trainval":
        raise Phase11BError("selection requires the train/validation-only materialization")
    layout = DriveLayout.from_root(drive_root)
    selected, pooled_truth, pooled_predictions = [], {}, {}
    candidate_registry: dict[str, Any] = {}
    configuration_sha256 = sha256_file(config_path)
    for seed in SEEDS:
        validate_completed_run(layout.run(seed), seed, configuration_sha256)
        candidates = checkpoint_candidates(layout.run(seed))
        choice = choose_checkpoint(candidates)
        checkpoint = Path(choice["path"])
        choice["seed"] = seed
        try:
            choice["drive_relative_path"] = checkpoint.relative_to(layout.root).as_posix()
        except ValueError as exc:
            raise Phase11BError("selected checkpoint is outside the declared Drive root") from exc
        choice.pop("path")
        selected.append(choice)
        candidate_registry[str(seed)] = candidates
        truth, predictions = _predict_validation(config, repo, data_root, checkpoint)
        for image_id, boxes in truth.items():
            pooled_truth[f"{seed}:{image_id}"] = boxes
            pooled_predictions[f"{seed}:{image_id}"] = predictions.get(image_id, [])
    candidates_path = layout.selection / "validation_checkpoint_candidates.json"
    atomic_json(candidates_path, candidate_registry)
    threshold_rows = threshold_metrics(
        pooled_truth, pooled_predictions, threshold_grid(config["scientific_protocol"]["threshold_grid"]),
        float(config["scientific_protocol"]["threshold_match_iou"]),
    )
    threshold_path = layout.selection / "validation_threshold_candidates.json"
    atomic_json(threshold_path, threshold_rows)
    threshold_choice = choose_threshold(threshold_rows)
    threshold_payload = {
        "value": threshold_choice["threshold"], "validation_f1": threshold_choice["f1"],
        "policy": config["scientific_protocol"]["threshold_selection"],
        "match_iou": config["scientific_protocol"]["threshold_match_iou"],
        "candidate_artifact_sha256": sha256_file(threshold_path),
    }
    nms_payload = dict(config["nms"])
    receipt = {
        "schema_version": "1.0", "status": "FROZEN_BEFORE_TEST",
        "configuration": {"path": config_path.relative_to(repo).as_posix(), "sha256": sha256_file(config_path)},
        "scientific_task": "class_agnostic_localization", "seeds": list(SEEDS),
        "checkpoints": selected, "threshold": threshold_payload, "nms": nms_payload,
        "validation_artifacts": {
            "checkpoint_candidates_sha256": sha256_file(candidates_path),
            "threshold_candidates_sha256": sha256_file(threshold_path),
        },
        "test_evaluated": False, "no_post_test_tuning": True,
    }
    receipt["frozen_hashes"] = {
        "configuration": receipt["configuration"]["sha256"],
        "checkpoints": canonical_hash(receipt["checkpoints"]),
        "threshold": canonical_hash(receipt["threshold"]),
        "nms": canonical_hash(receipt["nms"]),
    }
    atomic_json(receipt_output, receipt)
    return receipt


def generate_training_bundle(
    config: Mapping[str, Any], config_path: Path, repo: Path, drive_root: Path, receipt_path: Path,
) -> dict[str, Any]:
    """Create a byte-deterministic ZIP of locked training and selection artifacts."""

    layout = DriveLayout.from_root(drive_root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    files: dict[str, Path] = {
        "repository/configs/detection_phase11b.yaml": config_path,
        "repository/requirements-detection-colab.txt": repo / config["environment"]["requirements"],
        "repository/provenance/phase11b_selection_receipt.json": receipt_path,
        "drive/selection/validation_checkpoint_candidates.json": layout.selection / "validation_checkpoint_candidates.json",
        "drive/selection/validation_threshold_candidates.json": layout.selection / "validation_threshold_candidates.json",
    }
    weight_record = layout.root / config["colab"]["weight_record_relative_path"]
    files["drive/provenance/phase11b_weight_acquisition.json"] = weight_record
    for item in receipt["checkpoints"]:
        seed = int(item["seed"])
        checkpoint = layout.root / item["drive_relative_path"]
        if sha256_file(checkpoint) != item["sha256"]:
            raise Phase11BError(f"checkpoint changed before bundling: seed {seed}")
        files[f"drive/runs/seed_{seed}/selected.pt"] = checkpoint
        files[f"drive/runs/seed_{seed}/results.csv"] = layout.run(seed) / "results.csv"
        files[f"drive/runs/seed_{seed}/run_state.json"] = layout.run(seed) / "run_state.json"
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise Phase11BError("training bundle inputs are missing: " + ", ".join(missing))
    inventory = {name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in sorted(files.items())}
    manifest_path = layout.bundles / "phase11b_training_bundle_manifest.json"
    atomic_json(manifest_path, {"schema_version": "1.0", "files": inventory, "inventory_sha256": canonical_hash(inventory)})
    files["manifest.json"] = manifest_path
    output = layout.bundles / "phase11b_training_bundle.zip"
    digest = write_deterministic_zip(output, files)
    result = {"status": "PASS", "path": str(output), "sha256": digest, "size_bytes": output.stat().st_size}
    atomic_json(layout.bundles / "phase11b_training_bundle_record.json", result)
    return result


def run_final_test(
    config: Mapping[str, Any], config_path: Path, repo: Path, archive: Path,
    data_root: Path, drive_root: Path,
) -> dict[str, Any]:
    """Evaluate test once, but only after the committed selection receipt passes."""

    receipt = validate_receipt_firewall(config, config_path, repo, drive_root)
    layout = DriveLayout.from_root(drive_root)
    output = layout.selection / "final_test_metrics.json"
    if output.exists():
        raise Phase11BError("final test output already exists; reruns and post-test tuning are prohibited")
    test_root = data_root.with_name(data_root.name + "_test")
    materialize_dataset(config, repo, archive, test_root, scope="test")
    _ultralytics_settings_off()
    from ultralytics import YOLO
    rows = []
    for item in receipt["checkpoints"]:
        checkpoint = layout.root / item["drive_relative_path"]
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(test_root / "dataset" / "test.yaml"), split="test", imgsz=config["training"]["image_size"],
            batch=config["training"]["batch_size"], conf=config["nms"]["confidence_floor"],
            iou=config["nms"]["iou_threshold"], max_det=config["nms"]["maximum_detections"],
            agnostic_nms=config["nms"]["class_agnostic"], device=0, workers=config["training"]["workers"],
            plots=False, save_json=False, verbose=False,
        )
        truth, predictions = _predict_split(
            config, repo, test_root, checkpoint, "test", float(receipt["threshold"]["value"])
        )
        operating = threshold_metrics(
            truth, predictions, [float(receipt["threshold"]["value"])],
            float(config["scientific_protocol"]["threshold_match_iou"]),
        )[0]
        rows.append({
            "seed": item["seed"], "checkpoint_sha256": item["sha256"],
            "map_50_95": float(metrics.box.map), "map_50": float(metrics.box.map50),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "frozen_operating_threshold": receipt["threshold"]["value"],
            "frozen_threshold_f1": operating["f1"],
            "frozen_threshold_true_positive": operating["true_positive"],
            "frozen_threshold_false_positive": operating["false_positive"],
            "frozen_threshold_false_negative": operating["false_negative"],
        })
    result = {
        "schema_version": "1.0", "status": "FINAL_TEST_COMPLETE_NO_FURTHER_TUNING",
        "receipt_sha256": sha256_file(repo / config["firewall"]["committed_selection_receipt"]),
        "configuration_sha256": sha256_file(config_path), "per_seed": rows,
        "aggregate_map_50_95": {
            "mean": sum(row["map_50_95"] for row in rows) / 3,
            "sample_sd": math.sqrt(sum((row["map_50_95"] - sum(item["map_50_95"] for item in rows) / 3) ** 2 for row in rows) / 2),
        },
        "no_post_test_tuning": True,
    }
    atomic_json(output, result)
    return result
