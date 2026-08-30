"""Frozen-model evaluation primitives; this module contains no fitting API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
import torch
import yaml

from windblade.data.processed import LABELS, read_csv
from windblade.deep.checkpoints import load_checkpoint
from windblade.deep.dataset import canonical_transform
from windblade.deep.mobilenet import build_mobilenet
from windblade.deep.resnet import build_resnet18
from windblade.evaluation.metrics import classification_metrics
from windblade.features.hog import extract_hog
from windblade.features.lbp import extract_spatial_lbp


class RobustnessEvaluationError(RuntimeError):
    """Raised when frozen inference or clean reproduction fails."""


def _class_id(row: Mapping[str, Any]) -> int:
    return int(row["class_id"] if "class_id" in row else row["true_class_id"])


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RobustnessEvaluationError(f"invalid frozen YAML: {path}")
    return payload


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 162:
        raise RobustnessEvaluationError(f"evaluation requires 162 fixed test instances; received {len(rows)}")
    if len({str(row["instance_id"]) for row in rows}) != 162:
        raise RobustnessEvaluationError("evaluation rows contain duplicate instance IDs")
    for row in rows:
        class_id = _class_id(row)
        label = str(row.get("canonical_label", row.get("true_label", "")))
        if class_id not in range(6) or label != LABELS[class_id]:
            raise RobustnessEvaluationError(f"invalid label identity: {row.get('instance_id')}")


def load_frozen_traditional(method: str, config: Mapping[str, Any], root: str | Path) -> tuple[Any, dict[str, Any]]:
    if method not in {"hog", "lbp"}:
        raise RobustnessEvaluationError(f"unsupported traditional method: {method}")
    repository = Path(root).resolve()
    reference = config["models"][method]
    frozen = _load_yaml(repository / reference["config"])
    if frozen["processed_dataset_fingerprint"] != config["dataset"]["base_fingerprint"]:
        raise RobustnessEvaluationError(f"{method} frozen dataset fingerprint mismatch")
    model = joblib.load(repository / reference["model"])
    if list(getattr(model, "named_steps", {})) != ["scaler", "svm"]:
        raise RobustnessEvaluationError(f"{method} frozen artifact is not the expected scaler/SVM pipeline")
    scaler, svm = model.named_steps["scaler"], model.named_steps["svm"]
    if not hasattr(scaler, "mean_") or not hasattr(svm, "support_vectors_"):
        raise RobustnessEvaluationError(f"{method} artifact is not fitted")
    return model, frozen["feature_config"]


def evaluate_traditional_condition(
    method: str,
    model: Any,
    feature_config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    image_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Transform and predict only; StandardScaler/SVM fitting is impossible here."""

    _validate_rows(rows)
    if len(image_paths) != len(rows):
        raise RobustnessEvaluationError("traditional evaluation image count mismatch")
    extractor = extract_hog if method == "hog" else extract_spatial_lbp
    features = np.stack([extractor(Path(path), feature_config) for path in image_paths])
    predicted = np.asarray(model.predict(features), dtype=np.int64)
    true = np.asarray([_class_id(row) for row in rows], dtype=np.int64)
    metrics = classification_metrics(true, predicted, LABELS)
    predictions = []
    for row, prediction in zip(rows, predicted, strict=True):
        class_id = _class_id(row)
        predictions.append(
            {
                "instance_id": str(row["instance_id"]),
                "source_image_id": str(row["source_image_id"]),
                "true_class_id": class_id,
                "true_label": LABELS[class_id],
                "predicted_class_id": int(prediction),
                "predicted_label": LABELS[int(prediction)],
                "correct": bool(class_id == int(prediction)),
            }
        )
    return {"metrics": metrics, "predictions": predictions, "logits": None}


def load_frozen_cnn(
    method: str,
    seed: int,
    config: Mapping[str, Any],
    root: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    if method not in {"resnet18", "mobilenet_v3_small"} or seed not in {17, 29, 43}:
        raise RobustnessEvaluationError(f"unsupported frozen CNN identity: {method}/seed{seed}")
    repository = Path(root).resolve()
    reference = config["models"][method]
    checkpoint = repository / reference["result_root"] / f"seed_{seed}" / "best_state_dict.pt"
    metadata = checkpoint.with_suffix(".json")
    state, record = load_checkpoint(
        checkpoint,
        expected_dataset_fingerprint=config["dataset"]["base_fingerprint"],
        metadata_path=metadata,
        expected_architecture=reference["architecture"],
    )
    if int(record["seed"]) != seed or tuple(record["class_order"]) != LABELS:
        raise RobustnessEvaluationError(f"frozen checkpoint metadata mismatch: {method}/seed{seed}")
    model = build_resnet18(pretrained=False) if method == "resnet18" else build_mobilenet(pretrained=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if model.training:
        raise RobustnessEvaluationError("CNN failed to enter evaluation mode")
    return model


def evaluate_cnn_condition(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    image_paths: Sequence[str | Path],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    _validate_rows(rows)
    if len(image_paths) != len(rows) or batch_size != 64:
        raise RobustnessEvaluationError("CNN evaluation input count or frozen batch size changed")
    transform = canonical_transform()
    all_logits: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            tensors = []
            for path in image_paths[start : start + batch_size]:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (224, 224):
                        raise RobustnessEvaluationError(f"non-canonical evaluation image: {path}")
                    tensors.append(transform(image.copy()))
            logits = model(torch.stack(tensors).to(device)).detach().cpu()
            all_logits.extend(logits.tolist())
    logits_array = np.asarray(all_logits, dtype=np.float32)
    predicted = logits_array.argmax(axis=1).astype(np.int64)
    true = np.asarray([_class_id(row) for row in rows], dtype=np.int64)
    metrics = classification_metrics(true, predicted, LABELS)
    predictions = []
    for row, prediction, logits in zip(rows, predicted, all_logits, strict=True):
        class_id = _class_id(row)
        record = {
            "instance_id": str(row["instance_id"]),
            "source_image_id": str(row["source_image_id"]),
            "true_class_id": class_id,
            "true_label": LABELS[class_id],
            "predicted_class_id": int(prediction),
            "predicted_label": LABELS[int(prediction)],
            "correct": bool(class_id == int(prediction)),
        }
        record.update({f"logit_{label}": float(logits[index]) for index, label in enumerate(LABELS)})
        predictions.append(record)
    return {"metrics": metrics, "predictions": predictions, "logits": all_logits}


def _normalized_prediction_rows(rows: Sequence[Mapping[str, Any]], *, expect_logits: bool) -> list[dict[str, Any]]:
    fields = (
        "instance_id",
        "source_image_id",
        "true_class_id",
        "true_label",
        "predicted_class_id",
        "predicted_label",
        "correct",
    )
    normalized = []
    for row in rows:
        item: dict[str, Any] = {
            "instance_id": str(row["instance_id"]),
            "source_image_id": str(row["source_image_id"]),
            "true_class_id": int(row["true_class_id"]),
            "true_label": str(row["true_label"]),
            "predicted_class_id": int(row["predicted_class_id"]),
            "predicted_label": str(row["predicted_label"]),
            "correct": str(row["correct"]).lower() == "true" if isinstance(row["correct"], str) else bool(row["correct"]),
        }
        if expect_logits:
            item.update({f"logit_{label}": float(row[f"logit_{label}"]) for label in LABELS})
        selected_fields = (*fields, *(tuple(f"logit_{label}" for label in LABELS) if expect_logits else ()))
        normalized.append({key: item[key] for key in selected_fields})
    return sorted(normalized, key=lambda row: row["instance_id"])


def verify_clean_reproduction(
    method: str,
    seed: int | None,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Require exact predictions, stored logits (CNN), and metrics."""

    repository = Path(root).resolve()
    reference = config["models"][method]
    if seed is None:
        prediction_path = repository / reference["predictions"]
        metrics_path = repository / reference["metrics"]
        identity = method
        expect_logits = False
    else:
        prediction_path = repository / reference["summary_root"] / f"seed_{seed}" / "test_predictions.csv"
        metrics_path = repository / reference["summary_root"] / f"seed_{seed}" / "test_metrics.json"
        identity = f"{method}/seed{seed}"
        expect_logits = True
    expected_predictions = _normalized_prediction_rows(read_csv(prediction_path), expect_logits=expect_logits)
    observed_predictions = _normalized_prediction_rows(result["predictions"], expect_logits=expect_logits)
    if expected_predictions != observed_predictions:
        raise RobustnessEvaluationError(f"clean predictions/logits do not reproduce: {identity}")
    expected_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if expected_metrics != result["metrics"]:
        raise RobustnessEvaluationError(f"clean metrics do not reproduce: {identity}")
    return {
        "method": method,
        "seed": seed,
        "status": "PASS",
        "predictions_exact": True,
        "logits_exact": True if expect_logits else None,
        "metrics_exact": True,
        "instances": len(observed_predictions),
    }
