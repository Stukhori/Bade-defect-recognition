"""Training, selection, prediction, and three-seed aggregation primitives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from windblade.data.processed import LABELS
from windblade.evaluation.metrics import classification_metrics


def hyperparameter_grid() -> list[dict[str, float]]:
    return [
        {"learning_rate": lr, "weight_decay": decay}
        for lr in (1e-4, 3e-4)
        for decay in (0.0, 1e-4)
    ]


def candidate_key(row: Mapping[str, Any], tolerance: float = 1e-12) -> tuple[float, ...]:
    del tolerance
    return (
        -float(row["validation_macro_f1"]),
        -float(row["validation_balanced_accuracy"]),
        -float(row["validation_macro_recall"]),
        float(row["learning_rate"]),
        float(row["weight_decay"]),
    )


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValueError("Phase 5 selection requires exactly four validation candidates")
    return dict(min(rows, key=candidate_key))


@dataclass
class EarlyStopping:
    patience: int = 6
    min_delta: float = 1e-4
    tolerance: float = 1e-12
    best_epoch: int | None = None
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement: int = 0

    def update(self, epoch: int, metrics: Mapping[str, float]) -> tuple[bool, bool]:
        current = {key: float(metrics[key]) for key in ("macro_f1", "balanced_accuracy", "macro_recall")}
        improved = self.best_metrics is None
        if self.best_metrics is not None:
            delta = current["macro_f1"] - self.best_metrics["macro_f1"]
            if delta >= self.min_delta - self.tolerance:
                improved = True
            elif abs(delta) <= self.tolerance:
                improved = (
                    current["balanced_accuracy"], current["macro_recall"], -epoch
                ) > (
                    self.best_metrics["balanced_accuracy"],
                    self.best_metrics["macro_recall"],
                    -int(self.best_epoch),
                )
        if improved:
            self.best_epoch, self.best_metrics = epoch, current
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved, self.epochs_without_improvement >= self.patience


def run_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    true: list[int] = []
    predicted: list[int] = []
    records: list[dict[str, Any]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["class_id"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_predictions = logits.argmax(dim=1)
            losses.extend([float(loss.detach().cpu())] * len(labels))
            true.extend(labels.detach().cpu().tolist())
            predicted.extend(batch_predictions.detach().cpu().tolist())
            for index in range(len(labels)):
                records.append(
                    {
                        "instance_id": batch["instance_id"][index],
                        "source_image_id": batch["source_image_id"][index],
                        "true_class_id": int(labels[index].detach().cpu()),
                        "predicted_class_id": int(batch_predictions[index].detach().cpu()),
                        "logits": logits[index].detach().cpu().tolist(),
                    }
                )
    return float(np.mean(losses)), classification_metrics(true, predicted, LABELS), records


def train_with_validation(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    validation_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int = 30,
    patience: int = 6,
    min_delta: float = 1e-4,
) -> dict[str, Any]:
    """Train against train/validation only; this API intentionally accepts no test input."""

    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_records: list[dict[str, Any]] | None = None
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_metrics, validation_records = run_epoch(
            model, validation_loader, criterion, device
        )
        is_best, should_stop = stopper.update(epoch, validation_metrics)
        if is_best:
            best_state = deepcopy(model.state_dict())
            best_validation_records = validation_records
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_macro_precision": validation_metrics["macro_precision"],
                "validation_macro_recall": validation_metrics["macro_recall"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "is_best_epoch": False,
            }
        )
        if should_stop:
            break
    if best_state is None or best_validation_records is None or stopper.best_epoch is None:
        raise RuntimeError("training did not produce a best checkpoint")
    history[stopper.best_epoch - 1]["is_best_epoch"] = True
    model.load_state_dict(best_state)
    validation_loss, validation_metrics, validation_records = run_epoch(
        model, validation_loader, criterion, device
    )
    del validation_loss
    return {
        "model": model,
        "best_state_dict": best_state,
        "best_epoch": stopper.best_epoch,
        "best_validation_metrics": validation_metrics,
        "best_validation_records": validation_records,
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "epochs_executed": len(history),
    }


def aggregate_seed_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(metrics) != 3:
        raise ValueError("aggregate requires exactly three predeclared seeds")
    keys = ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall")
    overall = {}
    for key in keys:
        values = np.asarray([float(item[key]) for item in metrics])
        overall[key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    per_class = {}
    for label in LABELS:
        per_class[label] = {}
        for key in ("precision", "recall", "f1"):
            values = np.asarray([float(item["per_class"][label][key]) for item in metrics])
            per_class[label][key] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }
    matrices = np.asarray([item["confusion_matrix_row_normalized"] for item in metrics])
    return {"overall": overall, "per_class": per_class, "mean_normalized_confusion": matrices.mean(axis=0).tolist(), "standard_deviation": "sample (ddof=1)"}


def inference_latency(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    sample = tensor.unsqueeze(0).to(device)
    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        sync()
        values = []
        for _ in range(100):
            sync()
            started = time.perf_counter()
            model(sample)
            sync()
            values.append(time.perf_counter() - started)
    median = float(np.median(values))
    return {"warmup_passes": 20, "measured_passes": 100, "batch_size": 1, "median_seconds": median, "p95_seconds": float(np.percentile(values, 95)), "images_per_second_from_median": 1.0 / median}
