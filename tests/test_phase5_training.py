from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from windblade.deep.determinism import seed_torch
from windblade.deep.training import EarlyStopping, aggregate_seed_metrics, hyperparameter_grid, select_candidate, train_with_validation
from windblade.evaluation.metrics import classification_metrics


def test_grid_is_exact_and_selection_is_validation_only() -> None:
    assert hyperparameter_grid() == [
        {"learning_rate": 1e-4, "weight_decay": 0.0},
        {"learning_rate": 1e-4, "weight_decay": 1e-4},
        {"learning_rate": 3e-4, "weight_decay": 0.0},
        {"learning_rate": 3e-4, "weight_decay": 1e-4},
    ]
    rows = [{"learning_rate": row["learning_rate"], "weight_decay": row["weight_decay"], "validation_macro_f1": 0.5, "validation_balanced_accuracy": 0.5, "validation_macro_recall": 0.5} for row in hyperparameter_grid()]
    assert select_candidate(rows)["learning_rate"] == 1e-4
    assert select_candidate(rows)["weight_decay"] == 0.0
    assert "test" not in select_candidate.__annotations__


def test_early_stopping_threshold_patience_and_ties() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    assert stopper.update(1, {"macro_f1": .4, "balanced_accuracy": .4, "macro_recall": .4}) == (True, False)
    assert stopper.update(2, {"macro_f1": .45, "balanced_accuracy": .9, "macro_recall": .9}) == (False, False)
    assert stopper.update(3, {"macro_f1": .5, "balanced_accuracy": .4, "macro_recall": .4}) == (True, False)
    assert stopper.best_epoch == 3
    tied = EarlyStopping()
    tied.update(1, {"macro_f1": .5, "balanced_accuracy": .4, "macro_recall": .4})
    assert tied.update(2, {"macro_f1": .5, "balanced_accuracy": .5, "macro_recall": .4})[0]
    assert tied.best_epoch == 2
    assert not tied.update(3, {"macro_f1": .5, "balanced_accuracy": .5, "macro_recall": .4})[0]


def test_three_seed_aggregation_uses_sample_sd_and_class_order() -> None:
    metrics = [classification_metrics([0, 1, 2, 3, 4, 5], predictions, ("craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze")) for predictions in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 0, 0])]
    result = aggregate_seed_metrics(metrics)
    values = np.array([item["macro_f1"] for item in metrics])
    assert result["overall"]["macro_f1"]["sample_sd"] == values.std(ddof=1)
    assert list(result["per_class"]) == ["craze", "corrosion", "surface_injure", "thunderstrike", "crack", "hide_craze"]


class _Tiny(Dataset):
    def __init__(self):
        self.images = torch.arange(12 * 3 * 4 * 4, dtype=torch.float32).reshape(12, 3, 4, 4) / 100
    def __len__(self): return 12
    def __getitem__(self, index):
        return {"image": self.images[index], "class_id": index % 6, "instance_id": str(index), "source_image_id": str(index)}


def _tiny_run():
    seed_torch(17)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 6))
    initial = deepcopy(model[1].weight.detach())
    generator = torch.Generator().manual_seed(17)
    result = train_with_validation(model, DataLoader(_Tiny(), batch_size=4, shuffle=True, generator=generator), DataLoader(_Tiny(), batch_size=4, shuffle=False), device=torch.device("cpu"), class_weights=torch.ones(6), learning_rate=1e-3, weight_decay=0.0, max_epochs=2, patience=2)
    return initial, result


def test_tiny_training_repeats_exactly() -> None:
    initial_a, result_a = _tiny_run(); initial_b, result_b = _tiny_run()
    assert torch.equal(initial_a, initial_b)
    assert result_a["best_epoch"] == result_b["best_epoch"]
    assert result_a["best_validation_records"] == result_b["best_validation_records"]
    for key in result_a["best_state_dict"]:
        assert torch.equal(result_a["best_state_dict"][key], result_b["best_state_dict"][key])
