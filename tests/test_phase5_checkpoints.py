from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from windblade.deep.checkpoints import load_checkpoint, save_checkpoint, state_dict_fingerprint


def test_checkpoint_round_trip_fingerprint_and_logits(tmp_path: Path) -> None:
    model = nn.Linear(3, 2)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    expected = model(inputs).detach()
    fingerprint = state_dict_fingerprint(model.state_dict())
    metadata = save_checkpoint(tmp_path / "best.pt", model.state_dict(), {"processed_dataset_fingerprint": "abc", "class_order": ["a", "b"]})
    assert metadata["checkpoint_fingerprint"] == fingerprint
    state, loaded_metadata = load_checkpoint(tmp_path / "best.pt", expected_dataset_fingerprint="abc")
    restored = nn.Linear(3, 2); restored.load_state_dict(state)
    assert torch.equal(expected, restored(inputs))
    assert loaded_metadata["class_order"] == ["a", "b"]
    assert state_dict_fingerprint(restored.state_dict()) == fingerprint


def test_checkpoint_rejects_dataset_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    save_checkpoint(tmp_path / "best.pt", model.state_dict(), {"processed_dataset_fingerprint": "abc"})
    try:
        load_checkpoint(tmp_path / "best.pt", expected_dataset_fingerprint="wrong")
    except ValueError as error:
        assert "dataset fingerprint mismatch" in str(error)
    else:
        raise AssertionError("mismatched dataset checkpoint was accepted")


def test_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    save_checkpoint(tmp_path / "best.pt", model.state_dict(), {"processed_dataset_fingerprint": "abc", "architecture": "expected"})
    try:
        load_checkpoint(tmp_path / "best.pt", expected_dataset_fingerprint="abc", expected_architecture="wrong")
    except ValueError as error:
        assert "architecture mismatch" in str(error)
    else:
        raise AssertionError("mismatched architecture was accepted")
