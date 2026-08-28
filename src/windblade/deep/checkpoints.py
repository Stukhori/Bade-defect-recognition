"""Portable state-dictionary persistence with deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from windblade.data.processed import json_text
from windblade.utils import atomic_write_text


def state_dict_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path, state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = state_dict_fingerprint(state_dict)
    torch.save(dict(state_dict), target)
    record = {**dict(metadata), "checkpoint_fingerprint": fingerprint, "checkpoint_bytes": target.stat().st_size}
    atomic_write_text(target.with_suffix(".json"), json_text(record))
    return record


def load_checkpoint(
    path: str | Path,
    *,
    expected_dataset_fingerprint: str,
    metadata_path: str | Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    target = Path(path)
    meta_path = Path(metadata_path) if metadata_path else target.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["processed_dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("checkpoint processed dataset fingerprint mismatch")
    state = torch.load(target, map_location="cpu", weights_only=True)
    if state_dict_fingerprint(state) != metadata["checkpoint_fingerprint"]:
        raise ValueError("checkpoint state fingerprint mismatch")
    return state, metadata
