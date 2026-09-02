"""Read-only CPU inference with the frozen Phase 6 seed-17 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
import torch

from windblade.deep.checkpoints import load_checkpoint, state_dict_fingerprint
from windblade.deep.dataset import canonical_transform
from windblade.deep.mobilenet import build_mobilenet
from windblade_demo.constants import (
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_RELATIVE_PATH,
    CHECKPOINT_STATE_FINGERPRINT,
    CLASS_LABELS,
    DATASET_FINGERPRINT,
    EXPECTED_ARCHITECTURE,
    EXPECTED_SEED,
    MODEL_DISPLAY_NAME,
    MODEL_INPUT_SIZE,
)


class FrozenModelError(RuntimeError):
    """Raised when the required frozen model identity cannot be verified."""


@dataclass(frozen=True)
class LoadedFrozenModel:
    model: torch.nn.Module
    metadata: dict[str, Any]
    checkpoint_path: Path
    load_seconds: float


@dataclass(frozen=True)
class InferenceResult:
    predicted_class_id: int
    predicted_label: str
    logits: tuple[float, ...]
    scores: tuple[float, ...]
    preprocessing_seconds: float
    inference_seconds: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_model(root: str | Path | None = None) -> LoadedFrozenModel:
    """Load and verify the canonical full-data seed-17 model on CPU."""

    started = perf_counter()
    repository = Path(root) if root is not None else repository_root()
    checkpoint = repository / CHECKPOINT_RELATIVE_PATH
    metadata_path = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FrozenModelError(
            "The verified seed-17 model checkpoint and metadata are required to run the app."
        )
    if _sha256(checkpoint) != CHECKPOINT_FILE_SHA256:
        raise FrozenModelError("The frozen checkpoint file SHA-256 does not match the app contract.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("seed") != EXPECTED_SEED
        or metadata.get("architecture") != EXPECTED_ARCHITECTURE
        or tuple(metadata.get("class_order", ())) != CLASS_LABELS
        or metadata.get("checkpoint_fingerprint") != CHECKPOINT_STATE_FINGERPRINT
    ):
        raise FrozenModelError("The checkpoint metadata does not match the frozen app identity.")
    state, verified_metadata = load_checkpoint(
        checkpoint,
        expected_dataset_fingerprint=DATASET_FINGERPRINT,
        expected_architecture=EXPECTED_ARCHITECTURE,
    )
    if state_dict_fingerprint(state) != CHECKPOINT_STATE_FINGERPRINT:
        raise FrozenModelError("The decoded model state does not match the frozen fingerprint.")
    model = build_mobilenet(pretrained=False).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    return LoadedFrozenModel(model, verified_metadata, checkpoint, perf_counter() - started)


def preprocess_model_input(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB" or image.size != MODEL_INPUT_SIZE:
        raise FrozenModelError("Inference requires an RGB 224x224 image produced by the app crop policy.")
    tensor = canonical_transform()(image.copy()).unsqueeze(0).cpu()
    if tuple(tensor.shape) != (1, 3, 224, 224) or not torch.isfinite(tensor).all():
        raise FrozenModelError("Preprocessing produced an invalid tensor.")
    return tensor


def infer(loaded: LoadedFrozenModel, image: Image.Image) -> InferenceResult:
    """Run one read-only request and return all six softmax model scores."""

    preprocessing_started = perf_counter()
    tensor = preprocess_model_input(image)
    preprocessing_seconds = perf_counter() - preprocessing_started
    loaded.model.eval()
    inference_started = perf_counter()
    with torch.inference_mode():
        logits_tensor = loaded.model(tensor)[0]
        scores_tensor = torch.softmax(logits_tensor, dim=0)
    inference_seconds = perf_counter() - inference_started
    if tuple(logits_tensor.shape) != (len(CLASS_LABELS),) or not torch.isfinite(scores_tensor).all():
        raise FrozenModelError("The frozen model returned invalid output.")
    predicted = int(torch.argmax(scores_tensor).item())
    return InferenceResult(
        predicted_class_id=predicted,
        predicted_label=CLASS_LABELS[predicted],
        logits=tuple(float(value) for value in logits_tensor.cpu().tolist()),
        scores=tuple(float(value) for value in scores_tensor.cpu().tolist()),
        preprocessing_seconds=preprocessing_seconds,
        inference_seconds=inference_seconds,
    )


def model_status(loaded: LoadedFrozenModel) -> dict[str, str]:
    return {
        "model": MODEL_DISPLAY_NAME,
        "architecture": str(loaded.metadata["architecture"]),
        "seed": str(loaded.metadata["seed"]),
        "checkpoint_state": str(loaded.metadata["checkpoint_fingerprint"]),
        "status": "Frozen checkpoint verified; evaluation mode; CPU",
    }
