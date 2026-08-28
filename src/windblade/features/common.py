"""Shared deterministic image and feature helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


class FeatureValidationError(ValueError):
    """Raised when a fixed feature contract is violated."""


def grayscale_uint8(image_or_path: np.ndarray | str | Path) -> np.ndarray:
    """Return Pillow's deterministic ITU-R 601-2 luminance conversion."""

    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened:
            gray = np.asarray(opened.convert("RGB").convert("L"), dtype=np.uint8).copy()
    else:
        image = Image.fromarray(np.asarray(image_or_path, dtype=np.uint8), mode="RGB")
        gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if gray.shape != (224, 224):
        raise FeatureValidationError(f"expected a 224x224 crop; received {gray.shape}")
    return gray


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_batch(
    instance_ids: Sequence[str],
    image_paths: Sequence[str | Path],
    extractor: Callable[[str | Path], np.ndarray],
    expected_dimensions: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract features in identity-preserving order with strict validation."""

    if len(instance_ids) != len(image_paths):
        raise FeatureValidationError("instance IDs and image paths have different lengths")
    if len(instance_ids) != len(set(instance_ids)):
        raise FeatureValidationError("duplicate instance ID in feature batch")
    vectors = [np.asarray(extractor(path), dtype=np.float64) for path in image_paths]
    if not vectors:
        raise FeatureValidationError("cannot extract an empty feature batch")
    features = np.stack(vectors)
    if features.shape != (len(instance_ids), expected_dimensions):
        raise FeatureValidationError(
            f"feature matrix has shape {features.shape}; expected "
            f"({len(instance_ids)}, {expected_dimensions})"
        )
    if not np.isfinite(features).all():
        raise FeatureValidationError("feature matrix contains NaN or infinity")
    return features, tuple(str(value) for value in instance_ids)
