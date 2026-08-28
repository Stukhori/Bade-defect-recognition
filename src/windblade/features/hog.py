"""Frozen Phase 4 Histogram of Oriented Gradients representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.feature import hog

from windblade.features.common import FeatureValidationError, canonical_hash, grayscale_uint8


EXPECTED_HOG_DIMENSIONS = 6084


def hog_config_hash(config: Mapping[str, Any], library_versions: Mapping[str, str]) -> str:
    return canonical_hash({"family": "hog", "config": dict(config), "libraries": dict(library_versions)})


def extract_hog(image_or_path: np.ndarray | str | Path, config: Mapping[str, Any]) -> np.ndarray:
    gray = grayscale_uint8(image_or_path)
    vector = hog(
        gray,
        orientations=int(config["orientations"]),
        pixels_per_cell=tuple(int(value) for value in config["pixels_per_cell"]),
        cells_per_block=tuple(int(value) for value in config["cells_per_block"]),
        block_norm=str(config["block_norm"]),
        transform_sqrt=bool(config["transform_sqrt"]),
        feature_vector=bool(config["feature_vector"]),
    )
    vector = np.asarray(vector, dtype=np.float64)
    expected = int(config["expected_dimensions"])
    if vector.shape != (expected,) or expected != EXPECTED_HOG_DIMENSIONS:
        raise FeatureValidationError(
            f"HOG dimensionality is {vector.size}; frozen contract requires {EXPECTED_HOG_DIMENSIONS}"
        )
    if not np.isfinite(vector).all():
        raise FeatureValidationError("HOG feature contains NaN or infinity")
    return vector
