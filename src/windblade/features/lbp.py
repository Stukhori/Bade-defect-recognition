"""Frozen Phase 4 multi-scale spatial Local Binary Pattern representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.feature import local_binary_pattern

from windblade.features.common import FeatureValidationError, canonical_hash, grayscale_uint8


EXPECTED_LBP_DIMENSIONS = 1372


def lbp_config_hash(config: Mapping[str, Any], library_versions: Mapping[str, str]) -> str:
    return canonical_hash({"family": "lbp", "config": dict(config), "libraries": dict(library_versions)})


def normalized_lbp_histogram(
    cell: np.ndarray, *, points: int, radius: int, method: str, bins: int
) -> np.ndarray:
    codes = local_binary_pattern(cell, P=points, R=radius, method=method)
    histogram = np.bincount(codes.astype(np.int64).ravel(), minlength=bins)[:bins].astype(np.float64)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def extract_spatial_lbp(
    image_or_path: np.ndarray | str | Path, config: Mapping[str, Any]
) -> np.ndarray:
    gray = grayscale_uint8(image_or_path)
    grid_rows, grid_columns = (int(value) for value in config["grid"])
    if (grid_rows, grid_columns) != (7, 7):
        raise FeatureValidationError("frozen LBP grid must be 7x7")
    if gray.shape[0] % grid_rows or gray.shape[1] % grid_columns:
        raise FeatureValidationError("image dimensions are not divisible by the LBP grid")
    cell_height, cell_width = gray.shape[0] // grid_rows, gray.shape[1] // grid_columns
    if (cell_height, cell_width) != (32, 32):
        raise FeatureValidationError("frozen LBP cells must be 32x32 pixels")
    parts: list[np.ndarray] = []
    for row in range(grid_rows):
        for column in range(grid_columns):
            cell = gray[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ]
            for scale in config["scales"]:
                points, bins = int(scale["points"]), int(scale["bins"])
                if bins != points + 2 or scale["method"] != "uniform":
                    raise FeatureValidationError("LBP scale violates the frozen uniform-bin contract")
                parts.append(
                    normalized_lbp_histogram(
                        cell,
                        points=points,
                        radius=int(scale["radius"]),
                        method=str(scale["method"]),
                        bins=bins,
                    )
                )
    vector = np.concatenate(parts)
    expected = int(config["expected_dimensions"])
    if vector.shape != (expected,) or expected != EXPECTED_LBP_DIMENSIONS:
        raise FeatureValidationError(
            f"LBP dimensionality is {vector.size}; frozen contract requires {EXPECTED_LBP_DIMENSIONS}"
        )
    if not np.isfinite(vector).all():
        raise FeatureValidationError("LBP feature contains NaN or infinity")
    return vector
