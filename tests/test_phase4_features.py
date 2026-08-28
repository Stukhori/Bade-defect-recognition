from __future__ import annotations

import numpy as np

from windblade.features.common import extract_batch, grayscale_uint8
from windblade.features.hog import extract_hog, hog_config_hash
from windblade.features.lbp import extract_spatial_lbp, lbp_config_hash, normalized_lbp_histogram


HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": [16, 16],
    "cells_per_block": [2, 2],
    "block_norm": "L2-Hys",
    "transform_sqrt": True,
    "feature_vector": True,
    "expected_dimensions": 6084,
}
LBP_CONFIG = {
    "grid": [7, 7],
    "scales": [
        {"radius": 1, "points": 8, "method": "uniform", "bins": 10},
        {"radius": 2, "points": 16, "method": "uniform", "bins": 18},
    ],
    "expected_dimensions": 1372,
}


def _image():
    y, x = np.indices((224, 224))
    return np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype(np.uint8)


def test_grayscale_conversion_is_deterministic():
    assert np.array_equal(grayscale_uint8(_image()), grayscale_uint8(_image()))


def test_hog_dimension_finiteness_and_determinism():
    first = extract_hog(_image(), HOG_CONFIG)
    second = extract_hog(_image(), HOG_CONFIG)
    assert first.shape == (6084,)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_hog_zero_image_is_finite():
    feature = extract_hog(np.zeros((224, 224, 3), dtype=np.uint8), HOG_CONFIG)
    assert np.isfinite(feature).all()


def test_hog_batch_preserves_instance_ids():
    image = _image()
    features, ids = extract_batch(
        ["10_0", "10_1"], [image, image], lambda value: extract_hog(value, HOG_CONFIG), 6084
    )
    assert ids == ("10_0", "10_1")
    assert features.shape == (2, 6084)


def test_hog_config_change_changes_hash():
    versions = {"scikit-image": "x"}
    changed = dict(HOG_CONFIG, orientations=8)
    assert hog_config_hash(HOG_CONFIG, versions) != hog_config_hash(changed, versions)


def test_lbp_scales_bins_grid_and_dimension():
    assert [scale["bins"] for scale in LBP_CONFIG["scales"]] == [10, 18]
    assert LBP_CONFIG["grid"] == [7, 7]
    vector = extract_spatial_lbp(_image(), LBP_CONFIG)
    assert vector.shape == (1372,)
    assert np.isfinite(vector).all()


def test_lbp_local_histograms_are_normalized():
    cell = grayscale_uint8(_image())[:32, :32]
    first = normalized_lbp_histogram(cell, points=8, radius=1, method="uniform", bins=10)
    second = normalized_lbp_histogram(cell, points=16, radius=2, method="uniform", bins=18)
    assert first.shape == (10,)
    assert second.shape == (18,)
    assert np.isclose(first.sum(), 1.0)
    assert np.isclose(second.sum(), 1.0)


def test_lbp_is_deterministic_and_batch_preserves_ids():
    image = _image()
    first = extract_spatial_lbp(image, LBP_CONFIG)
    second = extract_spatial_lbp(image, LBP_CONFIG)
    assert np.array_equal(first, second)
    features, ids = extract_batch(
        ["1_0", "2_0"], [image, image], lambda value: extract_spatial_lbp(value, LBP_CONFIG), 1372
    )
    assert ids == ("1_0", "2_0")
    assert features.shape == (2, 1372)


def test_lbp_config_change_changes_hash():
    versions = {"scikit-image": "x"}
    changed = dict(LBP_CONFIG, grid=[8, 8])
    assert lbp_config_hash(LBP_CONFIG, versions) != lbp_config_hash(changed, versions)
