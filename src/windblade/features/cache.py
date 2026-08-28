"""Fingerprint-validated, regenerable handcrafted-feature caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from windblade.features.common import FeatureValidationError, canonical_hash, extract_batch
from windblade.utils import atomic_write_text


def cache_key(
    processed_fingerprint: str,
    family: str,
    feature_config_hash: str,
    library_versions: Mapping[str, str],
) -> str:
    return canonical_hash(
        {
            "processed_dataset_fingerprint": processed_fingerprint,
            "feature_family": family,
            "feature_config_hash": feature_config_hash,
            "library_versions": dict(library_versions),
        }
    )


def feature_matrix_fingerprint(instance_ids: Sequence[str], features: np.ndarray) -> str:
    digest = hashlib.sha256()
    for instance_id in instance_ids:
        digest.update(str(instance_id).encode("utf-8") + b"\0")
    contiguous = np.ascontiguousarray(features, dtype=np.float64)
    digest.update(str(contiguous.shape).encode("ascii") + b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_or_extract_features(
    *,
    cache_root: str | Path,
    family: str,
    key: str,
    instance_ids: Sequence[str],
    image_paths: Sequence[str | Path],
    labels: Sequence[int],
    source_ids: Sequence[str],
    extractor: Callable[[str | Path], np.ndarray],
    expected_dimensions: int,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / f"{family}_{key}.npz"
    metadata_path = root / f"{family}_{key}.json"
    expected_ids = tuple(str(value) for value in instance_ids)
    if array_path.is_file() and metadata_path.is_file():
        stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored_metadata.get("cache_key") == key:
            with np.load(array_path, allow_pickle=False) as stored:
                stored_ids = tuple(str(value) for value in stored["instance_ids"].tolist())
                features = np.asarray(stored["features"], dtype=np.float64)
                stored_labels = stored["labels"].astype(np.int64)
                stored_sources = tuple(str(value) for value in stored["source_ids"].tolist())
            if (
                stored_ids == expected_ids
                and np.array_equal(stored_labels, np.asarray(labels, dtype=np.int64))
                and stored_sources == tuple(str(value) for value in source_ids)
                and features.shape == (len(instance_ids), expected_dimensions)
                and np.isfinite(features).all()
                and feature_matrix_fingerprint(stored_ids, features)
                == stored_metadata.get("feature_fingerprint")
            ):
                stored_metadata["cache_hit"] = True
                return features, stored_metadata

    started = time.perf_counter()
    features, output_ids = extract_batch(instance_ids, image_paths, extractor, expected_dimensions)
    elapsed = time.perf_counter() - started
    fingerprint = feature_matrix_fingerprint(output_ids, features)
    np.savez_compressed(
        array_path,
        instance_ids=np.asarray(output_ids),
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        source_ids=np.asarray(source_ids),
    )
    record = {
        **dict(metadata),
        "cache_key": key,
        "cache_hit": False,
        "feature_family": family,
        "feature_dimensions": expected_dimensions,
        "row_count": len(instance_ids),
        "feature_fingerprint": fingerprint,
        "initial_extraction_seconds": elapsed,
        "array_file": array_path.name,
    }
    atomic_write_text(metadata_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return features, record
