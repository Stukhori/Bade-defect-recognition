"""Consistent warm-up and repeated median latency measurements."""

from __future__ import annotations

from statistics import median
import time
from typing import Any, Callable, Sequence

import numpy as np


def median_batch_latency(
    operation: Callable[[], Any], *, sample_count: int, warmup_runs: int, repeats: int
) -> dict[str, Any]:
    if sample_count <= 0 or warmup_runs < 0 or repeats <= 0:
        raise ValueError("invalid timing protocol")
    for _ in range(warmup_runs):
        operation()
    elapsed: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        elapsed.append(time.perf_counter() - started)
    return {
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "sample_count": sample_count,
        "batch_seconds": elapsed,
        "median_batch_seconds": median(elapsed),
        "median_seconds_per_image": median(elapsed) / sample_count,
    }


def prediction_timing(
    model: Any, features: np.ndarray, *, warmup_runs: int, repeats: int
) -> dict[str, Any]:
    warm_features = features[: min(32, len(features))]
    for _ in range(warmup_runs):
        model.predict(warm_features)
    return median_batch_latency(
        lambda: model.predict(features),
        sample_count=len(features),
        warmup_runs=0,
        repeats=repeats,
    ) | {"warmup_runs": warmup_runs}
