"""Random-seed and deterministic-setting utilities."""

from __future__ import annotations

import importlib
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> dict[str, Any]:
    """Seed required RNGs and return a record of applied settings.

    The record describes requested settings; it is not a promise of bitwise
    identity across different hardware, libraries, or every GPU operation.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be between 0 and 2^32 - 1")

    random.seed(seed)
    np.random.seed(seed)
    settings: dict[str, Any] = {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "pytorch_available": False,
        "torch_cpu_seeded": False,
        "cuda_seeded": False,
        "deterministic_algorithms_requested": False,
        "cudnn_benchmark_disabled": None,
        "guarantee": (
            "Deterministic settings improve repeatability on a fixed environment; "
            "they do not guarantee bitwise-identical results across all hardware "
            "or operations."
        ),
    }

    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        settings["pytorch_import_error"] = type(exc).__name__
        return settings

    settings["pytorch_available"] = True
    torch.manual_seed(seed)
    settings["torch_cpu_seeded"] = True

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)
        settings["cuda_seeded"] = True

    use_deterministic = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_deterministic):
        use_deterministic(True, warn_only=True)
        settings["deterministic_algorithms_requested"] = True

    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        settings["cudnn_benchmark_disabled"] = True

    return settings
