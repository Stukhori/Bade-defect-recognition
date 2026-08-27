from __future__ import annotations

import random

import numpy as np
import pytest

from windblade.reproducibility import set_global_seed


def test_same_seed_reproduces_python_random_values() -> None:
    set_global_seed(42)
    first = [random.random() for _ in range(5)]
    set_global_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_same_seed_reproduces_numpy_values() -> None:
    set_global_seed(42)
    first = np.random.random(5)
    set_global_seed(42)
    second = np.random.random(5)

    np.testing.assert_array_equal(first, second)


def test_different_seed_changes_sequences() -> None:
    set_global_seed(42)
    first = np.random.randint(0, 1_000_000, size=10)
    set_global_seed(43)
    second = np.random.randint(0, 1_000_000, size=10)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("invalid_seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_fails_loudly(invalid_seed) -> None:
    expected = TypeError if isinstance(invalid_seed, (bool, float)) else ValueError
    with pytest.raises(expected):
        set_global_seed(invalid_seed)
