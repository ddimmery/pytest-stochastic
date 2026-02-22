"""pytest plugin entry point for pytest-stochastic.

Registers the plugin hooks so that functions decorated with
``@stochastic_test`` are collected and executed by pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from .decorator import STOCHASTIC_TEST_MARKER


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag stochastic tests with a ``stochastic`` marker for easy filtering."""
    for item in items:
        if isinstance(item, pytest.Function):
            obj = item.obj
            if hasattr(obj, STOCHASTIC_TEST_MARKER):
                item.add_marker(pytest.mark.stochastic)


@pytest.fixture
def stochastic_rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Provide a seeded numpy RNG as a pytest fixture.

    The seed is derived from the test node id for reproducibility.
    """
    seed = hash(request.node.nodeid) % (2**32)
    return np.random.default_rng(seed)
