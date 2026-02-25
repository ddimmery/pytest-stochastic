"""Bound selection engine.

Given declared distributional properties, tolerance, failure probability and
test sidedness, selects the concentration bound requiring the fewest samples.
"""

from __future__ import annotations

from .bounds import applicable_bounds
from .types import BoundStrategy, NoApplicableBoundError


def select_bound(
    declared_properties: dict[str, object],
    tolerance: float,
    failure_prob: float,
    side: str = "two-sided",
) -> tuple[BoundStrategy, int]:
    """Select the tightest applicable bound and return it with the required *n*.

    Returns
    -------
    (bound, n)
        The :class:`BoundStrategy` requiring the fewest samples and the
        corresponding sample count.

    Raises
    ------
    NoApplicableBoundError
        If no bound in the registry is applicable with the given properties.
    """
    candidates = applicable_bounds(declared_properties, side)
    if not candidates:
        raise NoApplicableBoundError(
            "No concentration bound is applicable with the declared properties. "
            "Provide at least one of: bounds=(a, b), variance=sigma^2, "
            "moment_bound=(p, M), or sub_gaussian_param=sigma."
        )

    best_bound: BoundStrategy | None = None
    best_n = float("inf")

    for bound in candidates:
        n = bound.compute_n(tolerance, failure_prob, side=side, **declared_properties)
        if n < best_n:
            best_n = n
            best_bound = bound

    assert best_bound is not None  # guaranteed by non-empty candidates
    return best_bound, int(best_n)
