"""Parameter validation and tolerance computation for stochastic tests."""

from __future__ import annotations

from .types import (
    InvalidPropertyError,
    InvalidToleranceError,
    TestConfig,
)


def compute_tolerance(
    expected: float,
    atol: float,
    rtol: float,
) -> float:
    """Compute effective tolerance: tol = atol + rtol * |expected|.

    Raises :class:`InvalidToleranceError` when the inputs are invalid.
    """
    if atol < 0:
        raise InvalidToleranceError(f"atol must be non-negative, got {atol}")
    if rtol < 0:
        raise InvalidToleranceError(f"rtol must be non-negative, got {rtol}")
    if atol == 0 and rtol == 0:
        raise InvalidToleranceError("At least one of atol or rtol must be positive")
    if expected == 0 and rtol > 0 and atol == 0:
        raise InvalidToleranceError(
            "Relative tolerance alone is meaningless when expected=0; set atol > 0"
        )
    return atol + rtol * abs(expected)


def validate_and_build_config(
    *,
    expected: float,
    atol: float = 0.0,
    rtol: float = 0.0,
    failure_prob: float = 1e-8,
    bounds: tuple[float, float] | None = None,
    variance: float | None = None,
    sub_gaussian_param: float | None = None,
    symmetric: bool = False,
    moment_bound: tuple[float, float] | None = None,
    side: str = "two-sided",
) -> TestConfig:
    """Validate all decorator parameters and return a :class:`TestConfig`."""

    # --- tolerance ---
    tol = compute_tolerance(expected, atol, rtol)

    # --- failure_prob ---
    if not (0 < failure_prob < 1):
        raise InvalidPropertyError(f"failure_prob must be in (0, 1), got {failure_prob}")

    # --- side ---
    valid_sides = {"two-sided", "greater", "less"}
    if side not in valid_sides:
        raise InvalidPropertyError(f"side must be one of {valid_sides}, got {side!r}")

    # --- bounds ---
    if bounds is not None:
        if len(bounds) != 2:
            raise InvalidPropertyError("bounds must be a 2-tuple (a, b)")
        a, b = bounds
        if a >= b:
            raise InvalidPropertyError(f"bounds must satisfy a < b, got ({a}, {b})")

    # --- variance ---
    if variance is not None and variance <= 0:
        raise InvalidPropertyError(f"variance must be positive, got {variance}")

    # --- sub_gaussian_param ---
    if sub_gaussian_param is not None and sub_gaussian_param <= 0:
        raise InvalidPropertyError(
            f"sub_gaussian_param must be positive, got {sub_gaussian_param}"
        )

    # --- moment_bound ---
    if moment_bound is not None:
        if len(moment_bound) != 2:
            raise InvalidPropertyError("moment_bound must be a 2-tuple (p, M)")
        p, m = moment_bound
        if p <= 1:
            raise InvalidPropertyError(f"moment_bound p must be > 1, got {p}")
        if m <= 0:
            raise InvalidPropertyError(f"moment_bound M must be positive, got {m}")

    # --- at least one property ---
    has_property = any(
        [
            bounds is not None,
            variance is not None,
            sub_gaussian_param is not None,
            moment_bound is not None,
        ]
    )
    if not has_property:
        raise InvalidPropertyError(
            "At least one distributional property must be declared "
            "(bounds, variance, sub_gaussian_param, or moment_bound)"
        )

    # --- symmetric requires bounds ---
    if symmetric and bounds is None:
        raise InvalidPropertyError("symmetric=True requires bounds=(a, b) to be declared")

    return TestConfig(
        expected=expected,
        tol=tol,
        failure_prob=failure_prob,
        side=side,
        bounds=bounds,
        variance=variance,
        sub_gaussian_param=sub_gaussian_param,
        symmetric=symmetric,
        moment_bound=moment_bound,
    )
