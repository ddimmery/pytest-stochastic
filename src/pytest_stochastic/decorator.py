"""The ``@stochastic_test`` decorator."""

from __future__ import annotations

import functools
from typing import Any

from .runtime import (
    check_assertion,
    collect_samples,
    compute_estimate,
    make_rng,
)
from .selection import select_bound
from .validation import validate_and_build_config

# Marker attribute set on decorated functions so the pytest plugin can
# identify them during collection.
STOCHASTIC_TEST_MARKER = "_stochastic_test_config"


def stochastic_test(
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
    seed: int | None = None,
) -> Any:
    """Decorator that turns a scalar-returning function into a stochastic test.

    Parameters
    ----------
    expected:
        Expected value of the test statistic.
    atol:
        Absolute tolerance.
    rtol:
        Relative tolerance (fraction of ``|expected|``).
    failure_prob:
        Target false-failure probability.
    bounds:
        ``(a, b)`` guaranteeing each sample lies in ``[a, b]``.
    variance:
        Upper bound on ``Var(X_i)``.
    sub_gaussian_param:
        Sub-Gaussian parameter sigma.
    symmetric:
        ``True`` if the distribution is symmetric about its mean.
    moment_bound:
        ``(p, M)`` such that ``E[|X - μ|^p] ≤ M`` for some ``p > 1``.
    side:
        ``"two-sided"`` (default), ``"greater"``, or ``"less"``.
    seed:
        Optional fixed RNG seed for reproducibility.
    """
    # Validate eagerly at decoration time so misconfigurations surface
    # at import, not at test execution.
    config = validate_and_build_config(
        expected=expected,
        atol=atol,
        rtol=rtol,
        failure_prob=failure_prob,
        bounds=bounds,
        variance=variance,
        sub_gaussian_param=sub_gaussian_param,
        symmetric=symmetric,
        moment_bound=moment_bound,
        side=side,
    )

    # Select the best bound and required sample size.
    bound, n = select_bound(
        config.declared_properties,
        config.tol,
        config.failure_prob,
        config.side,
    )

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper() -> None:
            rng, actual_seed = make_rng(seed)
            samples = collect_samples(func, n, rng)
            estimate = compute_estimate(samples, bound.estimator_type, failure_prob)
            result = check_assertion(estimate, config, bound, n, actual_seed)
            wrapper._stochastic_result = result  # type: ignore[attr-defined]
            if not result.passed:
                raise AssertionError(result.message)

        # Remove __wrapped__ so pytest doesn't introspect the original
        # function's signature and try to inject fixtures for its parameters.
        del wrapper.__wrapped__

        # Attach metadata for introspection / plugin use.
        setattr(wrapper, STOCHASTIC_TEST_MARKER, config)
        wrapper._stochastic_bound = bound  # type: ignore[attr-defined]
        wrapper._stochastic_n = n  # type: ignore[attr-defined]
        return wrapper

    return decorator
