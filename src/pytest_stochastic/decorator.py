"""The ``@stochastic_test`` and ``@distributional_test`` decorators."""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from .runtime import (
    check_assertion,
    check_maurer_pontil,
    collect_samples,
    compute_estimate,
    make_rng,
)
from .selection import select_bound
from .types import ConfigurationError
from .validation import validate_and_build_config

# Marker attribute set on decorated functions so the pytest plugin can
# identify them during collection.
STOCHASTIC_TEST_MARKER = "_stochastic_test_config"

# Cache for tuned parameters (loaded once per process).
_tuned_params_cache: dict[str, dict[str, object]] | None = None


def _get_tuned_params() -> dict[str, dict[str, object]]:
    """Load and cache tuned parameters from .stochastic.toml."""
    global _tuned_params_cache
    if _tuned_params_cache is None:
        from .tune import load_tuned_params

        _tuned_params_cache = load_tuned_params()
    return _tuned_params_cache


def _load_tuned_variance(func: object, properties: dict[str, object]) -> None:
    """If tuned variance is available for *func*, add it to properties.

    Looks up the function in .stochastic.toml by matching the function's
    module and qualified name against stored test keys.
    """
    tuned = _get_tuned_params()
    if not tuned:
        return

    # Build candidate keys to match against stored test keys.
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or ""
    func_name = getattr(func, "__name__", "") or ""

    # The tune mode stores keys like "tests.test_module.test_fn" (derived from nodeid).
    # Try various key formats to find a match.
    candidates = {
        f"{module}.{qualname}",
        f"{module}.{func_name}",
        qualname,
        func_name,
    }

    for key, params in tuned.items():
        # Check if any candidate matches the stored key (or is a suffix of it)
        for candidate in candidates:
            if key.endswith(candidate) or candidate.endswith(key):
                variance = params.get("variance")
                if variance is not None and "variance_tuned" not in properties:
                    properties["variance_tuned"] = float(variance)  # type: ignore[arg-type]
                return


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

    def decorator(func: Any) -> Any:
        # Enrich declared properties with tuned variance if available.
        properties = dict(config.declared_properties)
        _load_tuned_variance(func, properties)

        # Select the best bound and required sample size.
        bound, n = select_bound(
            properties,
            config.tol,
            config.failure_prob,
            config.side,
        )

        @functools.wraps(func)
        def wrapper() -> None:
            rng, actual_seed = make_rng(seed)
            samples = collect_samples(func, n, rng)
            estimate = compute_estimate(samples, bound.estimator_type, failure_prob)
            result = check_assertion(estimate, config, bound, n, actual_seed)

            # Maurer-Pontil opportunistic upgrade: when using Hoeffding
            # (bounds declared, no variance), check if the empirical
            # Maurer-Pontil bound holds with fewer samples.
            if bound.name == "hoeffding" and config.bounds is not None:
                mp_n = check_maurer_pontil(samples, config, failure_prob)
                if mp_n is not None:
                    result.maurer_pontil_effective_n = mp_n

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


# Marker attribute for distributional tests.
DISTRIBUTIONAL_TEST_MARKER = "_distributional_test_config"

_VALID_DIST_TESTS = frozenset({"ks", "chi2", "anderson"})


def distributional_test(
    *,
    reference: object,
    test: str = "ks",
    significance: float = 1e-6,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> Any:
    """Decorator for testing that outputs match a reference distribution.

    Parameters
    ----------
    reference:
        A scipy continuous distribution (e.g. ``scipy.stats.norm(0, 1)``).
    test:
        Statistical test to use: ``"ks"`` (Kolmogorov-Smirnov),
        ``"chi2"`` (chi-squared goodness-of-fit), or ``"anderson"``.
    significance:
        Significance level alpha. The test asserts ``p-value > significance``.
    n_samples:
        Number of samples to draw from the test function.
    seed:
        Optional fixed RNG seed for reproducibility.
    """
    # Validate at decoration time.
    if test not in _VALID_DIST_TESTS:
        raise ConfigurationError(
            f"Unknown distributional test {test!r}. "
            f"Choose from: {', '.join(sorted(_VALID_DIST_TESTS))}"
        )
    if not 0 < significance < 1:
        raise ConfigurationError(f"significance must be in (0, 1), got {significance}")
    if n_samples < 1:
        raise ConfigurationError(f"n_samples must be positive, got {n_samples}")
    if not hasattr(reference, "cdf"):
        raise ConfigurationError("reference must be a scipy distribution with a .cdf method")

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper() -> None:
            rng, actual_seed = make_rng(seed)
            samples = collect_samples(func, n_samples, rng)

            if test == "ks":
                stat, pvalue = scipy_stats.kstest(samples, reference.cdf)
            elif test == "chi2":
                # Bin samples using quantiles of the reference distribution.
                n_bins = max(10, int(np.sqrt(n_samples)))
                edges = reference.ppf(np.linspace(0, 1, n_bins + 1))  # type: ignore[union-attr]
                observed, _ = np.histogram(samples, bins=edges)
                expected_counts = np.full(n_bins, n_samples / n_bins)
                stat, pvalue = scipy_stats.chisquare(observed, f_exp=expected_counts)
            else:  # anderson
                ref_samples = reference.rvs(size=n_samples, random_state=rng)  # type: ignore[union-attr]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="p-value capped",
                        category=UserWarning,
                    )
                    result = scipy_stats.anderson_ksamp(
                        [samples, ref_samples],
                        variant="midrank",
                    )
                stat = result.statistic
                pvalue = result.pvalue

            passed = pvalue > significance

            detail = (
                f"[{test}, n={n_samples}, stat={stat:.6g}, p={pvalue:.6g}, sig={significance:.6g}]"
            )
            wrapper._distributional_result = {  # type: ignore[attr-defined]
                "passed": passed,
                "test": test,
                "n_samples": n_samples,
                "statistic": stat,
                "pvalue": pvalue,
                "significance": significance,
                "seed": actual_seed,
                "detail": detail,
            }

            if not passed:
                raise AssertionError(f"Distributional test FAILED {detail} (seed={actual_seed})")

        # Remove __wrapped__ so pytest doesn't inject fixtures.
        del wrapper.__wrapped__

        setattr(
            wrapper,
            DISTRIBUTIONAL_TEST_MARKER,
            {
                "test": test,
                "significance": significance,
                "n_samples": n_samples,
            },
        )
        return wrapper

    return decorator
