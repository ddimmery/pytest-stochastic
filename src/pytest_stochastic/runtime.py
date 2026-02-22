"""Test execution runtime.

Handles RNG injection, sample collection, estimator computation, and
assertion checking for stochastic tests.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field

import numpy as np

from .types import BoundStrategy, EstimatorType, TestConfig

# ---------------------------------------------------------------------------
# RNG injection
# ---------------------------------------------------------------------------


def _wants_rng(func: object) -> bool:
    """Return True if *func*'s signature contains an ``rng`` parameter."""
    try:
        sig = inspect.signature(func)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return False
    return "rng" in sig.parameters


def make_rng(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a seeded :class:`numpy.random.Generator`.

    Returns ``(rng, seed)`` so the seed can be reported on failure.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)
    return rng, seed


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------


def collect_samples(
    func: object,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Call *func* n times and collect scalar return values.

    If the function signature contains an ``rng`` parameter, the generator is
    passed automatically.
    """
    inject_rng = _wants_rng(func)
    samples = np.empty(n, dtype=np.float64)

    for i in range(n):
        try:
            result = func(rng=rng) if inject_rng else func()  # type: ignore[operator]
        except Exception as exc:
            raise RuntimeError(
                f"Stochastic test function raised an exception on call {i + 1}/{n}: {exc}"
            ) from exc

        if not np.isscalar(result) or isinstance(result, (str, bytes, bool)):
            raise TypeError(
                f"Stochastic test function must return a numeric scalar, "
                f"got {type(result).__name__} on call {i + 1}/{n}"
            )
        samples[i] = float(result)  # type: ignore[arg-type]

    return samples


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def compute_estimate(
    samples: np.ndarray,
    estimator_type: EstimatorType,
    failure_prob: float,
) -> float:
    """Compute the point estimate using the appropriate estimator."""
    if estimator_type == EstimatorType.SAMPLE_MEAN:
        return float(np.mean(samples))

    if estimator_type == EstimatorType.MEDIAN_OF_MEANS:
        return _median_of_means(samples, failure_prob)

    if estimator_type == EstimatorType.CATONI_M_ESTIMATOR:
        return _catoni_estimator(samples)

    raise ValueError(f"Unknown estimator type: {estimator_type}")  # pragma: no cover


def _median_of_means(samples: np.ndarray, failure_prob: float) -> float:
    """Median-of-means estimator."""
    n = len(samples)
    k = math.ceil(8 * math.log(2 / failure_prob))
    k = min(k, n)  # can't have more blocks than samples
    block_size = n // k
    if block_size == 0:
        return float(np.mean(samples))

    block_means = np.array(
        [np.mean(samples[i * block_size : (i + 1) * block_size]) for i in range(k)]
    )
    return float(np.median(block_means))


def _catoni_estimator(samples: np.ndarray) -> float:
    """Catoni M-estimator (simplified).

    Uses the influence function psi(x) = sign(x) * log(1 + |x| + x^2/2)
    and finds mu_hat via bisection.
    """
    n = len(samples)
    alpha = math.sqrt(2 * math.log(2) / n) if n > 0 else 1.0

    def _psi(x: float) -> float:
        """Catoni's influence function."""
        if x >= 0:
            return math.log(1 + x + x * x / 2)
        return -math.log(1 - x + x * x / 2)

    def _objective(mu: float) -> float:
        return sum(_psi(alpha * (float(s) - mu)) for s in samples) / n

    # Bisection
    lo, hi = float(np.min(samples)), float(np.max(samples))
    if lo == hi:
        return lo

    for _ in range(100):
        mid = (lo + hi) / 2
        if _objective(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of a stochastic test execution."""

    passed: bool
    estimate: float
    expected: float
    tol: float
    n: int
    bound_name: str
    seed: int
    message: str
    maurer_pontil_effective_n: int | None = field(default=None)


def check_assertion(
    estimate: float,
    config: TestConfig,
    bound: BoundStrategy,
    n: int,
    seed: int,
) -> TestResult:
    """Check whether the estimate passes the stochastic test assertion."""
    expected = config.expected
    tol = config.tol
    side = config.side

    if side == "two-sided":
        passed = abs(estimate - expected) < tol
        direction = f"|{estimate:.6g} - {expected:.6g}| = {abs(estimate - expected):.6g}"
        condition = f"< {tol:.6g}"
    elif side == "greater":
        passed = estimate > expected - tol
        direction = f"{estimate:.6g}"
        condition = f"> {expected - tol:.6g} (expected - tol)"
    else:  # "less"
        passed = estimate < expected + tol
        direction = f"{estimate:.6g}"
        condition = f"< {expected + tol:.6g} (expected + tol)"

    if passed:
        message = f"PASSED [{bound.name}, n={n}]: {direction} {condition}"
    else:
        message = (
            f"FAILED [{bound.name}, n={n}, seed={seed}]: "
            f"{direction} not {condition} "
            f"(expected={expected:.6g}, tol={tol:.6g})"
        )

    return TestResult(
        passed=passed,
        estimate=estimate,
        expected=expected,
        tol=tol,
        n=n,
        bound_name=bound.name,
        seed=seed,
        message=message,
    )


# ---------------------------------------------------------------------------
# Maurer-Pontil opportunistic upgrade
# ---------------------------------------------------------------------------


def check_maurer_pontil(
    samples: np.ndarray,
    config: TestConfig,
    failure_prob: float,
) -> int | None:
    """Check the Maurer-Pontil empirical Bernstein bound post-hoc.

    Given the collected samples, find the smallest prefix length m such that
    the Maurer-Pontil bound holds at the given failure probability and
    tolerance.  Returns *m* if a tighter effective n is found (m < len(samples)),
    or ``None`` if no improvement over the full sample count.

    The Maurer-Pontil bound states:
        P(|mean - mu| >= sqrt(2*var_hat*ln(2/delta)/n) + 7*(b-a)*ln(2/delta)/(3*(n-1))) <= delta
    """
    if config.bounds is None:
        return None

    n = len(samples)
    a, b = config.bounds
    rng = b - a
    tol = config.tol
    log_term = math.log(2 / failure_prob)

    # Check from smallest possible n upward to find the earliest point
    # where the bound holds.  We need at least n=2 for sample variance.
    min_n = 2
    effective_n = None

    for m in range(min_n, n + 1):
        sub = samples[:m]
        sample_var = float(np.var(sub, ddof=1))
        # Maurer-Pontil threshold
        threshold = math.sqrt(2 * sample_var * log_term / m) + 7 * rng * log_term / (3 * (m - 1))
        if threshold <= tol:
            effective_n = m
            break

    if effective_n is not None and effective_n < n:
        return effective_n
    return None
