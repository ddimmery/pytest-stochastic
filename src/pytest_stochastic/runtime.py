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
    config: TestConfig,
) -> float:
    """Compute the point estimate using the appropriate estimator.

    The estimators must mirror the pre-allocation formulas in
    :mod:`pytest_stochastic.bounds` exactly (same block counts, same tuning
    constants), otherwise the computed sample sizes carry no guarantee.
    """
    if estimator_type == EstimatorType.SAMPLE_MEAN:
        return float(np.mean(samples))

    if estimator_type == EstimatorType.MEDIAN_OF_MEANS:
        return _median_of_means(samples, config.failure_prob, config.side)

    if estimator_type == EstimatorType.CATONI_M_ESTIMATOR:
        return _catoni_estimator(samples, config)

    raise ValueError(f"Unknown estimator type: {estimator_type}")  # pragma: no cover


def _mom_k(failure_prob: float, side: str) -> int:
    """Median-of-means block count, mirroring ``bounds._mom_num_blocks``."""
    k_side = 2 if side == "two-sided" else 1
    return math.ceil(8 * math.log(k_side / failure_prob))


def _median_of_means(
    samples: np.ndarray,
    failure_prob: float,
    side: str = "two-sided",
) -> float:
    """Median-of-means estimator with the same block count as pre-allocation."""
    n = len(samples)
    k = min(_mom_k(failure_prob, side), n)  # can't have more blocks than samples
    block_size = n // k
    if block_size == 0:
        return float(np.mean(samples))

    block_means = np.array(
        [np.mean(samples[i * block_size : (i + 1) * block_size]) for i in range(k)]
    )
    return float(np.median(block_means))


def _catoni_estimator(samples: np.ndarray, config: TestConfig) -> float:
    """Estimator backing the ``catoni`` bound.

    For a declared second central moment (p = 2, M = sigma^2) this is
    Catoni's M-estimator (Catoni 2012) with the influence function
    psi(x) = sign(x) * log(1 + |x| + x^2/2) and

        alpha = sqrt(2 ln(k_side/delta) / (n (M + tol^2))),

    matching the sample size computed by ``bounds._catoni_n``.  For
    1 < p < 2 the pre-allocation is based on median-of-means blocks, so the
    estimator is median-of-means with the identical block count.
    """
    n = len(samples)
    if n == 0:
        return float("nan")

    if config.moment_bound is None:  # pragma: no cover - catoni requires moment_bound
        raise ValueError("Catoni estimator requires a declared moment_bound=(p, M)")
    p, m = float(config.moment_bound[0]), float(config.moment_bound[1])
    if p < 2.0:
        return _median_of_means(samples, config.failure_prob, config.side)

    k_side = 2 if config.side == "two-sided" else 1
    alpha = math.sqrt(2 * math.log(k_side / config.failure_prob) / (n * (m + config.tol**2)))

    def _objective(mu: float) -> float:
        x = alpha * (samples - mu)
        ax = np.abs(x)
        return float(np.sum(np.sign(x) * np.log1p(ax + ax * ax / 2)) / n)

    # The objective is decreasing in mu with a root in [min, max]; bisect.
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

    Maurer & Pontil (2009, "Empirical Bernstein Bounds and Sample Variance
    Penalization") give the one-sided bound

        P(mu - mean >= sqrt(2 V_n ln(2/delta)/n) + 7 (b-a) ln(2/delta) / (3 (n-1))) <= delta

    — the factor 2 inside the log is intrinsic (a union over the mean and
    variance deviations), so a two-sided test needs ln(4/delta).  Because we
    scan every prefix length m, we additionally apply a union bound over the
    n - 1 prefixes considered, giving log(2 * k_side * (n-1) / delta).

    The returned value is purely informational (it never affects pass/fail):
    it reports how many samples would have sufficed had the empirical
    variance been trusted from the start.
    """
    if config.bounds is None:
        return None

    n = len(samples)
    if n < 2:
        return None

    a, b = config.bounds
    rng = b - a
    tol = config.tol
    k_side = 2 if config.side == "two-sided" else 1
    log_term = math.log(2 * k_side * (n - 1) / failure_prob)

    # Running mean/variance over all prefixes m = 2..n via cumulative sums.
    csum = np.cumsum(samples)
    csum2 = np.cumsum(samples * samples)
    m = np.arange(2, n + 1, dtype=np.float64)
    mean = csum[1:] / m
    # Unbiased sample variance; clip tiny negative values from cancellation.
    var = np.maximum((csum2[1:] - m * mean * mean) / (m - 1), 0.0)

    threshold = np.sqrt(2 * var * log_term / m) + 7 * rng * log_term / (3 * (m - 1))
    holds = threshold <= tol
    if not holds.any():
        return None

    effective_n = int(np.argmax(holds)) + 2
    if effective_n < n:
        return effective_n
    return None
