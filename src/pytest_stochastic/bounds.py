"""Concentration inequality registry.

Each bound computes the minimum number of samples *n* required to guarantee
that the chosen estimator is within *tol* of the true mean with probability at
least 1 - *failure_prob*, given the user's declared distributional properties.
"""

from __future__ import annotations

import math

from scipy import stats as sp_stats

from .types import BoundStrategy, EstimatorType

# ---------------------------------------------------------------------------
# Helper: side support predicates
# ---------------------------------------------------------------------------


def _supports_any_side(side: str) -> bool:
    return side in {"two-sided", "greater", "less"}


def _supports_two_sided_only(side: str) -> bool:
    return side == "two-sided"


def _supports_one_sided_only(side: str) -> bool:
    return side in {"greater", "less"}


# ---------------------------------------------------------------------------
# Individual bound implementations
# ---------------------------------------------------------------------------


def _median_of_means_n(tol: float, failure_prob: float, **props: object) -> int:
    """k = ceil(8 ln(2/delta)), n = k * ceil(2 sigma^2 / epsilon^2)."""
    variance = float(props["variance"])  # type: ignore[arg-type]
    k = math.ceil(8 * math.log(2 / failure_prob))
    block_size = math.ceil(2 * variance / tol**2)
    return k * block_size


def _catoni_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil(C_p * (M / eps^p)^(2/(p+1)) * ln(2/delta)^(p/(p+1)))."""
    p, m = props["moment_bound"]  # type: ignore[index]
    p = float(p)
    m = float(m)
    # C_p is a constant depending on p; use a standard choice
    c_p = 2.0 * (p / (p - 1)) ** (2 * p / (p + 1))
    return math.ceil(
        c_p * (m / tol**p) ** (2 / (p + 1)) * math.log(2 / failure_prob) ** (p / (p + 1))
    )


def _hoeffding_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil((b - a)^2 * ln(2/delta) / (2 * epsilon^2))."""
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    return math.ceil((b - a) ** 2 * math.log(2 / failure_prob) / (2 * tol**2))


def _anderson_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil((b - a)^2 * ln(1/delta) / (2 * epsilon^2)).

    Factor-of-2 improvement over Hoeffding for symmetric distributions.
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    return math.ceil((b - a) ** 2 * math.log(1 / failure_prob) / (2 * tol**2))


def _maurer_pontil_n(tol: float, failure_prob: float, **props: object) -> int:
    """Conservative pre-allocation using worst-case variance (b-a)^2/4.

    At runtime the framework checks whether the empirical Maurer-Pontil bound
    is tighter than Hoeffding.  For pre-allocation we fall back to Hoeffding's
    n so that Maurer-Pontil never *increases* the sample count.
    """
    # Pre-allocation is identical to Hoeffding; the benefit is post-hoc.
    return _hoeffding_n(tol, failure_prob, **props)


def _bentkus_n(tol: float, failure_prob: float, **props: object) -> int:
    """Numerically invert the Bentkus Binomial tail bound via bisection.

    For one-sided tests the full budget goes to one tail.
    For two-sided tests we split the budget (delta/2 per tail).
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    range_width = b - a

    # Bentkus constant
    c_bentkus = math.e / math.sqrt(2 * math.pi)

    # For one-sided: full delta; for two-sided: delta/2 per tail
    # The caller decides; we receive the effective delta directly.
    # However, we receive the full failure_prob always and handle sidedness
    # via the supports_side predicate.  For one-sided bounds the full budget
    # goes to one tail.
    delta = failure_prob

    # Bisection: find smallest n such that the Bentkus bound holds.
    lo, hi = 1, _hoeffding_n(tol, failure_prob, **props)
    # Ensure hi is sufficient
    hi = max(hi, 2)

    def _bentkus_holds(n: int) -> bool:
        # P(|S_n/n - mu| >= tol) <= c * P(Bin(n, p_star) >= k_star)
        p_star = tol / range_width + 0.5
        p_star = min(max(p_star, 0.0), 1.0)
        k_star = math.ceil(n * p_star)
        if k_star > n:
            return True
        binom_tail = 1.0 - sp_stats.binom.cdf(k_star - 1, n, 0.5)
        return c_bentkus * binom_tail <= delta

    while lo < hi:
        mid = (lo + hi) // 2
        if _bentkus_holds(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _bernstein_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil(2 sigma^2 ln(2/delta) / eps^2 + 2(b-a) ln(2/delta) / (3 eps))."""
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    variance = float(props["variance"])  # type: ignore[arg-type]
    log_term = math.log(2 / failure_prob)
    return math.ceil(2 * variance * log_term / tol**2 + 2 * (b - a) * log_term / (3 * tol))


def _bernstein_tuned_n(tol: float, failure_prob: float, **props: object) -> int:
    """Bernstein with machine-discovered variance from --stochastic-tune.

    Same formula as Bernstein, but uses variance_tuned (the UCB from tuning)
    instead of a user-declared variance.
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    variance = float(props["variance_tuned"])  # type: ignore[arg-type]
    log_term = math.log(2 / failure_prob)
    return math.ceil(2 * variance * log_term / tol**2 + 2 * (b - a) * log_term / (3 * tol))


def _sub_gaussian_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil(2 sigma^2 ln(2/delta) / epsilon^2)."""
    sigma = float(props["sub_gaussian_param"])  # type: ignore[arg-type]
    return math.ceil(2 * sigma**2 * math.log(2 / failure_prob) / tol**2)


# ---------------------------------------------------------------------------
# Bound registry
# ---------------------------------------------------------------------------

BOUND_REGISTRY: list[BoundStrategy] = [
    BoundStrategy(
        name="median_of_means",
        required_properties=frozenset({"variance"}),
        optional_properties=frozenset(),
        compute_n=_median_of_means_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.MEDIAN_OF_MEANS,
        description="Median-of-means; sub-Gaussian rate with only finite variance",
    ),
    BoundStrategy(
        name="catoni",
        required_properties=frozenset({"moment_bound"}),
        optional_properties=frozenset(),
        compute_n=_catoni_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.CATONI_M_ESTIMATOR,
        description="Catoni M-estimator; handles heavy-tailed distributions (p > 1)",
    ),
    BoundStrategy(
        name="hoeffding",
        required_properties=frozenset({"bounds"}),
        optional_properties=frozenset(),
        compute_n=_hoeffding_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Hoeffding's inequality for bounded random variables",
    ),
    BoundStrategy(
        name="anderson",
        required_properties=frozenset({"bounds", "symmetric"}),
        optional_properties=frozenset(),
        compute_n=_anderson_n,
        supports_side=_supports_two_sided_only,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Anderson's inequality; 2x improvement for symmetric distributions",
    ),
    BoundStrategy(
        name="maurer_pontil",
        required_properties=frozenset({"bounds"}),
        optional_properties=frozenset(),
        compute_n=_maurer_pontil_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Maurer-Pontil empirical Bernstein; data-adaptive, no declared variance",
    ),
    BoundStrategy(
        name="bentkus",
        required_properties=frozenset({"bounds"}),
        optional_properties=frozenset(),
        compute_n=_bentkus_n,
        supports_side=_supports_one_sided_only,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Bentkus inequality; ~20-40% fewer samples for one-sided bounded tests",
    ),
    BoundStrategy(
        name="bernstein",
        required_properties=frozenset({"bounds", "variance"}),
        optional_properties=frozenset(),
        compute_n=_bernstein_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Bernstein's inequality; tight when variance << range^2",
    ),
    BoundStrategy(
        name="bernstein_tuned",
        required_properties=frozenset({"bounds", "variance_tuned"}),
        optional_properties=frozenset(),
        compute_n=_bernstein_tuned_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Bernstein with machine-discovered variance from --stochastic-tune",
    ),
    BoundStrategy(
        name="sub_gaussian",
        required_properties=frozenset({"sub_gaussian_param"}),
        optional_properties=frozenset(),
        compute_n=_sub_gaussian_n,
        supports_side=_supports_any_side,
        estimator_type=EstimatorType.SAMPLE_MEAN,
        description="Sub-Gaussian tail bound",
    ),
]


def applicable_bounds(
    declared_properties: dict[str, object],
    side: str,
) -> list[BoundStrategy]:
    """Return all bounds whose requirements are met by *declared_properties*."""
    declared_keys = set(declared_properties.keys())
    return [
        b
        for b in BOUND_REGISTRY
        if b.required_properties <= declared_keys and b.supports_side(side)
    ]
