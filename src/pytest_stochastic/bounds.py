"""Concentration inequality registry.

Each bound computes the minimum number of samples *n* required to guarantee
that the chosen estimator is within *tol* of the true mean with probability at
least 1 - *failure_prob*, given the user's declared distributional properties.
"""

from __future__ import annotations

import math

from scipy import stats as sp_stats

from .types import BoundStrategy, EstimatorType, InvalidPropertyError

# ---------------------------------------------------------------------------
# Helper: side support predicates
# ---------------------------------------------------------------------------


def _supports_any_side(side: str) -> bool:
    return side in {"two-sided", "greater", "less"}


def _supports_two_sided_only(side: str) -> bool:
    return side == "two-sided"


def _supports_one_sided_only(side: str) -> bool:
    return side in {"greater", "less"}


def _log_inv_delta(failure_prob: float, side: str) -> float:
    """Return ``ln(k / δ)`` where *k* = 2 for two-sided and 1 for one-sided.

    Two-sided tests need a union bound over both tails (factor of 2).
    One-sided tests spend the full failure budget on a single tail.
    """
    k = 2 if side == "two-sided" else 1
    return math.log(k / failure_prob)


# ---------------------------------------------------------------------------
# Individual bound implementations
# ---------------------------------------------------------------------------


def _mom_num_blocks(failure_prob: float, side: str) -> int:
    """Number of median-of-means blocks: k = ceil(8 ln(k_side/delta)).

    Each block fails (deviates by more than epsilon) with probability at most
    1/4 by Chebyshev, so the median fails only if at least k/2 blocks fail;
    by Hoeffding on the Bernoulli(1/4) failure indicators,
    P(Bin(k, 1/4) >= k/2) <= exp(-2 k (1/4)^2) = exp(-k/8) <= delta.
    (The two-sided union factor inside the log is conservative: both tail
    events are contained in the single "half the blocks deviate" event.)
    """
    return math.ceil(8 * _log_inv_delta(failure_prob, side))


def _median_of_means_n(tol: float, failure_prob: float, **props: object) -> int:
    """k = ceil(8 ln(k_side/delta)) blocks of size ceil(4 sigma^2 / epsilon^2).

    Block size 4 sigma^2 / eps^2 makes the per-block Chebyshev failure
    probability at most 1/4, which the median argument requires (block size
    2 sigma^2 / eps^2 would give 1/2 and yield no guarantee at all).
    """
    variance = float(props["variance"])  # type: ignore[arg-type]
    side = str(props.get("side", "two-sided"))
    k = _mom_num_blocks(failure_prob, side)
    block_size = math.ceil(4 * variance / tol**2)
    return k * block_size


def _catoni_n(tol: float, failure_prob: float, **props: object) -> int:
    """Sample size under a central-moment bound E|X - mu|^p <= M, p in (1, 2].

    p = 2 (finite variance, M = sigma^2): Catoni's M-estimator (Catoni 2012,
    Ann. IHP 48(4)) achieves |mu_hat - mu| <= eps with probability
    >= 1 - delta once

        n = ceil(2 ln(k_side/delta) (M/eps^2 + 1)).

    The "+1" is the finite-n correction: the Chernoff argument evaluates
    E(X - m)^2 = sigma^2 + eps^2 at the test points m = mu +/- eps.

    1 < p < 2 (possibly infinite variance): median-of-means on blocks sized
    via the von Bahr-Esseen inequality (1965), E|sum (X_i - mu)|^p <= 2 n M,
    so a block of size B satisfies P(|block mean - mu| >= eps)
    <= 2 M / (B^(p-1) eps^p) <= 1/4 once B = ceil((8 M / eps^p)^(1/(p-1)));
    k = ceil(8 ln(k_side/delta)) blocks then give failure <= delta (cf.
    Bubeck, Cesa-Bianchi & Lugosi 2013, "Bandits with heavy tail").
    """
    p, m = props["moment_bound"]  # type: ignore[index]
    p = float(p)
    m = float(m)
    side = str(props.get("side", "two-sided"))
    if p == 2.0:
        return math.ceil(2 * _log_inv_delta(failure_prob, side) * (m / tol**2 + 1))
    block_size = math.ceil((8 * m / tol**p) ** (1 / (p - 1)))
    return _mom_num_blocks(failure_prob, side) * block_size


def _hoeffding_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil((b - a)^2 * ln(k_side/delta) / (2 * epsilon^2))."""
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    side = str(props.get("side", "two-sided"))
    return math.ceil((b - a) ** 2 * _log_inv_delta(failure_prob, side) / (2 * tol**2))


def _anderson_n(tol: float, failure_prob: float, **props: object) -> int:
    """Hoeffding on the reduced support of a symmetric distribution.

    Under the null hypothesis the mean is *expected*.  A distribution
    symmetric about its mean mu with support in [a, b] actually has
    ess sup |X - mu| <= w := min(b - mu, mu - a): any mass at mu + d with
    d > min(b - mu, mu - a) would require matching mass at mu - d outside
    [a, b].  Hoeffding therefore applies with range 2w:

        n = ceil((2w)^2 ln(2/delta) / (2 eps^2)) = ceil(2 w^2 ln(2/delta) / eps^2).

    Strictly better than Hoeffding when *expected* is off-center in [a, b];
    identical when centered.
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    expected = float(props["expected"])  # type: ignore[arg-type]
    w = min(b - expected, expected - a)
    if w <= 0:
        raise InvalidPropertyError(
            "The symmetric (anderson) bound requires a < expected < b, got "
            f"expected={expected} with bounds=({a}, {b})"
        )
    return math.ceil(2 * w**2 * math.log(2 / failure_prob) / tol**2)


def _maurer_pontil_n(tol: float, failure_prob: float, **props: object) -> int:
    """Conservative pre-allocation using worst-case variance (b-a)^2/4.

    At runtime the framework checks whether the empirical Maurer-Pontil bound
    is tighter than Hoeffding.  For pre-allocation we fall back to Hoeffding's
    n so that Maurer-Pontil never *increases* the sample count.
    """
    # Pre-allocation is identical to Hoeffding; the benefit is post-hoc.
    return _hoeffding_n(tol, failure_prob, **props)


def _bentkus_n(tol: float, failure_prob: float, **props: object) -> int:
    """Numerically invert the Bentkus binomial tail bound.

    Bentkus (2004, Ann. Probab. 32(2), Thm 1.1): for independent summands
    bounded in [a, b], P(S_n/n - mu >= tol) <= (e^2/2) P°(Bin(n, q) >= x)
    where P° is the least log-concave majorant of the binomial tail and the
    worst case over the unknown mean is q = 1/2.  Since P° interpolates the
    discrete tail, P°(x) <= P(Bin >= floor(x)); we therefore evaluate the
    tail at floor(n p*) to stay on the conservative side.

    This bound is registered one-sided only, so the full failure budget goes
    to the single tail being tested.
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    range_width = b - a

    # Bentkus (2004) constant
    c_bentkus = math.e**2 / 2
    delta = failure_prob

    def _bentkus_holds(n: int) -> bool:
        # P(S_n/n - mu >= tol) <= c * P(Bin(n, 1/2) >= floor(n p*))
        p_star = tol / range_width + 0.5
        p_star = min(max(p_star, 0.0), 1.0)
        k_star = math.floor(n * p_star)
        if k_star > n:
            return True
        binom_tail = float(sp_stats.binom.sf(k_star - 1, n, 0.5))
        return c_bentkus * binom_tail <= delta

    # Establish a verified upper bracket by doubling (the Hoeffding n is not
    # guaranteed to satisfy the predicate because of the e^2/2 constant).
    hi = 2
    while not _bentkus_holds(hi):
        hi *= 2
        if hi > 2**40:  # pragma: no cover - unreachable for valid inputs
            raise OverflowError("Bentkus bound inversion failed to bracket")

    # Bisect for the smallest n where the bound holds ...
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _bentkus_holds(mid):
            hi = mid
        else:
            lo = mid + 1

    # ... and guard against non-monotonicity of the discrete tail (parity
    # effects of the ceiling): soundness only requires the bound to hold at
    # the n actually used.
    while not _bentkus_holds(lo):
        lo += 1
    return lo


def _bernstein_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil(2 sigma^2 ln(k_side/delta) / eps^2 + 2(b-a) ln(k_side/delta) / (3 eps))."""
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    variance = float(props["variance"])  # type: ignore[arg-type]
    side = str(props.get("side", "two-sided"))
    log_term = _log_inv_delta(failure_prob, side)
    return math.ceil(2 * variance * log_term / tol**2 + 2 * (b - a) * log_term / (3 * tol))


def _bernstein_tuned_n(tol: float, failure_prob: float, **props: object) -> int:
    """Bernstein with machine-discovered variance from --stochastic-tune.

    Same formula as Bernstein, but uses variance_tuned (the UCB from tuning)
    instead of a user-declared variance.
    """
    a, b = props["bounds"]  # type: ignore[index]
    a, b = float(a), float(b)
    variance = float(props["variance_tuned"])  # type: ignore[arg-type]
    side = str(props.get("side", "two-sided"))
    log_term = _log_inv_delta(failure_prob, side)
    return math.ceil(2 * variance * log_term / tol**2 + 2 * (b - a) * log_term / (3 * tol))


def _sub_gaussian_n(tol: float, failure_prob: float, **props: object) -> int:
    """n = ceil(2 sigma^2 ln(k_side/delta) / epsilon^2)."""
    sigma = float(props["sub_gaussian_param"])  # type: ignore[arg-type]
    side = str(props.get("side", "two-sided"))
    return math.ceil(2 * sigma**2 * _log_inv_delta(failure_prob, side) / tol**2)


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
        description="Catoni M-estimator (p = 2) / median-of-means (p < 2) for heavy tails",
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
        description="Hoeffding on the reduced support of a symmetric distribution; "
        "improves on Hoeffding when expected is off-center in [a, b]",
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
        description="Bentkus inequality; ~5-10% fewer samples for one-sided bounded tests",
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
