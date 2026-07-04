"""Calibration tests: bounds must respect their failure budgets.

These tests simulate the actual estimator path at the sample sizes the
framework computes and check the observed false-failure rate against the
declared budget.  They are seeded, so they are deterministic; the budgets
use a moderate delta so a handful of trials has real detection power.

They exist because the unit tests merely codify the formulas — only a
simulation can catch a formula that is self-consistent but statistically
unsound (as the pre-fix Catoni bound was: 16.5% observed failures against a
1% budget).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats

from pytest_stochastic.bounds import _bentkus_n, _catoni_n, _median_of_means_n
from pytest_stochastic.runtime import compute_estimate
from pytest_stochastic.types import EstimatorType, TestConfig


def _failure_rate(sampler, n: int, estimator, trials: int, tol: float) -> float:
    failures = sum(abs(estimator(sampler(n))) >= tol for _ in range(trials))
    return failures / trials


class TestCatoniCalibration:
    def test_p2_heavy_tail_within_budget(self):
        """Catoni (p=2) on scaled Student-t(3): mean 0, variance 1 = M."""
        tol, delta, m = 0.1, 0.01, 1.0
        n = _catoni_n(tol, delta, moment_bound=(2.0, m))
        config = TestConfig(
            expected=0.0,
            tol=tol,
            failure_prob=delta,
            side="two-sided",
            moment_bound=(2.0, m),
        )
        rng = np.random.default_rng(1234)
        trials = 400

        def sampler(k: int) -> np.ndarray:
            return rng.standard_t(df=3, size=k) / math.sqrt(3)

        rate = _failure_rate(
            sampler,
            n,
            lambda s: compute_estimate(s, EstimatorType.CATONI_M_ESTIMATOR, config),
            trials,
            tol,
        )
        # 3-sigma slack above the budget for the finite number of trials.
        assert rate <= delta + 3 * math.sqrt(delta / trials)

    def test_p15_infinite_variance_within_budget(self):
        """Catoni (p=1.5) on a symmetric Pareto mixture with infinite
        variance but finite 1.5th central moment."""
        p, tail = 1.5, 1.9  # tail < 2 => infinite variance; E|X|^1.5 finite
        m = tail / (tail - p)  # E|X|^p for symmetric Pareto(alpha=tail), |X| >= 1
        tol, delta = 1.0, 0.01
        n = _catoni_n(tol, delta, moment_bound=(p, m))
        config = TestConfig(
            expected=0.0,
            tol=tol,
            failure_prob=delta,
            side="two-sided",
            moment_bound=(p, m),
        )
        rng = np.random.default_rng(99)
        trials = 100

        def sampler(k: int) -> np.ndarray:
            magnitude = rng.pareto(tail, size=k) + 1.0
            sign = rng.choice([-1.0, 1.0], size=k)
            return magnitude * sign  # symmetric about 0 => mean 0

        rate = _failure_rate(
            sampler,
            n,
            lambda s: compute_estimate(s, EstimatorType.CATONI_M_ESTIMATOR, config),
            trials,
            tol,
        )
        assert rate <= delta + 3 * math.sqrt(delta / trials)


class TestMedianOfMeansCalibration:
    def test_heavy_tail_within_budget(self):
        tol, delta, var = 0.2, 0.01, 1.0
        n = _median_of_means_n(tol, delta, variance=var)
        config = TestConfig(
            expected=0.0, tol=tol, failure_prob=delta, side="two-sided", variance=var
        )
        rng = np.random.default_rng(7)
        trials = 200

        def sampler(k: int) -> np.ndarray:
            return rng.standard_t(df=3, size=k) / math.sqrt(3)  # variance 1

        rate = _failure_rate(
            sampler,
            n,
            lambda s: compute_estimate(s, EstimatorType.MEDIAN_OF_MEANS, config),
            trials,
            tol,
        )
        assert rate <= delta + 3 * math.sqrt(delta / trials)


class TestBentkusCalibration:
    def test_exact_worst_case_within_budget(self):
        """No simulation needed: the worst case on [0, 1] with the mean at
        the midpoint is Bernoulli(1/2), whose tail is exactly binomial."""
        for delta in (1e-2, 1e-4, 1e-6, 1e-8):
            tol = 0.1
            n = _bentkus_n(tol, delta, bounds=(0.0, 1.0), side="less")
            # Test fails iff sample mean >= 0.5 + tol.
            k_star = math.ceil(n * (0.5 + tol))
            exact_failure = float(sp_stats.binom.sf(k_star - 1, n, 0.5))
            assert exact_failure <= delta
