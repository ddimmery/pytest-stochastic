"""Unit tests for the test execution runtime."""

from __future__ import annotations

import numpy as np
import pytest

from pytest_stochastic.runtime import (
    TestResult,
    _wants_rng,
    check_assertion,
    check_maurer_pontil,
    collect_samples,
    compute_estimate,
    make_rng,
)
from pytest_stochastic.types import BoundStrategy, EstimatorType, TestConfig


class TestWantsRng:
    def test_with_rng_param(self):
        def f(rng):
            pass

        assert _wants_rng(f) is True

    def test_without_rng_param(self):
        def f():
            pass

        assert _wants_rng(f) is False

    def test_with_other_params(self):
        def f(x, rng, y):
            pass

        assert _wants_rng(f) is True


class TestMakeRng:
    def test_returns_generator_and_seed(self):
        rng, seed = make_rng()
        assert isinstance(rng, np.random.Generator)
        assert isinstance(seed, int)

    def test_fixed_seed_reproducible(self):
        rng1, _ = make_rng(42)
        rng2, _ = make_rng(42)
        assert rng1.random() == rng2.random()


class TestCollectSamples:
    def test_basic_collection(self):
        def f():
            return 1.0

        samples = collect_samples(f, 10, np.random.default_rng(0))
        assert len(samples) == 10
        assert all(s == 1.0 for s in samples)

    def test_with_rng(self):
        def f(rng):
            return rng.random()

        samples = collect_samples(f, 100, np.random.default_rng(0))
        assert len(samples) == 100
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_exception_handling(self):
        def f():
            raise ValueError("oops")

        with pytest.raises(RuntimeError, match="call 1/5"):
            collect_samples(f, 5, np.random.default_rng(0))

    def test_non_numeric_raises(self):
        def f():
            return "not a number"

        with pytest.raises(TypeError, match="numeric scalar"):
            collect_samples(f, 1, np.random.default_rng(0))


class TestComputeEstimate:
    def test_sample_mean(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_estimate(samples, EstimatorType.SAMPLE_MEAN, 0.01)
        assert result == pytest.approx(3.0)

    def test_median_of_means(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0.0, 1.0, 10000)
        result = compute_estimate(samples, EstimatorType.MEDIAN_OF_MEANS, 0.01)
        assert abs(result) < 0.1  # should be near 0

    def test_catoni_estimator(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 1.0, 1000)
        result = compute_estimate(samples, EstimatorType.CATONI_M_ESTIMATOR, 0.01)
        assert abs(result - 5.0) < 0.2  # should be near 5.0


class TestCheckAssertion:
    def _make_config(self, side="two-sided"):
        return TestConfig(expected=0.5, tol=0.05, failure_prob=1e-8, side=side, bounds=(0.0, 1.0))

    def _make_bound(self):
        return BoundStrategy(
            name="test_bound",
            required_properties=frozenset({"bounds"}),
            optional_properties=frozenset(),
            compute_n=lambda *a, **kw: 100,
            supports_side=lambda s: True,
            estimator_type=EstimatorType.SAMPLE_MEAN,
            description="test",
        )

    def test_two_sided_pass(self):
        result = check_assertion(0.52, self._make_config(), self._make_bound(), 100, 42)
        assert result.passed is True
        assert "PASSED" in result.message

    def test_two_sided_fail(self):
        result = check_assertion(0.6, self._make_config(), self._make_bound(), 100, 42)
        assert result.passed is False
        assert "FAILED" in result.message
        assert "seed=42" in result.message

    def test_greater_pass(self):
        config = self._make_config(side="greater")
        result = check_assertion(0.46, config, self._make_bound(), 100, 42)
        # 0.46 > 0.5 - 0.05 = 0.45 => pass
        assert result.passed is True

    def test_greater_fail(self):
        config = self._make_config(side="greater")
        result = check_assertion(0.44, config, self._make_bound(), 100, 42)
        # 0.44 > 0.45 => fail
        assert result.passed is False

    def test_less_pass(self):
        config = self._make_config(side="less")
        result = check_assertion(0.54, config, self._make_bound(), 100, 42)
        # 0.54 < 0.5 + 0.05 = 0.55 => pass
        assert result.passed is True

    def test_less_fail(self):
        config = self._make_config(side="less")
        result = check_assertion(0.56, config, self._make_bound(), 100, 42)
        # 0.56 < 0.55 => fail
        assert result.passed is False

    def test_result_fields(self):
        result = check_assertion(0.52, self._make_config(), self._make_bound(), 100, 42)
        assert isinstance(result, TestResult)
        assert result.estimate == 0.52
        assert result.expected == 0.5
        assert result.tol == 0.05
        assert result.n == 100
        assert result.bound_name == "test_bound"
        assert result.seed == 42


class TestMaurerPontil:
    def test_low_variance_finds_effective_n(self):
        """With low-variance data, Maurer-Pontil should find an effective n < total n."""
        # Samples with very low variance and generous tolerance
        rng = np.random.default_rng(42)
        n = 5000
        samples = rng.normal(0.5, 0.01, n)
        config = TestConfig(expected=0.5, tol=0.05, failure_prob=1e-6, side="two-sided", bounds=(0.0, 1.0))
        effective_n = check_maurer_pontil(samples, config, failure_prob=1e-6)
        assert effective_n is not None
        assert effective_n < n

    def test_high_variance_no_improvement(self):
        """With high variance close to (b-a)^2/4, no improvement expected."""
        rng = np.random.default_rng(42)
        n = 100
        # Samples spanning the full range — variance near worst case
        samples = rng.choice([0.0, 1.0], size=n)
        config = TestConfig(expected=0.5, tol=0.05, failure_prob=1e-6, side="two-sided", bounds=(0.0, 1.0))
        effective_n = check_maurer_pontil(samples, config, failure_prob=1e-6)
        # With max variance and tight tolerance, no improvement from Maurer-Pontil
        assert effective_n is None

    def test_no_bounds_returns_none(self):
        """Without bounds declared, Maurer-Pontil is not applicable."""
        samples = np.ones(100)
        config = TestConfig(expected=1.0, tol=0.1, failure_prob=1e-6, side="two-sided", variance=0.01)
        effective_n = check_maurer_pontil(samples, config, failure_prob=1e-6)
        assert effective_n is None
