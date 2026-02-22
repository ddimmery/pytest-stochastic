"""Integration tests for the @stochastic_test decorator."""

from __future__ import annotations

import pytest

from pytest_stochastic import stochastic_test
from pytest_stochastic.decorator import STOCHASTIC_TEST_MARKER
from pytest_stochastic.types import (
    InvalidPropertyError,
    InvalidToleranceError,
)


class TestDecoratorBasic:
    def test_fair_coin(self):
        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=42)
        def test_coin(rng):
            return rng.random()

        # Should not raise
        test_coin()

    def test_zero_mean(self):
        @stochastic_test(expected=0.0, atol=0.1, bounds=(-1, 1), failure_prob=1e-6, seed=42)
        def test_zero(rng):
            return rng.uniform(-1, 1)

        test_zero()

    def test_deterministic_pass(self):
        @stochastic_test(expected=1.0, atol=0.01, bounds=(0.99, 1.01), failure_prob=1e-6)
        def test_const():
            return 1.0

        test_const()

    def test_deterministic_fail(self):
        @stochastic_test(expected=0.0, atol=0.01, bounds=(0.99, 1.01), failure_prob=1e-6)
        def test_const():
            return 1.0

        with pytest.raises(AssertionError, match="FAILED"):
            test_const()


class TestDecoratorMetadata:
    def test_marker_attached(self):
        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))
        def test_fn(rng):
            return rng.random()

        assert hasattr(test_fn, STOCHASTIC_TEST_MARKER)
        config = getattr(test_fn, STOCHASTIC_TEST_MARKER)
        assert config.expected == 0.5
        assert config.tol == 0.05

    def test_bound_info_attached(self):
        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))
        def test_fn(rng):
            return rng.random()

        assert hasattr(test_fn, "_stochastic_bound")
        assert hasattr(test_fn, "_stochastic_n")
        assert test_fn._stochastic_n > 0


class TestDecoratorValidation:
    def test_no_tolerance_raises(self):
        with pytest.raises(InvalidToleranceError):

            @stochastic_test(expected=0.5, bounds=(0, 1))
            def test_fn(rng):
                return rng.random()

    def test_no_properties_raises(self):
        with pytest.raises(InvalidPropertyError):

            @stochastic_test(expected=0.5, atol=0.05)
            def test_fn(rng):
                return rng.random()

    def test_invalid_bounds_raises(self):
        with pytest.raises(InvalidPropertyError):

            @stochastic_test(expected=0.5, atol=0.05, bounds=(1, 0))
            def test_fn(rng):
                return rng.random()


class TestDecoratorSides:
    def test_one_sided_greater(self):
        @stochastic_test(
            expected=0.5,
            atol=0.1,
            bounds=(0, 1),
            side="greater",
            failure_prob=1e-6,
            seed=42,
        )
        def test_fn(rng):
            return rng.random()

        test_fn()

    def test_one_sided_less(self):
        @stochastic_test(
            expected=0.5,
            atol=0.1,
            bounds=(0, 1),
            side="less",
            failure_prob=1e-6,
            seed=42,
        )
        def test_fn(rng):
            return rng.random()

        test_fn()


class TestDecoratorWithVariance:
    def test_bernstein_selected_with_low_variance(self):
        @stochastic_test(
            expected=0.5,
            atol=0.05,
            bounds=(0, 1),
            variance=0.01,
            failure_prob=1e-6,
            seed=42,
        )
        def test_fn(rng):
            return rng.random()

        assert test_fn._stochastic_bound.name == "bernstein"

    def test_with_variance_only(self):
        @stochastic_test(
            expected=0.0,
            atol=0.2,
            variance=1.0,
            failure_prob=1e-4,
            seed=42,
        )
        def test_fn(rng):
            return rng.normal(0.0, 1.0)

        test_fn()


class TestDecoratorReproducibility:
    def test_fixed_seed_deterministic(self):
        results = []
        for _ in range(3):

            @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=12345)
            def test_fn(rng):
                return rng.random()

            try:
                test_fn()
                results.append("pass")
            except AssertionError:
                results.append("fail")

        # All runs with the same seed should produce the same result
        assert len(set(results)) == 1
