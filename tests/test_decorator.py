"""Integration tests for the @stochastic_test and @distributional_test decorators."""

from __future__ import annotations

import pytest
from scipy import stats

from pytest_stochastic import distributional_test, stochastic_test
from pytest_stochastic.decorator import DISTRIBUTIONAL_TEST_MARKER, STOCHASTIC_TEST_MARKER
from pytest_stochastic.types import (
    ConfigurationError,
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


class TestDistributionalTestBasic:
    def test_normal_ks_pass(self):
        @distributional_test(
            reference=stats.norm(0, 1),
            test="ks",
            significance=1e-6,
            n_samples=1000,
            seed=42,
        )
        def test_fn(rng):
            return rng.normal(0, 1)

        test_fn()

    def test_normal_ks_fail(self):
        @distributional_test(
            reference=stats.norm(0, 1),
            test="ks",
            significance=0.05,
            n_samples=1000,
            seed=42,
        )
        def test_fn(rng):
            return rng.normal(5, 1)  # Wrong mean

        with pytest.raises(AssertionError, match="FAILED"):
            test_fn()

    def test_chi2_pass(self):
        @distributional_test(
            reference=stats.uniform(0, 1),
            test="chi2",
            significance=1e-6,
            n_samples=5000,
            seed=42,
        )
        def test_fn(rng):
            return rng.random()

        test_fn()

    def test_anderson_pass(self):
        @distributional_test(
            reference=stats.norm(0, 1),
            test="anderson",
            significance=0.01,
            n_samples=500,
            seed=42,
        )
        def test_fn(rng):
            return rng.normal(0, 1)

        test_fn()

    def test_anderson_fail_wrong_distribution(self):
        @distributional_test(
            reference=stats.norm(0, 1),
            test="anderson",
            significance=0.001,
            n_samples=500,
            seed=42,
        )
        def test_fn(rng):
            return rng.normal(3, 1)  # grossly wrong mean

        with pytest.raises(AssertionError, match="FAILED"):
            test_fn()


class TestDistributionalTestMetadata:
    def test_marker_attached(self):
        @distributional_test(
            reference=stats.norm(0, 1),
            test="ks",
            n_samples=100,
        )
        def test_fn(rng):
            return rng.normal(0, 1)

        assert hasattr(test_fn, DISTRIBUTIONAL_TEST_MARKER)
        meta = getattr(test_fn, DISTRIBUTIONAL_TEST_MARKER)
        assert meta["test"] == "ks"
        assert meta["n_samples"] == 100


class TestDistributionalTestValidation:
    def test_invalid_test_type_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown distributional test"):

            @distributional_test(
                reference=stats.norm(0, 1),
                test="invalid",
            )
            def test_fn(rng):
                return rng.normal(0, 1)

    def test_invalid_significance_raises(self):
        with pytest.raises(ConfigurationError, match="significance"):

            @distributional_test(
                reference=stats.norm(0, 1),
                significance=0.0,
            )
            def test_fn(rng):
                return rng.normal(0, 1)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ConfigurationError, match="n_samples"):

            @distributional_test(
                reference=stats.norm(0, 1),
                n_samples=0,
            )
            def test_fn(rng):
                return rng.normal(0, 1)

    def test_invalid_reference_raises(self):
        with pytest.raises(ConfigurationError, match="reference"):

            @distributional_test(
                reference="not a distribution",
            )
            def test_fn(rng):
                return rng.normal(0, 1)

    @pytest.mark.parametrize("significance", [1e-6, 1e-4, 0.25, 0.3])
    def test_anderson_significance_outside_scipy_range_raises(self, significance):
        """scipy's anderson_ksamp caps p-values to [0.001, 0.25]; outside
        that range the test could never fail (or would always fail), so the
        decorator must reject it loudly."""
        with pytest.raises(ConfigurationError, match="anderson"):

            @distributional_test(
                reference=stats.norm(0, 1),
                test="anderson",
                significance=significance,
            )
            def test_fn(rng):
                return rng.normal(0, 1)


class TestTunedVarianceMatching:
    """Unit tests for the .stochastic.toml key-matching logic."""

    def _with_tuned(self, monkeypatch, tuned):
        import pytest_stochastic.decorator as dec

        monkeypatch.setattr(dec, "_tuned_params_cache", tuned)

    def test_exact_module_qualname_match(self, monkeypatch):
        from pytest_stochastic.decorator import _tuned_variance_for

        def fn():
            return 0.0

        key = f"{fn.__module__}.{fn.__qualname__}"
        self._with_tuned(monkeypatch, {key: {"variance": 0.5}})
        assert _tuned_variance_for(fn) == 0.5

    def test_legacy_nodeid_key_with_py_segment_matches(self, monkeypatch):
        from pytest_stochastic.decorator import _normalize_test_key

        # Legacy keys were nodeid-derived: "tests.test_mod.py.test_fn"
        assert _normalize_test_key("tests.test_mod.py.test_fn") == "tests.test_mod.test_fn"

    def test_same_name_other_module_does_not_match(self, monkeypatch):
        from pytest_stochastic.decorator import _tuned_variance_for

        def fn():
            return 0.0

        # Two entries share fn's bare name but live in other modules: the
        # bare-name fallback must refuse the ambiguous match.
        self._with_tuned(
            monkeypatch,
            {
                f"other.module_a.{fn.__qualname__}": {"variance": 0.5},
                f"other.module_b.{fn.__qualname__}": {"variance": 0.7},
            },
        )
        assert _tuned_variance_for(fn) is None

    def test_unique_bare_name_fallback_matches(self, monkeypatch):
        from pytest_stochastic.decorator import _tuned_variance_for

        def fn():
            return 0.0

        self._with_tuned(monkeypatch, {f"other.module_a.{fn.__qualname__}": {"variance": 0.5}})
        assert _tuned_variance_for(fn) == 0.5

    def test_non_finite_variance_is_ignored(self, monkeypatch):
        from pytest_stochastic.decorator import _tuned_variance_for

        def fn():
            return 0.0

        key = f"{fn.__module__}.{fn.__qualname__}"
        self._with_tuned(monkeypatch, {key: {"variance": float("inf")}})
        assert _tuned_variance_for(fn) is None

    def test_inf_tuned_variance_does_not_break_decoration(self, monkeypatch):
        """A degenerate tune run (variance=inf) must not crash bound
        selection with an OverflowError at import time."""
        import pytest_stochastic.decorator as dec

        def sampler(rng):
            return rng.random()

        key = f"{sampler.__module__}.{sampler.__qualname__}"
        monkeypatch.setattr(dec, "_tuned_params_cache", {key: {"variance": float("inf")}})

        decorated = stochastic_test(expected=0.5, atol=0.1, bounds=(0, 1), seed=1)(sampler)
        assert decorated._stochastic_bound.name != "bernstein_tuned"


class TestDistributionalTestReproducibility:
    def test_fixed_seed_deterministic(self):
        results = []
        for _ in range(3):

            @distributional_test(
                reference=stats.norm(0, 1),
                test="ks",
                significance=1e-6,
                n_samples=500,
                seed=12345,
            )
            def test_fn(rng):
                return rng.normal(0, 1)

            try:
                test_fn()
                results.append("pass")
            except AssertionError:
                results.append("fail")

        assert len(set(results)) == 1
