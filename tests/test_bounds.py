"""Unit tests for concentration inequality bounds."""

from __future__ import annotations

import math

import pytest

from pytest_stochastic.bounds import (
    BOUND_REGISTRY,
    _anderson_n,
    _bentkus_n,
    _bernstein_n,
    _hoeffding_n,
    _log_inv_delta,
    _maurer_pontil_n,
    _median_of_means_n,
    _sub_gaussian_n,
    applicable_bounds,
)


class TestMedianOfMeans:
    def test_known_value(self):
        # k = ceil(8 * ln(2/0.01)) = ceil(8 * 5.298) = 43
        # block_size = ceil(2 * 1.0 / 0.1^2) = ceil(200) = 200
        # n = 43 * 200 = 8600
        n = _median_of_means_n(0.1, 0.01, variance=1.0)
        k = math.ceil(8 * math.log(2 / 0.01))
        block_size = math.ceil(2 * 1.0 / 0.1**2)
        assert n == k * block_size


class TestHoeffding:
    def test_known_value(self):
        # n = ceil((b-a)^2 * ln(2/delta) / (2 * eps^2))
        # bounds=(0,1), delta=0.01, tol=0.1
        # = ceil(1 * ln(200) / 0.02) = ceil(5.298 / 0.02) = ceil(264.9) = 265
        n = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        expected = math.ceil(1.0 * math.log(2 / 0.01) / (2 * 0.1**2))
        assert n == expected

    def test_wider_range_more_samples(self):
        n1 = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        n2 = _hoeffding_n(0.1, 0.01, bounds=(0.0, 2.0))
        assert n2 > n1


class TestAnderson:
    def test_improves_over_hoeffding(self):
        # Anderson uses ln(1/delta) vs Hoeffding's ln(2/delta)
        n_hoef = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        n_ander = _anderson_n(0.1, 0.01, bounds=(0.0, 1.0), symmetric=True)
        assert n_ander < n_hoef

    def test_known_value(self):
        n = _anderson_n(0.1, 0.01, bounds=(0.0, 1.0), symmetric=True)
        expected = math.ceil(1.0 * math.log(1 / 0.01) / (2 * 0.1**2))
        assert n == expected


class TestBernstein:
    def test_known_value(self):
        # n = ceil(2*var*ln(2/d)/eps^2 + 2*(b-a)*ln(2/d)/(3*eps))
        tol, delta = 0.1, 0.01
        var, a, b = 0.05, 0.0, 1.0
        log_term = math.log(2 / delta)
        expected = math.ceil(2 * var * log_term / tol**2 + 2 * (b - a) * log_term / (3 * tol))
        n = _bernstein_n(tol, delta, bounds=(a, b), variance=var)
        assert n == expected

    def test_beats_hoeffding_with_low_variance(self):
        # With small variance, Bernstein should require fewer samples
        n_hoef = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        n_bern = _bernstein_n(0.1, 0.01, bounds=(0.0, 1.0), variance=0.01)
        assert n_bern < n_hoef


class TestSubGaussian:
    def test_known_value(self):
        # n = ceil(2 * sigma^2 * ln(2/delta) / eps^2)
        sigma = 0.5
        tol, delta = 0.1, 0.01
        expected = math.ceil(2 * sigma**2 * math.log(2 / delta) / tol**2)
        n = _sub_gaussian_n(tol, delta, sub_gaussian_param=sigma)
        assert n == expected


class TestBentkus:
    def test_improves_over_hoeffding_one_sided(self):
        n_hoef = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        n_bent = _bentkus_n(0.1, 0.01, bounds=(0.0, 1.0))
        assert n_bent <= n_hoef

    def test_returns_positive(self):
        n = _bentkus_n(0.1, 0.01, bounds=(0.0, 1.0))
        assert n >= 1


class TestMaurerPontil:
    def test_equals_hoeffding_pre_allocation(self):
        # Maurer-Pontil pre-allocates conservatively using Hoeffding's n
        n_mp = _maurer_pontil_n(0.1, 0.01, bounds=(0.0, 1.0))
        n_hoef = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0))
        assert n_mp == n_hoef


class TestBoundRegistry:
    def test_registry_not_empty(self):
        assert len(BOUND_REGISTRY) > 0

    def test_all_names_unique(self):
        names = [b.name for b in BOUND_REGISTRY]
        assert len(names) == len(set(names))

    def test_applicable_with_bounds(self):
        result = applicable_bounds({"bounds": (0.0, 1.0)}, "two-sided")
        names = {b.name for b in result}
        assert "hoeffding" in names
        assert "maurer_pontil" in names

    def test_applicable_with_variance(self):
        result = applicable_bounds({"variance": 1.0}, "two-sided")
        names = {b.name for b in result}
        assert "median_of_means" in names

    def test_applicable_with_bounds_and_variance(self):
        result = applicable_bounds({"bounds": (0.0, 1.0), "variance": 0.1}, "two-sided")
        names = {b.name for b in result}
        assert "bernstein" in names
        assert "hoeffding" in names

    def test_anderson_requires_symmetric(self):
        result = applicable_bounds({"bounds": (0.0, 1.0)}, "two-sided")
        names = {b.name for b in result}
        assert "anderson" not in names

        result = applicable_bounds({"bounds": (0.0, 1.0), "symmetric": True}, "two-sided")
        names = {b.name for b in result}
        assert "anderson" in names

    def test_anderson_two_sided_only(self):
        result = applicable_bounds({"bounds": (0.0, 1.0), "symmetric": True}, "greater")
        names = {b.name for b in result}
        assert "anderson" not in names

    def test_bentkus_one_sided_only(self):
        result = applicable_bounds({"bounds": (0.0, 1.0)}, "two-sided")
        names = {b.name for b in result}
        assert "bentkus" not in names

        result = applicable_bounds({"bounds": (0.0, 1.0)}, "greater")
        names = {b.name for b in result}
        assert "bentkus" in names

    def test_no_properties_no_bounds(self):
        result = applicable_bounds({}, "two-sided")
        assert result == []

    def test_tighter_bounds_yield_smaller_n(self):
        """Bounds with more information should require fewer samples."""
        props_bounds_only = {"bounds": (0.0, 1.0)}
        props_bounds_var = {"bounds": (0.0, 1.0), "variance": 0.05}

        # Get best n for each property set
        tol, delta = 0.05, 1e-6
        ns_bounds = [
            b.compute_n(tol, delta, **props_bounds_only)
            for b in applicable_bounds(props_bounds_only, "two-sided")
        ]
        ns_bounds_var = [
            b.compute_n(tol, delta, **props_bounds_var)
            for b in applicable_bounds(props_bounds_var, "two-sided")
        ]
        assert min(ns_bounds_var) < min(ns_bounds)


class TestCatoni:
    def test_returns_positive(self):
        from pytest_stochastic.bounds import _catoni_n

        n = _catoni_n(0.1, 0.01, moment_bound=(1.5, 1.0))
        assert n >= 1

    def test_higher_p_fewer_samples(self):
        from pytest_stochastic.bounds import _catoni_n

        n1 = _catoni_n(0.1, 0.01, moment_bound=(1.2, 1.0))
        n2 = _catoni_n(0.1, 0.01, moment_bound=(1.8, 1.0))
        # Higher p (closer to 2) should generally give fewer samples
        # when the moment constant M is the same
        assert n2 <= n1


class TestEdgeCases:
    """Test edge cases across bounds."""

    @pytest.mark.parametrize(
        "bound_fn,props",
        [
            (_hoeffding_n, {"bounds": (0.0, 1.0)}),
            (_bernstein_n, {"bounds": (0.0, 1.0), "variance": 0.1}),
            (_sub_gaussian_n, {"sub_gaussian_param": 0.5}),
        ],
    )
    def test_all_return_positive(self, bound_fn, props):
        n = bound_fn(0.1, 0.01, **props)
        assert n >= 1

    @pytest.mark.parametrize(
        "bound_fn,props",
        [
            (_hoeffding_n, {"bounds": (0.0, 1.0)}),
            (_bernstein_n, {"bounds": (0.0, 1.0), "variance": 0.1}),
            (_sub_gaussian_n, {"sub_gaussian_param": 0.5}),
        ],
    )
    def test_smaller_tolerance_more_samples(self, bound_fn, props):
        n1 = bound_fn(0.1, 0.01, **props)
        n2 = bound_fn(0.01, 0.01, **props)
        assert n2 > n1

    @pytest.mark.parametrize(
        "bound_fn,props",
        [
            (_hoeffding_n, {"bounds": (0.0, 1.0)}),
            (_bernstein_n, {"bounds": (0.0, 1.0), "variance": 0.1}),
            (_sub_gaussian_n, {"sub_gaussian_param": 0.5}),
        ],
    )
    def test_smaller_delta_more_samples(self, bound_fn, props):
        n1 = bound_fn(0.1, 0.1, **props)
        n2 = bound_fn(0.1, 0.001, **props)
        assert n2 > n1


class TestLogInvDelta:
    """Tests for the _log_inv_delta helper."""

    def test_two_sided_uses_factor_2(self):
        assert _log_inv_delta(0.01, "two-sided") == math.log(2 / 0.01)

    def test_one_sided_uses_factor_1(self):
        assert _log_inv_delta(0.01, "greater") == math.log(1 / 0.01)
        assert _log_inv_delta(0.01, "less") == math.log(1 / 0.01)

    def test_one_sided_less_than_two_sided(self):
        assert _log_inv_delta(0.01, "greater") < _log_inv_delta(0.01, "two-sided")


class TestOneSidedOptimization:
    """One-sided tests should require fewer samples than two-sided."""

    @pytest.mark.parametrize(
        "bound_fn,props",
        [
            (_hoeffding_n, {"bounds": (0.0, 1.0)}),
            (_bernstein_n, {"bounds": (0.0, 1.0), "variance": 0.1}),
            (_sub_gaussian_n, {"sub_gaussian_param": 0.5}),
            (_median_of_means_n, {"variance": 1.0}),
        ],
    )
    def test_one_sided_fewer_samples(self, bound_fn, props):
        n_two = bound_fn(0.1, 0.01, side="two-sided", **props)
        n_greater = bound_fn(0.1, 0.01, side="greater", **props)
        n_less = bound_fn(0.1, 0.01, side="less", **props)
        assert n_greater < n_two
        assert n_less < n_two
        assert n_greater == n_less

    def test_hoeffding_one_sided_uses_log_1_over_delta(self):
        """Hoeffding one-sided: n = ceil((b-a)^2 * ln(1/delta) / (2 * eps^2))."""
        n = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0), side="greater")
        expected = math.ceil(1.0 * math.log(1 / 0.01) / (2 * 0.1**2))
        assert n == expected

    def test_maurer_pontil_matches_hoeffding_one_sided(self):
        n_mp = _maurer_pontil_n(0.1, 0.01, bounds=(0.0, 1.0), side="greater")
        n_hoef = _hoeffding_n(0.1, 0.01, bounds=(0.0, 1.0), side="greater")
        assert n_mp == n_hoef
