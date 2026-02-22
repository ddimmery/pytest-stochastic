"""Unit tests for bound selection engine."""

from __future__ import annotations

import pytest

from pytest_stochastic.selection import select_bound
from pytest_stochastic.types import NoApplicableBoundError


class TestSelectBound:
    def test_selects_hoeffding_with_bounds_only(self):
        bound, n = select_bound({"bounds": (0.0, 1.0)}, 0.1, 0.01)
        # With just bounds, hoeffding or maurer_pontil should be selected
        assert bound.name in {"hoeffding", "maurer_pontil"}
        assert n > 0

    def test_selects_bernstein_with_bounds_and_variance(self):
        bound, _n = select_bound({"bounds": (0.0, 1.0), "variance": 0.01}, 0.1, 0.01)
        # Bernstein should dominate with low variance
        assert bound.name == "bernstein"

    def test_raises_with_no_properties(self):
        with pytest.raises(NoApplicableBoundError):
            select_bound({}, 0.1, 0.01)

    def test_returns_positive_n(self):
        _, n = select_bound({"bounds": (0.0, 1.0)}, 0.1, 0.01)
        assert n >= 1

    def test_more_properties_fewer_samples(self):
        _, n_bounds = select_bound({"bounds": (0.0, 1.0)}, 0.05, 1e-6)
        _, n_bounds_var = select_bound({"bounds": (0.0, 1.0), "variance": 0.05}, 0.05, 1e-6)
        assert n_bounds_var < n_bounds

    def test_one_sided_may_differ_from_two_sided(self):
        _, n_two = select_bound({"bounds": (0.0, 1.0)}, 0.1, 0.01, side="two-sided")
        _, n_one = select_bound({"bounds": (0.0, 1.0)}, 0.1, 0.01, side="greater")
        # One-sided should not be worse; bentkus may help
        assert n_one <= n_two

    def test_sub_gaussian_with_param(self):
        bound, n = select_bound({"sub_gaussian_param": 0.5}, 0.1, 0.01)
        assert bound.name == "sub_gaussian"
        assert n > 0

    def test_variance_only(self):
        bound, n = select_bound({"variance": 1.0}, 0.1, 1e-6)
        # median-of-means is the sole variance-only bound
        assert bound.name == "median_of_means"
        assert n > 0

    def test_anderson_selected_for_symmetric(self):
        bound, _ = select_bound({"bounds": (0.0, 1.0), "symmetric": True}, 0.1, 0.01)
        # Anderson should be selected for symmetric two-sided
        assert bound.name == "anderson"
