"""Unit tests for parameter validation and tolerance computation."""

from __future__ import annotations

import pytest

from pytest_stochastic.types import (
    InvalidPropertyError,
    InvalidToleranceError,
)
from pytest_stochastic.validation import compute_tolerance, validate_and_build_config


class TestComputeTolerance:
    def test_pure_absolute(self):
        assert compute_tolerance(0.5, 0.05, 0.0) == 0.05

    def test_pure_relative(self):
        assert compute_tolerance(100.0, 0.0, 0.02) == pytest.approx(2.0)

    def test_combined(self):
        # tol = 0.01 + 0.01 * |5.0| = 0.06
        assert compute_tolerance(5.0, 0.01, 0.01) == pytest.approx(0.06)

    def test_negative_expected(self):
        # tol = 0.0 + 0.1 * |-10.0| = 1.0
        assert compute_tolerance(-10.0, 0.0, 0.1) == pytest.approx(1.0)

    def test_both_zero_raises(self):
        with pytest.raises(InvalidToleranceError, match=r"(?i)at least one"):
            compute_tolerance(0.5, 0.0, 0.0)

    def test_negative_atol_raises(self):
        with pytest.raises(InvalidToleranceError, match="non-negative"):
            compute_tolerance(0.5, -0.1, 0.0)

    def test_negative_rtol_raises(self):
        with pytest.raises(InvalidToleranceError, match="non-negative"):
            compute_tolerance(0.5, 0.0, -0.1)

    def test_zero_expected_with_rtol_only_raises(self):
        with pytest.raises(InvalidToleranceError, match="meaningless"):
            compute_tolerance(0.0, 0.0, 0.1)

    def test_zero_expected_with_atol_ok(self):
        assert compute_tolerance(0.0, 0.05, 0.1) == 0.05


class TestValidateAndBuildConfig:
    def test_basic_valid(self):
        config = validate_and_build_config(expected=0.5, atol=0.05, bounds=(0.0, 1.0))
        assert config.expected == 0.5
        assert config.tol == 0.05
        assert config.failure_prob == 1e-8
        assert config.side == "two-sided"
        assert config.bounds == (0.0, 1.0)

    def test_declared_properties(self):
        config = validate_and_build_config(
            expected=0.5, atol=0.05, bounds=(0.0, 1.0), variance=0.1
        )
        props = config.declared_properties
        assert "bounds" in props
        assert "variance" in props
        assert "sub_gaussian_param" not in props

    def test_invalid_failure_prob(self):
        with pytest.raises(InvalidPropertyError, match="failure_prob"):
            validate_and_build_config(expected=0.5, atol=0.05, bounds=(0.0, 1.0), failure_prob=0.0)

    def test_invalid_side(self):
        with pytest.raises(InvalidPropertyError, match="side"):
            validate_and_build_config(expected=0.5, atol=0.05, bounds=(0.0, 1.0), side="invalid")

    def test_invalid_bounds_order(self):
        with pytest.raises(InvalidPropertyError, match="a < b"):
            validate_and_build_config(expected=0.5, atol=0.05, bounds=(1.0, 0.0))

    def test_negative_variance(self):
        with pytest.raises(InvalidPropertyError, match="positive"):
            validate_and_build_config(expected=0.5, atol=0.05, variance=-1.0)

    def test_no_properties_raises(self):
        with pytest.raises(InvalidPropertyError, match="At least one"):
            validate_and_build_config(expected=0.5, atol=0.05)

    def test_invalid_moment_bound_p(self):
        with pytest.raises(InvalidPropertyError, match="p must be > 1"):
            validate_and_build_config(expected=0.5, atol=0.05, moment_bound=(0.5, 1.0))

    def test_invalid_moment_bound_m(self):
        with pytest.raises(InvalidPropertyError, match="M must be positive"):
            validate_and_build_config(expected=0.5, atol=0.05, moment_bound=(1.5, -1.0))

    def test_symmetric_without_bounds_raises(self):
        with pytest.raises(InvalidPropertyError, match="symmetric"):
            validate_and_build_config(expected=0.5, atol=0.05, variance=1.0, symmetric=True)

    def test_valid_one_sided(self):
        config = validate_and_build_config(
            expected=0.5, atol=0.05, bounds=(0.0, 1.0), side="greater"
        )
        assert config.side == "greater"
