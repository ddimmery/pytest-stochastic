"""Tests for the tune mode."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pytest_stochastic.tune import (
    TuneResult,
    compute_variance_ucb,
    load_tuned_params,
    run_tune,
    save_tuned_params,
    tune_test,
)


class TestComputeVarianceUCB:
    def test_returns_finite(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10_000)
        ucb, method = compute_variance_ucb(samples, confidence=1e-4)
        assert math.isfinite(ucb)
        assert method == "chi2_gaussian_approx"

    def test_upper_bound_on_true_variance(self):
        """UCB should be >= true variance with high probability."""
        rng = np.random.default_rng(42)
        true_var = 1.0
        # Run many times and check UCB >= true_var most of the time
        above = 0
        trials = 100
        for _i in range(trials):
            samples = rng.normal(0, math.sqrt(true_var), 1000)
            ucb, _ = compute_variance_ucb(samples, confidence=0.01)
            if ucb >= true_var:
                above += 1
        # With confidence=0.01, should be >= true_var at least 99% of the time
        assert above >= 95  # allow some slack

    def test_single_sample_returns_inf(self):
        samples = np.array([1.0])
        ucb, _ = compute_variance_ucb(samples, confidence=1e-4)
        assert math.isinf(ucb)

    def test_constant_samples_near_zero(self):
        samples = np.full(100, 5.0)
        ucb, _ = compute_variance_ucb(samples, confidence=1e-4)
        assert ucb == pytest.approx(0.0, abs=1e-10)

    def test_with_bounds_uses_maurer_pontil(self):
        """With declared bounds the UCB is the distribution-free
        Maurer-Pontil self-bounding interval."""
        rng = np.random.default_rng(42)
        samples = rng.random(10_000)  # uniform on [0, 1], var = 1/12
        confidence = 1e-8
        ucb, method = compute_variance_ucb(samples, confidence=confidence, bounds=(0.0, 1.0))
        assert method == "maurer_pontil"
        n = len(samples)
        sample_var = float(np.var(samples, ddof=1))
        slack = 1.0 * math.sqrt(2 * math.log(1 / confidence) / (n - 1))
        assert ucb == pytest.approx((math.sqrt(sample_var) + slack) ** 2)
        # It is a genuine upper bound on the true variance here
        assert ucb > 1 / 12

    def test_bounded_ucb_covers_true_variance(self):
        """Distribution-free coverage on non-Gaussian (Bernoulli) data."""
        rng = np.random.default_rng(0)
        true_var = 0.25  # Bernoulli(1/2)
        above = 0
        trials = 100
        for _ in range(trials):
            samples = (rng.random(2000) < 0.5).astype(float)
            ucb, _ = compute_variance_ucb(samples, confidence=0.01, bounds=(0.0, 1.0))
            if ucb >= true_var:
                above += 1
        assert above == trials


class TestRunTune:
    def test_collects_correct_count(self):
        def f(rng):
            return rng.random()

        samples, seed = run_tune(f, 100, seed=42)
        assert len(samples) == 100
        assert isinstance(seed, int)

    def test_no_rng(self):
        counter = {"n": 0}

        def f():
            counter["n"] += 1
            return 1.0

        samples, _ = run_tune(f, 50)
        assert len(samples) == 50
        assert counter["n"] == 50


class TestTuneTest:
    def test_returns_tune_result(self):
        def f(rng):
            return rng.normal(0, 1)

        result = tune_test(f, "test_module.test_fn", n_tune=1000, seed=42)
        assert isinstance(result, TuneResult)
        assert result.test_key == "test_module.test_fn"
        assert result.n_tune_samples == 1000
        assert result.variance > 0
        assert len(result.observed_range) == 2
        assert result.observed_range[0] < result.observed_range[1]


class TestTomlPersistence:
    def test_save_and_load(self, tmp_path: Path):
        results = [
            TuneResult(
                test_key="tests.test_foo.test_bar",
                variance=0.0832,
                observed_range=(0.003, 0.991),
                n_tune_samples=50000,
                tuned_at="2026-02-22T14:30:00+00:00",
            )
        ]
        path = save_tuned_params(results, root=tmp_path)
        assert path.exists()

        loaded = load_tuned_params(root=tmp_path)
        assert "tests.test_foo.test_bar" in loaded
        params = loaded["tests.test_foo.test_bar"]
        assert params["variance"] == pytest.approx(0.0832)
        assert params["n_tune_samples"] == 50000

    def test_merge_preserves_existing(self, tmp_path: Path):
        results1 = [
            TuneResult(
                test_key="test_a",
                variance=1.0,
                observed_range=(0.0, 1.0),
                n_tune_samples=1000,
                tuned_at="2026-01-01T00:00:00+00:00",
            )
        ]
        save_tuned_params(results1, root=tmp_path)

        results2 = [
            TuneResult(
                test_key="test_b",
                variance=2.0,
                observed_range=(0.0, 2.0),
                n_tune_samples=2000,
                tuned_at="2026-02-01T00:00:00+00:00",
            )
        ]
        save_tuned_params(results2, root=tmp_path)

        loaded = load_tuned_params(root=tmp_path)
        assert "test_a" in loaded
        assert "test_b" in loaded

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        loaded = load_tuned_params(root=tmp_path)
        assert loaded == {}

    def test_confidence_and_method_persisted(self, tmp_path: Path):
        results = [
            TuneResult(
                test_key="tests.test_foo.test_bar",
                variance=0.1,
                observed_range=(0.0, 1.0),
                n_tune_samples=1000,
                tuned_at="2026-02-22T14:30:00+00:00",
                confidence=1e-8,
                method="maurer_pontil",
            )
        ]
        save_tuned_params(results, root=tmp_path)
        loaded = load_tuned_params(root=tmp_path)
        params = loaded["tests.test_foo.test_bar"]
        assert params["confidence"] == pytest.approx(1e-8)
        assert params["method"] == "maurer_pontil"

    def test_set_project_root(self, tmp_path: Path):
        from pytest_stochastic.tune import set_project_root

        results = [
            TuneResult(
                test_key="test_root",
                variance=1.0,
                observed_range=(0.0, 1.0),
                n_tune_samples=100,
                tuned_at="2026-01-01T00:00:00+00:00",
            )
        ]
        set_project_root(tmp_path)
        try:
            save_tuned_params(results)
            assert (tmp_path / ".stochastic.toml").exists()
            assert "test_root" in load_tuned_params()
        finally:
            set_project_root(None)

    def test_update_existing_key(self, tmp_path: Path):
        results1 = [
            TuneResult(
                test_key="test_a",
                variance=1.0,
                observed_range=(0.0, 1.0),
                n_tune_samples=1000,
                tuned_at="2026-01-01T00:00:00+00:00",
            )
        ]
        save_tuned_params(results1, root=tmp_path)

        results2 = [
            TuneResult(
                test_key="test_a",
                variance=0.5,
                observed_range=(0.1, 0.9),
                n_tune_samples=5000,
                tuned_at="2026-02-01T00:00:00+00:00",
            )
        ]
        save_tuned_params(results2, root=tmp_path)

        loaded = load_tuned_params(root=tmp_path)
        assert loaded["test_a"]["variance"] == pytest.approx(0.5)
        assert loaded["test_a"]["n_tune_samples"] == 5000
