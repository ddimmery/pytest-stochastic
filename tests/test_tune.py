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
        ucb = compute_variance_ucb(samples, confidence=1e-4)
        assert math.isfinite(ucb)

    def test_upper_bound_on_true_variance(self):
        """UCB should be >= true variance with high probability."""
        rng = np.random.default_rng(42)
        true_var = 1.0
        # Run many times and check UCB >= true_var most of the time
        above = 0
        trials = 100
        for i in range(trials):
            samples = rng.normal(0, math.sqrt(true_var), 1000)
            ucb = compute_variance_ucb(samples, confidence=0.01)
            if ucb >= true_var:
                above += 1
        # With confidence=0.01, should be >= true_var at least 99% of the time
        assert above >= 95  # allow some slack

    def test_single_sample_returns_inf(self):
        samples = np.array([1.0])
        ucb = compute_variance_ucb(samples, confidence=1e-4)
        assert math.isinf(ucb)

    def test_constant_samples_near_zero(self):
        samples = np.full(100, 5.0)
        ucb = compute_variance_ucb(samples, confidence=1e-4)
        assert ucb == pytest.approx(0.0, abs=1e-10)


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
