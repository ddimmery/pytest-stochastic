"""Integration tests for pytest plugin hooks."""

from __future__ import annotations

import pytest


def test_stochastic_marker_applied(pytester: pytest.Pytester):
    """Verify that @stochastic_test decorated functions get the 'stochastic' marker."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.1, bounds=(0, 1), failure_prob=1e-4, seed=42)
        def test_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("-v", "-m", "stochastic")
    result.assert_outcomes(passed=1)


def test_stochastic_test_runs_via_pytest(pytester: pytest.Pytester):
    """Verify that a basic stochastic test runs end-to-end through pytest."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=42)
        def test_fair_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_stochastic_test_failure_via_pytest(pytester: pytest.Pytester):
    """Verify that a failing stochastic test shows the correct error."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.0, atol=0.01, bounds=(0.99, 1.01), failure_prob=1e-6)
        def test_always_one():
            return 1.0
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*FAILED*"])


def test_verbose_reporting_shows_stochastic_details(pytester: pytest.Pytester):
    """Verify that verbose mode shows bound name, n, and observed value."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=42)
        def test_fair_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
    # Verbose output should include bound name, n, and observed value
    result.stdout.fnmatch_lines(["*PASSED*n=*observed=*"])


def test_verbose_reporting_on_failure(pytester: pytest.Pytester):
    """Verify that verbose mode shows stochastic details on failure."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.0, atol=0.01, bounds=(0.99, 1.01), failure_prob=1e-6)
        def test_always_one():
            return 1.0
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*FAILED*n=*observed=*"])


def test_stochastic_tune_option_registered(pytester: pytest.Pytester):
    """Verify that --stochastic-tune is registered as a CLI option."""
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*--stochastic-tune*"])
    result.stdout.fnmatch_lines(["*--stochastic-tune-samples*"])


def test_configuration_error_at_import(pytester: pytest.Pytester):
    """Verify that misconfigured decorators fail at collection time."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.0, rtol=0.0, bounds=(0, 1))
        def test_bad_tolerance(rng):
            return rng.random()
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(errors=1)
