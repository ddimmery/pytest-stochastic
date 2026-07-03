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


def test_tune_mode_creates_toml(pytester: pytest.Pytester):
    """Verify that --stochastic-tune creates .stochastic.toml with correct schema."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=42)
        def test_fair_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("--stochastic-tune", "--stochastic-tune-samples=500")
    # The test should be skipped (tuning replaces normal execution)
    result.assert_outcomes(skipped=1)
    # .stochastic.toml should exist
    toml_path = pytester.path / ".stochastic.toml"
    assert toml_path.exists()
    content = toml_path.read_text()
    assert "variance" in content
    assert "n_tune_samples" in content
    assert "tuned_at" in content
    assert "observed_range" in content


def test_tune_mode_output_message(pytester: pytest.Pytester):
    """Verify that tune mode reports the tuned parameters."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-6, seed=42)
        def test_fair_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("--stochastic-tune", "--stochastic-tune-samples=500", "-v")
    result.stdout.fnmatch_lines(["*TUNED*variance_ucb=*"])
    result.stdout.fnmatch_lines(["*Tuned parameters written to*"])


def test_marker_is_registered(pytester: pytest.Pytester):
    """The 'stochastic' marker must be registered so user suites don't get
    PytestUnknownMarkWarning (and --strict-markers doesn't error)."""
    pytester.makepyfile("""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.1, bounds=(0, 1), failure_prob=1e-4, seed=42)
        def test_coin(rng):
            return rng.random()
    """)
    result = pytester.runpytest("--strict-markers", "-W", "error::pytest.PytestUnknownMarkWarning")
    result.assert_outcomes(passed=1)


def test_marker_applied_to_distributional_tests(pytester: pytest.Pytester):
    pytester.makepyfile("""
        from scipy import stats
        from pytest_stochastic import distributional_test

        @distributional_test(reference=stats.norm(0, 1), n_samples=200, seed=42)
        def test_normal(rng):
            return rng.normal(0, 1)
    """)
    result = pytester.runpytest("-m", "stochastic", "--strict-markers")
    result.assert_outcomes(passed=1)


def test_stochastic_rng_fixture_reproducible_across_processes(pytester: pytest.Pytester):
    """The fixture seed is a stable digest of the node id, so two separate
    pytest processes must draw identical values."""
    pytester.makepyfile("""
        def test_record_value(stochastic_rng, tmp_path_factory):
            value = stochastic_rng.random()
            out = tmp_path_factory.getbasetemp().parent / "rng_value.txt"
            with open(out, "a") as f:
                f.write(f"{value!r}\\n")
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)

    values = {
        line
        for path in pytester.path.rglob("rng_value.txt")
        for line in path.read_text().splitlines()
    }
    assert len(values) == 1


def test_tune_then_run_uses_bernstein_tuned(pytester: pytest.Pytester):
    """End-to-end: --stochastic-tune persists a variance whose key matches
    what the decorator looks up, so the next run selects bernstein_tuned."""
    pytester.makepyfile(
        test_roundtrip="""
        from pytest_stochastic import stochastic_test

        @stochastic_test(expected=0.5, atol=0.2, bounds=(0, 1), failure_prob=1e-4, seed=42)
        def test_uniform(rng):
            return rng.random()
    """
    )
    result = pytester.runpytest_subprocess("--stochastic-tune", "--stochastic-tune-samples=5000")
    result.assert_outcomes(skipped=1)
    toml_text = (pytester.path / ".stochastic.toml").read_text()
    assert "test_roundtrip.test_uniform" in toml_text

    pytester.makepyfile(
        test_verify="""
        import test_roundtrip

        def test_bound_is_tuned():
            assert test_roundtrip.test_uniform._stochastic_bound.name == "bernstein_tuned"
    """
    )
    result = pytester.runpytest_subprocess("-v", "test_verify.py")
    result.assert_outcomes(passed=1)


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
