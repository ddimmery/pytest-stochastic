"""pytest plugin entry point for pytest-stochastic.

Registers the plugin hooks so that functions decorated with
``@stochastic_test`` are collected and executed by pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .decorator import STOCHASTIC_TEST_MARKER

# Accumulated tune results for the session (written at the end).
_tune_results: list[object] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--stochastic-tune`` CLI options."""
    group = parser.getgroup("stochastic", "Stochastic testing options")
    group.addoption(
        "--stochastic-tune",
        action="store_true",
        default=False,
        help="Run in tuning mode: profile stochastic tests and write "
        "discovered parameters to .stochastic.toml",
    )
    group.addoption(
        "--stochastic-tune-samples",
        type=int,
        default=50_000,
        help="Number of samples to collect per test during tuning (default: 50000)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """In tune mode, replace normal execution with tuning."""
    if not item.config.getoption("stochastic_tune", default=False):
        return

    if not isinstance(item, pytest.Function):
        return

    obj = item.obj
    if not hasattr(obj, STOCHASTIC_TEST_MARKER):
        return

    from .tune import tune_test

    # The wrapper has the original function as __wrapped__ is deleted,
    # but we stored the original in the closure. We need to get the
    # underlying test function. The decorated wrapper calls the original
    # internally, so we can run tune on the original.
    # Get the original function from the wrapper's closure.
    original_fn = None
    if hasattr(obj, "__wrapped__"):
        original_fn = obj.__wrapped__
    else:
        # The wrapper deleted __wrapped__, but the closure contains
        # the original function as a free variable.
        for cell in obj.__closure__ or []:
            try:
                val = cell.cell_contents
                if callable(val) and val is not obj:
                    original_fn = val
                    break
            except ValueError:
                continue

    if original_fn is None:
        return

    n_tune = item.config.getoption("stochastic_tune_samples", default=50_000)
    test_key = item.nodeid.replace("/", ".").replace("::", ".")
    result = tune_test(original_fn, test_key, n_tune=n_tune)
    _tune_results.append(result)

    # Report the result
    reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"  TUNED {item.nodeid}: variance_ucb={result.variance:.6g}, "
            f"range=[{result.observed_range[0]:.6g}, {result.observed_range[1]:.6g}], "
            f"n={result.n_tune_samples}"
        )

    # Skip the normal test execution by marking this as passed via pytest.skip
    # Actually, we should just let it pass. The wrapper will be called but
    # we mark the function to skip its normal logic.
    # Instead, raise a special exception to prevent double-execution.
    pytest.skip("Tuning completed (skipping normal test execution)")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write accumulated tune results to .stochastic.toml at session end."""
    if not _tune_results:
        return

    from .tune import TuneResult, save_tuned_params

    results = [r for r in _tune_results if isinstance(r, TuneResult)]
    if not results:
        return

    root = Path(session.config.rootpath)
    path = save_tuned_params(results, root=root)

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(f"\nTuned parameters written to {path}")

    # Clear for next session (relevant in test scenarios)
    _tune_results.clear()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag stochastic tests with a ``stochastic`` marker for easy filtering."""
    for item in items:
        if isinstance(item, pytest.Function):
            obj = item.obj
            if hasattr(obj, STOCHASTIC_TEST_MARKER):
                item.add_marker(pytest.mark.stochastic)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> object:
    """Append stochastic test details to verbose output."""
    outcome = yield
    report = outcome.get_result()

    if call.when != "call" or report is None:
        return

    if not isinstance(item, pytest.Function):
        return

    obj = item.obj
    result = getattr(obj, "_stochastic_result", None)
    if result is None:
        return

    # Build a concise summary: [bound_name, n=..., observed=...]
    mp_n = getattr(result, "maurer_pontil_effective_n", None)
    mp_suffix = f", maurer_pontil_effective_n={mp_n}" if mp_n is not None else ""
    detail = f" [{result.bound_name}, n={result.n}, observed={result.estimate:.6g}{mp_suffix}]"

    # Attach the detail to the report sections so it appears in verbose output.
    # Using a "stochastic" section makes it available to terminal writers.
    report.sections.append(("stochastic", detail))

    # Also extend the longrepr for failures so details are visible in tracebacks.
    if report.failed and report.longrepr:
        report.sections.append(("Stochastic Test Details", result.message))

    # In verbose mode, modify the status word to include stochastic details.
    if item.config.option.verbose >= 1 and (report.passed or report.failed):
        report._stochastic_detail = detail


def pytest_report_teststatus(
    report: pytest.TestReport,
    config: pytest.Config,
) -> tuple[str, str, str] | None:
    """Customize the verbose status line for stochastic tests."""
    if report.when != "call":
        return None

    detail = getattr(report, "_stochastic_detail", None)
    if detail is None:
        return None

    if report.passed:
        return "passed", ".", f"PASSED{detail}"
    if report.failed:
        return "failed", "F", f"FAILED{detail}"
    return None


@pytest.fixture
def stochastic_rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Provide a seeded numpy RNG as a pytest fixture.

    The seed is derived from the test node id for reproducibility.
    """
    seed = hash(request.node.nodeid) % (2**32)
    return np.random.default_rng(seed)
