"""pytest plugin entry point for pytest-stochastic.

Registers the plugin hooks so that functions decorated with
``@stochastic_test`` are collected and executed by pytest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from .decorator import DISTRIBUTIONAL_TEST_MARKER, STOCHASTIC_TEST_MARKER

# Accumulated tune results for the session (written at the end).
_tune_results: list[object] = []


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``stochastic`` marker and pin tune-file resolution to rootdir."""
    config.addinivalue_line(
        "markers",
        "stochastic: test managed by pytest-stochastic (@stochastic_test or @distributional_test)",
    )

    # Resolve .stochastic.toml relative to the pytest rootdir rather than the
    # process cwd, and drop any parameters cached under a previous root.
    from .decorator import _reset_tuned_params_cache
    from .tune import set_project_root

    set_project_root(Path(config.rootpath))
    _reset_tuned_params_cache()


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

    # The decorator attaches the undecorated test function so tune mode can
    # profile it directly.
    original_fn = getattr(obj, "_stochastic_original", None)
    if original_fn is None:
        return

    n_tune = item.config.getoption("stochastic_tune_samples", default=50_000)
    # Key by module + qualname so that load-time lookup (which only knows the
    # function object, not the nodeid) can match exactly.
    test_key = f"{original_fn.__module__}.{original_fn.__qualname__}"
    config = getattr(obj, STOCHASTIC_TEST_MARKER)
    result = tune_test(original_fn, test_key, n_tune=n_tune, bounds=config.bounds)
    _tune_results.append(result)

    # Report the result
    reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"  TUNED {item.nodeid}: variance_ucb={result.variance:.6g}, "
            f"range=[{result.observed_range[0]:.6g}, {result.observed_range[1]:.6g}], "
            f"n={result.n_tune_samples}"
        )

    # Raising Skipped prevents the wrapped test body from also running.
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
    """Tag stochastic and distributional tests with the ``stochastic`` marker."""
    for item in items:
        if isinstance(item, pytest.Function):
            obj = item.obj
            if hasattr(obj, STOCHASTIC_TEST_MARKER) or hasattr(obj, DISTRIBUTIONAL_TEST_MARKER):
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
    dist_result = getattr(obj, "_distributional_result", None)
    if result is None and dist_result is None:
        return

    if result is not None:
        # Build a concise summary: [bound_name, n=..., observed=...]
        mp_n = getattr(result, "maurer_pontil_effective_n", None)
        mp_suffix = f", maurer_pontil_effective_n={mp_n}" if mp_n is not None else ""
        detail = f" [{result.bound_name}, n={result.n}, observed={result.estimate:.6g}{mp_suffix}]"
        failure_message = result.message
    elif dist_result is not None:
        detail = f" {dist_result['detail']}"
        failure_message = f"Distributional test FAILED {dist_result['detail']}"
    else:  # pragma: no cover - unreachable, guarded above
        return

    # Attach the detail to the report sections so it appears in verbose output.
    # Using a "stochastic" section makes it available to terminal writers.
    report.sections.append(("stochastic", detail))

    # Also extend the longrepr for failures so details are visible in tracebacks.
    if report.failed and report.longrepr:
        report.sections.append(("Stochastic Test Details", failure_message))

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

    The seed is a stable digest of the test node id, so the same test always
    gets the same stream — across processes and runs (unlike the builtin
    ``hash``, which is salted per process).
    """
    digest = hashlib.sha256(request.node.nodeid.encode()).digest()
    seed = int.from_bytes(digest[:4], "little")
    return np.random.default_rng(seed)
