"""pytest plugin entry point for pytest-stochastic.

Registers the plugin hooks so that functions decorated with
``@stochastic_test`` are collected and executed by pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from .decorator import STOCHASTIC_TEST_MARKER


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
    call: pytest.CallInfo[None],  # type: ignore[type-arg]
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
    detail = f" [{result.bound_name}, n={result.n}, observed={result.estimate:.6g}]"

    # Attach the detail to the report sections so it appears in verbose output.
    # Using a "stochastic" section makes it available to terminal writers.
    report.sections.append(("stochastic", detail))

    # Also extend the longrepr for failures so details are visible in tracebacks.
    if report.failed and report.longrepr:
        report.sections.append(("Stochastic Test Details", result.message))

    # In verbose mode, modify the status word to include stochastic details.
    if item.config.option.verbose >= 1:
        if report.passed:
            report._stochastic_detail = detail  # type: ignore[attr-defined]
        elif report.failed:
            report._stochastic_detail = detail  # type: ignore[attr-defined]


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
