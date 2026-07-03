"""Tune mode for stochastic tests.

When ``--stochastic-tune`` is passed to pytest, each stochastic test function
is run ``n_tune`` times (default 50,000).  The framework computes a rigorous
upper confidence bound on the variance and persists the result to
``.stochastic.toml`` so that subsequent test runs can use tighter bounds.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from .runtime import _wants_rng, make_rng

#: Default confidence for the variance UCB.  The UCB failing is an extra
#: source of test flakiness on top of each test's ``failure_prob`` (total
#: failure probability <= failure_prob + confidence), so it defaults to the
#: same order as the default failure_prob.
DEFAULT_TUNE_CONFIDENCE = 1e-8


@dataclass
class TuneResult:
    """Result of tuning a single stochastic test."""

    test_key: str
    variance: float
    observed_range: tuple[float, float]
    n_tune_samples: int
    tuned_at: str
    confidence: float = DEFAULT_TUNE_CONFIDENCE
    method: str = "maurer_pontil"


def compute_variance_ucb(
    samples: np.ndarray,
    confidence: float = DEFAULT_TUNE_CONFIDENCE,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, str]:
    """Compute a one-sided upper confidence bound on the true variance.

    Returns ``(var_upper, method)`` with
    ``P(var_true <= var_upper) >= 1 - confidence``.

    When *bounds* are declared the UCB is distribution-free, via the
    self-bounding variance concentration of Maurer & Pontil (2009, Thm 10):
    with probability at least 1 - confidence,

        sqrt(Var) <= sqrt(V_n) + (b - a) * sqrt(2 ln(1/confidence) / (n - 1)).

    Without bounds we fall back to the chi-squared interval
    ``var_upper = (n - 1) V_n / chi2_quantile(confidence, n - 1)``, which is
    exact only for Gaussian data — a documented approximation, tagged
    ``method="chi2_gaussian_approx"`` in the result.  (In practice the
    tuned variance is consumed by ``bernstein_tuned``, which requires
    declared bounds, so the rigorous branch is the one that matters.)
    """
    n = len(samples)
    if n < 2:
        return float("inf"), "insufficient_samples"

    sample_var = float(np.var(samples, ddof=1))

    if bounds is not None:
        a, b = float(bounds[0]), float(bounds[1])
        slack = (b - a) * math.sqrt(2 * math.log(1 / confidence) / (n - 1))
        return (math.sqrt(sample_var) + slack) ** 2, "maurer_pontil"

    # Lower quantile of chi-squared: smaller quantile → larger UCB
    chi2_quantile = sp_stats.chi2.ppf(confidence, df=n - 1)
    if chi2_quantile <= 0:
        return float("inf"), "chi2_gaussian_approx"

    return (n - 1) * sample_var / chi2_quantile, "chi2_gaussian_approx"


def run_tune(
    func: object,
    n_tune: int,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    """Run a test function n_tune times and collect samples.

    Returns (samples_array, seed_used).
    """
    rng, actual_seed = make_rng(seed)
    inject_rng = _wants_rng(func)

    samples = np.empty(n_tune, dtype=np.float64)
    for i in range(n_tune):
        result = func(rng=rng) if inject_rng else func()  # type: ignore[operator]
        samples[i] = float(result)

    return samples, actual_seed


def tune_test(
    func: object,
    test_key: str,
    n_tune: int = 50_000,
    confidence: float = DEFAULT_TUNE_CONFIDENCE,
    seed: int | None = None,
    bounds: tuple[float, float] | None = None,
) -> TuneResult:
    """Run the tuning procedure for a single test function.

    Collects n_tune samples, computes a variance UCB, and returns a
    TuneResult.  When the test declares *bounds*, the UCB is distribution-free
    (Maurer-Pontil); otherwise a chi-squared (Gaussian-approximation) interval
    is used.

    Note on budgets: a test that later relies on the tuned variance fails
    spuriously either when its own concentration bound fails (probability
    <= failure_prob) or when the UCB missed the true variance (probability
    <= confidence), so its total flakiness is at most
    ``failure_prob + confidence``.
    """
    samples, _ = run_tune(func, n_tune, seed=seed)

    variance_ucb, method = compute_variance_ucb(samples, confidence=confidence, bounds=bounds)
    observed_min = float(np.min(samples))
    observed_max = float(np.max(samples))

    return TuneResult(
        test_key=test_key,
        variance=variance_ucb,
        observed_range=(observed_min, observed_max),
        n_tune_samples=n_tune,
        tuned_at=datetime.now(UTC).isoformat(),
        confidence=confidence,
        method=method,
    )


# ---------------------------------------------------------------------------
# .stochastic.toml persistence
# ---------------------------------------------------------------------------

_TOML_FILENAME = ".stochastic.toml"

# Project root pinned by the pytest plugin (pytest_configure) so that load
# and save resolve the same file regardless of the process cwd.
_project_root: Path | None = None


def set_project_root(root: Path | None) -> None:
    """Pin the directory used to resolve .stochastic.toml (None resets to cwd)."""
    global _project_root
    _project_root = root


def _toml_path(root: Path | None = None) -> Path:
    """Return the path to .stochastic.toml."""
    if root is None:
        root = _project_root if _project_root is not None else Path.cwd()
    return root / _TOML_FILENAME


def load_tuned_params(root: Path | None = None) -> dict[str, dict[str, object]]:
    """Load tuned parameters from .stochastic.toml.

    Returns a dict mapping test keys to their tuned parameter dicts.
    Each entry has: variance, observed_range, tuned_at, n_tune_samples.
    """
    path = _toml_path(root)
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    tests = data.get("tests", {})
    result: dict[str, dict[str, object]] = {}
    for key, params in tests.items():
        result[key] = dict(params)
    return result


def save_tuned_params(
    results: list[TuneResult],
    root: Path | None = None,
) -> Path:
    """Write tuned parameters to .stochastic.toml.

    Merges with existing data — existing entries for different tests are
    preserved, and entries for tests in *results* are updated.
    """
    path = _toml_path(root)

    # Load existing data
    existing = load_tuned_params(root)

    # Merge results
    for r in results:
        existing[r.test_key] = {
            "variance": r.variance,
            "observed_range": list(r.observed_range),
            "tuned_at": r.tuned_at,
            "n_tune_samples": r.n_tune_samples,
            "confidence": r.confidence,
            "method": r.method,
        }

    # Write TOML manually (stdlib tomllib is read-only)
    lines = ["# Auto-generated by pytest-stochastic --stochastic-tune", ""]

    for test_key, params in sorted(existing.items()):
        # TOML section: [tests."module::test_name"]
        lines.append(f'[tests."{test_key}"]')
        for k, v in sorted(params.items()):
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    return path


def _toml_value(v: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if math.isnan(v):
            return "nan"
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        items = ", ".join(_toml_value(x) for x in v)
        return f"[{items}]"
    return repr(v)
