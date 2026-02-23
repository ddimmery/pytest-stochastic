# pytest-stochastic

[![PyPI version](https://img.shields.io/pypi/v/pytest-stochastic.svg)](https://pypi.org/project/pytest-stochastic/)
[![Python versions](https://img.shields.io/pypi/pyversions/pytest-stochastic.svg)](https://pypi.org/project/pytest-stochastic/)
[![License](https://img.shields.io/pypi/l/pytest-stochastic.svg)](https://github.com/ddimmery/pytest-stochastic/blob/main/LICENSE)

**Principled stochastic unit testing for pytest.**

pytest-stochastic eliminates flaky stochastic tests by replacing arbitrary sample
counts and thresholds with rigorous concentration inequalities. Declare statistical
properties of your test, and the framework automatically selects the tightest bound,
computes the required sample size for your flakiness budget, and runs the test.

## Why pytest-stochastic?

Stochastic tests are notoriously flaky. Common approaches either:

- Use **too few samples**, leading to false failures
- Use **far too many samples** "just to be safe," wasting CI time
- Pick thresholds by **trial and error** with no statistical justification

pytest-stochastic solves this by letting you specify a **failure probability**
(e.g., 10⁻⁸) and the framework computes exactly how many samples are needed using
the best available concentration inequality.

## Features

- **`@stochastic_test`** — Test that a statistic's mean matches an expected value
  within tolerance, with a mathematically guaranteed flakiness bound
- **`@distributional_test`** — Test that outputs follow a reference distribution
  using KS, chi-squared, or Anderson-Darling tests
- **Automatic bound selection** — Declare properties (bounds, variance, sub-Gaussian
  parameter) and the framework picks the tightest inequality from a registry of
  concentration bounds including Hoeffding, Bernstein, Bentkus, Anderson,
  Maurer-Pontil, sub-Gaussian, median-of-means, and Catoni
- **Tune mode** — Run `pytest --stochastic-tune` to empirically profile tests and
  persist discovered variance to `.stochastic.toml` for tighter bounds on subsequent
  runs
- **RNG injection** — Reproducible tests via automatic seed management and optional
  `rng` parameter injection

## Installation

```bash
pip install pytest-stochastic
```

The plugin registers automatically with pytest via the `pytest11` entry point — no
additional configuration is needed.

### Requirements

- Python 3.11+
- pytest 7.0+
- NumPy 1.24+
- SciPy 1.10+

## Quick Start

```python
from pytest_stochastic import stochastic_test

@stochastic_test(
    expected=0.5,
    atol=0.05,
    bounds=(0, 1),
)
def test_uniform_mean(rng):
    return rng.uniform(0, 1)
```

Run it:

```
$ pytest test_example.py -v
test_example.py::test_uniform_mean PASSED [hoeffding, n=185, observed=0.498]
```

The framework determined that Hoeffding's inequality is the tightest applicable
bound, computed the required sample size (n=185) for the default failure probability
of 10⁻⁸, called the test function 185 times with a seeded RNG, and verified the
sample mean falls within tolerance.

## Tighter Bounds with More Information

The more you tell the framework about your distribution, the fewer samples it needs:

```python
# Bounds only → Hoeffding (n ≈ 75,900)
@stochastic_test(expected=2.0, atol=0.1, bounds=(-2.46, 6.46))
def test_slope_hoeffding(rng): ...

# Bounds + variance → Bernstein (n ≈ 918, an 83x reduction)
@stochastic_test(expected=2.0, atol=0.1, bounds=(-2.46, 6.46), variance=0.029)
def test_slope_bernstein(rng): ...

# Sub-Gaussian parameter → Sub-Gaussian (n ≈ 113, a further 8x reduction)
@stochastic_test(expected=2.0, atol=0.1, sub_gaussian_param=0.172)
def test_slope_subgaussian(rng): ...
```

## Distributional Tests

Test that outputs follow a specific distribution:

```python
from pytest_stochastic import distributional_test
from scipy import stats

@distributional_test(
    reference=stats.norm(0, 1),
    test="ks",
    significance=1e-6,
    n_samples=10_000,
)
def test_standard_normal(rng):
    return rng.standard_normal()
```

Supported tests: Kolmogorov-Smirnov (`"ks"`), chi-squared (`"chi2"`), and
Anderson-Darling (`"anderson"`).

## Tune Mode

Automatically discover variance for tighter bounds without manual calculation:

```bash
# Profile tests and save results to .stochastic.toml
pytest --stochastic-tune

# Subsequent runs automatically use tuned variance
pytest
```

Tune mode collects 50,000 samples per test (configurable via
`--stochastic-tune-samples`), computes an upper confidence bound on the variance
using the chi-squared distribution, and persists the results. On subsequent runs,
the decorator loads the tuned variance and uses Bernstein's inequality for tighter
sample size requirements.

## Documentation

Full documentation is available at
[ddimmery.github.io/pytest-stochastic](https://ddimmery.github.io/pytest-stochastic/),
including:

- [Getting Started](https://ddimmery.github.io/pytest-stochastic/getting-started/) — Installation and your first test
- [User Guide](https://ddimmery.github.io/pytest-stochastic/guide/stochastic-test/) — Detailed decorator usage
- [Concentration Bounds](https://ddimmery.github.io/pytest-stochastic/bounds/) — The mathematics behind sample size selection
- [Tune Mode](https://ddimmery.github.io/pytest-stochastic/guide/tune-mode/) — Empirical profiling for tighter bounds
- [API Reference](https://ddimmery.github.io/pytest-stochastic/api/) — Complete API documentation

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
