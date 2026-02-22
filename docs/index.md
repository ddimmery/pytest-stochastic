# pytest-stochastic

**Principled stochastic unit testing for pytest.**

pytest-stochastic eliminates flaky stochastic tests by replacing arbitrary sample counts and thresholds with rigorous concentration inequalities. You declare statistical properties of your test, and the framework automatically selects the tightest bound, computes the required sample size for your flakiness budget, and runs the test.

## Why pytest-stochastic?

Stochastic tests are notoriously flaky. Common approaches either:

- Use too few samples, leading to false failures
- Use far too many samples "just to be safe," wasting CI time
- Pick thresholds by trial and error with no statistical justification

pytest-stochastic solves this by letting you specify a **failure probability** (e.g., $10^{-8}$) and the framework computes exactly how many samples are needed using the best available concentration inequality.

## Key Features

- **`@stochastic_test`** &mdash; Test that a statistic's mean matches an expected value within tolerance, with a mathematically guaranteed flakiness bound
- **`@distributional_test`** &mdash; Test that outputs follow a reference distribution using KS, chi-squared, or Anderson-Darling tests
- **Automatic bound selection** &mdash; Declares properties (bounds, variance, sub-Gaussian parameter) and the framework picks the tightest inequality from a registry of 10 bounds
- **Tune mode** &mdash; Run `--stochastic-tune` to empirically profile tests and persist discovered variance to `.stochastic.toml` for tighter bounds on subsequent runs
- **RNG injection** &mdash; Reproducible tests via automatic seed management and optional `rng` parameter injection

## Quick Example

```python
from pytest_stochastic import stochastic_test
import numpy as np

@stochastic_test(
    expected=0.5,
    atol=0.01,
    bounds=(0, 1),
    failure_prob=1e-8,
)
def test_uniform_mean(rng):
    return rng.uniform(0, 1)
```

The framework determines that Hoeffding's inequality applies, computes $n = 4{,}606$ samples, runs the test, and guarantees a false failure rate below $10^{-8}$.

## Installation

```bash
pip install pytest-stochastic
```

## Next Steps

- [Getting Started](getting-started.md) &mdash; Installation and your first test
- [User Guide](guide/stochastic-test.md) &mdash; Detailed decorator usage
- [Concentration Bounds](bounds.md) &mdash; The mathematics behind sample size selection
- [Tune Mode](guide/tune-mode.md) &mdash; Empirical profiling for tighter bounds
