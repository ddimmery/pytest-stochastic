# Getting Started

## Installation

Install pytest-stochastic with pip:

```bash
pip install pytest-stochastic
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pytest-stochastic
```

The plugin registers automatically with pytest via the `pytest11` entry point. No additional configuration is needed.

### Requirements

- Python 3.11+
- pytest 7.0+
- NumPy 1.24+
- SciPy 1.10+

## Your First Stochastic Test

Create a test file `test_example.py`:

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

```bash
pytest test_example.py -v
```

```
test_example.py::test_uniform_mean PASSED [hoeffding, n=185, observed=0.498]
```

### What Happened

1. The `@stochastic_test` decorator validated your parameters at import time
2. The framework determined that Hoeffding's inequality is the tightest applicable bound for a bounded random variable with no declared variance
3. It computed the required sample size ($n = 185$) for the default failure probability of $10^{-8}$
4. It called `test_uniform_mean` 185 times, injecting a seeded RNG each time
5. It computed the sample mean and checked whether it falls within 0.05 of the expected value 0.5

## Understanding the Parameters

### `expected`

The true expected value of the test statistic. Your test function should return values whose mean equals this.

### `atol` and `rtol`

Tolerance specification. The effective tolerance is:

$$\text{tol} = \text{atol} + \text{rtol} \times |\text{expected}|$$

At least one must be positive. Use `atol` for absolute tolerance and `rtol` for relative tolerance.

### `bounds`

A tuple `(a, b)` guaranteeing each sample lies in $[a, b]$. This enables Hoeffding, Bernstein, Bentkus, and other bounded-distribution inequalities.

### `failure_prob`

The target false-failure probability. Default is $10^{-8}$, meaning the test should falsely fail less than once in 100 million runs. Lower values require more samples.

## Adding More Information for Tighter Bounds

The more you tell the framework about your distribution, the fewer samples it needs:

```python
@stochastic_test(
    expected=0.5,
    atol=0.05,
    bounds=(0, 1),
    variance=1/12,          # Var(Uniform(0,1)) = 1/12
    failure_prob=1e-8,
)
def test_uniform_mean_tight(rng):
    return rng.uniform(0, 1)
```

With variance declared, the framework can use Bernstein's inequality, which typically requires fewer samples than Hoeffding when the variance is small relative to the range.

## Distributional Tests

To test that outputs follow a specific distribution:

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

See the [@distributional_test guide](guide/distributional-test.md) for details.

## Next Steps

- [@stochastic_test guide](guide/stochastic-test.md) &mdash; All decorator parameters and usage patterns
- [Concentration Bounds reference](bounds.md) &mdash; How bounds are selected and computed
- [Tune Mode](guide/tune-mode.md) &mdash; Automatically discover variance for tighter bounds
