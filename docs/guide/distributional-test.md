# @distributional_test

The `@distributional_test` decorator tests that a function's outputs follow a specified reference distribution, using a standard goodness-of-fit test.

## Basic Usage

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

The decorator:

1. Validates parameters at import time
2. At test time, draws `n_samples` from your function
3. Runs the chosen goodness-of-fit test against the reference distribution
4. Asserts the p-value exceeds the significance level

## Parameters

### `reference` (scipy distribution)

A frozen scipy continuous distribution providing a `.cdf` method (and `.rvs` + `.ppf` for some tests). This is the distribution you claim your function follows.

```python
from scipy import stats

# Standard normal
reference=stats.norm(0, 1)

# Exponential with rate 2
reference=stats.expon(scale=0.5)

# Beta(2, 5)
reference=stats.beta(2, 5)
```

### `test` (str, default: "ks")

The statistical test to use:

| Test | Description | Best for |
|------|-------------|----------|
| `"ks"` | Kolmogorov-Smirnov | General-purpose; sensitive to location and shape |
| `"chi2"` | Chi-squared goodness-of-fit | Discrete-like distributions or when CDF is expensive |
| `"anderson"` | Anderson-Darling (k-sample) | Better tail sensitivity than KS |

```python
# Kolmogorov-Smirnov (default)
@distributional_test(reference=stats.norm(0, 1), test="ks")

# Chi-squared
@distributional_test(reference=stats.norm(0, 1), test="chi2")

# Anderson-Darling
@distributional_test(reference=stats.norm(0, 1), test="anderson")
```

#### Test Details

**KS test** uses `scipy.stats.kstest(samples, reference.cdf)`. It compares the empirical CDF to the reference CDF and is the most commonly used option.

**Chi-squared test** bins samples using quantiles of the reference distribution ($\sqrt{n}$ bins, minimum 10), then compares observed vs. expected counts. Useful when the CDF is expensive to evaluate pointwise.

**Anderson test** uses `scipy.stats.anderson_ksamp` to compare your samples against an equal-sized sample from the reference distribution. It has better sensitivity in the distribution tails than KS.

### `significance` (float, default: 1e-6)

The significance level $\alpha$. The test asserts $\text{p-value} > \alpha$. Lower values make the test less likely to falsely fail (analogous to `failure_prob` in `@stochastic_test`).

```python
# Very strict
@distributional_test(reference=stats.norm(0, 1), significance=1e-8)

# Relaxed
@distributional_test(reference=stats.norm(0, 1), significance=0.01)
```

### `n_samples` (int, default: 10_000)

Number of samples to draw from the test function. More samples increase the test's power to detect distributional differences, but also increase runtime.

```python
@distributional_test(
    reference=stats.norm(0, 1),
    n_samples=50_000,  # More power to detect subtle differences
)
```

### `seed` (int | None, default: None)

Fix the RNG seed for reproducibility. When `None`, a random seed is generated and reported on failure.

## Test Function Signature

Like `@stochastic_test`, your function must return a numeric scalar and can optionally accept `rng`:

```python
@distributional_test(reference=stats.norm(0, 1))
def test_normal(rng):
    return rng.standard_normal()
```

## Output

In verbose mode, the test reports the test statistic and p-value:

```
test_dist.py::test_normal PASSED [ks, n=10000, stat=0.00812, p=0.523, sig=1e-06]
```

On failure:

```
Distributional test FAILED [ks, n=10000, stat=0.142, p=2.3e-08, sig=1e-06] (seed=12345)
```

## Choosing a Test

- **Start with `"ks"`** &mdash; It is the default and works well for most continuous distributions.
- **Use `"anderson"`** when tail behavior matters &mdash; Anderson-Darling gives more weight to the tails.
- **Use `"chi2"`** for discrete-like distributions or when you need explicit control over binning behavior.

## Differences from @stochastic_test

| | @stochastic_test | @distributional_test |
|---|---|---|
| Tests | Mean of a scalar statistic | Full distributional fit |
| Sample size | Computed from concentration bounds | User-specified `n_samples` |
| Bound selection | Automatic from declared properties | N/A |
| Output | Pass/fail on mean tolerance | Pass/fail on goodness-of-fit |
