# @stochastic_test

The `@stochastic_test` decorator turns a scalar-returning function into a stochastic test with mathematically guaranteed flakiness bounds.

## Basic Usage

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

The decorator:

1. Validates all parameters at import time (misconfigurations fail immediately)
2. Selects the tightest applicable concentration inequality
3. Computes the minimum sample size for the target failure probability
4. At test time, calls your function $n$ times and checks the result

## Parameters

### Required

#### `expected` (float)

The expected value of the test statistic.

#### Tolerance: `atol` and `rtol`

The effective tolerance is computed as:

$$\text{tol} = \text{atol} + \text{rtol} \times |\text{expected}|$$

At least one of `atol` or `rtol` must be positive. When `expected=0`, you must use `atol` since relative tolerance is meaningless.

```python
# Absolute tolerance of 0.01
@stochastic_test(expected=3.14, atol=0.01, bounds=(0, 10))

# 1% relative tolerance
@stochastic_test(expected=3.14, rtol=0.01, bounds=(0, 10))

# Combined: atol + rtol * |expected| = 0.001 + 0.01 * 3.14 = 0.0324
@stochastic_test(expected=3.14, atol=0.001, rtol=0.01, bounds=(0, 10))
```

### Distributional Properties

At least one must be declared. More properties enable tighter bounds.

#### `bounds` (tuple[float, float])

A tuple `(a, b)` such that every sample lies in $[a, b]$. Enables: Hoeffding, Anderson, Maurer-Pontil, Bentkus, Bernstein.

```python
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))
```

#### `variance` (float)

An upper bound on $\text{Var}(X_i)$. Enables: Chebyshev, Median-of-Means. Combined with `bounds`, also enables Bernstein.

```python
@stochastic_test(expected=0.5, atol=0.05, variance=1/12)
```

#### `sub_gaussian_param` (float)

The sub-Gaussian parameter $\sigma$ such that $\mathbb{E}[e^{\lambda(X - \mu)}] \leq e^{\lambda^2 \sigma^2 / 2}$. Enables: Sub-Gaussian bound.

```python
@stochastic_test(expected=0.0, atol=0.1, sub_gaussian_param=1.0)
```

#### `moment_bound` (tuple[float, float])

A tuple `(p, M)` with $p > 1$ such that $\mathbb{E}[|X - \mu|^p] \leq M$. Enables: Catoni M-estimator.

```python
@stochastic_test(expected=0.0, atol=0.1, moment_bound=(2, 1.0))
```

#### `symmetric` (bool)

Set to `True` if the distribution is symmetric about its mean. Requires `bounds`. Enables: Anderson's inequality (factor-of-2 improvement over Hoeffding for two-sided tests).

```python
@stochastic_test(
    expected=0.0, atol=0.1, bounds=(-1, 1), symmetric=True
)
```

### Test Configuration

#### `failure_prob` (float, default: 1e-8)

Target false-failure probability $\delta$. The test is guaranteed to falsely fail with probability at most $\delta$ (assuming the declared properties hold).

```python
# Very strict: fails less than 1 in 10 billion runs
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-10)

# Relaxed: fewer samples needed
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-4)
```

#### `side` (str, default: "two-sided")

Controls the test direction:

- `"two-sided"` &mdash; Tests $|\hat{\mu} - \mu| < \text{tol}$
- `"greater"` &mdash; Tests $\hat{\mu} > \mu - \text{tol}$
- `"less"` &mdash; Tests $\hat{\mu} < \mu + \text{tol}$

One-sided tests can use Bentkus (bounded distributions) which requires 20-40% fewer samples.

```python
@stochastic_test(
    expected=0.5, atol=0.05, bounds=(0, 1), side="greater"
)
```

#### `seed` (int | None, default: None)

Fix the RNG seed for deterministic reproducibility. When `None`, a random seed is generated and reported on failure for debugging.

```python
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), seed=42)
```

## Test Function Signature

Your test function must return a numeric scalar. It can optionally accept an `rng` parameter:

```python
# Without RNG injection
@stochastic_test(expected=6, atol=0.5, bounds=(2, 12))
def test_dice():
    import random
    return random.randint(2, 12)

# With RNG injection (recommended)
@stochastic_test(expected=3.5, atol=0.2, bounds=(1, 6))
def test_die(rng):
    return rng.integers(1, 7)
```

The `rng` parameter receives a `numpy.random.Generator` instance. This is the recommended approach for reproducibility.

## How Bound Selection Works

The framework maintains a registry of concentration inequalities. For each test:

1. It filters bounds whose required properties are a subset of your declared properties
2. It filters bounds that support the requested `side`
3. It evaluates each remaining bound's `compute_n` function
4. It selects the bound requiring the fewest samples

For example, if you declare `bounds=(0, 1)` and `variance=0.08`:

- Hoeffding is applicable (needs `bounds`)
- Bernstein is applicable (needs `bounds` + `variance`)
- Chebyshev is applicable (needs `variance`)
- Bernstein typically wins because it exploits both pieces of information

## Verbose Output

With `pytest -v`, stochastic tests show bound and sample information:

```
test_example.py::test_mean PASSED [bernstein, n=423, observed=0.501234]
```

On failure, the seed is reported for reproduction:

```
test_example.py::test_mean FAILED [bernstein, n=423, seed=12345]:
  |0.567 - 0.5| = 0.067 not < 0.05
```
