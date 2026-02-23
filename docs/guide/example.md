# Worked Example: Testing a Linear Regression Estimator

This end-to-end example shows every major feature of `pytest-stochastic` on a
realistic task: **verifying that OLS recovers the true slope of a linear
model**. Along the way you'll see how declaring progressively more distributional
properties lets the framework select tighter concentration inequalities and
dramatically reduce the number of repetitions.

## Setup

Consider the fixed-design linear model

$$Y_i = 1 + 2\,X_i + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2),$$

where $X_1, \dots, X_{100}$ are an equally-spaced grid on $[0, 1]$ and
$\sigma = 0.5$.  Each call to a test function draws fresh noise, fits OLS, and
returns the estimated slope $\hat\beta_1$.

Because the design is fixed, we can write the OLS slope in closed form:

$$\hat\beta_1 = \beta_1 + \frac{\sum (x_i - \bar{x})\,\varepsilon_i}{S_{XX}},
\qquad S_{XX} = \sum (x_i - \bar{x})^2 \approx 8.50.$$

Since the $\varepsilon_i$ are iid Gaussian, $\hat\beta_1$ is **exactly**
$N\!\big(\beta_1,\;\sigma^2 / S_{XX}\big)$.  That gives us three pieces of
information we can declare to the framework, each one enabling a tighter bound.

```python title="examples/test_ols_slope.py"
import numpy as np

from pytest_stochastic import stochastic_test

TRUE_SLOPE = 2.0
SIGMA = 0.5  # noise standard deviation
N_OBS = 100  # observations per regression

# Fixed design matrix
X = np.linspace(0, 1, N_OBS)
X_CENTERED = X - X.mean()
S_XX = float(X_CENTERED @ X_CENTERED)  # about 8.50

# Var(b1_hat) = sigma^2 / S_XX  (exact, from the fixed-design normal model)
VARIANCE = SIGMA**2 / S_XX  # about 0.029

# b1_hat is Gaussian, hence sub-Gaussian with parameter = std dev
SUB_GAUSSIAN_PARAM = np.sqrt(VARIANCE)  # about 0.172

# Bounds: b1_hat is Gaussian, so values beyond +/-6 std devs from the mean
# have probability < 2e-9.  Clipping at (1, 3) is about +/-5.8 std devs and
# introduces negligible bias (< 1e-8), safely below our tolerance of 0.1.
LOWER, UPPER = 1.0, 3.0


def _ols_slope(rng):
    """Draw fresh noise and return the OLS slope estimate."""
    y = 1.0 + TRUE_SLOPE * X + rng.normal(0, SIGMA, N_OBS)
    return float(X_CENTERED @ y / S_XX)
```

The module-level constants (`VARIANCE`, `SUB_GAUSSIAN_PARAM`, `LOWER`, `UPPER`)
are derived once from the model and reused by the tests below.

## Step 1 &mdash; Bounds only (Hoeffding)

The most basic declaration: we assert that every return value lies in $[1, 3]$
(enforced by clipping).  The framework selects **Hoeffding's inequality**, which
depends only on the range $b - a = 2$.

```python
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    failure_prob=1e-8,
)
def test_slope_bounds_only(rng):
    return np.clip(_ols_slope(rng), LOWER, UPPER)
```

$$n_{\text{Hoeffding}} = \left\lceil \frac{(b - a)^2 \ln(2/\delta)}{2\,\varepsilon^2} \right\rceil = 3{,}823$$

Hoeffding treats all $[1, 3]$-bounded random variables equally.  It doesn't
know that our estimator has small variance &mdash; it budgets for the worst case
($\text{Var} = (b-a)^2/4 = 1$).

## Step 2 &mdash; Bounds + variance (Bernstein)

Adding the variance $\text{Var}(\hat\beta_1) = \sigma^2 / S_{XX} \approx 0.029$
unlocks **Bernstein's inequality**, which combines the range and variance:

```python
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    variance=VARIANCE,
    failure_prob=1e-8,
)
def test_slope_bounds_and_variance(rng):
    return np.clip(_ols_slope(rng), LOWER, UPPER)
```

$$n_{\text{Bernstein}} = \left\lceil \frac{2\sigma^2 \ln(2/\delta)}{\varepsilon^2} + \frac{2(b-a)\ln(2/\delta)}{3\varepsilon} \right\rceil = 368$$

Because the true variance ($\approx 0.029$) is much smaller than the worst-case
variance implied by the range ($1.0$), Bernstein is **10&times; tighter** than
Hoeffding.

## Step 3 &mdash; Sub-Gaussian parameter (Sub-Gaussian)

The OLS slope is exactly Gaussian, so it is sub-Gaussian with parameter equal to
its standard deviation.  This is the tightest characterization of the tail
behavior, and unlike the bounded-distribution bounds, **no clipping is needed**:

```python
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    sub_gaussian_param=SUB_GAUSSIAN_PARAM,
    failure_prob=1e-8,
)
def test_slope_sub_gaussian(rng):
    return _ols_slope(rng)
```

$$n_{\text{Sub-Gaussian}} = \left\lceil \frac{2\sigma_{\text{sg}}^2 \ln(2/\delta)}{\varepsilon^2} \right\rceil = 113$$

The sub-Gaussian bound drops the second-order range term entirely, giving
another **3&times; reduction** over Bernstein.

## Running it

```bash
pytest examples/test_ols_slope.py -v
```

```{ .text .no-copy }
examples/test_ols_slope.py::test_slope_bounds_only PASSED
    [hoeffding, n=3823, observed=1.99845, maurer_pontil_effective_n=1271]
examples/test_ols_slope.py::test_slope_bounds_and_variance PASSED
    [bernstein, n=368, observed=1.9836]
examples/test_ols_slope.py::test_slope_sub_gaussian PASSED
    [sub_gaussian, n=113, observed=1.9911]
```

All three tests verify the same property &mdash; that $\hat\beta_1$ concentrates
around $2.0$ &mdash; but the sample counts range from **3,823 down to 113**: a
**34&times; reduction** from knowing nothing beyond bounds to knowing the exact
sub-Gaussian parameter.

!!! tip "Maurer-Pontil: a free upgrade"

    Notice the `maurer_pontil_effective_n=1271` on the first test.  The
    framework always runs the full $n$ required by the selected bound, but when
    bounds are declared it checks the **Maurer-Pontil empirical Bernstein
    bound** post-hoc.  Here it found that only 1,271 of the 3,823 samples
    were actually needed &mdash; the remaining samples confirmed that the
    estimator's empirical variance was small enough to tighten the bound at
    no extra cost.

## Summary

| Declared Properties | Bound Selected | Repetitions ($n$) | Reduction |
|---|---|--:|--:|
| `bounds` | Hoeffding | 3,823 | &mdash; |
| `bounds` + `variance` | Bernstein | 368 | 10&times; |
| `sub_gaussian_param` | Sub-Gaussian | 113 | 34&times; |

The pattern is general: **the more you know about your test statistic's
distribution, the fewer repetitions the framework needs** to guarantee the same
flakiness budget ($\delta = 10^{-8}$, i.e. less than one false failure in
100 million runs).

In practice, you might not know the sub-Gaussian parameter exactly.  That's
where [Tune Mode](tune-mode.md) comes in &mdash; run `pytest --stochastic-tune`
to empirically discover the variance and automatically unlock Bernstein's
inequality without manual derivation.
