# Worked Example: Testing a Linear Regression Estimator

This end-to-end example shows every major feature of `pytest-stochastic` on a
realistic task: **verifying that OLS recovers the true slope of a linear
model**. Along the way you'll see how declaring progressively more distributional
properties lets the framework select tighter concentration inequalities and
dramatically reduce the number of repetitions.

## Setup

Consider the fixed-design linear model

$$Y_i = 1 + 2\,X_i + \varepsilon_i, \qquad
\varepsilon_i \sim \text{TruncatedNormal}(0,\,\sigma^2,\,[-T,\,T]),$$

where $X_1, \dots, X_{100}$ are an equally-spaced grid on $[0, 1]$,
$\sigma = 0.5$, and $T = 3\sigma = 1.5$.  Each call to a test function draws
fresh noise, fits OLS, and returns the estimated slope $\hat\beta_1$.

Because the design is fixed, we can write the OLS slope in closed form:

$$\hat\beta_1 = \beta_1 + \frac{\sum (x_i - \bar{x})\,\varepsilon_i}{S_{XX}},
\qquad S_{XX} = \sum (x_i - \bar{x})^2 \approx 8.50.$$

Since the errors are bounded in $[-T, T]$, the estimator is **exactly** bounded:

$$\hat\beta_1 \in \Big[\beta_1 - \frac{T \sum |x_i - \bar{x}|}{S_{XX}},\;
\beta_1 + \frac{T \sum |x_i - \bar{x}|}{S_{XX}}\Big]
\approx [-2.46,\; 6.46].$$

No clipping is needed &mdash; the bounds hold exactly.  The variance of
$\hat\beta_1$ equals $\text{Var}(\varepsilon) / S_{XX}$, where
$\text{Var}(\varepsilon)$ is the exact variance of the truncated normal
(slightly less than $\sigma^2$).  And because a truncated Gaussian inherits
the sub-Gaussian property of its parent, the sub-Gaussian parameter is
$\sigma / \sqrt{S_{XX}}$.

```python title="examples/test_ols_slope.py"
import numpy as np
from scipy.stats import truncnorm

from pytest_stochastic import stochastic_test

TRUE_SLOPE = 2.0
SIGMA = 0.5  # noise scale parameter
TRUNC_MULT = 3  # truncate at +/- 3 sigma
TRUNC_HALF_WIDTH = TRUNC_MULT * SIGMA  # T = 1.5
N_OBS = 100  # observations per regression

# Fixed design matrix
X = np.linspace(0, 1, N_OBS)
X_CENTERED = X - X.mean()
S_XX = float(X_CENTERED @ X_CENTERED)  # about 8.50

# Exact bounds: T * sum(|x_i^c|) / S_XX
_MAX_DEVIATION = TRUNC_HALF_WIDTH * float(np.sum(np.abs(X_CENTERED))) / S_XX
LOWER = TRUE_SLOPE - _MAX_DEVIATION  # about -2.46
UPPER = TRUE_SLOPE + _MAX_DEVIATION  # about  6.46

# Exact variance: Var(eps_trunc) / S_XX
_EPS_DIST = truncnorm(-TRUNC_MULT, TRUNC_MULT, loc=0, scale=SIGMA)
VARIANCE = float(_EPS_DIST.var() / S_XX)  # about 0.029

# Sub-Gaussian parameter: sigma / sqrt(S_XX)
SUB_GAUSSIAN_PARAM = SIGMA / np.sqrt(S_XX)  # about 0.172


def _ols_slope(rng):
    """Draw fresh truncated-normal noise and return the OLS slope estimate."""
    eps = _EPS_DIST.rvs(size=N_OBS, random_state=rng)
    y = 1.0 + TRUE_SLOPE * X + eps
    return float(X_CENTERED @ y / S_XX)
```

The module-level constants (`VARIANCE`, `SUB_GAUSSIAN_PARAM`, `LOWER`, `UPPER`)
are derived once from the model and reused by the tests below.  Every declared
property is exact &mdash; no approximations or negligibility arguments needed.

## Step 1 &mdash; Bounds only (Hoeffding)

The most basic declaration: we assert that every return value lies in
$[\approx\!-2.46,\;\approx\!6.46]$ (which holds exactly by construction).
The framework selects **Hoeffding's inequality**, which depends only on the
range $b - a \approx 8.91$.

```python
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    failure_prob=1e-8,
)
def test_slope_bounds_only(rng):
    return _ols_slope(rng)
```

$$n_{\text{Hoeffding}} = \left\lceil \frac{(b - a)^2 \ln(2/\delta)}{2\,\varepsilon^2} \right\rceil = 75{,}886$$

Hoeffding treats all bounded random variables equally.  It doesn't know that
our estimator has small variance &mdash; it budgets for the worst case
($\text{Var} = (b-a)^2/4 \approx 19.8$).

## Step 2 &mdash; Bounds + variance (Bernstein)

Adding the variance $\text{Var}(\hat\beta_1) \approx 0.029$ unlocks
**Bernstein's inequality**, which combines the range and variance:

```python
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    variance=VARIANCE,
    failure_prob=1e-8,
)
def test_slope_bounds_and_variance(rng):
    return _ols_slope(rng)
```

$$n_{\text{Bernstein}} = \left\lceil \frac{2\sigma^2 \ln(2/\delta)}{\varepsilon^2} + \frac{2(b-a)\ln(2/\delta)}{3\varepsilon} \right\rceil = 918$$

Because the true variance ($\approx 0.029$) is much smaller than the worst-case
variance implied by the range ($\approx 19.8$), Bernstein is **83&times;
tighter** than Hoeffding.

## Step 3 &mdash; Sub-Gaussian parameter (Sub-Gaussian)

Because a truncated Gaussian inherits the sub-Gaussian property of its parent,
the OLS slope is sub-Gaussian with parameter $\sigma / \sqrt{S_{XX}}$.  This is
the tightest characterization of the tail behavior and avoids the looseness of
the bounded-range assumption entirely:

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
another **8&times; reduction** over Bernstein.

## Running it

```bash
pytest examples/test_ols_slope.py -v
```

```{ .text .no-copy }
examples/test_ols_slope.py::test_slope_bounds_only PASSED
    [hoeffding, n=75886, observed=2.00043, maurer_pontil_effective_n=4701]
examples/test_ols_slope.py::test_slope_bounds_and_variance PASSED
    [median_of_means, n=918, observed=1.99653]
examples/test_ols_slope.py::test_slope_sub_gaussian PASSED
    [sub_gaussian, n=113, observed=2.02819]
```

All three tests verify the same property &mdash; that $\hat\beta_1$ concentrates
around $2.0$ &mdash; but the sample counts range from **75,886 down to 113**: a
**672&times; reduction** from knowing nothing beyond bounds to knowing the exact
sub-Gaussian parameter.

!!! tip "Maurer-Pontil: a free upgrade"

    Notice the `maurer_pontil_effective_n=4701` on the first test.  The
    framework always runs the full $n$ required by the selected bound, but when
    bounds are declared it checks the **Maurer-Pontil empirical Bernstein
    bound** post-hoc.  Here it found that only 4,701 of the 75,886 samples
    were actually needed &mdash; the remaining samples confirmed that the
    estimator's empirical variance was small enough to tighten the bound at
    no extra cost.

## Summary

| Declared Properties | Bound Selected | Repetitions ($n$) | Reduction |
|---|---|--:|--:|
| `bounds` | Hoeffding | 75,886 | &mdash; |
| `bounds` + `variance` | Bernstein | 918 | 83&times; |
| `sub_gaussian_param` | Sub-Gaussian | 113 | 672&times; |

The pattern is general: **the more you know about your test statistic's
distribution, the fewer repetitions the framework needs** to guarantee the same
flakiness budget ($\delta = 10^{-8}$, i.e. less than one false failure in
100 million runs).

In practice, you might not know the sub-Gaussian parameter exactly.  That's
where [Tune Mode](tune-mode.md) comes in &mdash; run `pytest --stochastic-tune`
to empirically discover the variance and automatically unlock Bernstein's
inequality without manual derivation.
