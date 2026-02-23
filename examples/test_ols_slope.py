"""Example: testing that OLS recovers the true slope of a linear model.

Model
-----
    Y = 1 + 2*X + eps,    eps ~ TruncatedNormal(0, sigma^2, [-T, T])

with a fixed design X = linspace(0, 1, 100) and truncation at T = 3*sigma.
Each test call draws fresh noise, fits ordinary least squares, and returns
the estimated slope b1_hat.  Because the design is fixed:

    b1_hat = b1 + (1/S_XX) * sum(x_i^c * eps_i)

where x_i^c = x_i - x_bar and S_XX = sum(x_i^c ** 2).  Since the errors
are bounded, the estimator is bounded too:

    b1_hat in [b1 - T * sum(|x_i^c|) / S_XX,
               b1 + T * sum(|x_i^c|) / S_XX]

These bounds hold exactly --- no clipping needed.  The variance and
sub-Gaussian parameter are also exact, letting us show how each
additional declaration tightens the required sample size.
"""

import numpy as np
from scipy.stats import truncnorm

from pytest_stochastic import stochastic_test

# -- Model parameters ----------------------------------------------------------
TRUE_SLOPE = 2.0
SIGMA = 0.5  # noise scale parameter
TRUNC_MULT = 3  # truncate at +/- 3 sigma
TRUNC_HALF_WIDTH = TRUNC_MULT * SIGMA  # T = 1.5
N_OBS = 100  # observations per regression

# Fixed design matrix
X = np.linspace(0, 1, N_OBS)
X_CENTERED = X - X.mean()
S_XX = float(X_CENTERED @ X_CENTERED)  # about 8.50

# -- Exact bounds on the OLS slope estimator -----------------------------------
# The worst case for b1_hat is when each eps_i takes the sign of x_i^c (or its
# opposite), so the maximum deviation from b1 is T * sum(|x_i^c|) / S_XX.
_MAX_DEVIATION = TRUNC_HALF_WIDTH * float(np.sum(np.abs(X_CENTERED))) / S_XX
LOWER = TRUE_SLOPE - _MAX_DEVIATION  # about -2.46
UPPER = TRUE_SLOPE + _MAX_DEVIATION  # about  6.46

# -- Exact variance of the OLS slope estimator ---------------------------------
# Var(eps) for the truncated normal is slightly less than sigma^2.
# Var(b1_hat) = Var(eps) / S_XX  (from the fixed-design model).
_EPS_DIST = truncnorm(-TRUNC_MULT, TRUNC_MULT, loc=0, scale=SIGMA)
VARIANCE = float(_EPS_DIST.var() / S_XX)  # about 0.029

# -- Sub-Gaussian parameter ----------------------------------------------------
# A truncated Gaussian inherits the sub-Gaussian property of its parent,
# so eps_i is sub-Gaussian with parameter sigma.  By independence:
#
#   b1_hat - b1 = (1/S_XX) * sum(x_i^c * eps_i)
#
# is sub-Gaussian with parameter sigma / sqrt(S_XX).
SUB_GAUSSIAN_PARAM = SIGMA / np.sqrt(S_XX)  # about 0.172


def _ols_slope(rng):
    """Draw fresh truncated-normal noise and return the OLS slope estimate."""
    eps = _EPS_DIST.rvs(size=N_OBS, random_state=rng)
    y = 1.0 + TRUE_SLOPE * X + eps
    return float(X_CENTERED @ y / S_XX)


# -- Test 1: bounds only -> Hoeffding ------------------------------------------
# Declaring just the bounds, the framework selects Hoeffding's
# inequality, which depends only on the range (b - a) ~ 8.9.
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    failure_prob=1e-8,
)
def test_slope_bounds_only(rng):
    """Hoeffding: the most conservative option (n ~ 75900)."""
    return _ols_slope(rng)


# -- Test 2: bounds + variance -> Bernstein ------------------------------------
# Adding the variance lets the framework use Bernstein's inequality,
# which exploits the fact that Var(b1_hat) ~ 0.029 is much smaller
# than the worst-case (b-a)^2/4 ~ 19.8.
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    variance=VARIANCE,
    failure_prob=1e-8,
)
def test_slope_bounds_and_variance(rng):
    """Bernstein: an 83x reduction from Hoeffding (n ~ 918)."""
    return _ols_slope(rng)


# -- Test 3: sub-Gaussian parameter -> Sub-Gaussian ----------------------------
# Because eps is (truncated) Gaussian, b1_hat is sub-Gaussian with
# parameter sigma / sqrt(S_XX).  This is the tightest characterization
# and avoids the looseness of the bounded-range assumption entirely.
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    sub_gaussian_param=SUB_GAUSSIAN_PARAM,
    failure_prob=1e-8,
)
def test_slope_sub_gaussian(rng):
    """Sub-Gaussian: a further 8x reduction (n ~ 113)."""
    return _ols_slope(rng)
