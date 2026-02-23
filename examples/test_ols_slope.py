"""Example: testing that OLS recovers the true slope of a linear model.

Model
-----
    Y = 1 + 2*X + eps,    eps ~ N(0, sigma^2)

with a fixed design X = linspace(0, 1, 100).  Each test call draws
fresh noise, fits ordinary least squares, and returns the estimated
slope b1_hat.  Because the design is fixed, the estimator is exactly:

    b1_hat ~ N(b1, sigma^2 / S_XX)

where S_XX = sum((x_i - x_bar)^2).  This lets us derive exact
distributional properties and show how each additional declaration
tightens the bound.
"""

import numpy as np

from pytest_stochastic import stochastic_test

# -- Model parameters ----------------------------------------------------------
TRUE_SLOPE = 2.0
SIGMA = 0.5  # noise standard deviation
N_OBS = 100  # observations per regression

# Fixed design matrix
X = np.linspace(0, 1, N_OBS)
X_CENTERED = X - X.mean()
S_XX = float(X_CENTERED @ X_CENTERED)  # about 8.50

# -- Derived properties of the OLS slope estimator ----------------------------
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


# -- Test 1: bounds only -> Hoeffding ------------------------------------------
# Declaring just the bounds, the framework selects Hoeffding's
# inequality, which depends only on the range (b - a) = 2.
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    failure_prob=1e-8,
)
def test_slope_bounds_only(rng):
    """Hoeffding: the most conservative option (n ~ 3800)."""
    return np.clip(_ols_slope(rng), LOWER, UPPER)


# -- Test 2: bounds + variance -> Bernstein ------------------------------------
# Adding the variance lets the framework use Bernstein's inequality,
# which exploits the fact that Var(b1_hat) ~ 0.029 is much smaller
# than the worst-case (b-a)^2/4 = 1.
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    bounds=(LOWER, UPPER),
    variance=VARIANCE,
    failure_prob=1e-8,
)
def test_slope_bounds_and_variance(rng):
    """Bernstein: a 10x reduction from Hoeffding (n ~ 370)."""
    return np.clip(_ols_slope(rng), LOWER, UPPER)


# -- Test 3: sub-Gaussian parameter -> Sub-Gaussian ----------------------------
# Because b1_hat is exactly Gaussian, its sub-Gaussian parameter equals
# its standard deviation.  This is the tightest characterization
# and no clipping is needed (sub-Gaussian is a tail property, not a
# boundedness assumption).
@stochastic_test(
    expected=TRUE_SLOPE,
    atol=0.1,
    sub_gaussian_param=SUB_GAUSSIAN_PARAM,
    failure_prob=1e-8,
)
def test_slope_sub_gaussian(rng):
    """Sub-Gaussian: a further 3x reduction (n ~ 113)."""
    return _ols_slope(rng)
