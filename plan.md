# Design Document: `pytest-stochastic`

### A pytest plugin for principled stochastic unit testing

**Status:** Proposal  
**Author:** (draft)  
**Date:** February 2026

---

## 1. Problem Statement

Testing stochastic systems (ML models, randomized algorithms, simulations, sampling procedures) is painful. Teams currently choose between bad options:

- **Seed pinning** (`np.random.seed(42)`): Makes tests deterministic but doesn't actually verify distributional properties. A seeded test can pass while the underlying distribution is completely wrong.
- **Arbitrary thresholds** ("run 1000 times, assert success > 90%"): No principled relationship between sample count, tolerance, and flakiness rate. Tests are either too flaky or wastefully slow.
- **Ignoring the problem**: Marking stochastic tests as `xfail` or skipping them entirely.

What's missing is a framework that lets the user declare *what they know about their test statistic* and then automatically selects a concentration inequality, computes the required sample size for a target flakiness budget, runs the test, and reports results — all without requiring the user to know which bound to use.

---

## 2. Design Principles

1. **The user describes their test, not the math.** Users declare properties of their statistic (bounded? known variance? sub-Gaussian?) via a simple decorator API. The framework selects the tightest applicable bound.
2. **Flakiness is a budget, not an accident.** The user specifies a target false-failure probability (e.g., `1e-8`). The framework guarantees the test fails spuriously no more than this often, assuming the declared properties hold.
3. **Tight bounds → fast tests.** The more the user tells us, the fewer samples we need. Declaring a variance bound is rewarded with a smaller `n`. The framework always picks the tightest bound available from the information given.
4. **Zero magic numbers.** Sample counts, tolerances, and thresholds are all derived, not hand-tuned.
5. **Works with pytest.** This is a plugin, not a replacement. Tests look like normal pytest tests with a decorator.

---

## 3. User-Facing API

### 3.1 Core Decorator

```python
from pytest_stochastic import stochastic_test

@stochastic_test(
    # What do you want to assert?
    expected=0.5,              # Expected value of the statistic

    # How close is close enough? (combined as: tol = atol + rtol * |expected|)
    atol=0.05,                 # Absolute tolerance
    rtol=0.0,                  # Relative tolerance (fraction of |expected|)

    # Flakiness budget
    failure_prob=1e-8,         # P(test fails | code is correct) ≤ this

    # What can you guarantee about individual samples?
    bounds=(0, 1),             # Each sample ∈ [a, b]  (enables Hoeffding, Bentkus, Bernstein, ...)
    variance=None,             # Upper bound on Var(X_i) (enables Bernstein, Chebyshev, ...)
    sub_gaussian_param=None,   # σ s.t. E[exp(t(X-μ))] ≤ exp(t²σ²/2)
    symmetric=False,           # True if distribution is symmetric about its mean
    moment_bound=None,         # (p, M) s.t. E[|X-μ|^p] ≤ M for some p > 1 (enables Catoni)
)
def test_coin_is_fair(rng):
    return rng.random()  # single sample; framework handles repetition
```

The decorator calls the test function `n` times (where `n` is derived from the declared properties and target failure_prob), collects the return values, computes the sample mean, and asserts it falls within the effective tolerance of the expected value. The effective tolerance is `tol = atol + rtol * |expected|`, following the convention used by `numpy.allclose` and most numerical optimizers. This lets users express both absolute precision ("within 0.01") and relative precision ("within 1% of expected") or combine both.

### 3.2 Property Declaration Hierarchy

Users provide *what they can guarantee*. Each guarantee unlocks tighter bounds and thus requires fewer samples:

| User provides | Best bound available | Relative sample cost |
|---|---|---|
| `moment_bound=(p, M)` only | Catoni | Highest (but works for heavy-tailed, p > 1 suffices) |
| `variance=σ²` only | Median-of-means | High (sub-Gaussian rate, no boundedness needed) |
| `bounds=(a, b)` only | Maurer-Pontil | Moderate — data-adaptive, tighter than Hoeffding |
| `bounds=(a, b)` + `symmetric=True` | Anderson | ~2× fewer than Hoeffding |
| `bounds=(a, b)` + `variance=σ²` | Bernstein | Often 2–10× fewer than Hoeffding |
| `bounds=(a, b)` + `side="greater"` or `"less"` | Bentkus | ~20–40% fewer than Hoeffding (one-sided) |
| `bounds=(a, b)` + `variance=σ²` + `sub_gaussian_param=σ` | Best of all | Minimum of all applicable |

The framework evaluates **all** applicable bounds, computes the `n` each would require, and picks the one with the smallest `n`. The user never needs to know which bound was chosen — though verbose output always reports it. Note that at least one property must be provided; with no information about the distribution at all, no finite-sample guarantee is possible.

### 3.3 Tolerance Model

The effective tolerance used for the assertion is:

```
tol = atol + rtol * |expected|
```

At least one of `atol` or `rtol` must be positive. Examples:

```python
# Pure absolute: observed must be within 0.05 of expected
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))

# Pure relative: observed must be within 2% of expected
@stochastic_test(expected=100.0, rtol=0.02, bounds=(0, 200))

# Combined: useful when expected could be near zero
# tol = 0.01 + 0.01 * |expected|
@stochastic_test(expected=5.0, atol=0.01, rtol=0.01, bounds=(0, 10))
```

The framework computes `tol` once from the decorator arguments and uses it as `ε` in all concentration bound calculations. If `expected=0` and `rtol > 0` but `atol = 0`, the framework raises a configuration error (relative tolerance alone is meaningless at zero).

### 3.4 Test Function Contract

The test function returns a scalar. The framework computes the sample mean over `n` calls and checks `|mean - expected| < tol`.

```python
@stochastic_test(expected=0.0, atol=0.1, bounds=(-1, 1))
def test_zero_mean(rng):
    return rng.uniform(-1, 1)
```

For testing success probabilities, this is just a mean of booleans — no special mode is needed:

```python
@stochastic_test(expected=0.85, atol=0.03, bounds=(0, 1), failure_prob=1e-8)
def test_classifier_accuracy(rng):
    x = sample_input(rng)
    return float(model.predict(x) == ground_truth(x))
```

### 3.5 Two-Sided and One-Sided Tests

```python
# Two-sided (default): |observed - expected| < tol
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))

# One-sided: observed > expected - tol
@stochastic_test(expected=0.9, atol=0.05, bounds=(0, 1), side="greater")

# One-sided: observed < expected + tol
@stochastic_test(expected=0.1, atol=0.05, bounds=(0, 1), side="less")
```

For one-sided tests, the failure_prob budget is spent on one tail, giving a √2 improvement in sample efficiency.

### 3.6 Distribution Tests

For cases where you want to test that the full output distribution matches a reference, not just the mean:

```python
from pytest_stochastic import distributional_test

@distributional_test(
    reference=scipy.stats.norm(0, 1),   # reference distribution
    test="ks",                           # Kolmogorov-Smirnov
    significance=1e-6,                   # α level
    n_samples=10000,                     # sample count (user-specified for KS)
)
def test_normal_output(rng):
    return my_sampler(rng)
```

This is a thinner wrapper — the user is closer to the stats. Supported tests: `"ks"`, `"chi2"`, `"anderson"`. Since these are standard statistical tests with well-understood power characteristics, we don't auto-select `n` here; the user chooses it.

### 3.7 RNG Injection

The decorator injects a `rng` argument (a `numpy.random.Generator`) that is freshly seeded per test run. This ensures:

- Reproducibility on failure: the seed is reported so failing tests can be replayed deterministically.
- Independence between samples.

```python
@stochastic_test(expected=0.5, tolerance=0.05, bounds=(0, 1))
def test_my_function(rng):
    # rng is a numpy.random.Generator
    return my_function(rng.random())
```

If the user doesn't include `rng` in the function signature, the framework skips injection (the user manages their own randomness). Reproducibility is then the user's responsibility.

---

## 4. Bound Selection Engine

This is the core internal logic. The user never interacts with it directly.

### 4.1 Registry of Bounds

Each bound is a callable: `(tolerance, failure_prob, **properties) → n_required`.

```python
@dataclass
class BoundStrategy:
    name: str
    required_properties: set[str]       # e.g., {"bounds"}
    optional_properties: set[str]       # e.g., {"variance"}
    compute_n: Callable[..., int]       # (tol, delta, **props) → n
    description: str                    # for diagnostics/reporting
```

**Built-in bounds:**

The registry contains bounds ordered here roughly from weakest (fewest assumptions, most samples) to strongest (most assumptions, fewest samples). The selection algorithm tries all applicable bounds and picks the one requiring the fewest samples, so this ordering is for exposition only.

---

**Chebyshev.** Requires: `variance=σ²` (no boundedness needed).
```
n = ⌈ σ² / (δ · ε²) ⌉
```
The loosest bound in the registry. `P(|X̄ - μ| ≥ ε) ≤ σ²/(nε²)` by Chebyshev's inequality applied to the sample mean. Applicable to any distribution with finite variance, including heavy-tailed distributions where moments beyond the second may not exist. The `1/δ` dependence (vs. `ln(1/δ)` for sub-Gaussian bounds) makes this expensive for small `failure_prob`, but it is the only option when the distribution is heavy-tailed and no boundedness is available.

**Median-of-means.** Requires: `variance=σ²` (no boundedness needed).
```
k = ⌈ 8 ln(2/δ) ⌉            (number of blocks)
n = k · ⌈ 2σ² / ε² ⌉          (total samples)
```
Splits `n` samples into `k` blocks, computes the mean of each block, and takes the median. Achieves sub-Gaussian-style `ln(1/δ)` dependence on the failure probability while requiring only finite variance — a dramatic improvement over Chebyshev when `δ` is small. The estimator is the block median rather than the sample mean, so the framework uses this modified estimator automatically when median-of-means is selected. Strictly dominates Chebyshev for `δ < ~0.15` (which includes all practical `failure_prob` values).

**Catoni.** Requires: `moment_bound=(p, M)` where `E[|X - μ|^p] ≤ M` for some `p > 1` (a central absolute moment bound, consistent with how `variance` is `E[(X-μ)²]`).
```
n = ⌈ C_p · (M / ε^p)^(2/(p+1)) · ln(2/δ)^(p/(p+1)) ⌉
```
where `C_p` is a constant depending on `p`. Uses a truncated influence function (Catoni's M-estimator) instead of the sample mean. This is the right tool for genuinely heavy-tailed distributions (e.g., Pareto, log-normal) where only a fractional moment `p ∈ (1, 2)` may exist — situations where even Chebyshev is inapplicable because the variance is infinite. When `p = 2`, `M` is the variance and Catoni reduces to near-Bernstein rates. Like median-of-means, the framework substitutes the appropriate robust estimator automatically.

**Choosing `p`:** Catoni's rate improves as `p` increases (approaching the `n^{-1/2}` parametric rate as `p → 2`), but the moment `M_p = E[|X - μ|^p]` also grows with `p`. For a given distribution, there is an optimal `p` that minimizes the required `n` — it balances the better exponent against the larger moment constant. **Users declaring `moment_bound` manually should pick the largest `p` for which they can provide a finite bound**, since the framework will evaluate Catoni's `n` and only select it if it beats other applicable bounds. For distributions where variance exists (`p = 2`), Catoni at `p = 2` is evaluated alongside Bernstein, median-of-means, etc. — the framework picks whichever gives the smallest `n`.

**Hoeffding.** Requires: `bounds=(a, b)`.
```
n = ⌈ (b − a)² · ln(2/δ) / (2ε²) ⌉
```
The workhorse bound for bounded random variables. No variance knowledge needed. Always available when `bounds` is declared, and serves as the baseline that tighter bounds improve upon.

**Anderson.** Requires: `bounds=(a, b)`, `symmetric=True`.
```
n = ⌈ (b − a)² · ln(1/δ) / (2ε²) ⌉
```
Anderson's inequality gives a factor-of-2 improvement over Hoeffding for distributions symmetric about their mean (the `ln(2/δ)` in Hoeffding becomes `ln(1/δ)`). Applicable when the user can guarantee symmetry — e.g., testing that a symmetric random walk has zero mean, or that a centered noise distribution is unbiased. The user declares `symmetric=True` in the decorator.

**Maurer-Pontil.** Requires: `bounds=(a, b)`.
```
P(|X̄ - μ| ≥ √(2σ̂² ln(2/δ) / n) + 7(b-a) ln(2/δ) / (3(n-1))) ≤ δ
```
An empirical Bernstein-type bound that uses the *observed* sample variance `σ̂²` directly, without requiring the user to declare or tune variance. This is a single-pass finite-sample bound (not a two-phase procedure). It is strictly tighter than Hoeffding whenever the observed variance is meaningfully smaller than `(b-a)²/4`, which is common in practice.

Because the bound depends on the data, `n` cannot be solved in closed form a priori. The framework uses a conservative strategy: compute `n` assuming worst-case variance `(b-a)²/4` (i.e., Hoeffding's `n`), then check whether the Maurer-Pontil bound would have passed with fewer samples. If so, report the effective savings in verbose output. Alternatively, when tuned data is available, the framework can use `variance_upper` as a pilot estimate to compute a tighter initial `n`, then verify the Maurer-Pontil bound at runtime.

**Bentkus.** Requires: `bounds=(a, b)`. Most effective for one-sided tests (`side="greater"` or `side="less"`).
```
P(S_n - E[S_n] ≥ t) ≤ (e/√(2π)) · P(Bin(n, ?) ≥ ?)
```
Bentkus's inequality is a refinement of Hoeffding that gives the one-sided tail probability as at most `e/√(2π) ≈ 1.08` times the corresponding Binomial tail probability. In practice this yields ~20–40% reduction in `n` compared to Hoeffding for moderate tolerances. The improvement is most significant for one-sided tests; for two-sided tests, the union bound over both tails erodes most of the advantage. The framework automatically prefers Bentkus over Hoeffding for one-sided bounded tests.

Computing `n` requires numerically inverting the Binomial tail, which the framework does via bisection at configuration time (not at each sample).

**Bernstein.** Requires: `bounds=(a, b)`, `variance=σ²`.
```
n = ⌈ (2σ² · ln(2/δ)) / ε²  +  (2(b − a) · ln(2/δ)) / (3ε) ⌉
```
Tighter than Hoeffding when `σ² ≪ (b − a)²/4`. The first term dominates for small `ε` (sub-Gaussian rate with the true variance) and the second term captures the bounded range. This is the standard choice when both bounds and variance are known.

**Bernstein (tuned).** Requires: `bounds=(a, b)` + `variance_upper` from `--stochastic-tune` (see §4.3). Identical formula to Bernstein, but `σ²` is the machine-discovered upper confidence bound rather than a user-declared value.

**Sub-Gaussian.** Requires: `sub_gaussian_param=σ`.
```
n = ⌈ 2σ² · ln(2/δ) / ε² ⌉
```
For distributions satisfying `E[exp(t(X-μ))] ≤ exp(t²σ²/2)` for all `t`. This includes all bounded distributions (with `σ = (b-a)/2`), Gaussians, and any distribution with exponentially decaying tails. Useful when the distribution is unbounded but light-tailed and the user knows the sub-Gaussian parameter.

### 4.2 Selection Algorithm

```python
def select_bound(properties: dict, tolerance: float, failure_prob: float,
                 side: str = "two-sided") -> BoundStrategy:
    """
    Given declared properties, return the bound requiring fewest samples.
    """
    applicable = [
        b for b in BOUND_REGISTRY
        if b.required_properties <= set(properties.keys())
        and b.supports_side(side)
    ]

    if not applicable:
        raise ConfigurationError(
            "No concentration bound is applicable with the declared properties. "
            "Provide at least one of: bounds=(a, b), variance=σ², "
            "moment_bound=(p, M), or sub_gaussian_param=σ."
        )

    # Compute n for each applicable bound, pick the minimum
    best = min(applicable, key=lambda b: b.compute_n(
        tolerance, failure_prob, **properties
    ))

    return best
```

The key insight: the user doesn't choose. They just declare what they know, and the framework does the optimization.

**Estimator selection.** Most bounds use the ordinary sample mean, but two bounds require alternative estimators:

| Bound | Estimator | Why |
|---|---|---|
| Most bounds | Sample mean `X̄ = (1/n)ΣX_i` | Standard; concentration bounds apply directly |
| Median-of-means | Median of block means | Achieves sub-Gaussian rate with only finite variance |
| Catoni | Catoni's M-estimator `μ̂_α` | Handles distributions with only fractional moments (p < 2) |

When the selection algorithm picks a bound that requires a non-standard estimator, the framework automatically substitutes the appropriate estimator during execution. The user's test function is unchanged — it still returns scalars, and the framework handles aggregation.

**Maurer-Pontil as an opportunistic upgrade.** Maurer-Pontil is special because its `n` depends on the data. The framework handles this as follows: if `bounds` is declared but `variance` is not (and no tuned variance is available), the framework allocates `n` according to Hoeffding (the conservative baseline). After collecting all samples, it retroactively checks whether the Maurer-Pontil bound is satisfied — if so, the test passes at the Maurer-Pontil-certified confidence level, and verbose output reports the effective `n` at which the bound was already met. This doesn't reduce runtime (we still collect Hoeffding's `n` samples) but it provides a tighter post-hoc confidence statement. When tuned variance *is* available, the framework uses it to set a smaller initial `n` and verifies the Maurer-Pontil bound as a check.

### 4.3 Tune Mode

Rather than estimating variance at runtime (which adds complexity to every test run and makes execution non-deterministic), the framework provides a separate **tuning step** that profiles test functions and persists discovered properties for future runs.

#### Workflow

```
# Step 1: Write the test with just the properties you can guarantee
@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1), failure_prob=1e-8)
def test_my_sampler(rng):
    return my_sampler(rng)

# Step 2: Run tuning (once, or periodically)
$ pytest tests/test_stochastic.py --stochastic-tune

# Step 3: Tuning writes discovered properties to a config file
# .stochastic.toml (auto-generated)
# [tests.test_stochastic::test_my_sampler]
# variance = 0.0832
# observed_range = [0.003, 0.991]
# tuned_at = "2026-02-22T14:30:00Z"
# n_tune_samples = 50000

# Step 4: Subsequent test runs automatically load the tuned variance
$ pytest tests/test_stochastic.py -v
# test_my_sampler PASSED [bernstein (tuned), n=2104, observed=0.502]
```

#### Tune Execution

When `--stochastic-tune` is passed:

1. The framework runs each stochastic test function `n_tune` times (default: 50,000; configurable via `--stochastic-tune-samples`).
2. For each quantity that enters a concentration bound as a parameter, the framework computes a **rigorous one-sided upper confidence bound** — not a point estimate. The stored value must satisfy: with probability ≥ 1 − δ_tune, the true parameter is at or below the stored value. This is essential because these values feed directly into concentration inequalities; an underestimate would invalidate the guarantee on `failure_prob`.

**Variance upper bound.** The sample variance `σ̂²` from `n` i.i.d. samples satisfies `(n-1)σ̂²/σ² ~ χ²(n-1)`. The one-sided `(1 − δ_tune)` upper confidence bound on the true variance is:

```
σ²_upper = (n - 1) · σ̂² / χ²_{δ_tune}(n - 1)
```

where `χ²_{δ_tune
