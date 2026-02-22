# Concentration Bounds Reference

pytest-stochastic selects the tightest applicable concentration inequality from a registry of bounds. Each bound computes the minimum sample size $n$ required to guarantee that the estimator is within tolerance $\varepsilon$ of the true mean with probability at least $1 - \delta$.

## Bound Selection

For a given test, the framework:

1. Filters bounds whose required properties are a subset of the user's declared properties
2. Filters bounds that support the requested test side (two-sided, greater, or less)
3. Evaluates each remaining bound's sample size formula
4. Selects the bound requiring the fewest samples

## Available Bounds

### Median-of-Means

**Required properties:** `variance`
**Sides:** two-sided, greater, less
**Estimator:** median-of-means

$$k = \lceil 8 \ln(2/\delta) \rceil, \quad n = k \cdot \left\lceil \frac{2\sigma^2}{\varepsilon^2} \right\rceil$$

Achieves a sub-Gaussian $\ln(1/\delta)$ rate using only finite variance. Splits samples into $k$ blocks, computes each block's mean, and takes the median. The sole variance-only bound in the registry.

### Catoni M-Estimator

**Required properties:** `moment_bound` (tuple of $p > 1$ and $M$)
**Sides:** two-sided, greater, less
**Estimator:** Catoni M-estimator

$$n = \left\lceil C_p \cdot \left(\frac{M}{\varepsilon^p}\right)^{2/(p+1)} \cdot \left(\ln\frac{2}{\delta}\right)^{p/(p+1)} \right\rceil$$

Handles heavy-tailed distributions where only a $p$-th moment bound is known ($p > 1$). Uses a robust M-estimator with influence function $\psi(x) = \text{sign}(x) \cdot \ln(1 + |x| + x^2/2)$.

### Hoeffding

**Required properties:** `bounds`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{(b-a)^2 \ln(2/\delta)}{2\varepsilon^2} \right\rceil$$

The standard bound for bounded random variables $X_i \in [a, b]$. Does not require variance knowledge. Widely applicable but can be conservative when the actual variance is small relative to the range.

### Anderson

**Required properties:** `bounds`, `symmetric`
**Sides:** two-sided only
**Estimator:** sample mean

$$n = \left\lceil \frac{(b-a)^2 \ln(1/\delta)}{2\varepsilon^2} \right\rceil$$

A factor-of-2 improvement over Hoeffding for symmetric distributions. The $\ln(1/\delta)$ term (vs. Hoeffding's $\ln(2/\delta)$) comes from the symmetry assumption eliminating one tail.

### Maurer-Pontil

**Required properties:** `bounds`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

An empirical Bernstein bound that adapts to the data. Pre-allocates using Hoeffding's formula (same $n$), but at runtime checks whether the empirical variance yields a tighter bound. If so, it reports the effective sample count.

The runtime check uses:

$$P\!\left(|\bar{X} - \mu| \geq \sqrt{\frac{2\hat{\sigma}^2 \ln(2/\delta)}{n}} + \frac{7(b-a)\ln(2/\delta)}{3(n-1)}\right) \leq \delta$$

This bound is "free" &mdash; it requires the same samples as Hoeffding but may discover a tighter result post-hoc.

### Bentkus

**Required properties:** `bounds`
**Sides:** greater, less (one-sided only)
**Estimator:** sample mean

Numerically inverts the Bentkus binomial tail bound via bisection. Typically requires 20-40% fewer samples than Hoeffding for one-sided tests of bounded random variables.

Uses the bound:

$$P(\bar{X} - \mu \geq \varepsilon) \leq \frac{e}{\sqrt{2\pi}} \cdot P(\text{Bin}(n, p^*) \geq k^*)$$

where $p^* = \varepsilon/(b-a) + 1/2$.

### Bernstein

**Required properties:** `bounds`, `variance`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{2\sigma^2 \ln(2/\delta)}{\varepsilon^2} + \frac{2(b-a)\ln(2/\delta)}{3\varepsilon} \right\rceil$$

Exploits both bounded range and known variance. When $\sigma^2 \ll (b-a)^2/4$, Bernstein is significantly tighter than Hoeffding.

### Bernstein (Tuned)

**Required properties:** `bounds`, `variance_tuned`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

Same formula as Bernstein, but uses the machine-discovered variance from `--stochastic-tune` instead of a user-declared variance. The tuned variance is a rigorous upper confidence bound, so the bound remains valid.

### Sub-Gaussian

**Required properties:** `sub_gaussian_param`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{2\sigma^2 \ln(2/\delta)}{\varepsilon^2} \right\rceil$$

For distributions satisfying the sub-Gaussian tail condition with parameter $\sigma$. Many common distributions are sub-Gaussian: bounded distributions (with $\sigma = (b-a)/2$), Gaussian ($\sigma$ equals the standard deviation), and any distribution with bounded MGF.

## Comparison

The table below shows approximate sample sizes for $\varepsilon = 0.05$, $\delta = 10^{-8}$, and a $[0, 1]$-bounded distribution with variance $1/12$:

| Bound | Required Properties | Approximate $n$ |
|-------|-------------------|-----------------|
| Median-of-Means | variance | 29,440 |
| Hoeffding | bounds | 7,378 |
| Bernstein | bounds + variance | ~4,000 |
| Bentkus (one-sided) | bounds | ~4,500 |
| Anderson (symmetric) | bounds + symmetric | 3,689 |
| Sub-Gaussian | sub_gaussian_param=0.5 | 3,689 |

The ordering depends on the specific distribution parameters. Declaring more properties generally enables tighter bounds.
