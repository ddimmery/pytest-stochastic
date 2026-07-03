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

$$k = \lceil 8 \ln(k_s/\delta) \rceil, \quad n = k \cdot \left\lceil \frac{4\sigma^2}{\varepsilon^2} \right\rceil$$

where $k_s = 2$ for two-sided and $k_s = 1$ for one-sided tests. Achieves a sub-Gaussian $\ln(1/\delta)$ rate using only finite variance. Splits samples into $k$ blocks, computes each block's mean, and takes the median. The sole variance-only bound in the registry.

The block size $\lceil 4\sigma^2/\varepsilon^2 \rceil$ makes each block's failure probability at most $1/4$ by Chebyshev; the median then fails only if at least $k/2$ blocks fail, which by Hoeffding happens with probability at most $e^{-k/8} \leq \delta$.

### Catoni M-Estimator

**Required properties:** `moment_bound` (tuple of $p > 1$ and $M$ with $\mathbb{E}|X - \mu|^p \leq M$)
**Sides:** two-sided, greater, less
**Estimator:** Catoni M-estimator ($p = 2$) / median-of-means ($p < 2$)

For $p = 2$ ($M = \sigma^2$), Catoni's M-estimator (Catoni 2012) with influence function $\psi(x) = \text{sign}(x) \cdot \ln(1 + |x| + x^2/2)$ and $\alpha = \sqrt{2\ln(k/\delta) / (n(M + \varepsilon^2))}$ requires

$$n = \left\lceil 2 \ln\!\frac{k}{\delta} \left(\frac{M}{\varepsilon^2} + 1\right) \right\rceil$$

where $k = 2$ for two-sided and $k = 1$ for one-sided tests.

For $1 < p < 2$ (possibly infinite variance), the estimator is median-of-means with blocks sized via the von Bahr&ndash;Esseen inequality:

$$B = \left\lceil \left(\frac{8M}{\varepsilon^p}\right)^{1/(p-1)} \right\rceil, \quad k_b = \lceil 8 \ln(k/\delta) \rceil, \quad n = k_b \cdot B$$

Handles heavy-tailed distributions where only a $p$-th central moment bound is known.

### Hoeffding

**Required properties:** `bounds`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{(b-a)^2 \ln(k/\delta)}{2\varepsilon^2} \right\rceil$$

where $k = 2$ for two-sided tests (union bound over both tails) and $k = 1$ for one-sided tests.

The standard bound for bounded random variables $X_i \in [a, b]$. Does not require variance knowledge. Widely applicable but can be conservative when the actual variance is small relative to the range.

### Anderson

**Required properties:** `bounds`, `symmetric`
**Sides:** two-sided only
**Estimator:** sample mean

$$w = \min(b - \mu_0,\; \mu_0 - a), \quad n = \left\lceil \frac{(2w)^2 \ln(2/\delta)}{2\varepsilon^2} \right\rceil$$

where $\mu_0$ is the `expected` value. A distribution symmetric about its mean $\mu_0$ with support in $[a, b]$ actually has $\text{ess sup}|X - \mu_0| \leq w$: any mass at $\mu_0 + d$ with $d > w$ would require matching mass outside $[a, b]$. Hoeffding therefore applies with the reduced range $2w \leq b - a$.

This is a strict improvement over Hoeffding whenever `expected` is off-center in $[a, b]$, and identical when centered. Requires $a < \mu_0 < b$.

### Maurer-Pontil

**Required properties:** `bounds`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

An empirical Bernstein bound that adapts to the data. Pre-allocates using Hoeffding's formula (same $n$), but at runtime checks whether the empirical variance yields a tighter bound. If so, it reports the effective sample count.

The runtime check uses the Maurer&ndash;Pontil (2009) empirical Bernstein bound

$$P\!\left(\mu - \bar{X} \geq \sqrt{\frac{2\hat{\sigma}^2 L}{n}} + \frac{7(b-a)L}{3(n-1)}\right) \leq \delta, \quad L = \ln\!\frac{2 k_s (n-1)}{\delta}$$

where $k_s = 2$ for two-sided and $k_s = 1$ for one-sided tests. The factor 2 inside the log is intrinsic to the theorem (it union-bounds the mean and variance deviations), and the $(n-1)$ factor is a union bound over the prefix lengths scanned when searching for the smallest effective $n$. This bound is "free" &mdash; it requires the same samples as Hoeffding but may discover a tighter result post-hoc. The reported effective $n$ is informational only; it never affects pass/fail.

### Bentkus

**Required properties:** `bounds`
**Sides:** greater, less (one-sided only)
**Estimator:** sample mean

Numerically inverts the Bentkus binomial tail bound via bisection. Typically requires 5-10% fewer samples than one-sided Hoeffding for bounded random variables.

Uses Bentkus (2004, Thm 1.1) with the worst case over the unknown mean at $q = 1/2$:

$$P(\bar{X} - \mu \geq \varepsilon) \leq \frac{e^2}{2} \cdot P\big(\text{Bin}(n, 1/2) \geq \lfloor n p^* \rfloor\big)$$

where $p^* = \varepsilon/(b-a) + 1/2$. The floor keeps the evaluation on the conservative side of the theorem's log-concave-majorant interpolation, and the returned $n$ is re-verified against the inequality after bisection.

### Bernstein

**Required properties:** `bounds`, `variance`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{2\sigma^2 \ln(k/\delta)}{\varepsilon^2} + \frac{2(b-a)\ln(k/\delta)}{3\varepsilon} \right\rceil$$

where $k = 2$ for two-sided and $k = 1$ for one-sided tests. Exploits both bounded range and known variance. When $\sigma^2 \ll (b-a)^2/4$, Bernstein is significantly tighter than Hoeffding.

### Bernstein (Tuned)

**Required properties:** `bounds`, `variance_tuned`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

Same formula as Bernstein, but uses the machine-discovered variance from `--stochastic-tune` instead of a user-declared variance. When the test declares `bounds`, the tuned variance is a distribution-free upper confidence bound (Maurer-Pontil), so the resulting guarantee is $\text{failure\_prob} + \text{confidence}$ (the UCB's own failure budget, $10^{-8}$ by default). See [Tune Mode](guide/tune-mode.md).

### Sub-Gaussian

**Required properties:** `sub_gaussian_param`
**Sides:** two-sided, greater, less
**Estimator:** sample mean

$$n = \left\lceil \frac{2\sigma^2 \ln(k/\delta)}{\varepsilon^2} \right\rceil$$

where $k = 2$ for two-sided and $k = 1$ for one-sided tests. For distributions satisfying the sub-Gaussian tail condition with parameter $\sigma$. Many common distributions are sub-Gaussian: bounded distributions (with $\sigma = (b-a)/2$), Gaussian ($\sigma$ equals the standard deviation), and any distribution with bounded MGF.

## Comparison

The table below shows sample sizes for $\varepsilon = 0.05$, $\delta = 10^{-8}$, and a $[0, 1]$-bounded distribution with variance $1/12$ (expected value $0.3$ for the symmetric row, so the reduced support is $[0, 0.6]$):

| Bound | Required Properties | $n$ |
|-------|-------------------|-----------------|
| Median-of-Means | variance | 20,502 |
| Hoeffding | bounds | 3,823 |
| Sub-Gaussian ($\sigma = 0.5 = (b-a)/2$) | sub_gaussian_param | 3,823 |
| Bentkus (one-sided) | bounds | 3,439 (vs. 3,685 one-sided Hoeffding) |
| Bernstein | bounds + variance | 1,530 |
| Anderson (symmetric, expected=0.3) | bounds + symmetric | 1,377 |
| Catoni ($p=2$, $M = 1/12$) | moment_bound | 1,313 |
| Sub-Gaussian ($\sigma = \sqrt{1/12}$) | sub_gaussian_param | 1,275 |

The ordering depends on the specific distribution parameters. Declaring more properties generally enables tighter bounds. (Note that a bounded distribution is always sub-Gaussian with $\sigma = (b-a)/2$, which exactly recovers Hoeffding; the sub-Gaussian bound only helps when you can certify a smaller $\sigma$.)
