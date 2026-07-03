# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Catoni bound was statistically unsound.** The sample-size formula had
  incorrect exponents (scaling as ε^(-4/3) instead of ε^(-2) for p=2), producing
  a measured 16.5% false-failure rate against a 1% budget. It now uses Catoni's
  (2012) rate for p=2 and von Bahr-Esseen median-of-means blocks for 1 < p < 2,
  and the runtime estimator's tuning constant now depends on (n, M, δ) as the
  theorem requires.
- **Median-of-means block size was a factor of 2 too small** (per-block Chebyshev
  failure of exactly 1/2 makes the median argument vacuous); block size is now
  4σ²/ε², and the runtime estimator's block count matches the side-aware
  pre-allocation instead of hard-coding the two-sided constant.
- **The "anderson" bound dropped the two-sided union factor without
  justification** (true failure probability up to 2δ). It is now a legitimate
  reduced-range Hoeffding bound for symmetric distributions, exploiting that a
  distribution symmetric about its mean μ in [a, b] concentrates within
  μ ± min(b-μ, μ-a). Requires `expected` strictly inside `bounds`.
- **Bentkus used a non-citable constant** (e/√(2π) ≈ 1.08); it now uses the
  Bentkus (2004) constant e²/2, evaluates the binomial tail conservatively
  (floor instead of ceil), and verifies the inequality at the returned n.
  Honest savings vs. one-sided Hoeffding are ~5-10% (previously advertised
  as 20-40%).
- **Maurer-Pontil post-hoc check was O(n²)** (unusably slow at large n) and used
  a too-small log factor; it is now an O(n) vectorized scan with the theorem's
  intrinsic ln(2/δ) factor and a union bound over scanned prefixes.
- **`@distributional_test(test="anderson")` could never fail** at significance
  levels below 0.001 because scipy caps `anderson_ksamp` p-values to
  [0.001, 0.25]. Such configurations (including the former default,
  significance=1e-6) now raise `ConfigurationError` at import time.
- **Tuned-variance lookup never matched its keys** (stored keys contained a
  `.py` segment) and silently fell back to bare-function-name matching with
  cross-module collisions. Keys are now `{module}.{qualname}`, matched exactly
  (legacy keys still recognized; bare-name fallback only when unambiguous), and
  non-finite tuned variances no longer crash bound selection at import.
- **`stochastic_rng` was not reproducible across runs** (it used the
  process-salted builtin `hash`); the seed is now a stable sha256 digest of the
  test node ID, matching the documented behavior.
- **The `stochastic` marker was never registered**, causing
  `PytestUnknownMarkWarning` in user suites (and errors under
  `--strict-markers`). It is now registered, and distributional tests are
  tagged too.
- **The tune-mode variance UCB used a chi-squared interval that is only exact
  for Gaussian data.** When a test declares `bounds`, the UCB is now the
  distribution-free Maurer-Pontil self-bounding interval; the chi-squared
  interval remains as an explicitly labeled Gaussian approximation otherwise.
  The UCB confidence (default now 1e-8) and method are recorded in
  `.stochastic.toml`, and `.stochastic.toml` is resolved relative to the pytest
  root directory rather than the process working directory.
- Documentation example sample sizes recomputed from the corrected formulas
  (e.g. the quick-start Hoeffding example is n=3,823, not n=185; the OLS
  bounds+variance example selects Bernstein with n=1,245).

### Changed

- Median-of-means and Catoni tests now require more samples than before — the
  previous sample sizes did not actually deliver the declared failure
  probability.
- `@distributional_test(test="anderson")` requires
  `0.001 <= significance < 0.25`.
- Verbose reports now include distributional test details, mirroring stochastic
  tests.

## [0.1.0]

Initial release.

### Added

- `@stochastic_test` decorator — assert that a statistic's mean matches an
  expected value within tolerance, with a mathematically guaranteed flakiness
  bound.
- `@distributional_test` decorator — assert that outputs follow a reference
  distribution using Kolmogorov-Smirnov, chi-squared, or Anderson-Darling tests.
- Automatic bound selection across a registry of concentration inequalities:
  Hoeffding, Bernstein, tuned Bernstein, Bentkus, Anderson, Maurer-Pontil,
  sub-Gaussian, median-of-means, and Catoni. The framework evaluates every
  applicable bound and chooses the one requiring the fewest samples.
- One-sided tests (`side="greater"` / `side="less"`) with reduced sample sizes.
- Tune mode (`--stochastic-tune`) — empirically profile tests and persist a
  rigorous upper confidence bound on variance to `.stochastic.toml` for tighter
  bounds on subsequent runs. Sample count configurable via
  `--stochastic-tune-samples`.
- RNG injection — reproducible tests via automatic per-test seeding and optional
  `rng` parameter injection, with the seed reported on failure for replay.

[Unreleased]: https://github.com/ddimmery/pytest-stochastic/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ddimmery/pytest-stochastic/releases/tag/v0.1.0
