# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial release.

### Added

- `@stochastic_test` decorator — assert that a statistic's mean matches an
  expected value within tolerance, with a mathematically guaranteed flakiness
  bound.
- `@distributional_test` decorator — assert that outputs follow a reference
  distribution using Kolmogorov-Smirnov, chi-squared, or Anderson-Darling tests.
- Automatic bound selection across a registry of concentration inequalities:
  Hoeffding, Bernstein, Bentkus, Anderson, Maurer-Pontil, sub-Gaussian,
  median-of-means, Chebyshev, and Catoni. The framework evaluates every
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
