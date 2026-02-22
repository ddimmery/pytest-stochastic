# pytest-stochastic

A pytest plugin for principled stochastic testing. Users declare properties of their
test statistic (bounds, variance, sub-Gaussian parameter) via a decorator, and the
framework automatically selects the tightest concentration inequality, computes the
required sample size for a target flakiness budget, and runs the test. Includes an
offline tune mode (`--stochastic-tune`) that profiles tests and persists discovered
parameters to `.stochastic.toml`.

## Development

- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Formatting/linting:** [ruff](https://docs.astral.sh/ruff/)
- **Type checking:** [ty](https://docs.astral.sh/ty/)
- **Task tracking:** [beads](https://github.com/steveyegge/beads) (`bd`)

## Commands

```sh
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
bd ready                   # next tasks
```
