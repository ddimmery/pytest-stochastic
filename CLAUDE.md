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
- **Task tracking:** [beads](https://github.com/steveyegge/beads) (`bd`) — install with `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash`

## Commands

```sh
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
bd ready                   # next tasks
uv run mkdocs serve        # serve docs locally (http://127.0.0.1:8000)
uv run mkdocs build        # build static docs site to site/
```

## Documentation

Docs live in `docs/` and are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
Install docs dependencies with `uv sync --group docs`. The navigation structure is
defined in `mkdocs.yml`. API reference pages use
[mkdocstrings](https://mkdocstrings.github.io/) to auto-generate from docstrings.
