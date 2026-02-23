# Contributing to pytest-stochastic

Thanks for your interest in contributing! This guide will help you get set up and
explain the workflow.

## Development Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) for package and environment management

### Getting Started

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/pytest-stochastic.git
   cd pytest-stochastic
   ```

2. Install dependencies:

   ```bash
   uv sync --group dev
   ```

   To also work on documentation:

   ```bash
   uv sync --group docs
   ```

## Running Checks Locally

Before submitting a PR, make sure all checks pass. These mirror what CI runs.

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-format (fixes formatting in place)
uv run ruff format .

# Type check
uv run ty check
```

## Documentation

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and live in the `docs/` directory.

```bash
# Serve docs locally at http://127.0.0.1:8000
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

## Pull Request Workflow

1. Create a feature branch from `main`:

   ```bash
   git checkout -b your-branch-name
   ```

2. Make your changes and ensure all checks pass (see above).

3. Commit with a clear, concise message describing the change.

4. Open a pull request against `main`. CI will automatically run linting, type
   checking, and tests across Python 3.11, 3.12, and 3.13.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
