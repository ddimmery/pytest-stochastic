# RNG Injection & Reproducibility

pytest-stochastic provides built-in RNG (random number generator) management for reproducible stochastic tests.

## How RNG Injection Works

If your test function's signature includes an `rng` parameter, the framework automatically injects a seeded `numpy.random.Generator`:

```python
from pytest_stochastic import stochastic_test

@stochastic_test(expected=0.5, atol=0.05, bounds=(0, 1))
def test_uniform_mean(rng):
    return rng.uniform(0, 1)
```

The framework:

1. Creates a `numpy.random.Generator` via `numpy.random.default_rng(seed)`
2. If `seed` is specified in the decorator, uses that seed
3. Otherwise, generates a random seed from `numpy.random.SeedSequence`
4. Passes the same generator to every call of your function within that test run

This means all $n$ calls share one RNG stream, producing a deterministic sequence for a given seed.

## Signature Detection

The framework inspects your function's signature using `inspect.signature`. If it finds a parameter named `rng`, injection is enabled:

```python
# RNG injected
def test_with_rng(rng):
    return rng.normal()

# No RNG injection
def test_without_rng():
    import random
    return random.random()
```

You are free to use any randomness source in functions without `rng`, but reproducibility on failure then depends on your own seed management.

## Fixed Seeds

Pass `seed` to the decorator for fully deterministic tests:

```python
@stochastic_test(
    expected=0.5, atol=0.05, bounds=(0, 1), seed=42
)
def test_deterministic(rng):
    return rng.uniform(0, 1)
```

This test will produce identical results on every run, on every machine (assuming the same NumPy version).

## Seed Reporting on Failure

When no fixed seed is specified and a test fails, the failure message includes the seed:

```
FAILED [hoeffding, n=185, seed=7291038456123]:
  |0.567 - 0.5| = 0.067 not < 0.05
```

You can reproduce the failure by adding `seed=7291038456123` to the decorator.

## The `stochastic_rng` Fixture

For tests that don't use the `@stochastic_test` decorator but still need a seeded RNG, the plugin provides a `stochastic_rng` pytest fixture:

```python
def test_custom_logic(stochastic_rng):
    samples = [stochastic_rng.normal() for _ in range(100)]
    assert abs(sum(samples) / len(samples)) < 1.0
```

The fixture's seed is derived deterministically from the test's node ID (`hash(request.node.nodeid) % 2^32`), so it is reproducible across runs without specifying a seed.

## Best Practices

1. **Use `rng` injection** for all stochastic tests. It ensures reproducibility without manual seed management.

2. **Don't fix seeds in CI** unless you have a specific reason. Random seeds allow your tests to explore different random states across runs, while the concentration inequality guarantees keep the failure rate controlled.

3. **Fix seeds for debugging.** When a test fails, copy the reported seed into the decorator to reproduce the exact failure.

4. **Avoid global RNG state.** Functions using `numpy.random.random()` (the legacy global RNG) or `random.random()` are not reproducible through the framework. Use the injected `rng` parameter instead.
