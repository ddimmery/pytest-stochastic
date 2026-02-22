"""Core data types for pytest-stochastic."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class EstimatorType(enum.Enum):
    """Estimator used to aggregate samples."""

    SAMPLE_MEAN = "sample_mean"
    MEDIAN_OF_MEANS = "median_of_means"
    CATONI_M_ESTIMATOR = "catoni_m_estimator"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Base error for invalid stochastic test configuration."""


class NoApplicableBoundError(ConfigurationError):
    """No concentration bound is applicable with the declared properties."""


class InvalidToleranceError(ConfigurationError):
    """The tolerance specification is invalid."""


class InvalidPropertyError(ConfigurationError):
    """A declared distributional property is invalid."""


# ---------------------------------------------------------------------------
# BoundStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundStrategy:
    """A concentration inequality strategy.

    Each instance encapsulates one bound from the registry: the properties it
    requires, its sample-size formula, and which estimator it uses.
    """

    name: str
    required_properties: frozenset[str]
    optional_properties: frozenset[str]
    compute_n: Callable[..., int]
    supports_side: Callable[[str], bool]
    estimator_type: EstimatorType
    description: str


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestConfig:
    """Validated configuration for a single stochastic test."""

    expected: float
    tol: float
    failure_prob: float
    side: str

    # Distributional properties (None means not declared)
    bounds: tuple[float, float] | None = None
    variance: float | None = None
    sub_gaussian_param: float | None = None
    symmetric: bool = False
    moment_bound: tuple[float, float] | None = None

    @property
    def declared_properties(self) -> dict[str, object]:
        """Return a dict of all non-None declared distributional properties."""
        props: dict[str, object] = {}
        if self.bounds is not None:
            props["bounds"] = self.bounds
        if self.variance is not None:
            props["variance"] = self.variance
        if self.sub_gaussian_param is not None:
            props["sub_gaussian_param"] = self.sub_gaussian_param
        if self.symmetric:
            props["symmetric"] = self.symmetric
        if self.moment_bound is not None:
            props["moment_bound"] = self.moment_bound
        return props
