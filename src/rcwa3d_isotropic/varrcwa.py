from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .types import Layer


LayerFactoryResult = Union[Layer, complex, float, np.ndarray]
LayerFactory = Callable[[float], LayerFactoryResult]


@dataclass(frozen=True)
class AdaptiveLayerSpec:
    """Adaptive high-order z discretization for a smoothly varying layer.

    ``layerAt`` is called with the local z coordinate in ``[0, thickness]``.
    The layer is represented by piecewise-constant RCWA slices whose epsilon is
    computed by Gauss-Legendre averaging.  Intervals are split recursively until
    the one-panel and two-panel averages agree within ``tolerance``.
    """

    thickness: float
    layerAt: LayerFactory
    tolerance: float = 1e-3
    maxDepth: int = 8
    quadratureOrder: int = 4
    minDepth: int = 0
    name: str = "adaptive z layer"
    factorization: str = "auto"

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.maxDepth < 0:
            raise ValueError("maxDepth must be non-negative")
        if self.minDepth < 0 or self.minDepth > self.maxDepth:
            raise ValueError("minDepth must be between 0 and maxDepth")
        if self.quadratureOrder < 1:
            raise ValueError("quadratureOrder must be at least 1")

    def toLayers(self) -> list[Layer]:
        intervals: list[tuple[float, float]] = []
        self._refine(0.0, float(self.thickness), 0, intervals)
        layers: list[Layer] = []
        for index, (z0, z1) in enumerate(intervals):
            prototype = _asLayer(self.layerAt((z0 + z1) / 2), z1 - z0, self.name, self.factorization)
            epsilon = _averageEpsilon(self.layerAt, z0, z1, self.quadratureOrder)
            layers.append(
                Layer(
                    thickness=z1 - z0,
                    epsilon=epsilon,
                    name=f"{prototype.name or self.name} {index}",
                    normalField=getattr(prototype, "normalField", None),
                    factorization=getattr(prototype, "factorization", self.factorization),
                )
            )
        return layers

    def _refine(self, z0: float, z1: float, depth: int, intervals: list[tuple[float, float]]) -> None:
        if depth < self.minDepth:
            split = (z0 + z1) / 2
            self._refine(z0, split, depth + 1, intervals)
            self._refine(split, z1, depth + 1, intervals)
            return

        whole = _averageEpsilon(self.layerAt, z0, z1, self.quadratureOrder)
        split = (z0 + z1) / 2
        left = _averageEpsilon(self.layerAt, z0, split, self.quadratureOrder)
        right = _averageEpsilon(self.layerAt, split, z1, self.quadratureOrder)
        twoPanel = 0.5 * (left + right)
        error = _relativeError(whole, twoPanel)
        if error <= self.tolerance or depth >= self.maxDepth:
            intervals.append((z0, z1))
            return
        self._refine(z0, split, depth + 1, intervals)
        self._refine(split, z1, depth + 1, intervals)


def adaptiveZStack(
    thickness: float,
    layerAt: LayerFactory,
    *,
    tolerance: float = 1e-3,
    maxDepth: int = 8,
    quadratureOrder: int = 4,
    minDepth: int = 0,
    name: str = "adaptive z layer",
    factorization: str = "auto",
) -> list[Layer]:
    """Return adaptive high-order z slices for a smoothly varying layer."""

    return AdaptiveLayerSpec(
        thickness=thickness,
        layerAt=layerAt,
        tolerance=tolerance,
        maxDepth=maxDepth,
        quadratureOrder=quadratureOrder,
        minDepth=minDepth,
        name=name,
        factorization=factorization,
    ).toLayers()


def _averageEpsilon(layerAt: LayerFactory, z0: float, z1: float, order: int) -> np.ndarray | complex:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    center = (z0 + z1) / 2
    halfWidth = (z1 - z0) / 2
    values = []
    for node in nodes:
        layer = _asLayer(layerAt(center + halfWidth * float(node)), z1 - z0, "", "auto")
        values.append(np.asarray(layer.epsilon, dtype=complex))
    average = sum(float(weight) * value for weight, value in zip(weights, values)) / 2
    if np.ndim(average) == 0:
        return complex(np.asarray(average).item())
    return np.asarray(average, dtype=complex)


def _asLayer(
    value: LayerFactoryResult,
    thickness: float,
    name: str,
    factorization: str,
) -> Layer:
    if isinstance(value, Layer):
        return value
    return Layer(thickness=thickness, epsilon=value, name=name, factorization=factorization)


def _relativeError(reference: np.ndarray | complex, estimate: np.ndarray | complex) -> float:
    difference = np.asarray(estimate, dtype=complex) - np.asarray(reference, dtype=complex)
    scale = max(float(np.linalg.norm(np.asarray(estimate, dtype=complex))), 1e-14)
    return float(np.linalg.norm(difference) / scale)
