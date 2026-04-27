from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Union

import numpy as np

from .types import Layer
from .geometry import FactorizationMode, Pattern2D
from .materials import IsotropicMaterial


AxisSelector = Union[tuple[float, float], slice, np.ndarray, Callable[[np.ndarray], np.ndarray], None]
RegionSelector = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class PatternLayer:
    """Mutable sampled layer used by the high-level simulation builder."""

    period: tuple[float, float]
    thickness: float
    background: complex | float | IsotropicMaterial
    shape: tuple[int, int]
    name: str = ""
    factorization: FactorizationMode = "auto"
    pattern: Pattern2D = field(init=False)
    version: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.pattern = Pattern2D(period=self.period, shape=self.shape, background=self.background, name=self.name)

    @property
    def epsilon(self):
        return self.pattern.epsilon

    @property
    def normalField(self):
        return self.pattern.normalField

    def fill(self, mask, material: complex | float | IsotropicMaterial) -> "PatternLayer":
        self.pattern.fill(mask, material)
        self._changed()
        return self

    def circle(
        self,
        radius: float,
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.circle(radius, material, center=center, useNormal=useNormal)
        self._changed()
        return self

    def ellipse(
        self,
        radii: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.ellipse(radii, material, center=center, angle=angle, useNormal=useNormal)
        self._changed()
        return self

    def rectangle(
        self,
        size: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.rectangle(size, material, center=center, angle=angle, useNormal=useNormal)
        self._changed()
        return self

    def annulus(
        self,
        innerRadius: float,
        outerRadius: float,
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.annulus(innerRadius, outerRadius, material, center=center, useNormal=useNormal)
        self._changed()
        return self

    def cross(
        self,
        armLengths: tuple[float, float],
        armWidths: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.cross(armLengths, armWidths, material, center=center, angle=angle, useNormal=useNormal)
        self._changed()
        return self

    def stripes(
        self,
        fillFraction: float,
        material: complex | float | IsotropicMaterial,
        axis: str = "x",
        center: float = 0.0,
    ) -> "PatternLayer":
        self.pattern.stripes(fillFraction, material, axis=axis, center=center)
        self._changed()
        return self

    def polygon(
        self,
        vertices: Iterable[tuple[float, float]],
        material: complex | float | IsotropicMaterial,
        useNormal: bool = True,
    ) -> "PatternLayer":
        self.pattern.polygon(vertices, material, useNormal=useNormal)
        self._changed()
        return self

    def toLayer(self) -> Layer:
        return self.pattern.toLayer(self.thickness, name=self.name, factorization=self.factorization)

    def _changed(self) -> None:
        self.version += 1


@dataclass
class LayerStack:
    """Layer-first geometry builder for isotropic RCWA examples.

    Layers are ordered from the incident side toward the transmission side.
    The stack coordinate uses ``z=0`` at the top interface and positive ``z``
    downward through the finite layers.
    """

    period: tuple[float, float]
    shape: tuple[int, int]
    layers: list[PatternLayer] = field(default_factory=list)

    def addLayer(
        self,
        thickness: float,
        material: complex | float | IsotropicMaterial,
        *,
        name: str = "",
        factorization: FactorizationMode = "auto",
        shape: tuple[int, int] | None = None,
    ) -> PatternLayer:
        if thickness <= 0:
            raise ValueError("layer thickness must be positive")
        layer = PatternLayer(
            period=self.period,
            thickness=float(thickness),
            background=material,
            shape=self.shape if shape is None else shape,
            name=name,
            factorization=factorization,
        )
        self.layers.append(layer)
        return layer

    @property
    def totalThickness(self) -> float:
        return float(sum(layer.thickness for layer in self.layers))

    def layerBounds(self) -> tuple[tuple[float, float], ...]:
        start = 0.0
        bounds = []
        for layer in self.layers:
            end = start + float(layer.thickness)
            bounds.append((start, end))
            start = end
        return tuple(bounds)

    def setMaterial(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        x: AxisSelector = None,
        y: AxisSelector = None,
        z: AxisSelector = None,
        mask: RegionSelector | None = None,
        slices: int = 1,
    ) -> "LayerStack":
        if not self.layers:
            raise ValueError("add at least one layer before setting a material region")
        if isinstance(z, tuple):
            self._splitRange(float(min(z)), float(max(z)), slices)

        for layer, (zStart, zEnd) in zip(self.layers, self.layerBounds()):
            if not _zSelected(z, zStart, zEnd):
                continue
            xx, yy = layer.pattern.coordinates()
            region = _axisMask(xx, x) & _axisMask(yy, y)
            if mask is not None:
                zz = np.full(xx.shape, 0.5 * (zStart + zEnd), dtype=float)
                region &= np.asarray(mask(xx, yy, zz), dtype=bool)
            layer.fill(region, material)
        return self

    def addVolume(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        mask: RegionSelector,
        slices: int = 1,
    ) -> "LayerStack":
        """Write an arbitrary z-dependent 3D body into existing background layers."""

        return self.setMaterial(material, z=z, mask=mask, slices=slices)

    def addBox(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        size: tuple[float, float],
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        slices: int = 1,
    ) -> "LayerStack":
        return self.addVolume(
            material,
            z=z,
            slices=slices,
            mask=lambda xx, yy, _zz: _rectangleMask(xx, yy, size, center, angle),
        )

    def addCylinder(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        radius: float,
        center: tuple[float, float] = (0.0, 0.0),
        slices: int = 1,
    ) -> "LayerStack":
        return self.addVolume(
            material,
            z=z,
            slices=slices,
            mask=lambda xx, yy, _zz: _circleMask(xx, yy, radius, center),
        )

    def addCone(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        topRadius: float,
        bottomRadius: float,
        center: tuple[float, float] = (0.0, 0.0),
        slices: int = 20,
    ) -> "LayerStack":
        zTop, zBottom = _zRange(z)

        def mask(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
            fraction = np.clip((zz - zTop) / max(zBottom - zTop, 1e-30), 0.0, 1.0)
            radius = topRadius + (bottomRadius - topRadius) * fraction
            return _circleMask(xx, yy, radius, center)

        return self.addVolume(material, z=(zTop, zBottom), mask=mask, slices=slices)

    def addPyramid(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        topSize: float | tuple[float, float],
        bottomSize: float | tuple[float, float],
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        slices: int = 20,
    ) -> "LayerStack":
        zTop, zBottom = _zRange(z)
        top = _pairSize(topSize)
        bottom = _pairSize(bottomSize)

        def mask(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
            fraction = np.clip((zz - zTop) / max(zBottom - zTop, 1e-30), 0.0, 1.0)
            size = (
                top[0] + (bottom[0] - top[0]) * fraction,
                top[1] + (bottom[1] - top[1]) * fraction,
            )
            return _rectangleMask(xx, yy, size, center, angle)

        return self.addVolume(material, z=(zTop, zBottom), mask=mask, slices=slices)

    def addPolygonPrism(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        vertices: Iterable[tuple[float, float]],
        slices: int = 1,
    ) -> "LayerStack":
        points = _polygonPoints(vertices)
        return self.addVolume(
            material,
            z=z,
            slices=slices,
            mask=lambda xx, yy, _zz: _polygonMask(xx, yy, points),
        )

    def addPolygonPyramid(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        z: tuple[float, float],
        vertices: Iterable[tuple[float, float]],
        topScale: float = 0.0,
        bottomScale: float = 1.0,
        center: tuple[float, float] | None = None,
        slices: int = 20,
    ) -> "LayerStack":
        zTop, zBottom = _zRange(z)
        points = _polygonPoints(vertices)
        pivot = np.mean(points, axis=0) if center is None else np.asarray(center, dtype=float)

        def mask(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
            fraction = np.clip((zz - zTop) / max(zBottom - zTop, 1e-30), 0.0, 1.0)
            scale = topScale + (bottomScale - topScale) * fraction
            scaled = pivot + (points - pivot) * scale
            return _polygonMask(xx, yy, scaled)

        return self.addVolume(material, z=(zTop, zBottom), mask=mask, slices=slices)

    def addWaveBody(
        self,
        material: complex | float | IsotropicMaterial,
        *,
        baseZ: float,
        meanHeight: float,
        amplitude: float,
        axis: str = "x",
        periods: float = 1.0,
        phase: float = 0.0,
        slices: int = 20,
    ) -> "LayerStack":
        """Write a body bounded by a sinusoidal top surface."""

        if meanHeight <= 0:
            raise ValueError("meanHeight must be positive")
        if abs(amplitude) > meanHeight:
            raise ValueError("abs(amplitude) must not exceed meanHeight")
        zTop = float(baseZ)
        zBottom = zTop + meanHeight + abs(amplitude)
        period = self.period[0] if axis == "x" else self.period[1] if axis == "y" else None
        if period is None:
            raise ValueError("axis must be 'x' or 'y'")

        def mask(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
            coordinate = xx if axis == "x" else yy
            surface = zTop + meanHeight + amplitude * np.sin(2 * np.pi * periods * coordinate / period + phase)
            return zz <= surface

        return self.addVolume(material, z=(zTop, zBottom), mask=mask, slices=slices)

    def toLayers(self) -> list[Layer]:
        return [layer.toLayer() for layer in self.layers]

    def _splitAt(self, z: float) -> None:
        if z <= 0.0 or z >= self.totalThickness:
            return
        tolerance = 1e-12 * max(1.0, self.totalThickness)
        for index, (zStart, zEnd) in enumerate(self.layerBounds()):
            if abs(z - zStart) <= tolerance or abs(z - zEnd) <= tolerance:
                return
            if zStart + tolerance < z < zEnd - tolerance:
                layer = self.layers[index]
                upper = _copyPatternLayer(layer, z - zStart)
                lower = _copyPatternLayer(layer, zEnd - z)
                self.layers[index : index + 1] = [upper, lower]
                return

    def _splitRange(self, zStart: float, zEnd: float, slices: int) -> None:
        if slices < 1:
            raise ValueError("slices must be at least 1")
        low, high = _zRange((zStart, zEnd))
        self._splitAt(low)
        self._splitAt(high)
        for index in range(1, int(slices)):
            self._splitAt(low + (high - low) * index / int(slices))


def _copyPatternLayer(layer: PatternLayer, thickness: float) -> PatternLayer:
    copy = PatternLayer(
        period=layer.period,
        thickness=float(thickness),
        background=0.0,
        shape=layer.pattern.shape,
        name=layer.name,
        factorization=layer.factorization,
    )
    copy.pattern.epsilon = layer.pattern.epsilon.copy()
    copy.pattern.normalField = None if layer.pattern.normalField is None else layer.pattern.normalField.copy()
    return copy


def _zRange(z: tuple[float, float]) -> tuple[float, float]:
    low = float(min(z))
    high = float(max(z))
    if high <= low:
        raise ValueError("z range must have positive thickness")
    return low, high


def _rotatedCoordinates(
    xx: np.ndarray,
    yy: np.ndarray,
    center: tuple[float, float],
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = xx - center[0]
    y = yy - center[1]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def _rectangleMask(
    xx: np.ndarray,
    yy: np.ndarray,
    size: tuple[float, float],
    center: tuple[float, float],
    angle: float,
) -> np.ndarray:
    sx = np.asarray(size[0])
    sy = np.asarray(size[1])
    if np.any(sx < 0) or np.any(sy < 0):
        raise ValueError("rectangle size values must be non-negative")
    xr, yr = _rotatedCoordinates(xx, yy, center, angle)
    return (np.abs(xr) <= sx / 2) & (np.abs(yr) <= sy / 2)


def _circleMask(
    xx: np.ndarray,
    yy: np.ndarray,
    radius: float | np.ndarray,
    center: tuple[float, float],
) -> np.ndarray:
    if np.any(np.asarray(radius) < 0):
        raise ValueError("radius must be non-negative")
    x = xx - center[0]
    y = yy - center[1]
    return x * x + y * y <= np.asarray(radius) ** 2


def _polygonPoints(vertices: Iterable[tuple[float, float]]) -> np.ndarray:
    points = np.asarray(tuple(vertices), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("polygon vertices must be a sequence of at least three (x, y) points")
    return points


def _polygonMask(xx: np.ndarray, yy: np.ndarray, points: np.ndarray) -> np.ndarray:
    inside = np.zeros(xx.shape, dtype=bool)
    x1 = points[-1, 0]
    y1 = points[-1, 1]
    for x2, y2 in points:
        crosses = ((y1 > yy) != (y2 > yy)) & (xx < (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-300) + x1)
        inside ^= crosses
        x1, y1 = x2, y2
    return inside


def _pairSize(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        return (float(value[0]), float(value[1]))
    scalar = float(value)
    return (scalar, scalar)


def _axisMask(values: np.ndarray, selector: AxisSelector) -> np.ndarray:
    if selector is None:
        return np.ones(values.shape, dtype=bool)
    if callable(selector):
        return np.asarray(selector(values), dtype=bool)
    if isinstance(selector, slice):
        start = float("-inf") if selector.start is None else float(selector.start)
        stop = float("inf") if selector.stop is None else float(selector.stop)
        return (values >= min(start, stop)) & (values <= max(start, stop))
    array = np.asarray(selector)
    if array.shape == values.shape:
        return array.astype(bool)
    if array.size == 2:
        start, stop = float(array.flat[0]), float(array.flat[1])
        return (values >= min(start, stop)) & (values <= max(start, stop))
    raise ValueError("region selectors must be None, a two-value range, a slice, a mask array, or a callable")


def _zSelected(selector: AxisSelector, zStart: float, zEnd: float) -> bool:
    if selector is None:
        return True
    center = 0.5 * (zStart + zEnd)
    if callable(selector):
        return bool(np.asarray(selector(np.asarray([center])), dtype=bool)[0])
    if isinstance(selector, slice):
        start = float("-inf") if selector.start is None else float(selector.start)
        stop = float("inf") if selector.stop is None else float(selector.stop)
        return zEnd > min(start, stop) and zStart < max(start, stop)
    array = np.asarray(selector)
    if array.size == 2:
        start, stop = float(array.flat[0]), float(array.flat[1])
        low = min(start, stop)
        high = max(start, stop)
        tolerance = 1e-12 * max(1.0, abs(high), abs(low), zEnd)
        return zStart >= low - tolerance and zEnd <= high + tolerance and zEnd > low + tolerance and zStart < high - tolerance
    raise ValueError("z selector must be None, a two-value range, a slice, or a callable")
