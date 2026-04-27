from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .analytic import AnalyticDisk
from .solver import Layer


ComplexArray = np.ndarray
ShapeKind = Literal["circle", "ellipse", "rectangle"]
FactorizationMode = Literal["auto", "standard", "normal-vector", "jones"]


def _asTensor(value: complex | float | ComplexArray) -> ComplexArray:
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        return complex(array.item()) * np.eye(3, dtype=complex)
    if array.shape == (3, 3):
        return array.copy()
    raise ValueError("material must be a scalar or a (3, 3) tensor")


def _isTensorMaterial(value: complex | float | ComplexArray) -> bool:
    array = np.asarray(value)
    return bool(array.shape == (3, 3))


def _normalizeVectors(x: ComplexArray, y: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    length = np.sqrt(np.real(x) ** 2 + np.real(y) ** 2)
    safe = length > 1e-12
    return (
        np.where(safe, np.real(x) / np.where(safe, length, 1.0), 1.0),
        np.where(safe, np.real(y) / np.where(safe, length, 1.0), 0.0),
    )


def _scalarGrid(shape: tuple[int, int], value: complex | float) -> ComplexArray:
    return np.full(shape, complex(value), dtype=complex)


def _tensorGrid(shape: tuple[int, int], value: complex | float | ComplexArray) -> ComplexArray:
    tensor = _asTensor(value)
    grid = np.zeros(shape + (3, 3), dtype=complex)
    grid[...] = tensor
    return grid


@dataclass
class SampledPattern:
    """Sampled x-y unit cell for a scalar or tensor anisotropic RCWA layer."""

    period: tuple[float, float]
    shape: tuple[int, int]
    background: complex | float | ComplexArray
    name: str = ""
    epsilon: ComplexArray = field(init=False)
    normalField: ComplexArray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.period[0] <= 0 or self.period[1] <= 0:
            raise ValueError("period values must be positive")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be (ny, nx) with positive values")
        if _isTensorMaterial(self.background):
            self.epsilon = _tensorGrid(self.shape, self.background)
        else:
            self.epsilon = _scalarGrid(self.shape, complex(self.background))

    @property
    def nx(self) -> int:
        return int(self.shape[1])

    @property
    def ny(self) -> int:
        return int(self.shape[0])

    def coordinates(self) -> tuple[ComplexArray, ComplexArray]:
        periodX, periodY = self.period
        yIndex, xIndex = np.mgrid[0 : self.ny, 0 : self.nx]
        xx = (xIndex + 0.5) / self.nx * periodX - periodX / 2
        yy = (yIndex + 0.5) / self.ny * periodY - periodY / 2
        return xx, yy

    def fill(
        self,
        mask: ComplexArray,
        material: complex | float | ComplexArray,
        normal: tuple[ComplexArray, ComplexArray] | None = None,
    ) -> "SampledPattern":
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match pattern shape {self.shape}")

        materialIsTensor = _isTensorMaterial(material)
        if self.epsilon.ndim == 2 and materialIsTensor:
            self.epsilon = _tensorGrid(self.shape, self.background)

        if self.epsilon.ndim == 4:
            self.epsilon[mask] = _asTensor(material)
        else:
            self.epsilon[mask] = complex(material)

        if normal is not None:
            normalX, normalY = _normalizeVectors(normal[0], normal[1])
            self.normalField = np.zeros(self.shape + (2,), dtype=float)
            self.normalField[..., 0] = normalX
            self.normalField[..., 1] = normalY
        return self

    def circle(
        self,
        radius: float,
        material: complex | float | ComplexArray,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
    ) -> "SampledPattern":
        if radius < 0:
            raise ValueError("radius must be non-negative")
        xx, yy = self.coordinates()
        x = xx - center[0]
        y = yy - center[1]
        mask = x * x + y * y <= radius * radius
        return self.fill(mask, material, normal=(x, y) if useNormal else None)

    def ellipse(
        self,
        radii: tuple[float, float],
        material: complex | float | ComplexArray,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "SampledPattern":
        if radii[0] < 0 or radii[1] < 0:
            raise ValueError("ellipse radii must be non-negative")
        xx, yy = self.coordinates()
        x = xx - center[0]
        y = yy - center[1]
        c = np.cos(angle)
        s = np.sin(angle)
        xr = c * x + s * y
        yr = -s * x + c * y
        ax = max(float(radii[0]), 1e-30)
        ay = max(float(radii[1]), 1e-30)
        mask = (xr / ax) ** 2 + (yr / ay) ** 2 <= 1.0
        normalLocalX = xr / (ax * ax)
        normalLocalY = yr / (ay * ay)
        normalX = c * normalLocalX - s * normalLocalY
        normalY = s * normalLocalX + c * normalLocalY
        return self.fill(mask, material, normal=(normalX, normalY) if useNormal else None)

    def rectangle(
        self,
        size: tuple[float, float],
        material: complex | float | ComplexArray,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "SampledPattern":
        if size[0] < 0 or size[1] < 0:
            raise ValueError("rectangle size values must be non-negative")
        xx, yy = self.coordinates()
        x = xx - center[0]
        y = yy - center[1]
        c = np.cos(angle)
        s = np.sin(angle)
        xr = c * x + s * y
        yr = -s * x + c * y
        sx = max(float(size[0]), 1e-30)
        sy = max(float(size[1]), 1e-30)
        mask = (np.abs(xr) <= sx / 2) & (np.abs(yr) <= sy / 2)

        localX = np.where(np.abs(xr) / sx >= np.abs(yr) / sy, np.sign(xr), 0.0)
        localY = np.where(np.abs(xr) / sx >= np.abs(yr) / sy, 0.0, np.sign(yr))
        normalX = c * localX - s * localY
        normalY = s * localX + c * localY
        return self.fill(mask, material, normal=(normalX, normalY) if useNormal else None)

    def annulus(
        self,
        innerRadius: float,
        outerRadius: float,
        material: complex | float | ComplexArray,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
    ) -> "SampledPattern":
        if innerRadius < 0 or outerRadius < 0:
            raise ValueError("annulus radii must be non-negative")
        if outerRadius < innerRadius:
            raise ValueError("outerRadius must be greater than or equal to innerRadius")
        xx, yy = self.coordinates()
        x = xx - center[0]
        y = yy - center[1]
        radius = np.sqrt(x * x + y * y)
        mask = (radius >= innerRadius) & (radius <= outerRadius)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        sign = np.where(np.abs(radius - innerRadius) < np.abs(radius - outerRadius), -1.0, 1.0)
        return self.fill(mask, material, normal=(sign * x / safeRadius, sign * y / safeRadius) if useNormal else None)

    def cross(
        self,
        armLengths: tuple[float, float],
        armWidths: tuple[float, float],
        material: complex | float | ComplexArray,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
    ) -> "SampledPattern":
        if min(*armLengths, *armWidths) <= 0:
            raise ValueError("cross arm lengths and widths must be positive")
        xx, yy = self.coordinates()
        x = xx - center[0]
        y = yy - center[1]
        c = np.cos(angle)
        s = np.sin(angle)
        xr = c * x + s * y
        yr = -s * x + c * y
        horizontal = (np.abs(xr) <= armLengths[0] / 2) & (np.abs(yr) <= armWidths[0] / 2)
        vertical = (np.abs(xr) <= armWidths[1] / 2) & (np.abs(yr) <= armLengths[1] / 2)
        return self.fill(horizontal | vertical, material, normal=(xr, yr) if useNormal else None)

    def stripes(
        self,
        fillFraction: float,
        material: complex | float | ComplexArray,
        axis: str = "x",
        center: float = 0.0,
    ) -> "SampledPattern":
        if not 0 <= fillFraction <= 1:
            raise ValueError("fillFraction must be between 0 and 1")
        xx, yy = self.coordinates()
        if axis == "x":
            width = self.period[0] * fillFraction
            mask = np.abs(xx - center) <= width / 2
        elif axis == "y":
            width = self.period[1] * fillFraction
            mask = np.abs(yy - center) <= width / 2
        else:
            raise ValueError("axis must be 'x' or 'y'")
        return self.fill(mask, material)

    def polygon(
        self,
        vertices: Iterable[tuple[float, float]],
        material: complex | float | ComplexArray,
        useNormal: bool = True,
    ) -> "SampledPattern":
        points = np.asarray(tuple(vertices), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("polygon vertices must be a sequence of at least three (x, y) points")
        xx, yy = self.coordinates()
        mask = _polygonMask(xx, yy, points)
        normal = _polygonNormalField(xx, yy, points) if useNormal else None
        return self.fill(mask, material, normal=normal)

    def toLayer(
        self,
        thickness: float,
        name: str | None = None,
        factorization: FactorizationMode = "auto",
    ) -> Layer:
        normal = self.normalField if factorization in ("auto", "normal-vector", "jones") and self.epsilon.ndim == 2 else None
        return Layer(
            thickness=thickness,
            epsilon=self.epsilon.copy(),
            normalField=None if normal is None else normal.copy(),
            factorization=factorization,
            name=name if name is not None else self.name,
        )


def _polygonMask(xx: ComplexArray, yy: ComplexArray, points: ComplexArray) -> ComplexArray:
    x = xx
    y = yy
    inside = np.zeros(x.shape, dtype=bool)
    x1 = points[-1, 0]
    y1 = points[-1, 1]
    for x2, y2 in points:
        crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1)
        inside ^= crosses
        x1, y1 = x2, y2
    return inside


def _polygonNormalField(xx: ComplexArray, yy: ComplexArray, points: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    bestDistance = np.full(xx.shape, np.inf, dtype=float)
    bestX = np.ones(xx.shape, dtype=float)
    bestY = np.zeros(xx.shape, dtype=float)
    for index in range(points.shape[0]):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % points.shape[0]]
        edgeX = x2 - x1
        edgeY = y2 - y1
        lengthSquared = edgeX * edgeX + edgeY * edgeY
        if lengthSquared <= 1e-30:
            continue
        t = np.clip(((xx - x1) * edgeX + (yy - y1) * edgeY) / lengthSquared, 0.0, 1.0)
        closestX = x1 + t * edgeX
        closestY = y1 + t * edgeY
        distance = (xx - closestX) ** 2 + (yy - closestY) ** 2
        update = distance < bestDistance
        normalX = edgeY
        normalY = -edgeX
        bestDistance = np.where(update, distance, bestDistance)
        bestX = np.where(update, normalX, bestX)
        bestY = np.where(update, normalY, bestY)
    return _normalizeVectors(bestX, bestY)


def analyticCircularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float,
    post: complex | float,
    radius: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    jonesResolution: int = 512,
    name: str = "analytic circular post",
) -> Layer:
    if factorization in ("auto", "jones", "normal-vector"):
        diskFactorization = "jones"
    elif factorization == "standard":
        diskFactorization = "analytic"
    else:
        raise ValueError("factorization must be 'auto', 'standard', 'normal-vector', or 'jones'")
    return Layer(
        thickness=thickness,
        epsilon=AnalyticDisk(
            period=period,
            radius=radius,
            background=background,
            inclusion=post,
            center=center,
            factorization=diskFactorization,
            jonesResolution=jonesResolution,
        ),
        factorization=factorization,
        name=name,
    )


def circularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    radius: float,
    *,
    shape: tuple[int, int] = (128, 128),
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    jonesResolution: int = 512,
    name: str = "circular post",
) -> Layer:
    if analytic and not (_isTensorMaterial(background) or _isTensorMaterial(post)):
        return analyticCircularPostLayer(
            period,
            thickness,
            complex(background),
            complex(post),
            radius,
            center=center,
            factorization=factorization,
            jonesResolution=jonesResolution,
            name=name,
        )
    pattern = SampledPattern(period=period, shape=shape, background=background, name=name)
    return pattern.circle(radius, post, center=center).toLayer(thickness, factorization=factorization)


def ellipticalPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    radii: tuple[float, float],
    *,
    angle: float = 0.0,
    shape: tuple[int, int] = (128, 128),
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    name: str = "elliptical post",
) -> Layer:
    pattern = SampledPattern(period=period, shape=shape, background=background, name=name)
    return pattern.ellipse(radii, post, center=center, angle=angle).toLayer(thickness, factorization=factorization)


def rectangularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    size: tuple[float, float],
    *,
    angle: float = 0.0,
    shape: tuple[int, int] = (128, 128),
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    name: str = "rectangular post",
) -> Layer:
    pattern = SampledPattern(period=period, shape=shape, background=background, name=name)
    return pattern.rectangle(size, post, center=center, angle=angle).toLayer(thickness, factorization=factorization)


def rectangularHollowPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    size: tuple[float, float],
    holeRadius: float,
    *,
    holeMaterial: complex | float | ComplexArray | None = None,
    angle: float = 0.0,
    shape: tuple[int, int] = (128, 128),
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    name: str = "rectangular hollow post",
) -> Layer:
    """Sample a rectangular post with a circular through-hole."""

    if holeRadius < 0:
        raise ValueError("holeRadius must be non-negative")
    pattern = SampledPattern(period=period, shape=shape, background=background, name=name)
    pattern.rectangle(size, post, center=center, angle=angle, useNormal=False)
    pattern.circle(
        holeRadius,
        background if holeMaterial is None else holeMaterial,
        center=center,
        useNormal=False,
    )
    if factorization in ("auto", "normal-vector", "jones") and pattern.epsilon.ndim == 2:
        xx, yy = pattern.coordinates()
        pattern.normalField = _rectangularHollowNormalField(xx, yy, size, holeRadius, center, angle)
    return pattern.toLayer(thickness, factorization=factorization)


def _rectangularHollowNormalField(
    xx: ComplexArray,
    yy: ComplexArray,
    size: tuple[float, float],
    holeRadius: float,
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    x = xx - center[0]
    y = yy - center[1]
    c = np.cos(angle)
    s = np.sin(angle)
    xr = c * x + s * y
    yr = -s * x + c * y
    sx = max(float(size[0]), 1e-30)
    sy = max(float(size[1]), 1e-30)

    sideXCloser = np.abs(np.abs(xr) - sx / 2) <= np.abs(np.abs(yr) - sy / 2)
    localNx = np.where(sideXCloser, np.sign(xr), 0.0)
    localNy = np.where(sideXCloser, 0.0, np.sign(yr))
    rectNx = c * localNx - s * localNy
    rectNy = s * localNx + c * localNy

    radius = np.sqrt(x * x + y * y)
    safeRadius = np.where(radius > 1e-12, radius, 1.0)
    holeNx = x / safeRadius
    holeNy = y / safeRadius

    rectDistance = np.minimum(np.abs(np.abs(xr) - sx / 2), np.abs(np.abs(yr) - sy / 2))
    holeDistance = np.abs(radius - holeRadius)
    useHoleNormal = holeDistance < rectDistance
    nx = np.where(useHoleNormal, holeNx, rectNx)
    ny = np.where(useHoleNormal, holeNy, rectNy)

    normal = np.zeros(xx.shape + (2,), dtype=float)
    normal[..., 0], normal[..., 1] = _normalizeVectors(nx, ny)
    return normal


def polygonPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    vertices: Iterable[tuple[float, float]],
    *,
    shape: tuple[int, int] = (128, 128),
    factorization: FactorizationMode = "auto",
    name: str = "polygon post",
) -> Layer:
    pattern = SampledPattern(period=period, shape=shape, background=background, name=name)
    return pattern.polygon(vertices, post).toLayer(thickness, factorization=factorization)


def slicedTaperStack(
    period: tuple[float, float],
    height: float,
    background: complex | float | ComplexArray,
    post: complex | float | ComplexArray,
    bottomSize: float | tuple[float, float],
    topSize: float | tuple[float, float],
    *,
    kind: ShapeKind = "rectangle",
    slices: int = 20,
    angle: float = 0.0,
    shape: tuple[int, int] = (128, 128),
    factorization: FactorizationMode = "auto",
    name: str = "taper",
) -> list[Layer]:
    """Approximate a z-varying post by a stack of constant cross sections."""

    if slices <= 0:
        raise ValueError("slices must be positive")
    if height <= 0:
        raise ValueError("height must be positive")

    layers: list[Layer] = []
    dz = height / slices
    for index in range(slices):
        fraction = (index + 0.5) / slices
        size = _lerpSize(bottomSize, topSize, fraction)
        layerName = f"{name} {index + 1}/{slices}"
        if kind == "circle":
            layers.append(
                circularPostLayer(
                    period,
                    dz,
                    background,
                    post,
                    float(size),
                    shape=shape,
                    factorization=factorization,
                    name=layerName,
                )
            )
        elif kind == "ellipse":
            layers.append(
                ellipticalPostLayer(
                    period,
                    dz,
                    background,
                    post,
                    _pairSize(size),
                    angle=angle,
                    shape=shape,
                    factorization=factorization,
                    name=layerName,
                )
            )
        elif kind == "rectangle":
            layers.append(
                rectangularPostLayer(
                    period,
                    dz,
                    background,
                    post,
                    _pairSize(size),
                    angle=angle,
                    shape=shape,
                    factorization=factorization,
                    name=layerName,
                )
            )
        else:
            raise ValueError("slicedTaperStack supports kind='circle', 'ellipse', or 'rectangle'")
    return layers


def stack(*layers: Layer | Iterable[Layer]) -> list[Layer]:
    result: list[Layer] = []
    for item in layers:
        if isinstance(item, Layer):
            result.append(item)
        else:
            result.extend(item)
    return result


def _lerpSize(
    bottom: float | tuple[float, float],
    top: float | tuple[float, float],
    fraction: float,
) -> float | tuple[float, float]:
    if isinstance(bottom, tuple) or isinstance(top, tuple):
        b = _pairSize(bottom)
        t = _pairSize(top)
        return (b[0] + (t[0] - b[0]) * fraction, b[1] + (t[1] - b[1]) * fraction)
    return float(bottom) + (float(top) - float(bottom)) * fraction


def _pairSize(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        return (float(value[0]), float(value[1]))
    scalar = float(value)
    return (scalar, scalar)
