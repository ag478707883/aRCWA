from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .materials import IsotropicMaterial
from .analytic import (
    AnalyticAnnulus,
    AnalyticComposite,
    AnalyticDisk,
    AnalyticEllipse,
    AnalyticRectangle,
    AnalyticTerm,
)
from .types import Layer


ComplexArray = np.ndarray
FactorizationMode = Literal["auto", "standard", "normal-vector", "jones"]


def resolveEpsilon(value: complex | float | IsotropicMaterial) -> complex:
    if isinstance(value, IsotropicMaterial):
        return value.epsilon()
    array = np.asarray(value)
    if array.ndim == 2 and array.shape == (3, 3):
        tensor = np.asarray(value, dtype=complex)
        diagonal = np.diag(tensor)
        offDiagonal = tensor - np.diag(diagonal)
        if not np.allclose(offDiagonal, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("isotropic geometry accepts only diagonal 3x3 epsilon tensors")
        if not np.allclose(diagonal, diagonal[0], rtol=1e-12, atol=1e-12):
            raise ValueError("isotropic geometry accepts only scalar 3x3 epsilon tensors")
        return complex(diagonal[0])
    return complex(value)


def normalizeVectors(x: ComplexArray, y: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    length = np.sqrt(np.real(x) ** 2 + np.real(y) ** 2)
    safe = length > 1e-12
    return (
        np.where(safe, np.real(x) / np.where(safe, length, 1.0), 1.0),
        np.where(safe, np.real(y) / np.where(safe, length, 1.0), 0.0),
    )


@dataclass
class Pattern2D:
    """Sampled 2D periodic unit cell used as one RCWA layer.

    ``period`` may be non-square, e.g. ``(0.42, 0.78)`` for rectangular
    metasurfaces. Coordinates and shape sizes use the same unit as the period.
    """

    period: tuple[float, float]
    shape: tuple[int, int]
    background: complex | float | IsotropicMaterial
    name: str = ""
    supersample: int = 1
    epsilon: np.ndarray = field(init=False)
    normalField: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.period[0] <= 0 or self.period[1] <= 0:
            raise ValueError("period values must be positive")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be (ny, nx) with positive values")
        if int(self.supersample) < 1:
            raise ValueError("supersample must be at least 1")
        self.supersample = int(self.supersample)
        self.epsilon = np.full(self.shape, resolveEpsilon(self.background), dtype=complex)

    @property
    def nx(self) -> int:
        return int(self.shape[1])

    @property
    def ny(self) -> int:
        return int(self.shape[0])

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sampleCoordinates()

    def sampleCoordinates(self, supersample: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        supersample = self.supersample if supersample is None else int(supersample)
        if supersample < 1:
            raise ValueError("supersample must be at least 1")
        periodX, periodY = self.period
        yIndex, xIndex = np.mgrid[0 : self.ny * supersample, 0 : self.nx * supersample]
        xx = (xIndex + 0.5) / (self.nx * supersample) * periodX - periodX / 2
        yy = (yIndex + 0.5) / (self.ny * supersample) * periodY - periodY / 2
        return xx, yy

    def fill(
        self,
        mask: np.ndarray,
        material: complex | float | IsotropicMaterial,
        normal: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "Pattern2D":
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} does not match pattern shape {self.shape}")
        self.epsilon[mask] = resolveEpsilon(material)
        if normal is not None:
            normalX, normalY = normalizeVectors(normal[0], normal[1])
            if normalX.shape != self.shape:
                normalX = blockAverage(normalX, self.shape).real
                normalY = blockAverage(normalY, self.shape).real
                normalX, normalY = normalizeVectors(normalX, normalY)
            self.normalField = np.zeros(self.shape + (2,), dtype=float)
            self.normalField[..., 0] = normalX
            self.normalField[..., 1] = normalY
        return self

    def fillFraction(
        self,
        fraction: np.ndarray,
        material: complex | float | IsotropicMaterial,
        normal: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "Pattern2D":
        if fraction.shape != self.shape:
            raise ValueError(f"fraction shape {fraction.shape} does not match pattern shape {self.shape}")
        values = np.clip(np.asarray(fraction, dtype=float), 0.0, 1.0)
        materialValue = resolveEpsilon(material)
        self.epsilon = (1.0 - values) * self.epsilon + values * materialValue
        if normal is not None:
            normalX, normalY = normalizeVectors(normal[0], normal[1])
            if normalX.shape != self.shape:
                normalX = blockAverage(normalX, self.shape).real
                normalY = blockAverage(normalY, self.shape).real
                normalX, normalY = normalizeVectors(normalX, normalY)
            self.normalField = np.zeros(self.shape + (2,), dtype=float)
            self.normalField[..., 0] = normalX
            self.normalField[..., 1] = normalY
        return self

    def circle(
        self,
        radius: float,
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        x = xx - center[0]
        y = yy - center[1]
        mask = x**2 + y**2 <= radius**2
        return self.fillSampled(mask, material, normal=(x, y) if useNormal else None, supersample=supersample)

    def ellipse(
        self,
        radii: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        x = xx - center[0]
        y = yy - center[1]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        xRotated = cosine * x + sine * y
        yRotated = -sine * x + cosine * y
        ax = max(float(radii[0]), 1e-30)
        ay = max(float(radii[1]), 1e-30)
        mask = (xRotated / ax) ** 2 + (yRotated / ay) ** 2 <= 1.0
        normalLocalX = xRotated / (ax * ax)
        normalLocalY = yRotated / (ay * ay)
        normalX = cosine * normalLocalX - sine * normalLocalY
        normalY = sine * normalLocalX + cosine * normalLocalY
        return self.fillSampled(
            mask,
            material,
            normal=(normalX, normalY) if useNormal else None,
            supersample=supersample,
        )

    def rectangle(
        self,
        size: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        x = xx - center[0]
        y = yy - center[1]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        xRotated = cosine * x + sine * y
        yRotated = -sine * x + cosine * y
        sx = max(float(size[0]), 1e-30)
        sy = max(float(size[1]), 1e-30)
        mask = (np.abs(xRotated) <= sx / 2) & (np.abs(yRotated) <= sy / 2)
        tolerance = 1e-12 * max(1.0, self.period[0], self.period[1], sx, sy)
        spansX = sx >= self.period[0] - tolerance
        spansY = sy >= self.period[1] - tolerance
        signX = np.where(xRotated >= 0.0, 1.0, -1.0)
        signY = np.where(yRotated >= 0.0, 1.0, -1.0)
        if spansY and not spansX:
            localX = signX
            localY = np.zeros_like(signY)
        elif spansX and not spansY:
            localX = np.zeros_like(signX)
            localY = signY
        else:
            useX = np.abs(xRotated) / sx >= np.abs(yRotated) / sy
            localX = np.where(useX, signX, 0.0)
            localY = np.where(useX, 0.0, signY)
        normalX = cosine * localX - sine * localY
        normalY = sine * localX + cosine * localY
        return self.fillSampled(
            mask,
            material,
            normal=(normalX, normalY) if useNormal else None,
            supersample=supersample,
        )

    def annulus(
        self,
        innerRadius: float,
        outerRadius: float,
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        x = xx - center[0]
        y = yy - center[1]
        radius = np.sqrt(x * x + y * y)
        radiusSquared = radius**2
        mask = (radiusSquared >= innerRadius**2) & (radiusSquared <= outerRadius**2)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        sign = np.where(np.abs(radius - innerRadius) < np.abs(radius - outerRadius), -1.0, 1.0)
        return self.fillSampled(
            mask,
            material,
            normal=(sign * x / safeRadius, sign * y / safeRadius) if useNormal else None,
            supersample=supersample,
        )

    def cross(
        self,
        armLengths: tuple[float, float],
        armWidths: tuple[float, float],
        material: complex | float | IsotropicMaterial,
        center: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        x = xx - center[0]
        y = yy - center[1]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        xRotated = cosine * x + sine * y
        yRotated = -sine * x + cosine * y
        horizontalMask = (np.abs(xRotated) <= armLengths[0] / 2) & (np.abs(yRotated) <= armWidths[0] / 2)
        verticalMask = (np.abs(xRotated) <= armWidths[1] / 2) & (np.abs(yRotated) <= armLengths[1] / 2)
        return self.fillSampled(
            horizontalMask | verticalMask,
            material,
            normal=(xRotated, yRotated) if useNormal else None,
            supersample=supersample,
        )

    def stripes(
        self,
        fillFraction: float,
        material: complex | float | IsotropicMaterial,
        axis: str = "x",
        center: float = 0.0,
    ) -> "Pattern2D":
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
        material: complex | float | IsotropicMaterial,
        useNormal: bool = True,
        supersample: int | None = None,
    ) -> "Pattern2D":
        points = np.asarray(tuple(vertices), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("polygon vertices must be a sequence of at least three (x, y) points")
        supersample = self.supersample if supersample is None else int(supersample)
        xx, yy = self.sampleCoordinates(supersample)
        mask = polygonMask(xx, yy, points)
        normal = polygonNormalField(xx, yy, points) if useNormal else None
        return self.fillSampled(mask, material, normal=normal, supersample=supersample)

    def toLayer(
        self,
        thickness: float,
        name: str | None = None,
        factorization: FactorizationMode = "auto",
    ) -> Layer:
        normal = self.normalField if factorization in ("auto", "normal-vector", "jones") else None
        return Layer(
            thickness=thickness,
            epsilon=self.epsilon.copy(),
            name=name if name is not None else self.name,
            normalField=None if normal is None else normal.copy(),
            factorization=factorization,
            sampleShape=self.shape,
        )

    def fillSampled(
        self,
        mask: np.ndarray,
        material: complex | float | IsotropicMaterial,
        *,
        normal: tuple[np.ndarray, np.ndarray] | None,
        supersample: int,
    ) -> "Pattern2D":
        if supersample == 1:
            return self.fill(mask, material, normal=normal)
        fraction = blockAverage(mask.astype(float), self.shape).real
        averagedNormal = None
        if normal is not None:
            averagedNormal = (
                blockAverage(np.asarray(normal[0]) * mask, self.shape).real,
                blockAverage(np.asarray(normal[1]) * mask, self.shape).real,
            )
        return self.fillFraction(fraction, material, normal=averagedNormal)


def polygonMask(xx: ComplexArray, yy: ComplexArray, points: ComplexArray) -> ComplexArray:
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


def blockAverage(values: ComplexArray, shape: tuple[int, int]) -> ComplexArray:
    array = np.asarray(values)
    ny, nx = shape
    if array.shape == shape:
        return array
    if array.ndim < 2 or array.shape[0] % ny != 0 or array.shape[1] % nx != 0:
        raise ValueError(f"cannot block-average shape {array.shape} to {shape}")
    sy = array.shape[0] // ny
    sx = array.shape[1] // nx
    trailing = array.shape[2:]
    return array.reshape(ny, sy, nx, sx, *trailing).mean(axis=(1, 3))


def polygonNormalField(xx: ComplexArray, yy: ComplexArray, points: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
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
    return normalizeVectors(bestX, bestY)


def homogeneousLayer(thickness: float, material: object, name: str = "", factorization: FactorizationMode = "auto") -> object:
    if callable(material):
        from .simulation import LayerSpec

        return LayerSpec(thickness=thickness, epsilon=material, name=name, factorization=factorization)
    return Layer(thickness=thickness, epsilon=resolveEpsilon(material), name=name, factorization=factorization)


def photonicCrystalSlab(
    period: tuple[float, float],
    thickness: float,
    slab: complex | float | IsotropicMaterial,
    hole: complex | float | IsotropicMaterial,
    radius: float,
    shape: tuple[int, int] = (96, 96),
    name: str = "photonic crystal slab",
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    jonesResolution: int = 512,
) -> Layer:
    if analytic:
        return Layer(
            thickness=thickness,
            epsilon=AnalyticDisk(
                period=period,
                radius=radius,
                background=resolveEpsilon(slab),
                inclusion=resolveEpsilon(hole),
                factorization="jones" if factorization in ("auto", "jones") else "analytic",
                jonesResolution=jonesResolution,
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=slab, name=name)
    pattern.circle(radius=radius, material=hole)
    return pattern.toLayer(thickness, factorization=factorization)


def analyticCircularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    jonesResolution: int = 512,
    name: str = "analytic circular post layer",
) -> Layer:
    return Layer(
        thickness=thickness,
        epsilon=AnalyticDisk(
            period=period,
            radius=radius,
            background=resolveEpsilon(background),
            inclusion=resolveEpsilon(post),
            center=center,
            factorization="jones" if factorization in ("auto", "jones") else "analytic",
            jonesResolution=jonesResolution,
        ),
        name=name,
        factorization=factorization,
    )


def circularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    radius: float,
    shape: tuple[int, int] = (96, 96),
    name: str = "circular post layer",
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    jonesResolution: int = 512,
) -> Layer:
    if analytic:
        return analyticCircularPostLayer(
            period=period,
            thickness=thickness,
            background=background,
            post=post,
            radius=radius,
            center=center,
            factorization=factorization,
            jonesResolution=jonesResolution,
            name=name,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.circle(radius=radius, material=post, center=center)
    return pattern.toLayer(thickness, factorization=factorization)


def ellipticalPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    radii: tuple[float, float],
    angle: float = 0.0,
    shape: tuple[int, int] = (96, 96),
    name: str = "elliptical post layer",
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    jonesResolution: int = 512,
) -> Layer:
    if analytic:
        return Layer(
            thickness=thickness,
            epsilon=AnalyticEllipse(
                period=period,
                radii=radii,
                background=resolveEpsilon(background),
                inclusion=resolveEpsilon(post),
                center=center,
                angle=angle,
                factorization="jones" if factorization in ("auto", "jones") else "analytic",
                jonesResolution=jonesResolution,
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.ellipse(radii=radii, angle=angle, material=post, center=center)
    return pattern.toLayer(thickness, factorization=factorization)


def rectangularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    size: tuple[float, float],
    angle: float = 0.0,
    shape: tuple[int, int] = (96, 96),
    name: str = "rectangular post layer",
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    jonesResolution: int = 512,
) -> Layer:
    if analytic:
        return Layer(
            thickness=thickness,
            epsilon=AnalyticRectangle(
                period=period,
                size=size,
                background=resolveEpsilon(background),
                inclusion=resolveEpsilon(post),
                center=center,
                angle=angle,
                factorization="jones" if factorization in ("auto", "jones") else "analytic",
                jonesResolution=jonesResolution,
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.rectangle(size=size, angle=angle, material=post, center=center)
    return pattern.toLayer(thickness, factorization=factorization)


def annularPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    ring: complex | float | IsotropicMaterial,
    innerRadius: float,
    outerRadius: float,
    shape: tuple[int, int] = (96, 96),
    name: str = "annular post layer",
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
    holeMaterial: complex | float | IsotropicMaterial | None = None,
    jonesResolution: int = 512,
) -> Layer:
    if analytic:
        return Layer(
            thickness=thickness,
            epsilon=AnalyticAnnulus(
                period=period,
                innerRadius=innerRadius,
                outerRadius=outerRadius,
                background=resolveEpsilon(background),
                ring=resolveEpsilon(ring),
                hole=None if holeMaterial is None else resolveEpsilon(holeMaterial),
                center=center,
                factorization="jones" if factorization in ("auto", "jones") else "analytic",
                jonesResolution=jonesResolution,
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.annulus(innerRadius, outerRadius, ring, center=center)
    if holeMaterial is not None:
        pattern.circle(innerRadius, holeMaterial, center=center, useNormal=False)
    return pattern.toLayer(thickness, factorization=factorization)


def rectangularHollowPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    size: tuple[float, float],
    holeRadius: float,
    shape: tuple[int, int] = (96, 96),
    name: str = "rectangular hollow post layer",
    holeMaterial: complex | float | IsotropicMaterial | None = None,
    angle: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
) -> Layer:
    if holeRadius < 0:
        raise ValueError("holeRadius must be non-negative")
    if analytic:
        if holeRadius > min(size[0], size[1]) / 2 + 1e-15:
            raise ValueError("analytic rectangular hollow requires the circular hole to lie inside the rectangle")
        if factorization in ("normal-vector", "jones"):
            raise ValueError("analytic rectangular hollow layers support 'auto' or 'standard' factorization")
        backgroundValue = resolveEpsilon(background)
        postValue = resolveEpsilon(post)
        holeValue = backgroundValue if holeMaterial is None else resolveEpsilon(holeMaterial)
        rectangle = AnalyticRectangle(
            period=period,
            size=size,
            background=0.0,
            inclusion=1.0,
            center=center,
            angle=angle,
        )
        terms = [AnalyticTerm(rectangle, postValue - backgroundValue)]
        if holeRadius > 0:
            hole = AnalyticDisk(
                period=period,
                radius=holeRadius,
                background=0.0,
                inclusion=1.0,
                center=center,
            )
            terms.append(AnalyticTerm(hole, holeValue - postValue))
        return Layer(
            thickness=thickness,
            epsilon=AnalyticComposite(
                period=period,
                background=backgroundValue,
                terms=tuple(terms),
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.rectangle(size, post, center=center, angle=angle, useNormal=False)
    pattern.circle(
        holeRadius,
        background if holeMaterial is None else holeMaterial,
        center=center,
        useNormal=False,
    )
    if factorization in ("auto", "normal-vector", "jones"):
        xx, yy = pattern.coordinates()
        pattern.normalField = rectangularHollowNormalField(xx, yy, size, holeRadius, center, angle)
    return pattern.toLayer(thickness, factorization=factorization)


def crossPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    armLengths: tuple[float, float],
    armWidths: tuple[float, float],
    shape: tuple[int, int] = (96, 96),
    name: str = "cross post layer",
    center: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    factorization: FactorizationMode = "auto",
    analytic: bool = False,
) -> Layer:
    if min(*armLengths, *armWidths) <= 0:
        raise ValueError("cross arm lengths and widths must be positive")
    if analytic:
        if factorization in ("normal-vector", "jones"):
            raise ValueError("analytic cross layers support 'auto' or 'standard' factorization")
        backgroundValue = resolveEpsilon(background)
        postValue = resolveEpsilon(post)
        horizontal = AnalyticRectangle(
            period=period,
            size=(armLengths[0], armWidths[0]),
            background=0.0,
            inclusion=1.0,
            center=center,
            angle=angle,
        )
        vertical = AnalyticRectangle(
            period=period,
            size=(armWidths[1], armLengths[1]),
            background=0.0,
            inclusion=1.0,
            center=center,
            angle=angle,
        )
        overlap = AnalyticRectangle(
            period=period,
            size=(armWidths[1], armWidths[0]),
            background=0.0,
            inclusion=1.0,
            center=center,
            angle=angle,
        )
        delta = postValue - backgroundValue
        return Layer(
            thickness=thickness,
            epsilon=AnalyticComposite(
                period=period,
                background=backgroundValue,
                terms=(
                    AnalyticTerm(horizontal, delta),
                    AnalyticTerm(vertical, delta),
                    AnalyticTerm(overlap, -delta),
                ),
            ),
            name=name,
            factorization=factorization,
        )
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.cross(armLengths, armWidths, post, center=center, angle=angle)
    return pattern.toLayer(thickness, factorization=factorization)


def polygonPostLayer(
    period: tuple[float, float],
    thickness: float,
    background: complex | float | IsotropicMaterial,
    post: complex | float | IsotropicMaterial,
    vertices: Iterable[tuple[float, float]],
    shape: tuple[int, int] = (96, 96),
    name: str = "polygon post layer",
    factorization: FactorizationMode = "auto",
) -> Layer:
    pattern = Pattern2D(period=period, shape=shape, background=background, name=name)
    pattern.polygon(vertices, post)
    return pattern.toLayer(thickness, factorization=factorization)


def rectangularHollowNormalField(
    xx: ComplexArray,
    yy: ComplexArray,
    size: tuple[float, float],
    holeRadius: float,
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    x = xx - center[0]
    y = yy - center[1]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    xRotated = cosine * x + sine * y
    yRotated = -sine * x + cosine * y
    sx = max(float(size[0]), 1e-30)
    sy = max(float(size[1]), 1e-30)

    sideXCloser = np.abs(np.abs(xRotated) - sx / 2) <= np.abs(np.abs(yRotated) - sy / 2)
    localNx = np.where(sideXCloser, np.sign(xRotated), 0.0)
    localNy = np.where(sideXCloser, 0.0, np.sign(yRotated))
    rectNx = cosine * localNx - sine * localNy
    rectNy = sine * localNx + cosine * localNy

    radius = np.sqrt(x * x + y * y)
    safeRadius = np.where(radius > 1e-12, radius, 1.0)
    holeNx = x / safeRadius
    holeNy = y / safeRadius

    rectDistance = np.minimum(np.abs(np.abs(xRotated) - sx / 2), np.abs(np.abs(yRotated) - sy / 2))
    holeDistance = np.abs(radius - holeRadius)
    useHoleNormal = holeDistance < rectDistance
    nx = np.where(useHoleNormal, holeNx, rectNx)
    ny = np.where(useHoleNormal, holeNy, rectNy)

    normal = np.zeros(xx.shape + (2,), dtype=float)
    normal[..., 0], normal[..., 1] = normalizeVectors(nx, ny)
    return normal


def stack(*layers: Layer | Iterable[Layer]) -> list[Layer]:
    result: list[Layer] = []
    for item in layers:
        if isinstance(item, Layer):
            result.append(item)
        else:
            result.extend(item)
    return result
