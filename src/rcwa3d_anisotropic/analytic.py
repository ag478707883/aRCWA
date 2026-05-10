from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from .fourier import epsilonConvolutionMatrix


ComplexArray = np.ndarray


def validatePeriod(period: tuple[float, float]) -> None:
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")


def validateResolution(value: int) -> None:
    if int(value) < 8:
        raise ValueError("normal-vector resolution must be at least 8")


def validateAnalyticFactorization(value: str) -> None:
    if value not in ("analytic", "normal-vector"):
        raise ValueError("analytic factorization must be 'analytic' or 'normal-vector'")


def scalarMaterial(value: complex | float | ComplexArray) -> complex:
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        return complex(array.item())
    if array.shape == (3, 3):
        diagonal = np.diag(array)
        offDiagonal = array - np.diag(diagonal)
        if np.allclose(offDiagonal, 0.0, rtol=0.0, atol=1e-12) and np.allclose(
            diagonal,
            diagonal[0],
            rtol=1e-12,
            atol=1e-12,
        ):
            return complex(diagonal[0])
    raise ValueError("analytic scalar geometry requires scalar materials or scalar 3x3 tensors")


def sinc(value: ComplexArray) -> ComplexArray:
    result = np.ones_like(value, dtype=complex)
    nonzero = np.abs(value) > 1e-14
    result[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return result


def besselRatio(argument: ComplexArray) -> ComplexArray:
    ratio = np.empty_like(argument, dtype=complex)
    small = np.abs(argument) < 1e-5
    x = argument[small]
    ratio[small] = 1.0 - x * x / 8.0 + x**4 / 192.0 - x**6 / 9216.0
    ratio[~small] = 2.0 * j1(argument[~small]) / argument[~small]
    return ratio


def phase(
    harmonics: object,
    period: tuple[float, float],
    center: tuple[float, float],
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    dmx = getattr(harmonics, "deltaMx", None)
    dmy = getattr(harmonics, "deltaMy", None)
    if dmx is None or dmy is None:
        mx = getattr(harmonics, "mx")
        my = getattr(harmonics, "my")
        dmx = mx[:, None] - mx[None, :]
        dmy = my[:, None] - my[None, :]
    gx = 2.0 * np.pi * dmx / period[0]
    gy = 2.0 * np.pi * dmy / period[1]
    return gx, gy, np.exp(-1j * (gx * center[0] + gy * center[1]))


def rotatedComponents(
    x: ComplexArray,
    y: ComplexArray,
    angle: float,
) -> tuple[ComplexArray, ComplexArray]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def rectangleIndicator(
    harmonics: object,
    period: tuple[float, float],
    size: tuple[float, float],
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    gx, gy, phaseFactor = phase(harmonics, period, center)
    gxLocal, gyLocal = rotatedComponents(gx, gy, angle)
    fill = size[0] * size[1] / (period[0] * period[1])
    return fill * sinc(gxLocal * size[0] / 2.0) * sinc(gyLocal * size[1] / 2.0) * phaseFactor


def diskIndicator(
    harmonics: object,
    period: tuple[float, float],
    radius: float,
    center: tuple[float, float],
) -> ComplexArray:
    gx, gy, phaseFactor = phase(harmonics, period, center)
    argument = np.sqrt(gx * gx + gy * gy) * radius
    fill = np.pi * radius * radius / (period[0] * period[1])
    return fill * besselRatio(argument) * phaseFactor


def ellipseIndicator(
    harmonics: object,
    period: tuple[float, float],
    radii: tuple[float, float],
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    gx, gy, phaseFactor = phase(harmonics, period, center)
    gxLocal, gyLocal = rotatedComponents(gx, gy, angle)
    argument = np.sqrt((gxLocal * radii[0]) ** 2 + (gyLocal * radii[1]) ** 2)
    fill = np.pi * radii[0] * radii[1] / (period[0] * period[1])
    return fill * besselRatio(argument) * phaseFactor


def twoMaterialConvolution(
    indicator: ComplexArray,
    size: int,
    background: complex,
    inclusion: complex,
) -> ComplexArray:
    return complex(background) * np.eye(size, dtype=complex) + (complex(inclusion) - complex(background)) * indicator


def reciprocal(value: complex) -> complex:
    return 1.0 / complex(value)


def sampleCoordinates(
    period: tuple[float, float],
    center: tuple[float, float],
    samples: int,
) -> tuple[ComplexArray, ComplexArray]:
    periodX, periodY = period
    yIndex, xIndex = np.mgrid[0:samples, 0:samples]
    xx = (xIndex + 0.5) / samples * periodX - periodX / 2.0 - center[0]
    yy = (yIndex + 0.5) / samples * periodY - periodY / 2.0 - center[1]
    xx = (xx + periodX / 2.0) % periodX - periodX / 2.0
    yy = (yy + periodY / 2.0) % periodY - periodY / 2.0
    return xx, yy


def localCoordinates(
    period: tuple[float, float],
    center: tuple[float, float],
    angle: float,
    samples: int,
) -> tuple[ComplexArray, ComplexArray, float, float]:
    xx, yy = sampleCoordinates(period, center, samples)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return cosine * xx + sine * yy, -sine * xx + cosine * yy, cosine, sine


def normalizeVectorField(x: ComplexArray, y: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    length = np.sqrt(np.real(x) ** 2 + np.real(y) ** 2)
    safe = length > 1e-12
    return (
        np.where(safe, np.real(x) / np.where(safe, length, 1.0), 1.0).astype(complex),
        np.where(safe, np.real(y) / np.where(safe, length, 1.0), 0.0).astype(complex),
    )


def rotateLocalVector(
    xLocal: ComplexArray,
    yLocal: ComplexArray,
    cosine: float,
    sine: float,
) -> tuple[ComplexArray, ComplexArray]:
    return cosine * xLocal - sine * yLocal, sine * xLocal + cosine * yLocal


def normalVectorMatricesFromField(
    normalX: ComplexArray,
    normalY: ComplexArray,
    harmonics: object,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    tangentX = -normalY
    tangentY = normalX
    return (
        epsilonConvolutionMatrix(normalX, harmonics),
        epsilonConvolutionMatrix(normalY, harmonics),
        epsilonConvolutionMatrix(tangentX, harmonics),
        epsilonConvolutionMatrix(tangentY, harmonics),
    )


def alignedRectangleInvariantAxes(
    period: tuple[float, float],
    size: tuple[float, float],
    angle: float,
) -> tuple[str, ...]:
    angleMod = float(angle) % np.pi
    tolerance = 1e-12 * max(1.0, period[0], period[1], size[0], size[1])
    axes: list[str] = []
    if min(abs(angleMod), abs(angleMod - np.pi)) <= tolerance:
        if size[1] >= period[1] - tolerance:
            axes.append("y")
        if size[0] >= period[0] - tolerance:
            axes.append("x")
    elif abs(angleMod - np.pi / 2.0) <= tolerance:
        if size[0] >= period[1] - tolerance:
            axes.append("y")
        if size[1] >= period[0] - tolerance:
            axes.append("x")
    return tuple(axes)


class TwoMaterialMixin:
    background: complex
    inclusion: complex

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        raise NotImplementedError

    def convolutionMatrix(
        self,
        harmonics: object,
        background: complex | None = None,
        inclusion: complex | None = None,
    ) -> ComplexArray:
        return twoMaterialConvolution(
            self.indicatorMatrix(harmonics),
            getattr(harmonics, "count"),
            self.background if background is None else background,
            self.inclusion if inclusion is None else inclusion,
        )

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        return self.convolutionMatrix(harmonics, reciprocal(self.background), reciprocal(self.inclusion))

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return normalVectorMatricesFromField(normalX, normalY, harmonics)


class RadialNormalMixin:
    period: tuple[float, float]
    center: tuple[float, float]
    jonesResolution: int

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xx, yy = sampleCoordinates(self.period, self.center, self.jonesResolution)
        radius = np.sqrt(xx * xx + yy * yy)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        normalX = np.where(radius > 1e-12, xx / safeRadius, 1.0)
        normalY = np.where(radius > 1e-12, yy / safeRadius, 0.0)
        return normalizeVectorField(normalX, normalY)


@dataclass(frozen=True)
class AnalyticTerm:
    shape: object
    delta: complex


@dataclass(frozen=True)
class AnalyticComposite:
    period: tuple[float, float]
    background: complex
    terms: tuple[AnalyticTerm, ...]

    def __post_init__(self) -> None:
        validatePeriod(self.period)

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        matrix = complex(self.background) * np.eye(getattr(harmonics, "count"), dtype=complex)
        for term in self.terms:
            matrix = matrix + complex(term.delta) * term.shape.indicatorMatrix(harmonics)
        return matrix


@dataclass(frozen=True)
class AnalyticDisk(RadialNormalMixin, TwoMaterialMixin):
    period: tuple[float, float]
    radius: float
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    factorization: str = "normal-vector"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        validateAnalyticFactorization(self.factorization)
        validateResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return diskIndicator(harmonics, self.period, self.radius, self.center)


@dataclass(frozen=True)
class AnalyticEllipse(TwoMaterialMixin):
    period: tuple[float, float]
    radii: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "normal-vector"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.radii[0] <= 0 or self.radii[1] <= 0:
            raise ValueError("ellipse radii must be positive")
        validateAnalyticFactorization(self.factorization)
        validateResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return ellipseIndicator(harmonics, self.period, self.radii, self.center, self.angle)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xLocal, yLocal, cosine, sine = localCoordinates(
            self.period,
            self.center,
            self.angle,
            self.jonesResolution,
        )
        localX = xLocal / (self.radii[0] * self.radii[0])
        localY = yLocal / (self.radii[1] * self.radii[1])
        normalX, normalY = rotateLocalVector(localX, localY, cosine, sine)
        return normalizeVectorField(normalX, normalY)


@dataclass(frozen=True)
class AnalyticRectangle(TwoMaterialMixin):
    period: tuple[float, float]
    size: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "normal-vector"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("rectangle size values must be positive")
        validateAnalyticFactorization(self.factorization)
        validateResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return rectangleIndicator(harmonics, self.period, self.size, self.center, self.angle)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xLocal, yLocal, cosine, sine = localCoordinates(
            self.period,
            self.center,
            self.angle,
            self.jonesResolution,
        )
        signX = np.where(xLocal >= 0.0, 1.0, -1.0)
        signY = np.where(yLocal >= 0.0, 1.0, -1.0)
        angleMod = float(self.angle) % np.pi
        tolerance = 1e-12 * max(1.0, self.period[0], self.period[1], self.size[0], self.size[1])
        if min(abs(angleMod), abs(angleMod - np.pi)) <= tolerance:
            localPeriod = self.period
        elif abs(angleMod - np.pi / 2.0) <= tolerance:
            localPeriod = (self.period[1], self.period[0])
        else:
            localPeriod = (np.inf, np.inf)
        spansLocalX = self.size[0] >= localPeriod[0] - tolerance
        spansLocalY = self.size[1] >= localPeriod[1] - tolerance
        if spansLocalY and not spansLocalX:
            localX = signX
            localY = np.zeros_like(signY)
        elif spansLocalX and not spansLocalY:
            localX = np.zeros_like(signX)
            localY = signY
        else:
            distanceX = np.abs(np.abs(xLocal) - self.size[0] / 2.0)
            distanceY = np.abs(np.abs(yLocal) - self.size[1] / 2.0)
            useX = distanceX <= distanceY
            localX = np.where(useX, signX, 0.0)
            localY = np.where(useX, 0.0, signY)
        normalX, normalY = rotateLocalVector(localX, localY, cosine, sine)
        return normalizeVectorField(normalX, normalY)

    def invariantAxes(self) -> tuple[str, ...]:
        return alignedRectangleInvariantAxes(self.period, self.size, self.angle)
