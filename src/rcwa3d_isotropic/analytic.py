from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import j1


ComplexArray = np.ndarray


def _validatePeriod(period: tuple[float, float]) -> None:
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")


def _sinc(value: ComplexArray) -> ComplexArray:
    result = np.ones_like(value, dtype=complex)
    nonzero = np.abs(value) > 1e-14
    result[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return result


def _besselRatio(argument: ComplexArray) -> ComplexArray:
    ratio = np.empty_like(argument, dtype=complex)
    small = np.abs(argument) < 1e-5
    x = argument[small]
    ratio[small] = 1.0 - x * x / 8.0 + x**4 / 192.0 - x**6 / 9216.0
    ratio[~small] = 2 * j1(argument[~small]) / argument[~small]
    return ratio


def _validateFactorization(value: str) -> None:
    if value not in ("analytic", "jones"):
        raise ValueError("factorization must be 'analytic' or 'jones'")


def _validateJonesResolution(value: int) -> None:
    if value < 8:
        raise ValueError("jonesResolution must be at least 8")


def _phase(
    harmonics: object,
    period: tuple[float, float],
    center: tuple[float, float],
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    dmx = getattr(harmonics, "mx")[:, None] - getattr(harmonics, "mx")[None, :]
    dmy = getattr(harmonics, "my")[:, None] - getattr(harmonics, "my")[None, :]
    gx = 2 * np.pi * dmx / period[0]
    gy = 2 * np.pi * dmy / period[1]
    return gx, gy, np.exp(-1j * (gx * center[0] + gy * center[1]))


def _diskIndicator(
    harmonics: object,
    period: tuple[float, float],
    radius: float,
    center: tuple[float, float],
) -> ComplexArray:
    gx, gy, phase = _phase(harmonics, period, center)
    gr = np.sqrt(gx * gx + gy * gy)
    fill = np.pi * radius * radius / (period[0] * period[1])
    coeff = np.empty_like(gr, dtype=complex)
    zero = np.abs(gr) < 1e-14
    argument = gr * radius
    coeff[zero] = fill
    coeff[~zero] = fill * _besselRatio(argument[~zero])
    return coeff * phase


def _ellipseIndicator(
    harmonics: object,
    period: tuple[float, float],
    radii: tuple[float, float],
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    gx, gy, phase = _phase(harmonics, period, center)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    gxLocal = cosine * gx + sine * gy
    gyLocal = -sine * gx + cosine * gy
    argument = np.sqrt((gxLocal * radii[0]) ** 2 + (gyLocal * radii[1]) ** 2)
    fill = np.pi * radii[0] * radii[1] / (period[0] * period[1])
    coeff = np.empty_like(argument, dtype=complex)
    zero = np.abs(argument) < 1e-14
    coeff[zero] = fill
    coeff[~zero] = fill * _besselRatio(argument[~zero])
    return coeff * phase


def _rectangleIndicator(
    harmonics: object,
    period: tuple[float, float],
    size: tuple[float, float],
    center: tuple[float, float],
    angle: float,
) -> ComplexArray:
    gx, gy, phase = _phase(harmonics, period, center)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    gxLocal = cosine * gx + sine * gy
    gyLocal = -sine * gx + cosine * gy
    fill = size[0] * size[1] / (period[0] * period[1])
    return fill * _sinc(gxLocal * size[0] / 2) * _sinc(gyLocal * size[1] / 2) * phase


def _twoMaterialConvolution(
    indicator: ComplexArray,
    size: int,
    background: complex,
    inclusion: complex,
) -> ComplexArray:
    return complex(background) * np.eye(size, dtype=complex) + (complex(inclusion) - complex(background)) * indicator


def _sampleCoordinates(
    period: tuple[float, float],
    center: tuple[float, float],
    samples: int,
) -> tuple[ComplexArray, ComplexArray]:
    periodX, periodY = period
    yIndex, xIndex = np.mgrid[0:samples, 0:samples]
    xx = (xIndex + 0.5) / samples * periodX - periodX / 2 - center[0]
    yy = (yIndex + 0.5) / samples * periodY - periodY / 2 - center[1]
    xx = (xx + periodX / 2) % periodX - periodX / 2
    yy = (yy + periodY / 2) % periodY - periodY / 2
    return xx, yy


def _localCoordinates(
    period: tuple[float, float],
    center: tuple[float, float],
    angle: float,
    samples: int,
) -> tuple[ComplexArray, ComplexArray, float, float]:
    xx, yy = _sampleCoordinates(period, center, samples)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return cosine * xx + sine * yy, -sine * xx + cosine * yy, cosine, sine


def _normalizeVectorField(x: ComplexArray, y: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    length = np.sqrt(np.real(x) ** 2 + np.real(y) ** 2)
    safe = length > 1e-12
    return (
        np.where(safe, np.real(x) / np.where(safe, length, 1.0), 1.0).astype(complex),
        np.where(safe, np.real(y) / np.where(safe, length, 1.0), 0.0).astype(complex),
    )


def _rotateLocalVector(
    xLocal: ComplexArray,
    yLocal: ComplexArray,
    cosine: float,
    sine: float,
) -> tuple[ComplexArray, ComplexArray]:
    return cosine * xLocal - sine * yLocal, sine * xLocal + cosine * yLocal


def _normalVectorMatricesFromField(
    normalX: ComplexArray,
    normalY: ComplexArray,
    harmonics: object,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    tangentX = -normalY
    tangentY = normalX
    from .fourier import epsilonConvolutionMatrix

    return (
        epsilonConvolutionMatrix(normalX, harmonics),
        epsilonConvolutionMatrix(normalY, harmonics),
        epsilonConvolutionMatrix(tangentX, harmonics),
        epsilonConvolutionMatrix(tangentY, harmonics),
    )


def _alignedRectangleInvariantAxes(
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
    elif abs(angleMod - np.pi / 2) <= tolerance:
        if size[0] >= period[1] - tolerance:
            axes.append("y")
        if size[1] >= period[0] - tolerance:
            axes.append("x")
    return tuple(axes)


@dataclass(frozen=True)
class AnalyticTerm:
    shape: object
    delta: complex


@dataclass(frozen=True)
class AnalyticComposite:
    period: tuple[float, float]
    background: complex
    terms: Sequence[AnalyticTerm]

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        for term in self.terms:
            if not hasattr(term.shape, "indicatorMatrix"):
                raise TypeError("AnalyticComposite terms require shapes with indicatorMatrix")
            if getattr(term.shape, "period", self.period) != self.period:
                raise ValueError("all analytic composite shapes must use the same period")

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        matrix = complex(self.background) * np.eye(getattr(harmonics, "count"), dtype=complex)
        for term in self.terms:
            matrix += complex(term.delta) * term.shape.indicatorMatrix(harmonics)
        return matrix


@dataclass(frozen=True)
class AnalyticDisk:
    period: tuple[float, float]
    radius: float
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        _validateFactorization(self.factorization)
        _validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return _diskIndicator(harmonics, self.period, self.radius, self.center)

    def convolutionMatrix(
        self,
        harmonics: object,
        background: complex | None = None,
        inclusion: complex | None = None,
    ) -> ComplexArray:
        return _twoMaterialConvolution(
            self.indicatorMatrix(harmonics),
            getattr(harmonics, "count"),
            self.background if background is None else background,
            self.inclusion if inclusion is None else inclusion,
        )

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        return self.convolutionMatrix(harmonics, 1.0 / complex(self.background), 1.0 / complex(self.inclusion))

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return _normalVectorMatricesFromField(normalX, normalY, harmonics)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xx, yy = _sampleCoordinates(self.period, self.center, self.jonesResolution)
        radius = np.sqrt(xx * xx + yy * yy)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        normalX = np.where(radius > 1e-12, xx / safeRadius, 1.0)
        normalY = np.where(radius > 1e-12, yy / safeRadius, 0.0)
        return _normalizeVectorField(normalX, normalY)


@dataclass(frozen=True)
class AnalyticEllipse:
    period: tuple[float, float]
    radii: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.radii[0] <= 0 or self.radii[1] <= 0:
            raise ValueError("ellipse radii must be positive")
        _validateFactorization(self.factorization)
        _validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return _ellipseIndicator(harmonics, self.period, self.radii, self.center, self.angle)

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        indicator = self.indicatorMatrix(harmonics)
        return _twoMaterialConvolution(indicator, getattr(harmonics, "count"), self.background, self.inclusion)

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        indicator = self.indicatorMatrix(harmonics)
        return _twoMaterialConvolution(
            indicator,
            getattr(harmonics, "count"),
            1.0 / complex(self.background),
            1.0 / complex(self.inclusion),
        )

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return _normalVectorMatricesFromField(normalX, normalY, harmonics)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xLocal, yLocal, cosine, sine = _localCoordinates(
            self.period,
            self.center,
            self.angle,
            self.jonesResolution,
        )
        localX = xLocal / (self.radii[0] * self.radii[0])
        localY = yLocal / (self.radii[1] * self.radii[1])
        normalX, normalY = _rotateLocalVector(localX, localY, cosine, sine)
        return _normalizeVectorField(normalX, normalY)


@dataclass(frozen=True)
class AnalyticRectangle:
    period: tuple[float, float]
    size: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("rectangle size values must be positive")
        _validateFactorization(self.factorization)
        _validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return _rectangleIndicator(harmonics, self.period, self.size, self.center, self.angle)

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        indicator = self.indicatorMatrix(harmonics)
        return _twoMaterialConvolution(indicator, getattr(harmonics, "count"), self.background, self.inclusion)

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        indicator = self.indicatorMatrix(harmonics)
        return _twoMaterialConvolution(
            indicator,
            getattr(harmonics, "count"),
            1.0 / complex(self.background),
            1.0 / complex(self.inclusion),
        )

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return _normalVectorMatricesFromField(normalX, normalY, harmonics)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xLocal, yLocal, cosine, sine = _localCoordinates(
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
        elif abs(angleMod - np.pi / 2) <= tolerance:
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
            distanceX = np.abs(np.abs(xLocal) - self.size[0] / 2)
            distanceY = np.abs(np.abs(yLocal) - self.size[1] / 2)
            useX = distanceX <= distanceY
            localX = np.where(useX, signX, 0.0)
            localY = np.where(useX, 0.0, signY)
        normalX, normalY = _rotateLocalVector(localX, localY, cosine, sine)
        return _normalizeVectorField(normalX, normalY)

    def invariantAxes(self) -> tuple[str, ...]:
        return _alignedRectangleInvariantAxes(self.period, self.size, self.angle)


@dataclass(frozen=True)
class AnalyticAnnulus:
    period: tuple[float, float]
    innerRadius: float
    outerRadius: float
    background: complex
    ring: complex
    hole: complex | None = None
    center: tuple[float, float] = (0.0, 0.0)
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.innerRadius < 0 or self.outerRadius <= 0 or self.innerRadius >= self.outerRadius:
            raise ValueError("annulus requires 0 <= innerRadius < outerRadius")
        _validateFactorization(self.factorization)
        _validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        outer = _diskIndicator(harmonics, self.period, self.outerRadius, self.center)
        if self.innerRadius <= 0:
            return outer
        return outer - _diskIndicator(harmonics, self.period, self.innerRadius, self.center)

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        count = getattr(harmonics, "count")
        outer = _diskIndicator(harmonics, self.period, self.outerRadius, self.center)
        matrix = complex(self.background) * np.eye(count, dtype=complex)
        matrix += (complex(self.ring) - complex(self.background)) * outer
        if self.innerRadius > 0:
            inner = _diskIndicator(harmonics, self.period, self.innerRadius, self.center)
            holeValue = self.background if self.hole is None else self.hole
            matrix += (complex(holeValue) - complex(self.ring)) * inner
        return matrix

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        count = getattr(harmonics, "count")
        outer = _diskIndicator(harmonics, self.period, self.outerRadius, self.center)
        matrix = (1.0 / complex(self.background)) * np.eye(count, dtype=complex)
        matrix += (1.0 / complex(self.ring) - 1.0 / complex(self.background)) * outer
        if self.innerRadius > 0:
            inner = _diskIndicator(harmonics, self.period, self.innerRadius, self.center)
            holeValue = self.background if self.hole is None else self.hole
            matrix += (1.0 / complex(holeValue) - 1.0 / complex(self.ring)) * inner
        return matrix

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return _normalVectorMatricesFromField(normalX, normalY, harmonics)

    def normalVectorField(self) -> tuple[ComplexArray, ComplexArray]:
        xx, yy = _sampleCoordinates(self.period, self.center, self.jonesResolution)
        radius = np.sqrt(xx * xx + yy * yy)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        normalX = np.where(radius > 1e-12, xx / safeRadius, 1.0)
        normalY = np.where(radius > 1e-12, yy / safeRadius, 0.0)
        return _normalizeVectorField(normalX, normalY)
