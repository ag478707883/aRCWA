from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.special import j1

from .fourier import epsilonConvolutionMatrix, epsilonConvolutionMatrixTorch


ComplexArray = np.ndarray


def validatePeriod(period: tuple[float, float]) -> None:
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")


def sinc(value: ComplexArray) -> ComplexArray:
    result = np.ones_like(value, dtype=complex)
    nonzero = np.abs(value) > 1e-14
    result[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return result


def sincTorch(value: Any, torch: Any) -> Any:
    result = torch.ones_like(value, dtype=torch.complex128)
    valueComplex = value.to(torch.complex128)
    nonzero = torch.abs(valueComplex) > 1e-14
    return torch.where(nonzero, torch.sin(valueComplex) / valueComplex, result)


def besselRatio(argument: ComplexArray) -> ComplexArray:
    ratio = np.empty_like(argument, dtype=complex)
    small = np.abs(argument) < 1e-5
    x = argument[small]
    ratio[small] = 1.0 - x * x / 8.0 + x**4 / 192.0 - x**6 / 9216.0
    ratio[~small] = 2 * j1(argument[~small]) / argument[~small]
    return ratio


def besselRatioTorch(argument: Any, torch: Any) -> Any:
    argument = argument.to(torch.float64)
    ratio = torch.empty_like(argument, dtype=torch.complex128)
    seriesMask = torch.abs(argument) <= 8.0
    x = argument[seriesMask].to(torch.complex128)
    term = torch.ones_like(x, dtype=torch.complex128)
    series = term.clone()
    x2Over4 = x * x / 4.0
    for order in range(1, 36):
        term = term * (-x2Over4) / (order * (order + 1))
        series = series + term
    ratio[seriesMask] = series
    large = ~seriesMask
    if bool(torch.any(large).item()):
        xLarge = argument[large]
        ratio[large] = (2 * torch.special.bessel_j1(xLarge) / xLarge).to(torch.complex128)
    return ratio


def validateFactorization(value: str) -> None:
    if value not in ("analytic", "jones"):
        raise ValueError("factorization must be 'analytic' or 'jones'")


def validateJonesResolution(value: int) -> None:
    if value < 8:
        raise ValueError("jonesResolution must be at least 8")


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
    gx = 2 * np.pi * dmx / period[0]
    gy = 2 * np.pi * dmy / period[1]
    return gx, gy, np.exp(-1j * (gx * center[0] + gy * center[1]))


def phaseTorch(
    harmonics: object,
    period: tuple[float, float],
    center: tuple[float, float],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    dmx = asTorchReal(getattr(harmonics, "deltaMx"), torch, device)
    dmy = asTorchReal(getattr(harmonics, "deltaMy"), torch, device)
    pi = torch.as_tensor(np.pi, dtype=torch.float64, device=device)
    gx = 2 * pi * dmx / period[0]
    gy = 2 * pi * dmy / period[1]
    return gx, gy, torch.exp(-1j * (gx * center[0] + gy * center[1]))


def radialIndicator(argument: ComplexArray, fill: float, phase: ComplexArray) -> ComplexArray:
    coeff = np.empty_like(argument, dtype=complex)
    zero = np.abs(argument) < 1e-14
    coeff[zero] = fill
    coeff[~zero] = fill * besselRatio(argument[~zero])
    return coeff * phase


def radialIndicatorTorch(argument: Any, fill: float, phase: Any, torch: Any) -> Any:
    return complex(fill) * besselRatioTorch(argument, torch) * phase


def rotatedComponents(
    x: ComplexArray,
    y: ComplexArray,
    angle: float,
) -> tuple[ComplexArray, ComplexArray]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def rotatedComponentsTorch(x: Any, y: Any, angle: float, torch: Any, device: Any) -> tuple[Any, Any]:
    angleTensor = torch.as_tensor(float(angle), dtype=torch.float64, device=device)
    cosine = torch.cos(angleTensor)
    sine = torch.sin(angleTensor)
    return cosine * x + sine * y, -sine * x + cosine * y


def diskIndicator(
    harmonics: object,
    period: tuple[float, float],
    radius: float,
    center: tuple[float, float],
) -> ComplexArray:
    gx, gy, phaseFactor = phase(harmonics, period, center)
    gr = np.sqrt(gx * gx + gy * gy)
    fill = np.pi * radius * radius / (period[0] * period[1])
    return radialIndicator(gr * radius, fill, phaseFactor)


def diskIndicatorTorch(
    harmonics: object,
    period: tuple[float, float],
    radius: float,
    center: tuple[float, float],
    torch: Any,
    device: Any,
) -> Any:
    gx, gy, phaseFactor = phaseTorch(harmonics, period, center, torch, device)
    gr = torch.sqrt(gx * gx + gy * gy)
    fill = np.pi * radius * radius / (period[0] * period[1])
    return radialIndicatorTorch(gr * radius, fill, phaseFactor, torch)


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
    return radialIndicator(argument, fill, phaseFactor)


def ellipseIndicatorTorch(
    harmonics: object,
    period: tuple[float, float],
    radii: tuple[float, float],
    center: tuple[float, float],
    angle: float,
    torch: Any,
    device: Any,
) -> Any:
    gx, gy, phaseFactor = phaseTorch(harmonics, period, center, torch, device)
    gxLocal, gyLocal = rotatedComponentsTorch(gx, gy, angle, torch, device)
    argument = torch.sqrt((gxLocal * radii[0]) ** 2 + (gyLocal * radii[1]) ** 2)
    fill = np.pi * radii[0] * radii[1] / (period[0] * period[1])
    return radialIndicatorTorch(argument, fill, phaseFactor, torch)


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
    return fill * sinc(gxLocal * size[0] / 2) * sinc(gyLocal * size[1] / 2) * phaseFactor


def rectangleIndicatorTorch(
    harmonics: object,
    period: tuple[float, float],
    size: tuple[float, float],
    center: tuple[float, float],
    angle: float,
    torch: Any,
    device: Any,
) -> Any:
    gx, gy, phaseFactor = phaseTorch(harmonics, period, center, torch, device)
    gxLocal, gyLocal = rotatedComponentsTorch(gx, gy, angle, torch, device)
    fill = size[0] * size[1] / (period[0] * period[1])
    return complex(fill) * sincTorch(gxLocal * size[0] / 2, torch) * sincTorch(
        gyLocal * size[1] / 2, torch
    ) * phaseFactor


def twoMaterialConvolution(
    indicator: ComplexArray,
    size: int,
    background: complex,
    inclusion: complex,
) -> ComplexArray:
    return complex(background) * np.eye(size, dtype=complex) + (complex(inclusion) - complex(background)) * indicator


def twoMaterialConvolutionTorch(
    indicator: Any,
    size: int,
    background: complex,
    inclusion: complex,
    torch: Any,
    device: Any,
) -> Any:
    return complex(background) * torch.eye(size, dtype=torch.complex128, device=device) + (
        complex(inclusion) - complex(background)
    ) * indicator


def inverse(value: complex) -> complex:
    return 1.0 / complex(value)


def sampleCoordinates(
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


def sampleCoordinatesTorch(
    period: tuple[float, float],
    center: tuple[float, float],
    samples: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    periodX, periodY = period
    yIndex, xIndex = torch.meshgrid(
        torch.arange(samples, dtype=torch.float64, device=device),
        torch.arange(samples, dtype=torch.float64, device=device),
        indexing="ij",
    )
    xx = (xIndex + 0.5) / samples * periodX - periodX / 2 - center[0]
    yy = (yIndex + 0.5) / samples * periodY - periodY / 2 - center[1]
    xx = torch.remainder(xx + periodX / 2, periodX) - periodX / 2
    yy = torch.remainder(yy + periodY / 2, periodY) - periodY / 2
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


def localCoordinatesTorch(
    period: tuple[float, float],
    center: tuple[float, float],
    angle: float,
    samples: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    xx, yy = sampleCoordinatesTorch(period, center, samples, torch, device)
    angleTensor = torch.as_tensor(float(angle), dtype=torch.float64, device=device)
    cosine = torch.cos(angleTensor)
    sine = torch.sin(angleTensor)
    return cosine * xx + sine * yy, -sine * xx + cosine * yy, cosine, sine


def normalizeVectorField(x: ComplexArray, y: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    length = np.sqrt(np.real(x) ** 2 + np.real(y) ** 2)
    safe = length > 1e-12
    return (
        np.where(safe, np.real(x) / np.where(safe, length, 1.0), 1.0).astype(complex),
        np.where(safe, np.real(y) / np.where(safe, length, 1.0), 0.0).astype(complex),
    )


def normalizeVectorFieldTorch(x: Any, y: Any, torch: Any) -> tuple[Any, Any]:
    xReal = x.real if torch.is_complex(x) else x
    yReal = y.real if torch.is_complex(y) else y
    length = torch.sqrt(xReal * xReal + yReal * yReal)
    safe = length > 1e-12
    normalX = torch.where(safe, xReal / torch.where(safe, length, torch.ones_like(length)), torch.ones_like(length))
    normalY = torch.where(safe, yReal / torch.where(safe, length, torch.ones_like(length)), torch.zeros_like(length))
    return normalX.to(torch.complex128), normalY.to(torch.complex128)


def rotateLocalVector(
    xLocal: ComplexArray,
    yLocal: ComplexArray,
    cosine: float,
    sine: float,
) -> tuple[ComplexArray, ComplexArray]:
    return cosine * xLocal - sine * yLocal, sine * xLocal + cosine * yLocal


def rotateLocalVectorTorch(xLocal: Any, yLocal: Any, cosine: Any, sine: Any) -> tuple[Any, Any]:
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


def normalVectorMatricesFromFieldTorch(
    normalX: Any,
    normalY: Any,
    harmonics: object,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    tangentX = -normalY
    tangentY = normalX
    return (
        epsilonConvolutionMatrixTorch(normalX, harmonics, torch, device),
        epsilonConvolutionMatrixTorch(normalY, harmonics, torch, device),
        epsilonConvolutionMatrixTorch(tangentX, harmonics, torch, device),
        epsilonConvolutionMatrixTorch(tangentY, harmonics, torch, device),
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
    elif abs(angleMod - np.pi / 2) <= tolerance:
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
        return self.convolutionMatrix(harmonics, inverse(self.background), inverse(self.inclusion))

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return normalVectorMatricesFromField(normalX, normalY, harmonics)

    def convolutionMatrixTorch(
        self,
        harmonics: object,
        torch: Any,
        device: Any,
        background: complex | None = None,
        inclusion: complex | None = None,
    ) -> Any:
        return twoMaterialConvolutionTorch(
            self.indicatorMatrixTorch(harmonics, torch, device),
            getattr(harmonics, "count"),
            self.background if background is None else background,
            self.inclusion if inclusion is None else inclusion,
            torch,
            device,
        )

    def reciprocalConvolutionMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        return self.convolutionMatrixTorch(
            harmonics,
            torch,
            device,
            background=inverse(self.background),
            inclusion=inverse(self.inclusion),
        )

    def normalVectorMatricesTorch(
        self,
        harmonics: object,
        torch: Any,
        device: Any,
    ) -> tuple[Any, Any, Any, Any]:
        normalX, normalY = self.normalVectorFieldTorch(torch, device)
        return normalVectorMatricesFromFieldTorch(normalX, normalY, harmonics, torch, device)


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

    def normalVectorFieldTorch(self, torch: Any, device: Any) -> tuple[Any, Any]:
        xx, yy = sampleCoordinatesTorch(self.period, self.center, self.jonesResolution, torch, device)
        radius = torch.sqrt(xx * xx + yy * yy)
        safeRadius = torch.where(radius > 1e-12, radius, torch.ones_like(radius))
        normalX = torch.where(radius > 1e-12, xx / safeRadius, torch.ones_like(radius))
        normalY = torch.where(radius > 1e-12, yy / safeRadius, torch.zeros_like(radius))
        return normalizeVectorFieldTorch(normalX, normalY, torch)


def diskIndicators(
    harmonics: object,
    period: tuple[float, float],
    center: tuple[float, float],
    outerRadius: float,
    innerRadius: float = 0.0,
) -> tuple[ComplexArray, ComplexArray | None]:
    outer = diskIndicator(harmonics, period, outerRadius, center)
    inner = diskIndicator(harmonics, period, innerRadius, center) if innerRadius > 0 else None
    return outer, inner


def diskIndicatorsTorch(
    harmonics: object,
    period: tuple[float, float],
    center: tuple[float, float],
    outerRadius: float,
    innerRadius: float,
    torch: Any,
    device: Any,
) -> tuple[Any, Any | None]:
    outer = diskIndicatorTorch(harmonics, period, outerRadius, center, torch, device)
    inner = (
        diskIndicatorTorch(harmonics, period, innerRadius, center, torch, device)
        if innerRadius > 0
        else None
    )
    return outer, inner


def asTorchReal(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(value), dtype=torch.float64, device=device)


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
        validatePeriod(self.period)
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

    def convolutionMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        matrix = complex(self.background) * torch.eye(getattr(harmonics, "count"), dtype=torch.complex128, device=device)
        for term in self.terms:
            if hasattr(term.shape, "indicatorMatrixTorch"):
                indicator = term.shape.indicatorMatrixTorch(harmonics, torch, device)
            else:
                indicator = torch.as_tensor(
                    np.asarray(term.shape.indicatorMatrix(harmonics), dtype=complex),
                    dtype=torch.complex128,
                    device=device,
                )
            matrix = matrix + complex(term.delta) * indicator
        return matrix


@dataclass(frozen=True)
class AnalyticDisk(RadialNormalMixin, TwoMaterialMixin):
    period: tuple[float, float]
    radius: float
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        validateFactorization(self.factorization)
        validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return diskIndicator(harmonics, self.period, self.radius, self.center)

    def indicatorMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        return diskIndicatorTorch(harmonics, self.period, self.radius, self.center, torch, device)


@dataclass(frozen=True)
class AnalyticEllipse(TwoMaterialMixin):
    period: tuple[float, float]
    radii: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.radii[0] <= 0 or self.radii[1] <= 0:
            raise ValueError("ellipse radii must be positive")
        validateFactorization(self.factorization)
        validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return ellipseIndicator(harmonics, self.period, self.radii, self.center, self.angle)

    def indicatorMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        return ellipseIndicatorTorch(harmonics, self.period, self.radii, self.center, self.angle, torch, device)

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

    def normalVectorFieldTorch(self, torch: Any, device: Any) -> tuple[Any, Any]:
        xLocal, yLocal, cosine, sine = localCoordinatesTorch(
            self.period,
            self.center,
            self.angle,
            self.jonesResolution,
            torch,
            device,
        )
        localX = xLocal / (self.radii[0] * self.radii[0])
        localY = yLocal / (self.radii[1] * self.radii[1])
        normalX, normalY = rotateLocalVectorTorch(localX, localY, cosine, sine)
        return normalizeVectorFieldTorch(normalX, normalY, torch)


@dataclass(frozen=True)
class AnalyticRectangle(TwoMaterialMixin):
    period: tuple[float, float]
    size: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0
    factorization: str = "analytic"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        validatePeriod(self.period)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("rectangle size values must be positive")
        validateFactorization(self.factorization)
        validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        return rectangleIndicator(harmonics, self.period, self.size, self.center, self.angle)

    def indicatorMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        return rectangleIndicatorTorch(harmonics, self.period, self.size, self.center, self.angle, torch, device)

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
        normalX, normalY = rotateLocalVector(localX, localY, cosine, sine)
        return normalizeVectorField(normalX, normalY)

    def invariantAxes(self) -> tuple[str, ...]:
        return alignedRectangleInvariantAxes(self.period, self.size, self.angle)

    def normalVectorFieldTorch(self, torch: Any, device: Any) -> tuple[Any, Any]:
        xLocal, yLocal, cosine, sine = localCoordinatesTorch(
            self.period,
            self.center,
            self.angle,
            self.jonesResolution,
            torch,
            device,
        )
        signX = torch.where(xLocal >= 0.0, torch.ones_like(xLocal), -torch.ones_like(xLocal))
        signY = torch.where(yLocal >= 0.0, torch.ones_like(yLocal), -torch.ones_like(yLocal))
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
            localY = torch.zeros_like(signY)
        elif spansLocalX and not spansLocalY:
            localX = torch.zeros_like(signX)
            localY = signY
        else:
            distanceX = torch.abs(torch.abs(xLocal) - self.size[0] / 2)
            distanceY = torch.abs(torch.abs(yLocal) - self.size[1] / 2)
            useX = distanceX <= distanceY
            localX = torch.where(useX, signX, torch.zeros_like(signX))
            localY = torch.where(useX, torch.zeros_like(signY), signY)
        normalX, normalY = rotateLocalVectorTorch(localX, localY, cosine, sine)
        return normalizeVectorFieldTorch(normalX, normalY, torch)


@dataclass(frozen=True)
class AnalyticAnnulus(RadialNormalMixin):
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
        validatePeriod(self.period)
        if self.innerRadius < 0 or self.outerRadius <= 0 or self.innerRadius >= self.outerRadius:
            raise ValueError("annulus requires 0 <= innerRadius < outerRadius")
        validateFactorization(self.factorization)
        validateJonesResolution(self.jonesResolution)

    def indicatorMatrix(self, harmonics: object) -> ComplexArray:
        outer, inner = diskIndicators(harmonics, self.period, self.center, self.outerRadius, self.innerRadius)
        return outer if inner is None else outer - inner

    def indicatorMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        outer, inner = diskIndicatorsTorch(
            harmonics,
            self.period,
            self.center,
            self.outerRadius,
            self.innerRadius,
            torch,
            device,
        )
        return outer if inner is None else outer - inner

    def convolutionMatrix(self, harmonics: object) -> ComplexArray:
        count = getattr(harmonics, "count")
        outer, inner = diskIndicators(harmonics, self.period, self.center, self.outerRadius, self.innerRadius)
        matrix = complex(self.background) * np.eye(count, dtype=complex)
        matrix += (complex(self.ring) - complex(self.background)) * outer
        if inner is not None:
            holeValue = self.background if self.hole is None else self.hole
            matrix += (complex(holeValue) - complex(self.ring)) * inner
        return matrix

    def reciprocalConvolutionMatrix(self, harmonics: object) -> ComplexArray:
        count = getattr(harmonics, "count")
        outer, inner = diskIndicators(harmonics, self.period, self.center, self.outerRadius, self.innerRadius)
        background = inverse(self.background)
        ring = inverse(self.ring)
        matrix = background * np.eye(count, dtype=complex)
        matrix += (ring - background) * outer
        if inner is not None:
            holeValue = self.background if self.hole is None else self.hole
            matrix += (inverse(holeValue) - ring) * inner
        return matrix

    def normalVectorMatrices(self, harmonics: object) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
        normalX, normalY = self.normalVectorField()
        return normalVectorMatricesFromField(normalX, normalY, harmonics)

    def convolutionMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        count = getattr(harmonics, "count")
        outer, inner = diskIndicatorsTorch(
            harmonics,
            self.period,
            self.center,
            self.outerRadius,
            self.innerRadius,
            torch,
            device,
        )
        matrix = complex(self.background) * torch.eye(count, dtype=torch.complex128, device=device)
        matrix = matrix + (complex(self.ring) - complex(self.background)) * outer
        if inner is not None:
            holeValue = self.background if self.hole is None else self.hole
            matrix = matrix + (complex(holeValue) - complex(self.ring)) * inner
        return matrix

    def reciprocalConvolutionMatrixTorch(self, harmonics: object, torch: Any, device: Any) -> Any:
        count = getattr(harmonics, "count")
        outer, inner = diskIndicatorsTorch(
            harmonics,
            self.period,
            self.center,
            self.outerRadius,
            self.innerRadius,
            torch,
            device,
        )
        background = inverse(self.background)
        ring = inverse(self.ring)
        matrix = background * torch.eye(count, dtype=torch.complex128, device=device)
        matrix = matrix + (ring - background) * outer
        if inner is not None:
            holeValue = self.background if self.hole is None else self.hole
            matrix = matrix + (inverse(holeValue) - ring) * inner
        return matrix

    def normalVectorMatricesTorch(
        self,
        harmonics: object,
        torch: Any,
        device: Any,
    ) -> tuple[Any, Any, Any, Any]:
        normalX, normalY = self.normalVectorFieldTorch(torch, device)
        return normalVectorMatricesFromFieldTorch(normalX, normalY, harmonics, torch, device)
