from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from .fourier import Harmonics, epsilonConvolutionMatrix


ComplexArray = np.ndarray


def _validatePeriod(period: tuple[float, float]) -> None:
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")


def _sinc(value: ComplexArray) -> ComplexArray:
    result = np.ones_like(value, dtype=complex)
    nonzero = np.abs(value) > 1e-14
    result[nonzero] = np.sin(value[nonzero]) / value[nonzero]
    return result


def _solveIdentity(matrix: ComplexArray) -> ComplexArray:
    return np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=complex))


def _phase(
    harmonics: Harmonics,
    period: tuple[float, float],
    center: tuple[float, float],
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    dmx = harmonics.mx[:, None] - harmonics.mx[None, :]
    dmy = harmonics.my[:, None] - harmonics.my[None, :]
    gx = 2 * np.pi * dmx / period[0]
    gy = 2 * np.pi * dmy / period[1]
    return gx, gy, np.exp(-1j * (gx * center[0] + gy * center[1]))


def _besselRatio(argument: ComplexArray) -> ComplexArray:
    ratio = np.empty_like(argument, dtype=complex)
    small = np.abs(argument) < 1e-5
    x = argument[small]
    ratio[small] = 1.0 - x * x / 8.0 + x**4 / 192.0 - x**6 / 9216.0
    ratio[~small] = 2 * j1(argument[~small]) / argument[~small]
    return ratio


def _twoMaterialConvolution(
    indicator: ComplexArray,
    size: int,
    background: complex,
    inclusion: complex,
) -> ComplexArray:
    return complex(background) * np.eye(size, dtype=complex) + (complex(inclusion) - complex(background)) * indicator


@dataclass(frozen=True)
class AnalyticDisk:
    """Analytic circular inclusion in one rectangular periodic unit cell.

    The Fourier coefficients of the disk indicator are evaluated in closed
    form, avoiding staircasing errors from pixelized circular boundaries.
    ``factorization="jones"`` uses analytic epsilon and inverse-epsilon
    coefficients with a sampled continuous Jones-vector field for the local
    normal/tangent decomposition.
    """

    period: tuple[float, float]
    radius: float
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    factorization: str = "jones"
    jonesResolution: int = 512

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.jonesResolution < 8:
            raise ValueError("jonesResolution must be at least 8")
        if self.factorization not in ("analytic", "jones"):
            raise ValueError("factorization must be 'analytic' or 'jones'")

    def indicatorMatrix(self, harmonics: Harmonics) -> ComplexArray:
        return diskIndicatorConvolution(self, harmonics)

    def convolutionMatrix(
        self,
        harmonics: Harmonics,
        background: complex | None = None,
        inclusion: complex | None = None,
    ) -> ComplexArray:
        return _twoMaterialConvolution(
            self.indicatorMatrix(harmonics),
            harmonics.count,
            self.background if background is None else background,
            self.inclusion if inclusion is None else inclusion,
        )


def analyticDiskConvolution(
    disk: AnalyticDisk,
    harmonics: Harmonics,
    background: complex | None = None,
    inclusion: complex | None = None,
) -> ComplexArray:
    """Convolution matrix for a two-material analytic disk pattern."""

    return disk.convolutionMatrix(harmonics, background=background, inclusion=inclusion)


def diskIndicatorConvolution(disk: AnalyticDisk, harmonics: Harmonics) -> ComplexArray:
    dmx = harmonics.mx[:, None] - harmonics.mx[None, :]
    dmy = harmonics.my[:, None] - harmonics.my[None, :]
    return diskIndicatorCoefficients(disk, dmx, dmy)


def diskIndicatorCoefficients(disk: AnalyticDisk, mx: ComplexArray, my: ComplexArray) -> ComplexArray:
    gx = 2 * np.pi * mx / disk.period[0]
    gy = 2 * np.pi * my / disk.period[1]
    gr = np.sqrt(gx * gx + gy * gy)
    fill = np.pi * disk.radius**2 / (disk.period[0] * disk.period[1])
    coeff = np.empty_like(gr, dtype=complex)
    zero = np.abs(gr) < 1e-14
    argument = gr[~zero] * disk.radius
    coeff[zero] = fill
    coeff[~zero] = fill * _besselRatio(argument)
    phase = np.exp(-1j * (gx * disk.center[0] + gy * disk.center[1]))
    return coeff * phase


@dataclass(frozen=True)
class AnalyticEllipse:
    period: tuple[float, float]
    radii: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.radii[0] <= 0 or self.radii[1] <= 0:
            raise ValueError("ellipse radii must be positive")

    def indicatorMatrix(self, harmonics: Harmonics) -> ComplexArray:
        gx, gy, phase = _phase(harmonics, self.period, self.center)
        cosine = np.cos(self.angle)
        sine = np.sin(self.angle)
        gxLocal = cosine * gx + sine * gy
        gyLocal = -sine * gx + cosine * gy
        argument = np.sqrt((gxLocal * self.radii[0]) ** 2 + (gyLocal * self.radii[1]) ** 2)
        fill = np.pi * self.radii[0] * self.radii[1] / (self.period[0] * self.period[1])
        coeff = np.empty_like(argument, dtype=complex)
        zero = np.abs(argument) < 1e-14
        coeff[zero] = fill
        coeff[~zero] = fill * _besselRatio(argument[~zero])
        return coeff * phase

    def convolutionMatrix(self, harmonics: Harmonics) -> ComplexArray:
        return _twoMaterialConvolution(
            self.indicatorMatrix(harmonics),
            harmonics.count,
            self.background,
            self.inclusion,
        )


@dataclass(frozen=True)
class AnalyticRectangle:
    period: tuple[float, float]
    size: tuple[float, float]
    background: complex
    inclusion: complex
    center: tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0

    def __post_init__(self) -> None:
        _validatePeriod(self.period)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("rectangle size values must be positive")

    def indicatorMatrix(self, harmonics: Harmonics) -> ComplexArray:
        gx, gy, phase = _phase(harmonics, self.period, self.center)
        cosine = np.cos(self.angle)
        sine = np.sin(self.angle)
        gxLocal = cosine * gx + sine * gy
        gyLocal = -sine * gx + cosine * gy
        fill = self.size[0] * self.size[1] / (self.period[0] * self.period[1])
        return fill * _sinc(gxLocal * self.size[0] / 2) * _sinc(gyLocal * self.size[1] / 2) * phase

    def convolutionMatrix(self, harmonics: Harmonics) -> ComplexArray:
        return _twoMaterialConvolution(
            self.indicatorMatrix(harmonics),
            harmonics.count,
            self.background,
            self.inclusion,
        )


def analyticDiskJonesMatrices(
    disk: AnalyticDisk,
    harmonics: Harmonics,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return Li/Jones-factorized in-plane epsilon blocks and epsilon_zz."""

    direct = analyticDiskConvolution(disk, harmonics)
    inverseRule = _solveIdentity(
        analyticDiskConvolution(
            disk,
            harmonics,
            background=1.0 / complex(disk.background),
            inclusion=1.0 / complex(disk.inclusion),
        )
    )
    nx, ny, tx, ty = _jonesVectorMatrices(disk, harmonics)

    cxx = nx @ inverseRule @ nx + tx @ direct @ tx
    cxy = nx @ inverseRule @ ny + tx @ direct @ ty
    cyx = ny @ inverseRule @ nx + ty @ direct @ tx
    cyy = ny @ inverseRule @ ny + ty @ direct @ ty
    return cxx, cxy, cyx, cyy, direct


def _jonesVectorMatrices(
    disk: AnalyticDisk,
    harmonics: Harmonics,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    normalX, normalY = _jonesVectorField(disk)
    tangentX = -normalY
    tangentY = normalX
    return (
        epsilonConvolutionMatrix(normalX, harmonics),
        epsilonConvolutionMatrix(normalY, harmonics),
        epsilonConvolutionMatrix(tangentX, harmonics),
        epsilonConvolutionMatrix(tangentY, harmonics),
    )


def _jonesVectorField(disk: AnalyticDisk) -> tuple[ComplexArray, ComplexArray]:
    periodX, periodY = disk.period
    samples = disk.jonesResolution
    yIndex, xIndex = np.mgrid[0:samples, 0:samples]
    xx = (xIndex + 0.5) / samples * periodX - periodX / 2 - disk.center[0]
    yy = (yIndex + 0.5) / samples * periodY - periodY / 2 - disk.center[1]

    # Periodic shortest-image radial direction.  This gives a continuous Jones
    # field around a centered disk and keeps the correct normal on the cylinder.
    xx = (xx + periodX / 2) % periodX - periodX / 2
    yy = (yy + periodY / 2) % periodY - periodY / 2
    radius = np.sqrt(xx * xx + yy * yy)
    safeRadius = np.where(radius > 1e-12, radius, 1.0)
    normalX = np.where(radius > 1e-12, xx / safeRadius, 1.0)
    normalY = np.where(radius > 1e-12, yy / safeRadius, 0.0)
    return normalX.astype(complex), normalY.astype(complex)
