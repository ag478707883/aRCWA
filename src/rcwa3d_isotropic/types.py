from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


ComplexArray = np.ndarray


@dataclass(frozen=True)
class Layer:
    """One finite isotropic RCWA layer."""

    thickness: float
    epsilon: complex | ArrayLike
    name: str = ""
    normalField: ArrayLike | None = None
    factorization: str = "auto"
    sampleShape: tuple[int, int] | None = None


@dataclass(frozen=True)
class CompiledLayer:
    """Layer with fixed-order Fourier convolution data."""

    thickness: float
    epsilonMatrix: ComplexArray
    epsilonInverse: ComplexArray
    orders: tuple[int, int]
    truncation: str = "rectangular"
    name: str = ""
    displacementMatrices: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray] | None = None
    factorization: str = "standard"
    homogeneousEpsilon: complex | None = None
    sampleShape: tuple[int, int] | None = None


@dataclass(frozen=True)
class DiffractionOrder:
    mx: int
    my: int
    kx: complex
    ky: complex
    kzReflected: complex
    kzTransmitted: complex
    reflectedPower: float
    transmittedPower: float
    reflectedPropagating: bool
    transmittedPropagating: bool


@dataclass(frozen=True)
class LayerFieldSolution:
    name: str
    thickness: float
    wavelength: float
    period: tuple[float, float]
    orders: tuple[int, int]
    mx: ComplexArray
    my: ComplexArray
    kx: ComplexArray
    ky: ComplexArray
    qValues: ComplexArray
    modeMatrix: ComplexArray
    coefficients: ComplexArray
    epsilonInverse: ComplexArray | None = None
    backwardCoefficientsRight: ComplexArray | None = None


@dataclass(frozen=True)
class LayerEigTiming:
    layerIndex: int
    name: str
    kind: str
    matrixShape: tuple[int, ...]
    eigTimeSeconds: float
    factorizationTimeSeconds: float = 0.0
    inverseTimeSeconds: float = 0.0
    pqTimeSeconds: float = 0.0
    totalTimeSeconds: float = 0.0
    minAbsQ: float | None = None
    safeQThreshold: float | None = None
    nearZeroModeCount: int = 0


@dataclass(frozen=True)
class StackTiming:
    interfaceTimeSeconds: float = 0.0
    cascadeTimeSeconds: float = 0.0
    totalPrepareTimeSeconds: float = 0.0
    interfaceConditionNumbers: tuple[float, ...] = ()
    maxInterfaceCondition: float | None = None
    stabilityWarnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RCWAResult:
    reflection: float
    transmission: float
    conservation: float
    absorption: float
    energyError: float | None
    rAmplitudes: ComplexArray
    tAmplitudes: ComplexArray
    orders: tuple[DiffractionOrder, ...]
    incidentFlux: float
    solvedBy: str
    layerSolutions: tuple[LayerFieldSolution, ...] = ()
    layerEigTimings: tuple[LayerEigTiming, ...] = ()
    stackTiming: StackTiming | None = None
    epsIncident: complex = 1.0
    epsTransmission: complex = 1.0
    sAmplitude: complex = 1.0
    pAmplitude: complex = 0.0
    powerWarning: str | None = None
    diagnostics: tuple[str, ...] = ()
