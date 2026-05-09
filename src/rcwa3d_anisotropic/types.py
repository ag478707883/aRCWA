from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .factorization import TensorConvolutionData
from .fourier import Harmonics as Harmonics


ComplexArray = np.ndarray
TensorLike = object


@dataclass(frozen=True)
class BackendTensorConvolutionData:
    components: tuple[tuple[object, object, object], ...]
    etaZz: object


@dataclass(frozen=True)
class Layer:
    """One finite tensor-material layer in a z-stacked RCWA structure.

    Parameters
    ----------
    thickness:
        Layer thickness in the same length unit as ``wavelength`` and
        ``period``.
    epsilon:
        Relative-permittivity tensor. Accepted forms are scalar/2D-grid
        isotropic inputs, constant ``(3, 3)`` tensors, sampled
        ``(ny, nx, 3, 3)`` tensor fields, or component mappings such as
        ``{"xx": grid, "yy": grid, "zz": grid, "xz": grid, "zx": grid}``.
    normalField:
        Optional sampled in-plane normal-vector field with shape ``(ny, nx, 2)``
        for scalar 2D gratings. When supplied, the in-plane displacement
        blocks use normal-vector Li factorization, while the z-normal
        elimination remains driven by epsilon_zz.
    factorization:
        Requested scalar-boundary factorization path. ``"auto"`` enables the
        vector-field Li route for sampled piecewise-constant scalar grids.
    name:
        Optional label used only by callers/debugging.
    """

    thickness: float
    epsilon: TensorLike
    normalField: ArrayLike | None = None
    factorization: str = "auto"
    name: str = ""
    sampleShape: tuple[int, int] | None = None


@dataclass(frozen=True)
class CompiledLayer:
    """Layer with precomputed tensor Fourier convolution data."""

    thickness: float
    tensorData: TensorConvolutionData
    orders: tuple[int, int]
    truncation: str = "circular"
    normalField: ArrayLike | None = None
    factorization: str = "auto"
    name: str = ""
    sampleShape: tuple[int, int] | None = None


@dataclass(frozen=True)
class BatchedHomogeneousLayer:
    """Homogeneous layer whose tensor is materialized for each wavelength."""

    thickness: float
    tensors: ComplexArray
    name: str = ""


@dataclass(frozen=True)
class SMatrix:
    s11: object
    s12: object
    s21: object
    s22: object
    isIdentity: bool = False
    isPropagation: bool = False


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


@dataclass(frozen=True)
class LayerEigTiming:
    layerIndex: int
    name: str
    kind: str
    matrixShape: tuple[int, ...]
    eigTimeSeconds: float
    factorizationTimeSeconds: float = 0.0
    matrixBuildTimeSeconds: float = 0.0
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
    rAmplitudes: ComplexArray
    tAmplitudes: ComplexArray
    orders: tuple[DiffractionOrder, ...]
    incidentFlux: float
    solvedBy: str
    layerSolutions: tuple[LayerFieldSolution, ...] = ()
    layerEigTimings: tuple[LayerEigTiming, ...] = ()
    stackTiming: StackTiming | None = None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedStack:
    """Reusable anisotropic S-matrix data for one wavelength/angle state."""

    layers: tuple[Layer | CompiledLayer, ...]
    wavelength: float
    period: tuple[float, float]
    orders: tuple[int, int]
    epsIncident: complex
    epsTransmission: complex
    theta: float
    phi: float
    truncation: str
    backend: str
    precision: str
    harmonics: Harmonics
    components: tuple[SMatrix, ...]
    total: SMatrix
    layerModes: tuple[tuple[object, object], ...]
    incidentBackward: object
    transmissionForward: object
    zeroIndex: int
    layerEigTimings: tuple[LayerEigTiming, ...] = ()
    stackTiming: StackTiming | None = None

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class BatchedHarmonics:
    mx: ComplexArray
    my: ComplexArray
    kx: object
    ky: object
    orders: tuple[int, int]
    truncation: str

    @property
    def count(self) -> int:
        return int(self.mx.size)

    @property
    def batchSize(self) -> int:
        return int(self.kx.shape[0])


@dataclass(frozen=True)
class PreparedBatchStack:
    """Reusable CUDA S-matrix data for a wavelength batch."""

    layers: tuple[Layer | CompiledLayer, ...]
    wavelengths: ComplexArray
    period: tuple[float, float]
    orders: tuple[int, int]
    epsIncident: complex
    epsTransmission: complex
    theta: float
    phi: float
    truncation: str
    backend: str
    precision: str
    harmonics: BatchedHarmonics
    total: SMatrix
    incidentForward: object
    incidentBackward: object
    transmissionForward: object
    zeroIndex: int

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class AutomaticFastPathPlan:
    label: str
    reducedOrders: tuple[int, int]
    fullHarmonics: Harmonics
    keptIndices: ComplexArray


@dataclass(frozen=True)
class PatternedHomogeneousInterfaceWork:
    homogeneousLeft: bool
    homogeneousMatrix: object
    patternedMatrix: object
    reducedMatrix: object
    reducedRhs: object
    fullRhs: object
    columns: object
    rowIndex: object
    leftInverse: object
    size: int
