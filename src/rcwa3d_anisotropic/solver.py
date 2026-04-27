from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Iterable, Mapping, Sequence
import weakref

import numpy as np
from numpy.typing import ArrayLike

from .backend import ArrayBackend as _ArrayBackend
from .backend import cpuBackend as _cpuBackend
from .backend import resolveBackend as _resolveBackend
from .factorization import TensorConvolutionData
from .factorization import constantTensor as _constantTensor
from .factorization import layerTensorData as _layerTensorData
from .factorization import liFactorizedSystemMatrix as _liFactorizedSystemMatrix
from .factorization import tensorConvolutionData as _tensorConvolutionData
from .fourier import Harmonics as _Harmonics
from .fourier import makeHarmonics as _makeHarmonics
from .fourier import normalizeOrders as _normalizeOrders
from .phase import flux as _flux
from .phase import forwardKz as _forwardKz
from .phase import planeWaveFields as _planeWaveFields
from .phase import putOrderField as _putOrderField
from .phase import singleOrderVector as _singleOrderVector


ComplexArray = np.ndarray
TensorLike = object
_CPU_BACKEND = _cpuBackend()
_BackendTensorCacheEntry = tuple[weakref.ReferenceType[TensorConvolutionData], "_BackendTensorConvolutionData"]
_BACKEND_TENSOR_DATA_CACHE: dict[tuple[int, str, str], _BackendTensorCacheEntry] = {}


@dataclass(frozen=True)
class _BackendTensorConvolutionData:
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
        for scalar 2D gratings.  When supplied, the in-plane displacement
        blocks use normal-vector Li factorization, while the z-normal
        elimination remains driven by epsilon_zz.
    factorization:
        Requested scalar-boundary factorization path. ``"auto"`` keeps the
        analytic-disk Jones path when available and otherwise enables the
        vector-field Li route for sampled piecewise-constant scalar grids.
    name:
        Optional label used only by callers/debugging.
    """

    thickness: float
    epsilon: TensorLike
    normalField: ArrayLike | None = None
    factorization: str = "auto"
    name: str = ""


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


@dataclass(frozen=True)
class _SMatrix:
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
    harmonics: _Harmonics
    components: tuple[_SMatrix, ...]
    total: _SMatrix
    layerModes: tuple[tuple[object, object], ...]
    incidentBackward: object
    transmissionForward: object
    zeroIndex: int
    layerEigTimings: tuple[LayerEigTiming, ...] = ()

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class _AutomaticFastPathPlan:
    label: str
    reducedOrders: tuple[int, int]
    fullHarmonics: _Harmonics
    keptIndices: ComplexArray


def prepareStackSMatrix(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    truncation: str = "circular",
    backend: str | _ArrayBackend | None = "cuda",
    fullTotal: bool = True,
    profile: bool = False,
) -> PreparedStack:
    _validateInputs(wavelength, period, orders)
    arrayBackend = _resolveBackend(backend)
    xp = arrayBackend.xp

    harmonics = _makeHarmonics(wavelength, period, orders, epsIncident, theta, phi, truncation=truncation)
    nOrders = harmonics.count
    nPorts = 2 * nOrders
    k0 = 2 * np.pi / wavelength

    incidentForward = arrayBackend.asarray(_homogeneousBasis(harmonics, epsIncident, direction=1))
    incidentBackward = arrayBackend.asarray(_homogeneousBasis(harmonics, epsIncident, direction=-1))
    transmissionForward = arrayBackend.asarray(_homogeneousBasis(harmonics, epsTransmission, direction=1))
    transmissionBackward = arrayBackend.asarray(_homogeneousBasis(harmonics, epsTransmission, direction=-1))

    profileTimings = _profileTimingsEnabled(profile)
    layerModeList: list[tuple[object, object]] = []
    layerEigTimings: list[LayerEigTiming] = []
    for layerIndex, layer in enumerate(layers):
        modes, timing = _layerModesWithTiming(
            layer,
            harmonics,
            arrayBackend,
            layerIndex=layerIndex,
            collectTiming=profileTimings,
        )
        layerModeList.append(modes)
        if timing is not None:
            layerEigTimings.append(timing)
    layerModes = tuple(layerModeList)
    layerHomogeneous = tuple(_isHomogeneousLayer(layer) for layer in layers)
    regionForward = [incidentForward]
    regionBackward = [incidentBackward]
    for _qValues, modeMatrix in layerModes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)
    regionHomogeneous = (True, *layerHomogeneous, True)

    interfaces = _interfaceSMatrices(
        regionForward,
        regionBackward,
        regionHomogeneous,
        nOrders,
        arrayBackend,
    )
    components: list[_SMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(interfaces[regionIndex])
        if regionIndex < len(layers):
            qValues = layerModes[regionIndex][0]
            qForward = qValues[:nPorts]
            qBackward = qValues[nPorts:]
            propagationForward = xp.diag(xp.exp(1j * qForward * k0 * layers[regionIndex].thickness))
            propagationBackward = xp.diag(xp.exp(-1j * qBackward * k0 * layers[regionIndex].thickness))
            components.append(_propagationSMatrixBidirectional(propagationForward, propagationBackward, arrayBackend))

    componentTuple = tuple(components)
    return PreparedStack(
        layers=tuple(layers),
        wavelength=float(wavelength),
        period=period,
        orders=tuple(_normalizeOrders(orders)),
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=float(theta),
        phi=float(phi),
        truncation=harmonics.truncation,
        backend=arrayBackend.name,
        harmonics=harmonics,
        components=componentTuple,
        total=(
            _cascadeMany(componentTuple, nPorts, arrayBackend)
            if fullTotal
            else _reflectionTransmissionOnlySMatrix(componentTuple, nPorts, arrayBackend)
        ),
        layerModes=layerModes,
        incidentBackward=incidentBackward,
        transmissionForward=transmissionForward,
        zeroIndex=_zeroOrderIndex(harmonics),
        layerEigTimings=tuple(layerEigTimings),
    )


def evaluatePreparedStack(
    prepared: PreparedStack,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    *,
    solvedBy: str = "smatrix",
    returnFields: bool = False,
) -> RCWAResult:
    arrayBackend = _resolveBackend(prepared.backend)
    incident = _incidentField(
        harmonics=prepared.harmonics,
        eps=prepared.epsIncident,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )
    incidentFlux = _checkedIncidentFlux(incident)
    incidentAmplitudes = _incidentAmplitudes(prepared, sAmplitude, pAmplitude, arrayBackend)
    rAmplitudes = prepared.total.s11 @ incidentAmplitudes
    tAmplitudes = prepared.total.s21 @ incidentAmplitudes

    if returnFields and prepared.layers:
        layerSolutions = _smatrixLayerSolutions(
            layers=prepared.layers,
            layerModes=prepared.layerModes,
            components=prepared.components,
            incidentAmplitudes=incidentAmplitudes,
            harmonics=prepared.harmonics,
            wavelength=prepared.wavelength,
            period=prepared.period,
            orders=prepared.orders,
            backend=arrayBackend,
        )
    else:
        layerSolutions = ()

    return _result(
        harmonics=prepared.harmonics,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        reflectedBasis=arrayBackend.toNumpy(prepared.incidentBackward),
        transmittedBasis=arrayBackend.toNumpy(prepared.transmissionForward),
        rAmplitudes=arrayBackend.toNumpy(rAmplitudes),
        tAmplitudes=arrayBackend.toNumpy(tAmplitudes),
        incidentFlux=incidentFlux,
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        layerEigTimings=prepared.layerEigTimings,
    )


def evaluatePreparedBatch(
    prepared: PreparedStack,
    excitations: Mapping[str, tuple[complex, complex]],
    *,
    solvedBy: str = "smatrix-batch",
) -> dict[str, RCWAResult]:
    labels = tuple(excitations)
    if not labels:
        return {}

    arrayBackend = _resolveBackend(prepared.backend)
    incidentColumns = arrayBackend.asarray(
        np.column_stack([_incidentAmplitudeVector(prepared, *excitations[label]) for label in labels])
    )
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    reflectedBasis = arrayBackend.toNumpy(prepared.incidentBackward)
    transmittedBasis = arrayBackend.toNumpy(prepared.transmissionForward)
    reflectedNumpy = arrayBackend.toNumpy(reflected)
    transmittedNumpy = arrayBackend.toNumpy(transmitted)

    results: dict[str, RCWAResult] = {}
    for column, label in enumerate(labels):
        sAmplitude, pAmplitude = excitations[label]
        incident = _incidentField(
            harmonics=prepared.harmonics,
            eps=prepared.epsIncident,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
        )
        results[label] = _result(
            harmonics=prepared.harmonics,
            epsIncident=prepared.epsIncident,
            epsTransmission=prepared.epsTransmission,
            reflectedBasis=reflectedBasis,
            transmittedBasis=transmittedBasis,
            rAmplitudes=reflectedNumpy[:, column],
            tAmplitudes=transmittedNumpy[:, column],
            incidentFlux=_checkedIncidentFlux(incident),
            solvedBy=solvedBy,
            layerSolutions=(),
            layerEigTimings=prepared.layerEigTimings,
        )
    return results


def solveStack(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    returnFields: bool = False,
    method: str = "smatrix",
    truncation: str = "circular",
    backend: str | _ArrayBackend | None = "cuda",
    profile: bool = False,
) -> RCWAResult:
    """Solve a 2D-periodic stack with tensor permittivity in finite layers.

    The finite layers may include non-zero xz/zx tensor components.  The layer
    eigenproblem uses the full 4N first-order Maxwell matrix with Li inverse
    factorization through ``epsilon_zz``.  Angles are in radians. Harmonic
    orders are ``(nx, ny)``, producing ``mx=-nx..nx`` and ``my=-ny..ny``.
    """

    _requireSMatrixMethod(method)
    return _solveStackSMatrix(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        returnFields=returnFields,
        truncation=truncation,
        backend=backend,
        profile=profile,
    )


def solveStackBatch(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    method: str = "smatrix",
    truncation: str = "circular",
    backend: str | _ArrayBackend | None = "cuda",
    profile: bool = False,
) -> dict[str, RCWAResult]:
    """Solve several incident polarizations while reusing layer modes.

    ``excitations`` maps result labels to ``(sAmplitude, pAmplitude)``.  The
    S-matrix path builds layer eigenmodes and cascades the stack once, then
    applies the resulting network to each incident vector.
    """

    _requireSMatrixMethod(method)
    return _solveStackSMatrixBatch(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        excitations=excitations,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=backend,
        profile=profile,
    )


def compileLayers(
    layers: Sequence[Layer],
    orders: int | tuple[int, int],
    truncation: str = "circular",
) -> tuple[CompiledLayer, ...]:
    """Precompute tensor convolution matrices for a fixed harmonic truncation."""

    normalizedOrders = _normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")
    harmonics = _makeHarmonics(
        wavelength=1.0,
        period=(1.0, 1.0),
        orders=normalizedOrders,
        epsIncident=1.0,
        theta=0.0,
        phi=0.0,
        truncation=truncation,
    )
    compiledLayers = []
    for layer in layers:
        compiledLayers.append(
            CompiledLayer(
                thickness=layer.thickness,
                tensorData=_tensorConvolutionData(
                    layer.epsilon,
                    harmonics,
                    normalField=layer.normalField,
                    factorization=getattr(layer, "factorization", "auto"),
                ),
                orders=normalizedOrders,
                truncation=harmonics.truncation,
                normalField=layer.normalField,
                factorization=getattr(layer, "factorization", "auto"),
                name=layer.name,
            )
        )
    return tuple(compiledLayers)


def _solveStackSMatrix(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    returnFields: bool = False,
    truncation: str = "circular",
    backend: str | _ArrayBackend | None = "cuda",
    profile: bool = False,
) -> RCWAResult:
    fastPlan = _automaticFastPathPlan(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        theta=theta,
        phi=phi,
        truncation=truncation,
        returnFields=returnFields,
    )
    if fastPlan is not None:
        return _solveStackReducedSMatrix(
            layers=layers,
            wavelength=wavelength,
            period=period,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            truncation=truncation,
            backend=backend,
            plan=fastPlan,
            profile=profile,
        )

    prepared = prepareStackSMatrix(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=backend,
        fullTotal=returnFields,
        profile=profile,
    )
    return evaluatePreparedStack(
        prepared,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{prepared.backend}",
        returnFields=returnFields,
    )


def _solveStackSMatrixBatch(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    truncation: str = "circular",
    backend: str | _ArrayBackend | None = "cuda",
    profile: bool = False,
) -> dict[str, RCWAResult]:
    fastPlan = _automaticFastPathPlan(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        theta=theta,
        phi=phi,
        truncation=truncation,
        returnFields=False,
    )
    if fastPlan is not None:
        return _solveStackReducedSMatrixBatch(
            layers=layers,
            wavelength=wavelength,
            period=period,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            excitations=excitations,
            truncation=truncation,
            backend=backend,
            plan=fastPlan,
            profile=profile,
        )

    prepared = prepareStackSMatrix(
        layers=layers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=backend,
        fullTotal=False,
        profile=profile,
    )
    return evaluatePreparedBatch(prepared, excitations, solvedBy=f"smatrix-batch-{prepared.backend}")


def _solveStackReducedSMatrix(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    epsIncident: complex,
    epsTransmission: complex,
    theta: float,
    phi: float,
    sAmplitude: complex,
    pAmplitude: complex,
    truncation: str,
    backend: str | _ArrayBackend | None,
    plan: _AutomaticFastPathPlan,
    profile: bool = False,
) -> RCWAResult:
    reducedLayers = _reducedFastPathLayers(layers, plan)
    prepared = prepareStackSMatrix(
        layers=reducedLayers,
        wavelength=wavelength,
        period=period,
        orders=plan.reducedOrders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=backend,
        fullTotal=False,
        profile=profile,
    )
    reduced = evaluatePreparedStack(
        prepared,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{plan.label}-{prepared.backend}",
        returnFields=False,
    )
    return _embedReducedResult(
        reduced,
        fullHarmonics=plan.fullHarmonics,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{plan.label}-{prepared.backend}",
    )


def _solveStackReducedSMatrixBatch(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    epsIncident: complex,
    epsTransmission: complex,
    theta: float,
    phi: float,
    excitations: Mapping[str, tuple[complex, complex]],
    truncation: str,
    backend: str | _ArrayBackend | None,
    plan: _AutomaticFastPathPlan,
    profile: bool = False,
) -> dict[str, RCWAResult]:
    reducedLayers = _reducedFastPathLayers(layers, plan)
    prepared = prepareStackSMatrix(
        layers=reducedLayers,
        wavelength=wavelength,
        period=period,
        orders=plan.reducedOrders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=backend,
        fullTotal=False,
        profile=profile,
    )
    reducedResults = evaluatePreparedBatch(
        prepared,
        excitations,
        solvedBy=f"smatrix-batch-{plan.label}-{prepared.backend}",
    )
    return {
        label: _embedReducedResult(
            reduced,
            fullHarmonics=plan.fullHarmonics,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            sAmplitude=excitations[label][0],
            pAmplitude=excitations[label][1],
            solvedBy=f"smatrix-batch-{plan.label}-{prepared.backend}",
        )
        for label, reduced in reducedResults.items()
    }


def _automaticFastPathPlan(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex,
    theta: float,
    phi: float,
    truncation: str,
    returnFields: bool,
) -> _AutomaticFastPathPlan | None:
    if returnFields:
        return None

    normalizedOrders = _normalizeOrders(orders)
    if normalizedOrders == (0, 0):
        return None

    fullHarmonics = _makeHarmonics(
        wavelength,
        period,
        normalizedOrders,
        epsIncident,
        theta,
        phi,
        truncation=truncation,
    )
    if _hasMismatchedCompiledLayer(layers, fullHarmonics):
        return None
    if all(_isHomogeneousLayer(layer) for layer in layers):
        return _makeFastPathPlan("homogeneous-4x4", (0, 0), fullHarmonics)

    nx, ny = normalizedOrders
    if ny > 0 and _stackInvariantAlong("y", layers, fullHarmonics):
        return _makeFastPathPlan("1d-x-4x4", (nx, 0), fullHarmonics)
    if nx > 0 and _stackInvariantAlong("x", layers, fullHarmonics):
        return _makeFastPathPlan("1d-y-4x4", (0, ny), fullHarmonics)
    return None


def _makeFastPathPlan(
    label: str,
    reducedOrders: tuple[int, int],
    fullHarmonics: _Harmonics,
) -> _AutomaticFastPathPlan:
    kept = _reducedHarmonicIndices(fullHarmonics, reducedOrders)
    return _AutomaticFastPathPlan(
        label=label,
        reducedOrders=reducedOrders,
        fullHarmonics=fullHarmonics,
        keptIndices=kept,
    )


def _reducedFastPathLayers(
    layers: Sequence[Layer | CompiledLayer],
    plan: _AutomaticFastPathPlan,
) -> tuple[Layer | CompiledLayer, ...]:
    reduced: list[Layer | CompiledLayer] = []
    for layer in layers:
        if _isHomogeneousLayer(layer):
            reduced.append(_homogeneousEquivalentLayer(layer))
        elif isinstance(layer, CompiledLayer):
            reduced.append(_sliceCompiledLayer(layer, plan))
        else:
            reduced.append(layer)
    return tuple(reduced)


def _homogeneousEquivalentLayer(layer: Layer | CompiledLayer) -> Layer:
    if isinstance(layer, CompiledLayer):
        tensor = layer.tensorData.constantTensor
        if tensor is None:
            raise RuntimeError("compiled layer is not homogeneous")
        return Layer(thickness=layer.thickness, epsilon=tensor, name=layer.name)
    tensor = _constantTensor(getattr(layer, "epsilon"))
    if tensor is None:
        raise RuntimeError("layer is not homogeneous")
    return Layer(thickness=layer.thickness, epsilon=tensor, name=getattr(layer, "name", ""))


def _sliceCompiledLayer(layer: CompiledLayer, plan: _AutomaticFastPathPlan) -> CompiledLayer:
    indices = plan.keptIndices
    indexer = np.ix_(indices, indices)
    tensorData = layer.tensorData
    components = tuple(
        tuple(np.asarray(component)[indexer].copy() for component in row)
        for row in tensorData.components
    )
    reducedData = TensorConvolutionData(
        components=components,
        etaZz=np.asarray(tensorData.etaZz)[indexer].copy(),
        constantTensor=tensorData.constantTensor,
        factorization=tensorData.factorization,
    )
    return CompiledLayer(
        thickness=layer.thickness,
        tensorData=reducedData,
        orders=plan.reducedOrders,
        truncation=plan.fullHarmonics.truncation,
        normalField=None,
        factorization=layer.factorization,
        name=layer.name,
    )


def _embedReducedResult(
    reduced: RCWAResult,
    *,
    fullHarmonics: _Harmonics,
    epsIncident: complex,
    epsTransmission: complex,
    sAmplitude: complex,
    pAmplitude: complex,
    solvedBy: str,
) -> RCWAResult:
    nPorts = 2 * fullHarmonics.count
    rAmplitudes = np.zeros(nPorts, dtype=complex)
    tAmplitudes = np.zeros(nPorts, dtype=complex)

    fullOrderToIndex = {
        (int(mx), int(my)): index
        for index, (mx, my) in enumerate(zip(fullHarmonics.mx, fullHarmonics.my))
    }
    for reducedIndex, order in enumerate(reduced.orders):
        fullIndex = fullOrderToIndex[(order.mx, order.my)]
        rAmplitudes[2 * fullIndex : 2 * fullIndex + 2] = reduced.rAmplitudes[
            2 * reducedIndex : 2 * reducedIndex + 2
        ]
        tAmplitudes[2 * fullIndex : 2 * fullIndex + 2] = reduced.tAmplitudes[
            2 * reducedIndex : 2 * reducedIndex + 2
        ]

    incident = _incidentField(
        harmonics=fullHarmonics,
        eps=epsIncident,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )
    return _result(
        harmonics=fullHarmonics,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        reflectedBasis=_homogeneousBasis(fullHarmonics, epsIncident, direction=-1),
        transmittedBasis=_homogeneousBasis(fullHarmonics, epsTransmission, direction=1),
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        incidentFlux=_checkedIncidentFlux(incident),
        solvedBy=solvedBy,
        layerSolutions=(),
        layerEigTimings=reduced.layerEigTimings,
    )


def _reducedHarmonicIndices(harmonics: _Harmonics, reducedOrders: tuple[int, int]) -> ComplexArray:
    nx, ny = reducedOrders
    mask = (np.abs(harmonics.mx) <= nx) & (np.abs(harmonics.my) <= ny)
    return np.flatnonzero(mask)


def _stackInvariantAlong(axis: str, layers: Sequence[Layer | CompiledLayer], harmonics: _Harmonics) -> bool:
    return all(_layerInvariantAlong(axis, layer, harmonics) for layer in layers)


def _hasMismatchedCompiledLayer(layers: Sequence[Layer | CompiledLayer], harmonics: _Harmonics) -> bool:
    return any(
        isinstance(layer, CompiledLayer)
        and (layer.orders != harmonics.orders or layer.truncation != harmonics.truncation)
        for layer in layers
    )


def _layerInvariantAlong(axis: str, layer: Layer | CompiledLayer, harmonics: _Harmonics) -> bool:
    if _isHomogeneousLayer(layer):
        return True
    if isinstance(layer, CompiledLayer):
        return _compiledLayerInvariantAlong(axis, layer, harmonics)
    return _rawLayerInvariantAlong(axis, layer)


def _rawLayerInvariantAlong(axis: str, layer: Layer) -> bool:
    epsilon = getattr(layer, "epsilon")
    if _constantTensor(epsilon) is not None:
        return True
    if not _epsilonInvariantAlong(axis, epsilon):
        return False
    normal = getattr(layer, "normalField", None)
    return normal is None or _arrayInvariantAlong(axis, np.asarray(normal))


def _epsilonInvariantAlong(axis: str, epsilon: TensorLike) -> bool:
    if isinstance(epsilon, Mapping):
        return all(_arrayInvariantAlong(axis, np.asarray(value)) for value in epsilon.values())
    if np.isscalar(epsilon):
        return True
    array = np.asarray(epsilon)
    if array.ndim == 0 or (array.ndim == 2 and array.shape == (3, 3)):
        return True
    if array.ndim in (2, 4):
        return _arrayInvariantAlong(axis, array)
    return False


def _arrayInvariantAlong(axis: str, array: ComplexArray) -> bool:
    if array.ndim < 2:
        return True
    if axis == "y":
        reference = array[:1, ...]
        return bool(np.allclose(array, reference, rtol=1e-10, atol=1e-12))
    if axis == "x":
        reference = array[:, :1, ...]
        return bool(np.allclose(array, reference, rtol=1e-10, atol=1e-12))
    raise ValueError("axis must be 'x' or 'y'")


def _compiledLayerInvariantAlong(axis: str, layer: CompiledLayer, harmonics: _Harmonics) -> bool:
    if layer.orders != harmonics.orders or layer.truncation != harmonics.truncation:
        return False
    if layer.tensorData.constantTensor is not None:
        return True
    if axis == "y":
        uncoupled = harmonics.my[:, None] != harmonics.my[None, :]
    elif axis == "x":
        uncoupled = harmonics.mx[:, None] != harmonics.mx[None, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")

    matrices = [
        *[component for row in layer.tensorData.components for component in row],
        layer.tensorData.etaZz,
    ]
    scale = max(1.0, max(float(np.max(np.abs(matrix))) for matrix in matrices if matrix.size))
    tolerance = 1e-10 * scale
    return all(np.max(np.abs(matrix[uncoupled])) <= tolerance for matrix in matrices)


def _layerModes(
    layer: Layer | CompiledLayer,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
) -> tuple[object, object]:
    modes, _timing = _layerModesWithTiming(
        layer,
        harmonics,
        backend,
        layerIndex=0,
        collectTiming=False,
    )
    return modes


def _layerModesWithTiming(
    layer: Layer | CompiledLayer,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
    *,
    layerIndex: int,
    collectTiming: bool,
) -> tuple[tuple[object, object], LayerEigTiming | None]:
    if not isinstance(layer, CompiledLayer) and getattr(layer, "normalField", None) is None:
        tensor = _constantTensor(getattr(layer, "epsilon"))
        if tensor is not None:
            qValues, vectors, matrixShape, eigTime = _homogeneousTensorLayerModesMeasured(
                tensor,
                harmonics,
                collectTiming=collectTiming,
            )
            modes = (backend.asarray(qValues), backend.asarray(vectors))
            return modes, _layerEigTiming(layerIndex, layer, "homogeneous-4x4", matrixShape, eigTime, collectTiming)

    factorized = _layerTensorData(layer, harmonics)
    if factorized.constantTensor is not None:
        qValues, vectors, matrixShape, eigTime = _homogeneousTensorLayerModesMeasured(
            factorized.constantTensor,
            harmonics,
            collectTiming=collectTiming,
        )
        modes = (backend.asarray(qValues), backend.asarray(vectors))
        return modes, _layerEigTiming(layerIndex, layer, "homogeneous-4x4", matrixShape, eigTime, collectTiming)
    if _hasNoLongitudinalCoupling(factorized):
        timing: dict[str, object] | None = {} if collectTiming else None
        modes = _transverseBlockLayerModes(factorized, harmonics, backend, timing=timing)
        return modes, _layerEigTimingFromDict(layerIndex, layer, "transverse-2N", timing)
    if backend.isTorch:
        system = _liFactorizedSystemMatrixBackend(factorized, harmonics, backend)
    else:
        system = backend.asarray(_liFactorizedSystemMatrix(factorized, harmonics))
    start = _startTimedOperation(backend) if collectTiming else None
    qValues, vectors = backend.eig(system)
    eigTime = _finishTimedOperation(backend, start) if start is not None else 0.0
    vectors = _normalizeModes(vectors, backend)
    modes = _splitForwardBackward(qValues, vectors, 2 * harmonics.count, backend)
    return modes, _layerEigTiming(layerIndex, layer, "full-4N", tuple(system.shape), eigTime, collectTiming)


def _profileTimingsEnabled(profile: bool) -> bool:
    if profile:
        return True
    value = os.environ.get("RCWA3D_PROFILE_EIG", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _layerEigTiming(
    layerIndex: int,
    layer: Layer | CompiledLayer,
    kind: str,
    matrixShape: tuple[int, ...],
    eigTime: float,
    collectTiming: bool,
) -> LayerEigTiming | None:
    if not collectTiming:
        return None
    return LayerEigTiming(
        layerIndex=layerIndex,
        name=getattr(layer, "name", ""),
        kind=kind,
        matrixShape=tuple(int(value) for value in matrixShape),
        eigTimeSeconds=float(eigTime),
    )


def _layerEigTimingFromDict(
    layerIndex: int,
    layer: Layer | CompiledLayer,
    kind: str,
    timing: dict[str, object] | None,
) -> LayerEigTiming | None:
    if timing is None:
        return None
    return _layerEigTiming(
        layerIndex,
        layer,
        kind,
        tuple(timing.get("matrixShape", ())),
        float(timing.get("eigTimeSeconds", 0.0)),
        True,
    )


def _startTimedOperation(backend: _ArrayBackend) -> float:
    backend.synchronize()
    return time.perf_counter()


def _finishTimedOperation(backend: _ArrayBackend, start: float) -> float:
    backend.synchronize()
    return time.perf_counter() - start


def _hasNoLongitudinalCoupling(data: TensorConvolutionData) -> bool:
    longitudinal = (
        data.components[0][2],
        data.components[1][2],
        data.components[2][0],
        data.components[2][1],
    )
    scaleMatrices = (*longitudinal, data.components[2][2])
    scale = max(1.0, max(float(np.max(np.abs(matrix))) for matrix in scaleMatrices if matrix.size))
    tolerance = 1e-14 * scale
    return all(float(np.max(np.abs(matrix))) <= tolerance for matrix in longitudinal if matrix.size)


def _transverseBlockLayerModes(
    data: TensorConvolutionData,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
    timing: dict[str, object] | None = None,
) -> tuple[object, object]:
    if backend.isTorch:
        return _transverseBlockLayerModesBackend(data, harmonics, backend, timing=timing)
    return _transverseBlockLayerModesNumpy(data, harmonics, backend, timing=timing)


def _transverseBlockLayerModesBackend(
    data: TensorConvolutionData,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
    timing: dict[str, object] | None = None,
) -> tuple[object, object]:
    xp = backend.xp
    torch = xp._torch
    n = harmonics.count
    diagonal = xp.arange(n)
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)
    cached = _backendTensorConvolutionData(data, backend)
    c = cached.components
    eta = cached.etaZz
    cxx, cxy = c[0][0], c[0][1]
    cyx, cyy = c[1][0], c[1][1]

    identity = xp.eye(n, dtype=complex)
    p11 = kx[:, None] * eta * ky[None, :]
    p12 = identity - kx[:, None] * eta * kx[None, :]
    p21 = ky[:, None] * eta * ky[None, :] - identity
    p22 = -ky[:, None] * eta * kx[None, :]

    q11 = backend.copy(-cyx)
    q11[diagonal, diagonal] = q11[diagonal, diagonal] - kx * ky
    q12 = backend.copy(-cyy)
    q12[diagonal, diagonal] = q12[diagonal, diagonal] + kx * kx
    q21 = backend.copy(cxx)
    q21[diagonal, diagonal] = q21[diagonal, diagonal] - ky * ky
    q22 = backend.copy(cxy)
    q22[diagonal, diagonal] = q22[diagonal, diagonal] + ky * kx

    pMatrix = xp.empty((2 * n, 2 * n), dtype=complex)
    pMatrix[0:n, 0:n] = p11
    pMatrix[0:n, n : 2 * n] = p12
    pMatrix[n : 2 * n, 0:n] = p21
    pMatrix[n : 2 * n, n : 2 * n] = p22
    qMatrix = xp.empty((2 * n, 2 * n), dtype=complex)
    qMatrix[0:n, 0:n] = q11
    qMatrix[0:n, n : 2 * n] = q12
    qMatrix[n : 2 * n, 0:n] = q21
    qMatrix[n : 2 * n, n : 2 * n] = q22

    eigenMatrix = pMatrix @ qMatrix
    if timing is not None:
        timing["matrixShape"] = tuple(eigenMatrix.shape)
    start = _startTimedOperation(backend) if timing is not None else None
    qSquared, electricModes = backend.eig(eigenMatrix)
    if start is not None:
        timing["eigTimeSeconds"] = _finishTimedOperation(backend, start)
    qValues = _forwardKzTorch(qSquared, torch)
    safeQ = torch.where(
        torch.abs(qValues) < 1e-13,
        torch.as_tensor(1e-13, dtype=qValues.dtype, device=qValues.device),
        qValues,
    )
    magneticModes = qMatrix @ (electricModes * (1.0 / safeQ)[None, :])
    vectors = xp.concatenate(
        [
            xp.concatenate([electricModes, electricModes], axis=1),
            xp.concatenate([magneticModes, -magneticModes], axis=1),
        ],
        axis=0,
    )
    qAll = xp.concatenate([qValues, -qValues])
    return qAll, _normalizeModes(vectors, backend)


def _transverseBlockLayerModesNumpy(
    data: TensorConvolutionData,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
    timing: dict[str, object] | None = None,
) -> tuple[object, object]:
    n = harmonics.count
    diagonal = np.arange(n)
    kx = harmonics.kx
    ky = harmonics.ky
    c = data.components
    eta = data.etaZz
    cxx, cxy = c[0][0], c[0][1]
    cyx, cyy = c[1][0], c[1][1]

    identity = np.eye(n, dtype=complex)
    p11 = kx[:, None] * eta * ky[None, :]
    p12 = identity - kx[:, None] * eta * kx[None, :]
    p21 = ky[:, None] * eta * ky[None, :] - identity
    p22 = -ky[:, None] * eta * kx[None, :]

    q11 = -cyx.copy()
    q11[diagonal, diagonal] -= kx * ky
    q12 = -cyy.copy()
    q12[diagonal, diagonal] += kx * kx
    q21 = cxx.copy()
    q21[diagonal, diagonal] -= ky * ky
    q22 = cxy.copy()
    q22[diagonal, diagonal] += ky * kx

    pMatrix = np.block([[p11, p12], [p21, p22]])
    qMatrix = np.block([[q11, q12], [q21, q22]])
    eigenMatrix = pMatrix @ qMatrix
    if timing is not None:
        timing["matrixShape"] = tuple(eigenMatrix.shape)
    start = _startTimedOperation(backend) if timing is not None else None
    qSquared, electricModes = np.linalg.eig(eigenMatrix)
    if start is not None:
        timing["eigTimeSeconds"] = _finishTimedOperation(backend, start)
    qValues = _forwardKz(qSquared)
    safeQ = qValues.copy()
    safeQ[np.abs(safeQ) < 1e-13] = 1e-13
    magneticModes = qMatrix @ (electricModes * (1.0 / safeQ)[None, :])
    vectors = np.block([[electricModes, electricModes], [magneticModes, -magneticModes]])
    qAll = np.concatenate([qValues, -qValues])
    return backend.asarray(qAll), backend.asarray(_normalizeModes(vectors, _CPU_BACKEND))


def _forwardKzTorch(values: object, torch: object) -> object:
    roots = torch.sqrt(values)
    flip = (torch.imag(roots) < -1e-14) | (
        (torch.abs(torch.imag(roots)) <= 1e-14) & (torch.real(roots) < 0)
    )
    return torch.where(flip, -roots, roots)


def _liFactorizedSystemMatrixBackend(
    data: TensorConvolutionData,
    harmonics: _Harmonics,
    backend: _ArrayBackend,
) -> object:
    xp = backend.xp
    n = harmonics.count
    diagonal = xp.arange(n)
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)
    c = _backendTensorConvolutionData(data, backend).components
    eta = _backendTensorConvolutionData(data, backend).etaZz

    cxx, cxy, cxz = c[0]
    cyx, cyy, cyz = c[1]
    czx, czy, _czz = c[2]

    cxzEta = cxz @ eta
    cyzEta = cyz @ eta
    etaCzx = eta @ czx
    etaCzy = eta @ czy

    dxEx = cxx - cxzEta @ czx
    dxEy = cxy - cxzEta @ czy
    dxHx = cxzEta * ky[None, :]
    dxHy = -cxzEta * kx[None, :]

    dyEx = cyx - cyzEta @ czx
    dyEy = cyy - cyzEta @ czy
    dyHx = cyzEta * ky[None, :]
    dyHy = -cyzEta * kx[None, :]

    identity = xp.eye(n, dtype=complex)
    a11 = -kx[:, None] * etaCzx
    a12 = -kx[:, None] * etaCzy
    a13 = kx[:, None] * eta * ky[None, :]
    a14 = identity - kx[:, None] * eta * kx[None, :]

    a21 = -ky[:, None] * etaCzx
    a22 = -ky[:, None] * etaCzy
    a23 = ky[:, None] * eta * ky[None, :] - identity
    a24 = -ky[:, None] * eta * kx[None, :]

    a31 = backend.copy(-dyEx)
    a31[diagonal, diagonal] = a31[diagonal, diagonal] - kx * ky
    a32 = backend.copy(-dyEy)
    a32[diagonal, diagonal] = a32[diagonal, diagonal] + kx * kx
    a33 = -dyHx
    a34 = -dyHy

    a41 = backend.copy(dxEx)
    a41[diagonal, diagonal] = a41[diagonal, diagonal] - ky * ky
    a42 = backend.copy(dxEy)
    a42[diagonal, diagonal] = a42[diagonal, diagonal] + ky * kx
    a43 = dxHx
    a44 = dxHy

    system = xp.empty((4 * n, 4 * n), dtype=complex)
    system[0:n, 0:n] = a11
    system[0:n, n : 2 * n] = a12
    system[0:n, 2 * n : 3 * n] = a13
    system[0:n, 3 * n : 4 * n] = a14
    system[n : 2 * n, 0:n] = a21
    system[n : 2 * n, n : 2 * n] = a22
    system[n : 2 * n, 2 * n : 3 * n] = a23
    system[n : 2 * n, 3 * n : 4 * n] = a24
    system[2 * n : 3 * n, 0:n] = a31
    system[2 * n : 3 * n, n : 2 * n] = a32
    system[2 * n : 3 * n, 2 * n : 3 * n] = a33
    system[2 * n : 3 * n, 3 * n : 4 * n] = a34
    system[3 * n : 4 * n, 0:n] = a41
    system[3 * n : 4 * n, n : 2 * n] = a42
    system[3 * n : 4 * n, 2 * n : 3 * n] = a43
    system[3 * n : 4 * n, 3 * n : 4 * n] = a44
    return system


def _backendTensorConvolutionData(
    data: TensorConvolutionData,
    backend: _ArrayBackend,
) -> _BackendTensorConvolutionData:
    key = (id(data), backend.name, str(backend.device))
    entry = _BACKEND_TENSOR_DATA_CACHE.get(key)
    if entry is not None:
        dataRef, cached = entry
        if dataRef() is data:
            return cached
        _BACKEND_TENSOR_DATA_CACHE.pop(key, None)

    cached = _BackendTensorConvolutionData(
        components=tuple(tuple(backend.asarray(component) for component in row) for row in data.components),
        etaZz=backend.asarray(data.etaZz),
    )

    def forget(_ref: weakref.ReferenceType[TensorConvolutionData], cacheKey: tuple[int, str, str] = key) -> None:
        _BACKEND_TENSOR_DATA_CACHE.pop(cacheKey, None)

    _BACKEND_TENSOR_DATA_CACHE[key] = (weakref.ref(data, forget), cached)
    return cached


def _isHomogeneousLayer(layer: Layer | CompiledLayer) -> bool:
    if isinstance(layer, CompiledLayer):
        return layer.tensorData.constantTensor is not None
    if getattr(layer, "normalField", None) is not None:
        return False
    return _constantTensor(getattr(layer, "epsilon")) is not None


def _homogeneousTensorLayerModes(tensor: ComplexArray, harmonics: _Harmonics) -> tuple[ComplexArray, ComplexArray]:
    qValues, vectors, _matrixShape, _eigTime = _homogeneousTensorLayerModesMeasured(
        tensor,
        harmonics,
        collectTiming=False,
    )
    return qValues, vectors


def _homogeneousTensorLayerModesMeasured(
    tensor: ComplexArray,
    harmonics: _Harmonics,
    *,
    collectTiming: bool,
) -> tuple[ComplexArray, ComplexArray, tuple[int, ...], float]:
    nOrders = harmonics.count
    systems = np.stack(
        [_homogeneousOrderSystemMatrix(tensor, kx, ky) for kx, ky in zip(harmonics.kx, harmonics.ky)],
        axis=0,
    )
    start = time.perf_counter() if collectTiming else None
    orderQ, orderVectors = np.linalg.eig(systems)
    eigTime = time.perf_counter() - start if start is not None else 0.0

    qValues = orderQ.reshape(-1)
    vectors = np.zeros((4 * nOrders, 4 * nOrders), dtype=complex)
    baseColumns = 4 * np.arange(nOrders)
    for localIndex in range(4):
        columns = baseColumns + localIndex
        vectors[np.arange(nOrders), columns] = orderVectors[:, 0, localIndex]
        vectors[nOrders + np.arange(nOrders), columns] = orderVectors[:, 1, localIndex]
        vectors[2 * nOrders + np.arange(nOrders), columns] = orderVectors[:, 2, localIndex]
        vectors[3 * nOrders + np.arange(nOrders), columns] = orderVectors[:, 3, localIndex]
    vectors = _normalizeModes(vectors, _CPU_BACKEND)
    qValues, vectors = _splitForwardBackward(qValues, vectors, 2 * nOrders, _CPU_BACKEND)
    return qValues, vectors, tuple(systems.shape), eigTime


def _homogeneousOrderSystemMatrix(tensor: ComplexArray, kx: complex, ky: complex) -> ComplexArray:
    exx, exy, exz = tensor[0]
    eyx, eyy, eyz = tensor[1]
    ezx, ezy, ezz = tensor[2]
    if abs(ezz) < 1e-14:
        raise ValueError("epsilon_zz is near zero in a homogeneous anisotropic layer")
    eta = 1.0 / ezz

    dxEx = exx - exz * eta * ezx
    dxEy = exy - exz * eta * ezy
    dxHx = exz * eta * ky
    dxHy = -exz * eta * kx

    dyEx = eyx - eyz * eta * ezx
    dyEy = eyy - eyz * eta * ezy
    dyHx = eyz * eta * ky
    dyHy = -eyz * eta * kx

    return np.array(
        [
            [-kx * eta * ezx, -kx * eta * ezy, kx * eta * ky, 1.0 - kx * eta * kx],
            [-ky * eta * ezx, -ky * eta * ezy, ky * eta * ky - 1.0, -ky * eta * kx],
            [-kx * ky - dyEx, kx * kx - dyEy, -dyHx, -dyHy],
            [dxEx - ky * ky, dxEy + ky * kx, dxHx, dxHy],
        ],
        dtype=complex,
    )


def _splitForwardBackward(
    qValues: object,
    vectors: object,
    nForward: int,
    backend: _ArrayBackend,
) -> tuple[object, object]:
    fluxes = _modeFluxes(vectors, backend)
    qNumpy = backend.toNumpy(qValues)
    fluxNumpy = backend.toNumpy(fluxes)
    forwardIndices = _forwardModeIndices(qNumpy, fluxNumpy, nForward)
    forwardSet = set(int(index) for index in forwardIndices)
    backwardIndices = np.array([index for index in range(qNumpy.size) if index not in forwardSet], dtype=int)
    if backwardIndices.size != nForward:
        raise RuntimeError("anisotropic layer mode split did not produce equal forward/backward spaces")

    forwardIndices = _sortModes(forwardIndices, qNumpy, fluxNumpy, forward=True)
    backwardIndices = _sortModes(backwardIndices, qNumpy, fluxNumpy, forward=False)
    indices = np.concatenate([forwardIndices, backwardIndices])
    return qValues[indices], vectors[:, indices]


def _forwardModeIndices(qValues: ComplexArray, fluxes: ComplexArray, nForward: int) -> ComplexArray:
    fluxScale = max(1.0, float(np.max(np.abs(fluxes))) if fluxes.size else 1.0)
    imagScale = max(1.0, float(np.max(np.abs(np.imag(qValues)))) if qValues.size else 1.0)
    fluxTolerance = 1e-9 * fluxScale

    scores = fluxes / fluxScale
    evanescent = np.abs(fluxes) <= fluxTolerance
    scores = np.asarray(scores, dtype=float)
    scores[evanescent] = np.imag(qValues[evanescent]) / imagScale

    if np.max(np.abs(scores)) <= 1e-14:
        scores = np.real(qValues) / max(1.0, float(np.max(np.abs(np.real(qValues)))))

    # Highest score means positive z-directed power or decay in +z.
    return np.argsort(scores)[-nForward:]


def _modeFluxes(vectors: object, backend: _ArrayBackend) -> object:
    xp = backend.xp
    nOrders = vectors.shape[0] // 4
    ex = vectors[:nOrders, :]
    ey = vectors[nOrders : 2 * nOrders, :]
    hx = vectors[2 * nOrders : 3 * nOrders, :]
    hy = vectors[3 * nOrders :, :]
    return 0.5 * xp.real(xp.sum(ex * xp.conj(hy) - ey * xp.conj(hx), axis=0))


def _sortModes(indices: ComplexArray, qValues: ComplexArray, fluxes: ComplexArray, forward: bool) -> ComplexArray:
    sign = -1.0 if forward else 1.0
    return np.array(
        sorted(
            [int(index) for index in indices],
            key=lambda index: (
                sign * fluxes[index],
                sign * np.imag(qValues[index]),
                sign * np.real(qValues[index]),
                index,
            ),
        ),
        dtype=int,
    )


def _normalizeModes(vectors: object, backend: _ArrayBackend) -> object:
    xp = backend.xp
    amplitudes = xp.max(xp.abs(vectors), axis=0)
    fluxes = xp.abs(_modeFluxes(vectors, backend))
    fluxful = fluxes > 1e-12
    scales = xp.where(fluxful, xp.sqrt(fluxes), amplitudes)
    scales = xp.where(scales > 0, scales, 1.0)
    normalized = vectors / scales

    pivotIndices = xp.argmax(xp.abs(normalized), axis=0)
    pivotValues = normalized[pivotIndices, xp.arange(normalized.shape[1])]
    phases = xp.where(xp.abs(pivotValues) > 0, pivotValues / xp.abs(pivotValues), 1.0 + 0.0j)
    return normalized / phases


def _propagationSMatrixBidirectional(forward: object, backward: object, backend: _ArrayBackend) -> _SMatrix:
    zeroForward = backend.xp.zeros_like(forward)
    return _SMatrix(
        s11=backend.copy(zeroForward),
        s12=backend.copy(backward),
        s21=backend.copy(forward),
        s22=backend.copy(zeroForward),
        isPropagation=True,
    )


def _identitySMatrix(size: int, backend: _ArrayBackend) -> _SMatrix:
    xp = backend.xp
    zero = xp.zeros((size, size), dtype=complex)
    identity = xp.eye(size, dtype=complex)
    return _SMatrix(
        s11=backend.copy(zero),
        s12=backend.copy(identity),
        s21=backend.copy(identity),
        s22=backend.copy(zero),
        isIdentity=True,
    )


def _interfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    backend: _ArrayBackend,
    *,
    homogeneousLeft: bool = False,
    homogeneousRight: bool = False,
    nOrders: int | None = None,
) -> _SMatrix:
    if homogeneousLeft and homogeneousRight and nOrders is not None and (not backend.isGpu or nOrders >= 180):
        blockResult = _homogeneousInterfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            nOrders,
            backend,
        )
        if blockResult is not None:
            return blockResult

    xp = backend.xp
    size = leftForward.shape[1]
    matrix = xp.concatenate([leftBackward, -rightForward], axis=1)
    rhsLeft = -leftForward
    rhsRight = rightBackward
    solved = _solveInterfaceBlocks(backend, matrix, rhsLeft, rhsRight)
    return _SMatrix(
        s11=solved[:size, :size],
        s12=solved[:size, size:],
        s21=solved[size:, :size],
        s22=solved[size:, size:],
    )


def _interfaceSMatrices(
    regionForward: Sequence[object],
    regionBackward: Sequence[object],
    regionHomogeneous: Sequence[bool],
    nOrders: int,
    backend: _ArrayBackend,
) -> tuple[_SMatrix, ...]:
    if not backend.isTorch:
        return tuple(
            _interfaceSMatrix(
                regionForward[index],
                regionBackward[index],
                regionForward[index + 1],
                regionBackward[index + 1],
                backend,
                homogeneousLeft=regionHomogeneous[index],
                homogeneousRight=regionHomogeneous[index + 1],
                nOrders=nOrders,
            )
            for index in range(len(regionForward) - 1)
        )

    xp = backend.xp
    results: list[_SMatrix | None] = [None] * (len(regionForward) - 1)
    batchedMatrices = []
    batchedRhs = []
    batchedIndices: list[int] = []
    for index in range(len(regionForward) - 1):
        homogeneousLeft = regionHomogeneous[index]
        homogeneousRight = regionHomogeneous[index + 1]
        if homogeneousLeft and homogeneousRight and nOrders >= 180:
            results[index] = _interfaceSMatrix(
                regionForward[index],
                regionBackward[index],
                regionForward[index + 1],
                regionBackward[index + 1],
                backend,
                homogeneousLeft=homogeneousLeft,
                homogeneousRight=homogeneousRight,
                nOrders=nOrders,
            )
            continue

        batchedMatrices.append(xp.concatenate([regionBackward[index], -regionForward[index + 1]], axis=1))
        batchedRhs.append(xp.concatenate([-regionForward[index], regionBackward[index + 1]], axis=1))
        batchedIndices.append(index)

    if batchedMatrices:
        matrixBatch = xp.stack(batchedMatrices, axis=0)
        rhsBatch = xp.stack(batchedRhs, axis=0)
        solvedBatch = _solveInterfaceBlocks(backend, matrixBatch, rhsBatch)
        size = regionForward[0].shape[1]
        for batchIndex, regionIndex in enumerate(batchedIndices):
            solved = solvedBatch[batchIndex]
            results[regionIndex] = _SMatrix(
                s11=solved[:size, :size],
                s12=solved[:size, size:],
                s21=solved[size:, :size],
                s22=solved[size:, size:],
            )

    if any(result is None for result in results):
        raise RuntimeError("batched interface solve did not fill every interface result")
    return tuple(result for result in results if result is not None)


def _homogeneousInterfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: _ArrayBackend,
) -> _SMatrix | None:
    leftForwardArray = backend.toNumpy(leftForward)
    leftBackwardArray = backend.toNumpy(leftBackward)
    rightForwardArray = backend.toNumpy(rightForward)
    rightBackwardArray = backend.toNumpy(rightBackward)
    leftForwardOrders = _homogeneousColumnOrdersFromArray(leftForwardArray, nOrders)
    leftBackwardOrders = _homogeneousColumnOrdersFromArray(leftBackwardArray, nOrders)
    rightForwardOrders = _homogeneousColumnOrdersFromArray(rightForwardArray, nOrders)
    rightBackwardOrders = _homogeneousColumnOrdersFromArray(rightBackwardArray, nOrders)
    if (
        leftForwardOrders is None
        or leftBackwardOrders is None
        or rightForwardOrders is None
        or rightBackwardOrders is None
    ):
        return None

    size = leftForward.shape[1]
    s11 = np.zeros((size, size), dtype=complex)
    s12 = np.zeros((size, size), dtype=complex)
    s21 = np.zeros((size, size), dtype=complex)
    s22 = np.zeros((size, size), dtype=complex)

    matrices = np.empty((nOrders, 4, 4), dtype=complex)
    rightHandSides = np.empty((nOrders, 4, 4), dtype=complex)
    columnGroups: list[tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]] = []
    for orderIndex in range(nOrders):
        leftForwardColumns = _columnsForOrder(leftForwardOrders, orderIndex)
        leftBackwardColumns = _columnsForOrder(leftBackwardOrders, orderIndex)
        rightForwardColumns = _columnsForOrder(rightForwardOrders, orderIndex)
        rightBackwardColumns = _columnsForOrder(rightBackwardOrders, orderIndex)
        if (
            leftForwardColumns.size != 2
            or leftBackwardColumns.size != 2
            or rightForwardColumns.size != 2
            or rightBackwardColumns.size != 2
        ):
            return None

        rows = _fieldRowsForOrder(nOrders, orderIndex)
        matrices[orderIndex] = np.concatenate(
            [
                leftBackwardArray[rows[:, None], leftBackwardColumns],
                -rightForwardArray[rows[:, None], rightForwardColumns],
            ],
            axis=1,
        )
        rightHandSides[orderIndex] = np.concatenate(
            [
                -leftForwardArray[rows[:, None], leftForwardColumns],
                rightBackwardArray[rows[:, None], rightBackwardColumns],
            ],
            axis=1,
        )
        columnGroups.append((leftForwardColumns, leftBackwardColumns, rightForwardColumns, rightBackwardColumns))

    try:
        solvedBlocks = np.linalg.solve(matrices, rightHandSides)
    except np.linalg.LinAlgError:
        return None

    for solved, (leftForwardColumns, leftBackwardColumns, rightForwardColumns, rightBackwardColumns) in zip(
        solvedBlocks,
        columnGroups,
    ):
        s11[leftBackwardColumns[:, None], leftForwardColumns] = solved[:2, :2]
        s12[leftBackwardColumns[:, None], rightBackwardColumns] = solved[:2, 2:]
        s21[rightForwardColumns[:, None], leftForwardColumns] = solved[2:, :2]
        s22[rightForwardColumns[:, None], rightBackwardColumns] = solved[2:, 2:]

    return _SMatrix(
        s11=backend.asarray(s11),
        s12=backend.asarray(s12),
        s21=backend.asarray(s21),
        s22=backend.asarray(s22),
    )


def _homogeneousColumnOrders(matrix: object, nOrders: int) -> ComplexArray | None:
    return _homogeneousColumnOrdersFromArray(np.asarray(matrix), nOrders)


def _homogeneousColumnOrdersFromArray(array: ComplexArray, nOrders: int) -> ComplexArray | None:
    if array.shape[0] != 4 * nOrders:
        return None
    assignments = np.full(array.shape[1], -1, dtype=int)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    for column in range(array.shape[1]):
        activeRows = np.flatnonzero(np.abs(array[:, column]) > tolerance)
        if activeRows.size == 0:
            return None
        orders = np.unique(activeRows % nOrders)
        if orders.size != 1:
            return None
        assignments[column] = int(orders[0])
    return assignments


def _columnsForOrder(assignments: ComplexArray, orderIndex: int) -> ComplexArray:
    return np.flatnonzero(assignments == orderIndex)


def _fieldRowsForOrder(nOrders: int, orderIndex: int) -> ComplexArray:
    return np.array([orderIndex, nOrders + orderIndex, 2 * nOrders + orderIndex, 3 * nOrders + orderIndex])


def _redhefferStar(left: _SMatrix, right: _SMatrix, backend: _ArrayBackend) -> _SMatrix:
    if left.isIdentity:
        return right
    if right.isIdentity:
        return left
    if right.isPropagation:
        forward = _matrixDiagonal(right.s21)
        backward = _matrixDiagonal(right.s12)
        return _SMatrix(
            s11=left.s11,
            s12=left.s12 * backward[None, :],
            s21=forward[:, None] * left.s21,
            s22=forward[:, None] * left.s22 * backward[None, :],
        )
    if left.isPropagation:
        forward = _matrixDiagonal(left.s21)
        backward = _matrixDiagonal(left.s12)
        return _SMatrix(
            s11=backward[:, None] * right.s11 * forward[None, :],
            s12=backward[:, None] * right.s12,
            s21=right.s21 * forward[None, :],
            s22=right.s22,
        )

    xp = backend.xp
    size = left.s11.shape[0]
    identity = xp.eye(size, dtype=complex)
    leftFactor = identity - right.s11 @ left.s22
    rightFactor = identity - left.s22 @ right.s11
    leftSolved = _solveFactoredBlocks(backend, leftFactor, right.s11, right.s12)
    rightSolved = _solveFactoredBlocks(backend, rightFactor, left.s22, left.s21)
    leftDenominator = leftSolved[:, :size]
    leftTransmission = leftSolved[:, size:]
    rightDenominator = rightSolved[:, :size]
    rightTransmission = rightSolved[:, size:]
    return _SMatrix(
        s11=left.s11 + left.s12 @ leftDenominator @ left.s21,
        s12=left.s12 @ leftTransmission,
        s21=right.s21 @ rightTransmission,
        s22=right.s22 + right.s21 @ rightDenominator @ right.s12,
    )


def _matrixDiagonal(matrix: object) -> object:
    return matrix.diagonal()


def _solveFactored(backend: _ArrayBackend, matrix: object, rhs: object) -> object:
    return backend.solveFactored(backend.factor(matrix), rhs)


def _solveFactoredBlocks(backend: _ArrayBackend, matrix: object, *rhsBlocks: object) -> object:
    if len(rhsBlocks) == 1:
        return _solveFactored(backend, matrix, rhsBlocks[0])
    rhs = backend.xp.concatenate(rhsBlocks, axis=1)
    return backend.solveFactored(backend.factor(matrix), rhs)


def _solveInterfaceBlocks(backend: _ArrayBackend, matrix: object, *rhsBlocks: object) -> object:
    if len(rhsBlocks) == 1:
        rhs = rhsBlocks[0]
    else:
        rhs = backend.xp.concatenate(rhsBlocks, axis=1)

    method = _interfaceSolveMethod()
    if method == "solve":
        return backend.solve(matrix, rhs)
    if method == "lu":
        return backend.solveFactored(backend.factor(matrix), rhs)
    raise ValueError("RCWA3D_INTERFACE_SOLVER must be 'lu' or 'solve'")


def _interfaceSolveMethod() -> str:
    return os.environ.get("RCWA3D_INTERFACE_SOLVER", "lu").strip().lower()


def _cascadeMany(components: Sequence[_SMatrix], size: int, backend: _ArrayBackend) -> _SMatrix:
    result = _identitySMatrix(size, backend)
    for component in components:
        result = _redhefferStar(result, component, backend)
    return result


def _reflectionTransmissionOnlySMatrix(
    components: Sequence[_SMatrix],
    size: int,
    backend: _ArrayBackend,
) -> _SMatrix:
    reflection, transmission = _enhancedReflectionTransmission(components, size, backend)
    zero = backend.xp.zeros_like(reflection)
    return _SMatrix(
        s11=reflection,
        s12=backend.copy(zero),
        s21=transmission,
        s22=backend.copy(zero),
    )


def _enhancedReflectionTransmission(
    components: Sequence[_SMatrix],
    size: int,
    backend: _ArrayBackend,
) -> tuple[object, object]:
    xp = backend.xp
    identity = xp.eye(size, dtype=complex)
    reflection = xp.zeros((size, size), dtype=complex)
    transmission = backend.copy(identity)
    for component in reversed(components):
        if component.isIdentity:
            continue
        if component.isPropagation:
            forward = _matrixDiagonal(component.s21)
            backward = _matrixDiagonal(component.s12)
            reflection = backward[:, None] * reflection * forward[None, :]
            transmission = transmission * forward[None, :]
            continue
        internalReflection = _solveFactored(
            backend,
            identity - reflection @ component.s22,
            reflection @ component.s21,
        )
        forward = component.s21 + component.s22 @ internalReflection
        transmission = transmission @ forward
        reflection = component.s11 + component.s12 @ internalReflection
    return reflection, transmission


def _prefixSMatrices(components: Sequence[_SMatrix], size: int, backend: _ArrayBackend) -> list[_SMatrix]:
    prefixes = [_identitySMatrix(size, backend)]
    current = prefixes[0]
    for component in components:
        current = _redhefferStar(current, component, backend)
        prefixes.append(current)
    return prefixes


def _suffixSMatrices(components: Sequence[_SMatrix], size: int, backend: _ArrayBackend) -> list[_SMatrix]:
    suffixes = [_identitySMatrix(size, backend) for _ in range(len(components) + 1)]
    current = _identitySMatrix(size, backend)
    suffixes[len(components)] = current
    for index in range(len(components) - 1, -1, -1):
        current = _redhefferStar(components[index], current, backend)
        suffixes[index] = current
    return suffixes


def _incidentField(
    harmonics: _Harmonics,
    eps: complex,
    sAmplitude: complex,
    pAmplitude: complex,
) -> ComplexArray:
    field = np.zeros(4 * harmonics.count, dtype=complex)
    index = _zeroOrderIndex(harmonics)

    sField, pField = _planeWaveFields(
        harmonics.kx[index],
        harmonics.ky[index],
        _forwardKz(eps - harmonics.kx[index] ** 2 - harmonics.ky[index] ** 2),
        eps,
    )
    _putOrderField(field, index, sAmplitude * sField + pAmplitude * pField)
    return field


def _homogeneousBasis(harmonics: _Harmonics, eps: complex, direction: int) -> ComplexArray:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    nOrders = harmonics.count
    basis = np.zeros((4 * nOrders, 2 * nOrders), dtype=complex)
    kzForward = _forwardKz(eps - harmonics.kx**2 - harmonics.ky**2)
    for index, (kx, ky, kzPositive) in enumerate(zip(harmonics.kx, harmonics.ky, kzForward)):
        kz = kzPositive if direction > 0 else -kzPositive
        sField, pField = _planeWaveFields(kx, ky, kz, eps)
        sColumn = 2 * index
        pColumn = sColumn + 1
        basis[index, sColumn] = sField[0]
        basis[nOrders + index, sColumn] = sField[1]
        basis[2 * nOrders + index, sColumn] = sField[2]
        basis[3 * nOrders + index, sColumn] = sField[3]
        basis[index, pColumn] = pField[0]
        basis[nOrders + index, pColumn] = pField[1]
        basis[2 * nOrders + index, pColumn] = pField[2]
        basis[3 * nOrders + index, pColumn] = pField[3]
    return basis


def _orderResults(
    harmonics: _Harmonics,
    epsReflected: complex,
    epsTransmitted: complex,
    reflectedBasis: ComplexArray,
    transmittedBasis: ComplexArray,
    rAmplitudes: ComplexArray,
    tAmplitudes: ComplexArray,
    incidentFlux: float,
) -> Iterable[DiffractionOrder]:
    kzReflectedForward = _forwardKz(epsReflected - harmonics.kx**2 - harmonics.ky**2)
    kzTransmittedForward = _forwardKz(epsTransmitted - harmonics.kx**2 - harmonics.ky**2)
    reflectedFluxes = _homogeneousOrderFluxes(reflectedBasis, rAmplitudes, harmonics.count)
    transmittedFluxes = _homogeneousOrderFluxes(transmittedBasis, tAmplitudes, harmonics.count)
    for index, (mx, my, kx, ky) in enumerate(zip(harmonics.mx, harmonics.my, harmonics.kx, harmonics.ky)):
        reflectedPower = -reflectedFluxes[index] / incidentFlux
        transmittedPower = transmittedFluxes[index] / incidentFlux
        yield DiffractionOrder(
            mx=int(mx),
            my=int(my),
            kx=kx,
            ky=ky,
            kzReflected=-kzReflectedForward[index],
            kzTransmitted=kzTransmittedForward[index],
            reflectedPower=float(reflectedPower),
            transmittedPower=float(transmittedPower),
            reflectedPropagating=_isPropagating(kzReflectedForward[index]),
            transmittedPropagating=_isPropagating(kzTransmittedForward[index]),
        )


def _homogeneousOrderFluxes(basis: ComplexArray, amplitudes: ComplexArray, nOrders: int) -> ComplexArray:
    orderIndices = np.arange(nOrders)
    sColumns = 2 * orderIndices
    pColumns = sColumns + 1
    sAmplitudes = amplitudes[sColumns]
    pAmplitudes = amplitudes[pColumns]

    ex = basis[orderIndices, sColumns] * sAmplitudes + basis[orderIndices, pColumns] * pAmplitudes
    ey = (
        basis[nOrders + orderIndices, sColumns] * sAmplitudes
        + basis[nOrders + orderIndices, pColumns] * pAmplitudes
    )
    hx = (
        basis[2 * nOrders + orderIndices, sColumns] * sAmplitudes
        + basis[2 * nOrders + orderIndices, pColumns] * pAmplitudes
    )
    hy = (
        basis[3 * nOrders + orderIndices, sColumns] * sAmplitudes
        + basis[3 * nOrders + orderIndices, pColumns] * pAmplitudes
    )
    return 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))


def _smatrixLayerSolutions(
    layers: Sequence[Layer | CompiledLayer],
    layerModes: Sequence[tuple[ComplexArray, ComplexArray]],
    components: Sequence[_SMatrix],
    incidentAmplitudes: ComplexArray,
    harmonics: _Harmonics,
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    backend: _ArrayBackend,
) -> tuple[LayerFieldSolution, ...]:
    xp = backend.xp
    nPorts = 2 * harmonics.count
    prefix = _prefixSMatrices(components, nPorts, backend)
    suffix = _suffixSMatrices(components, nPorts, backend)
    identity = xp.eye(nPorts, dtype=complex)
    layerSolutionList = []
    for layerIndex, (layer, (qValues, modeMatrix)) in enumerate(zip(layers, layerModes)):
        boundaryIndex = 2 * layerIndex + 1
        leftNetwork = prefix[boundaryIndex]
        rightNetwork = suffix[boundaryIndex]
        rhs = rightNetwork.s11 @ (leftNetwork.s21 @ incidentAmplitudes)
        backwardAtLeft = _solveFactored(backend, identity - rightNetwork.s11 @ leftNetwork.s22, rhs)
        forwardAtLeft = leftNetwork.s21 @ incidentAmplitudes + leftNetwork.s22 @ backwardAtLeft
        coefficients = xp.concatenate([forwardAtLeft, backwardAtLeft])
        layerSolutionList.append(
            _layerSolution(
                layer,
                backend.toNumpy(qValues),
                backend.toNumpy(modeMatrix),
                backend.toNumpy(coefficients),
                harmonics,
                wavelength,
                period,
                orders,
            )
        )
    return tuple(layerSolutionList)


def _incidentAmplitudeVector(prepared: PreparedStack, sAmplitude: complex, pAmplitude: complex) -> ComplexArray:
    amplitudes = np.zeros(prepared.nPorts, dtype=complex)
    amplitudes[2 * prepared.zeroIndex] = sAmplitude
    amplitudes[2 * prepared.zeroIndex + 1] = pAmplitude
    return amplitudes


def _incidentAmplitudes(
    prepared: PreparedStack,
    sAmplitude: complex,
    pAmplitude: complex,
    backend: _ArrayBackend,
) -> object:
    return backend.asarray(_incidentAmplitudeVector(prepared, sAmplitude, pAmplitude))


def _layerSolution(
    layer: Layer | CompiledLayer,
    qValues: ComplexArray,
    modeMatrix: ComplexArray,
    coefficients: ComplexArray,
    harmonics: _Harmonics,
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
) -> LayerFieldSolution:
    return LayerFieldSolution(
        name=getattr(layer, "name", ""),
        thickness=float(layer.thickness),
        wavelength=float(wavelength),
        period=period,
        orders=orders,
        mx=harmonics.mx.copy(),
        my=harmonics.my.copy(),
        kx=harmonics.kx.copy(),
        ky=harmonics.ky.copy(),
        qValues=qValues.copy(),
        modeMatrix=modeMatrix.copy(),
        coefficients=coefficients.copy(),
    )


def _result(
    harmonics: _Harmonics,
    epsIncident: complex,
    epsTransmission: complex,
    reflectedBasis: ComplexArray,
    transmittedBasis: ComplexArray,
    rAmplitudes: ComplexArray,
    tAmplitudes: ComplexArray,
    incidentFlux: float,
    solvedBy: str,
    layerSolutions: tuple[LayerFieldSolution, ...],
    layerEigTimings: tuple[LayerEigTiming, ...] = (),
) -> RCWAResult:
    orderResults = tuple(
        _orderResults(
            harmonics=harmonics,
            epsReflected=epsIncident,
            epsTransmitted=epsTransmission,
            reflectedBasis=reflectedBasis,
            transmittedBasis=transmittedBasis,
            rAmplitudes=rAmplitudes,
            tAmplitudes=tAmplitudes,
            incidentFlux=incidentFlux,
        )
    )
    reflection = float(sum(order.reflectedPower for order in orderResults))
    transmission = float(sum(order.transmittedPower for order in orderResults))

    return RCWAResult(
        reflection=reflection,
        transmission=transmission,
        conservation=reflection + transmission,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=orderResults,
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        layerEigTimings=layerEigTimings,
    )


def _checkedIncidentFlux(field: ComplexArray) -> float:
    incidentFlux = _flux(field)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def _zeroOrderIndex(harmonics: _Harmonics) -> int:
    zeroIndices = np.where((harmonics.mx == 0) & (harmonics.my == 0))[0]
    if zeroIndices.size != 1:
        raise RuntimeError("zero diffraction order was not found")
    return int(zeroIndices[0])


def _validateInputs(wavelength: float, period: tuple[float, float], orders: int | tuple[int, int]) -> None:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")
    normalizedOrders = _normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")


def _requireSMatrixMethod(method: str) -> None:
    if str(method).lower() != "smatrix":
        raise ValueError("the anisotropic solver now supports only method='smatrix'")


def _isPropagating(kz: complex) -> bool:
    return bool(abs(np.imag(kz)) < 1e-10 and np.real(kz) > 1e-12)
