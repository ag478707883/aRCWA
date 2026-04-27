from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import eig

from .backend import ArrayBackend, resolveBackend
from .fourier import Harmonics, epsilonConvolutionMatrix, makeHarmonics, normalizeOrders
from .phase import flux, forwardKz, planeWaveFields, putOrderField, singleOrderVector
from .smatrix import (
    SMatrix,
    cascadeMany,
    interfaceSMatrix,
    prefixSMatrices,
    propagationSMatrix,
    suffixSMatrices,
)
from .types import ComplexArray, CompiledLayer, DiffractionOrder, Layer, LayerFieldSolution, RCWAResult
from .varrcwa import AdaptiveLayerSpec


@dataclass(frozen=True)
class FactorizationData:
    epsilonMatrix: ComplexArray
    epsilonInverse: ComplexArray
    displacementMatrices: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray] | None = None
    factorization: str = "standard"
    homogeneousEpsilon: complex | None = None


@dataclass(frozen=True)
class PreparedStack:
    layers: tuple[Layer | CompiledLayer, ...]
    wavelength: float
    period: tuple[float, float]
    orders: tuple[int, int]
    epsIncident: complex
    epsTransmission: complex
    theta: float
    phi: float
    truncation: str
    harmonics: Harmonics
    components: tuple[SMatrix, ...]
    total: SMatrix
    layerModes: tuple[tuple[ComplexArray, ComplexArray], ...]
    incidentBackward: ComplexArray
    transmissionForward: ComplexArray
    zeroIndex: int

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class _TorchSMatrix:
    s11: Any
    s12: Any
    s21: Any
    s22: Any


@dataclass(frozen=True)
class _TorchLayerData:
    epsilonMatrix: ComplexArray
    epsilonInverse: ComplexArray | None
    displacementMatrices: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray] | None
    homogeneousEpsilon: complex | None


@dataclass(frozen=True)
class PreparedTorchStack:
    layers: tuple[Layer | CompiledLayer, ...]
    wavelength: float
    period: tuple[float, float]
    orders: tuple[int, int]
    epsIncident: complex
    epsTransmission: complex
    truncation: str
    harmonics: Harmonics
    layerModes: tuple[tuple[Any, Any], ...]
    components: tuple[_TorchSMatrix, ...]
    total: _TorchSMatrix
    incidentForward: ComplexArray
    incidentBackward: ComplexArray
    transmissionForward: ComplexArray
    zeroIndex: int
    backend: ArrayBackend

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class _OrderReductionPlan:
    label: str
    reducedOrders: tuple[int, int]
    fullHarmonics: Harmonics
    keptIndices: ComplexArray


def solveStack(
    layers: Sequence[Layer | CompiledLayer | AdaptiveLayerSpec],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    returnFields: bool = False,
    method: str = "smatrix",
    truncation: str = "circular",
    backend: str | ArrayBackend | None = "cuda",
) -> RCWAResult:
    """Solve a 2D-periodic isotropic stack with the S-matrix RCWA path."""

    _requireSMatrixMethod(method)
    expandedLayers = _expandAdaptiveLayers(layers)
    _validateIsotropicLayers(expandedLayers, kind="solve")
    arrayBackend = resolveBackend(backend)
    reductionPlan = _automaticOrderReductionPlan(
        layers=expandedLayers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        theta=theta,
        phi=phi,
        truncation=truncation,
        returnFields=returnFields,
    )
    if reductionPlan is not None:
        return _solveStackReducedTorch(
            layers=expandedLayers,
            wavelength=wavelength,
            period=period,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            truncation=truncation,
            backend=arrayBackend,
            plan=reductionPlan,
        )

    preparedTorch = prepareStackTorch(
        layers=expandedLayers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=arrayBackend,
    )
    return evaluatePreparedStackTorch(
        preparedTorch,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{_backendSuffix(arrayBackend)}",
        returnFields=returnFields,
    )


def solveStackBatch(
    layers: Sequence[Layer | CompiledLayer | AdaptiveLayerSpec],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    method: str = "smatrix",
    truncation: str = "circular",
    backend: str | ArrayBackend | None = "cuda",
) -> dict[str, RCWAResult]:
    """Solve several incident polarizations while reusing the same S-matrix."""

    _requireSMatrixMethod(method)
    if not excitations:
        return {}
    expandedLayers = _expandAdaptiveLayers(layers)
    _validateIsotropicLayers(expandedLayers, kind="solve")
    arrayBackend = resolveBackend(backend)
    reductionPlan = _automaticOrderReductionPlan(
        layers=expandedLayers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        theta=theta,
        phi=phi,
        truncation=truncation,
        returnFields=False,
    )
    if reductionPlan is not None:
        return _solveBatchReducedTorch(
            layers=expandedLayers,
            wavelength=wavelength,
            period=period,
            excitations=excitations,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            truncation=truncation,
            backend=arrayBackend,
            plan=reductionPlan,
        )

    preparedTorch = prepareStackTorch(
        layers=expandedLayers,
        wavelength=wavelength,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=theta,
        phi=phi,
        truncation=truncation,
        backend=arrayBackend,
    )
    return evaluatePreparedBatchTorch(
        preparedTorch,
        excitations,
        solvedBy=f"smatrix-batch-{_backendSuffix(arrayBackend)}",
    )


def compileLayers(
    layers: Sequence[Layer | AdaptiveLayerSpec],
    orders: int | tuple[int, int],
    truncation: str = "circular",
) -> tuple[CompiledLayer, ...]:
    """Precompute Fourier convolution data for fixed orders/truncation."""

    expandedLayers = _expandAdaptiveLayers(layers)
    _validateIsotropicLayers(expandedLayers, kind="compile")

    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")
    harmonics = makeHarmonics(
        wavelength=1.0,
        period=(1.0, 1.0),
        orders=normalizedOrders,
        epsIncident=1.0,
        theta=0.0,
        phi=0.0,
        truncation=truncation,
    )
    compiledLayers: list[CompiledLayer] = []
    for layer in expandedLayers:
        factorized = layerEpsilonData(layer, harmonics)
        compiledLayers.append(
            CompiledLayer(
                thickness=layer.thickness,
                epsilonMatrix=factorized.epsilonMatrix,
                epsilonInverse=factorized.epsilonInverse,
                orders=normalizedOrders,
                truncation=harmonics.truncation,
                name=layer.name,
                displacementMatrices=factorized.displacementMatrices,
                factorization=factorized.factorization,
                homogeneousEpsilon=factorized.homogeneousEpsilon,
            )
        )
    return tuple(compiledLayers)


def prepareStack(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    truncation: str = "circular",
) -> PreparedStack:
    _validateGeometry(wavelength, period, orders)
    normalizedOrders = normalizeOrders(orders)
    harmonics = makeHarmonics(wavelength, period, normalizedOrders, epsIncident, theta, phi, truncation=truncation)
    nPorts = 2 * harmonics.count
    k0 = 2 * np.pi / wavelength

    incidentForward = homogeneousBasis(harmonics, epsIncident, direction=1)
    incidentBackward = homogeneousBasis(harmonics, epsIncident, direction=-1)
    transmissionForward = homogeneousBasis(harmonics, epsTransmission, direction=1)
    transmissionBackward = homogeneousBasis(harmonics, epsTransmission, direction=-1)

    modes = tuple(layerModes(layer, harmonics) for layer in layers)
    regionForward: list[ComplexArray] = [incidentForward]
    regionBackward: list[ComplexArray] = [incidentBackward]
    for _qValues, modeMatrix in modes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)

    components: list[SMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(
            interfaceSMatrix(
                regionForward[regionIndex],
                regionBackward[regionIndex],
                regionForward[regionIndex + 1],
                regionBackward[regionIndex + 1],
            )
        )
        if regionIndex < len(layers):
            qForward = modes[regionIndex][0][:nPorts]
            propagation = np.diag(np.exp(1j * qForward * k0 * layers[regionIndex].thickness))
            components.append(propagationSMatrix(propagation))

    componentTuple = tuple(components)
    return PreparedStack(
        layers=tuple(layers),
        wavelength=float(wavelength),
        period=period,
        orders=normalizedOrders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=float(theta),
        phi=float(phi),
        truncation=harmonics.truncation,
        harmonics=harmonics,
        components=componentTuple,
        total=cascadeMany(componentTuple, nPorts),
        layerModes=modes,
        incidentBackward=incidentBackward,
        transmissionForward=transmissionForward,
        zeroIndex=zeroOrderIndex(harmonics),
    )


def evaluatePreparedStack(
    prepared: PreparedStack,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    *,
    solvedBy: str = "smatrix",
    returnFields: bool = False,
) -> RCWAResult:
    incident = incidentField(prepared.harmonics, prepared.epsIncident, sAmplitude, pAmplitude)
    incidentFlux = _checkedFlux(incident)
    incidentAmplitudes = _incidentAmplitudes(prepared, sAmplitude, pAmplitude)
    rAmplitudes = prepared.total.s11 @ incidentAmplitudes
    tAmplitudes = prepared.total.s21 @ incidentAmplitudes
    return _result(
        prepared,
        rAmplitudes,
        tAmplitudes,
        incidentFlux,
        solvedBy,
        _layerSolutions(prepared, incidentAmplitudes) if returnFields and prepared.layers else (),
        sAmplitude,
        pAmplitude,
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

    incidentColumns = np.column_stack([_incidentAmplitudes(prepared, *excitations[label]) for label in labels])
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    results: dict[str, RCWAResult] = {}
    for column, label in enumerate(labels):
        sAmplitude, pAmplitude = excitations[label]
        incident = incidentField(prepared.harmonics, prepared.epsIncident, sAmplitude, pAmplitude)
        results[label] = _result(
            prepared,
            reflected[:, column],
            transmitted[:, column],
            _checkedFlux(incident),
            solvedBy,
            (),
            sAmplitude,
            pAmplitude,
        )
    return results


def prepareStackTorch(
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    truncation: str = "circular",
    backend: str | ArrayBackend = "cuda",
) -> PreparedTorchStack:
    _validateGeometry(wavelength, period, orders)
    arrayBackend = resolveBackend(backend)
    if not arrayBackend.isTorch or not arrayBackend.isCuda:
        raise ValueError("prepareStackTorch requires the CUDA backend")

    torch = arrayBackend.xp
    device = arrayBackend.device if arrayBackend.device is not None else torch.device("cuda")
    normalizedOrders = normalizeOrders(orders)
    harmonics = makeHarmonics(wavelength, period, normalizedOrders, epsIncident, theta, phi, truncation=truncation)
    nPorts = 2 * harmonics.count
    k0 = 2 * np.pi / wavelength

    incidentForwardNp = homogeneousBasis(harmonics, epsIncident, direction=1)
    incidentBackwardNp = homogeneousBasis(harmonics, epsIncident, direction=-1)
    transmissionForwardNp = homogeneousBasis(harmonics, epsTransmission, direction=1)
    transmissionBackwardNp = homogeneousBasis(harmonics, epsTransmission, direction=-1)

    incidentForward = _toTorchComplex(incidentForwardNp, torch, device)
    incidentBackward = _toTorchComplex(incidentBackwardNp, torch, device)
    transmissionForward = _toTorchComplex(transmissionForwardNp, torch, device)
    transmissionBackward = _toTorchComplex(transmissionBackwardNp, torch, device)

    modes = tuple(_layerModesTorch(layer, harmonics, torch, device) for layer in layers)
    regionForward: list[Any] = [incidentForward]
    regionBackward: list[Any] = [incidentBackward]
    for _qValues, modeMatrix in modes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)

    components: list[_TorchSMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(
            _interfaceSMatrixTorch(
                regionForward[regionIndex],
                regionBackward[regionIndex],
                regionForward[regionIndex + 1],
                regionBackward[regionIndex + 1],
                torch,
            )
        )
        if regionIndex < len(layers):
            qForward = modes[regionIndex][0][:nPorts]
            propagation = torch.diag(torch.exp(1j * qForward * k0 * layers[regionIndex].thickness))
            components.append(_propagationSMatrixTorch(propagation, torch))

    componentTuple = tuple(components)
    return PreparedTorchStack(
        layers=tuple(layers),
        wavelength=float(wavelength),
        period=period,
        orders=normalizedOrders,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        truncation=harmonics.truncation,
        harmonics=harmonics,
        layerModes=modes,
        components=componentTuple,
        total=_cascadeManyTorch(componentTuple, nPorts, torch, device),
        incidentForward=incidentForwardNp,
        incidentBackward=incidentBackwardNp,
        transmissionForward=transmissionForwardNp,
        zeroIndex=zeroOrderIndex(harmonics),
        backend=arrayBackend,
    )


def evaluatePreparedStackTorch(
    prepared: PreparedTorchStack,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    *,
    solvedBy: str = "smatrix-cuda",
    returnFields: bool = False,
) -> RCWAResult:
    torch = prepared.backend.xp
    incident = _incidentAmplitudesTorch(prepared, sAmplitude, pAmplitude, torch)
    incidentFlux = _checkedIncidentFluxTorch(
        prepared,
        _incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, sAmplitude, pAmplitude),
    )
    rAmplitudes = prepared.total.s11 @ incident
    tAmplitudes = prepared.total.s21 @ incident
    return _resultFromTorchAmplitudes(
        prepared=prepared,
        rAmplitudes=prepared.backend.asnumpy(rAmplitudes),
        tAmplitudes=prepared.backend.asnumpy(tAmplitudes),
        incidentFlux=incidentFlux,
        solvedBy=solvedBy,
        layerSolutions=_layerSolutionsTorch(prepared, incident) if returnFields and prepared.layers else (),
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


def evaluatePreparedBatchTorch(
    prepared: PreparedTorchStack,
    excitations: Mapping[str, tuple[complex, complex]],
    *,
    solvedBy: str = "smatrix-batch-cuda",
) -> dict[str, RCWAResult]:
    labels = tuple(excitations)
    if not labels:
        return {}

    torch = prepared.backend.xp
    columns = [_incidentAmplitudesTorch(prepared, *excitations[label], torch) for label in labels]
    incidentColumns = torch.column_stack(columns)
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    incidentColumnsNp = np.column_stack(
        [_incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, *excitations[label]) for label in labels]
    )
    reflectedNp = prepared.backend.asnumpy(reflected)
    transmittedNp = prepared.backend.asnumpy(transmitted)

    results: dict[str, RCWAResult] = {}
    for column, label in enumerate(labels):
        sAmplitude, pAmplitude = excitations[label]
        incidentFlux = _checkedIncidentFluxTorch(prepared, incidentColumnsNp[:, column])
        results[label] = _resultFromTorchAmplitudes(
            prepared=prepared,
            rAmplitudes=reflectedNp[:, column],
            tAmplitudes=transmittedNp[:, column],
            incidentFlux=incidentFlux,
            solvedBy=solvedBy,
            layerSolutions=(),
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
        )
    return results


def _automaticOrderReductionPlan(
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
) -> _OrderReductionPlan | None:
    if returnFields:
        return None
    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders == (0, 0):
        return None

    fullHarmonics = makeHarmonics(
        wavelength=wavelength,
        period=period,
        orders=normalizedOrders,
        epsIncident=epsIncident,
        theta=theta,
        phi=phi,
        truncation=truncation,
    )
    if _hasMismatchedCompiledLayer(layers, fullHarmonics):
        return None
    if all(_isHomogeneousLayer(layer) for layer in layers):
        return _makeOrderReductionPlan("homogeneous", (0, 0), fullHarmonics)

    nx, ny = normalizedOrders
    if ny > 0 and _stackInvariantAlong("y", layers, fullHarmonics):
        return _makeOrderReductionPlan("1d-x", (nx, 0), fullHarmonics)
    if nx > 0 and _stackInvariantAlong("x", layers, fullHarmonics):
        return _makeOrderReductionPlan("1d-y", (0, ny), fullHarmonics)
    return None


def _makeOrderReductionPlan(
    label: str,
    reducedOrders: tuple[int, int],
    fullHarmonics: Harmonics,
) -> _OrderReductionPlan:
    return _OrderReductionPlan(
        label=label,
        reducedOrders=reducedOrders,
        fullHarmonics=fullHarmonics,
        keptIndices=_reducedHarmonicIndices(fullHarmonics, reducedOrders),
    )


def _solveStackReducedTorch(
    *,
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
    backend: ArrayBackend,
    plan: _OrderReductionPlan,
) -> RCWAResult:
    reducedLayers = _reducedLayers(layers, plan)
    prepared = prepareStackTorch(
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
    )
    reduced = evaluatePreparedStackTorch(
        prepared,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{_backendSuffix(backend)}",
        returnFields=False,
    )
    return _embedReducedResult(
        reduced,
        fullHarmonics=plan.fullHarmonics,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{plan.label}-{_backendSuffix(backend)}",
    )


def _solveBatchReducedTorch(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex,
    epsTransmission: complex,
    theta: float,
    phi: float,
    truncation: str,
    backend: ArrayBackend,
    plan: _OrderReductionPlan,
) -> dict[str, RCWAResult]:
    reducedLayers = _reducedLayers(layers, plan)
    prepared = prepareStackTorch(
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
    )
    reducedResults = evaluatePreparedBatchTorch(
        prepared,
        excitations,
        solvedBy=f"smatrix-batch-{_backendSuffix(backend)}",
    )
    return {
        label: _embedReducedResult(
            result,
            fullHarmonics=plan.fullHarmonics,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            sAmplitude=excitations[label][0],
            pAmplitude=excitations[label][1],
            solvedBy=f"smatrix-batch-{plan.label}-{_backendSuffix(backend)}",
        )
        for label, result in reducedResults.items()
    }


def _reducedLayers(
    layers: Sequence[Layer | CompiledLayer],
    plan: _OrderReductionPlan,
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
        if layer.homogeneousEpsilon is None:
            raise RuntimeError("compiled layer is not homogeneous")
        return Layer(thickness=layer.thickness, epsilon=layer.homogeneousEpsilon, name=layer.name)
    epsilon = _homogeneousEpsilon(getattr(layer, "epsilon"))
    if epsilon is None:
        raise RuntimeError("layer is not homogeneous")
    return Layer(thickness=layer.thickness, epsilon=epsilon, name=getattr(layer, "name", ""))


def _sliceCompiledLayer(layer: CompiledLayer, plan: _OrderReductionPlan) -> CompiledLayer:
    indexer = np.ix_(plan.keptIndices, plan.keptIndices)
    displacement = layer.displacementMatrices
    return CompiledLayer(
        thickness=layer.thickness,
        epsilonMatrix=np.asarray(layer.epsilonMatrix)[indexer].copy(),
        epsilonInverse=np.asarray(layer.epsilonInverse)[indexer].copy(),
        orders=plan.reducedOrders,
        truncation=plan.fullHarmonics.truncation,
        name=layer.name,
        displacementMatrices=None
        if displacement is None
        else tuple(np.asarray(matrix)[indexer].copy() for matrix in displacement),
        factorization=layer.factorization,
        homogeneousEpsilon=layer.homogeneousEpsilon,
    )


def _embedReducedResult(
    reduced: RCWAResult,
    *,
    fullHarmonics: Harmonics,
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

    incidentBackward = homogeneousBasis(fullHarmonics, epsIncident, direction=-1)
    transmissionForward = homogeneousBasis(fullHarmonics, epsTransmission, direction=1)
    incident = incidentField(fullHarmonics, epsIncident, sAmplitude, pAmplitude)
    incidentFlux = _checkedFlux(incident)
    reflection = float(-flux(incidentBackward @ rAmplitudes) / incidentFlux)
    transmission = float(flux(transmissionForward @ tAmplitudes) / incidentFlux)
    orders = orderResults(
        harmonics=fullHarmonics,
        epsReflected=epsIncident,
        epsTransmitted=epsTransmission,
        reflectedBasis=incidentBackward,
        transmittedBasis=transmissionForward,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        incidentFlux=incidentFlux,
    )
    return RCWAResult(
        reflection=reflection,
        transmission=transmission,
        conservation=reflection + transmission,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=tuple(orders),
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=(),
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


def _reducedHarmonicIndices(harmonics: Harmonics, reducedOrders: tuple[int, int]) -> ComplexArray:
    nx, ny = reducedOrders
    mask = (np.abs(harmonics.mx) <= nx) & (np.abs(harmonics.my) <= ny)
    return np.flatnonzero(mask)


def _hasMismatchedCompiledLayer(layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return any(
        isinstance(layer, CompiledLayer)
        and (layer.orders != harmonics.orders or layer.truncation != harmonics.truncation)
        for layer in layers
    )


def _stackInvariantAlong(axis: str, layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return all(_layerInvariantAlong(axis, layer, harmonics) for layer in layers)


def _layerInvariantAlong(axis: str, layer: Layer | CompiledLayer, harmonics: Harmonics) -> bool:
    if _isHomogeneousLayer(layer):
        return True
    if isinstance(layer, CompiledLayer):
        return _compiledLayerInvariantAlong(axis, layer, harmonics)
    return _rawLayerInvariantAlong(axis, layer)


def _rawLayerInvariantAlong(axis: str, layer: Layer) -> bool:
    epsilon = getattr(layer, "epsilon")
    if _homogeneousEpsilon(epsilon) is not None:
        return True
    if hasattr(epsilon, "invariantAxes"):
        epsilonInvariant = axis in epsilon.invariantAxes()
    elif hasattr(epsilon, "convolutionMatrix"):
        epsilonInvariant = False
    else:
        epsilonInvariant = _arrayInvariantAlong(axis, np.asarray(epsilon))
    if not epsilonInvariant:
        return False

    normal = getattr(layer, "normalField", None)
    return normal is None or _arrayInvariantAlong(axis, np.asarray(normal))


def _arrayInvariantAlong(axis: str, array: ComplexArray) -> bool:
    if array.ndim < 2:
        return True
    if axis == "y":
        return bool(np.allclose(array, array[:1, ...], rtol=1e-10, atol=1e-12))
    if axis == "x":
        return bool(np.allclose(array, array[:, :1, ...], rtol=1e-10, atol=1e-12))
    raise ValueError("axis must be 'x' or 'y'")


def _compiledLayerInvariantAlong(axis: str, layer: CompiledLayer, harmonics: Harmonics) -> bool:
    if layer.orders != harmonics.orders or layer.truncation != harmonics.truncation:
        return False
    if layer.homogeneousEpsilon is not None and layer.displacementMatrices is None:
        return True
    if axis == "y":
        uncoupled = harmonics.my[:, None] != harmonics.my[None, :]
    elif axis == "x":
        uncoupled = harmonics.mx[:, None] != harmonics.mx[None, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")

    matrices = [layer.epsilonMatrix, layer.epsilonInverse]
    if layer.displacementMatrices is not None:
        matrices.extend(layer.displacementMatrices)
    scale = max(1.0, max(float(np.max(np.abs(matrix))) for matrix in matrices if matrix.size))
    tolerance = 1e-10 * scale
    return all(float(np.max(np.abs(np.asarray(matrix)[uncoupled]))) <= tolerance for matrix in matrices)


def _isHomogeneousLayer(layer: Layer | CompiledLayer) -> bool:
    if isinstance(layer, CompiledLayer):
        return layer.homogeneousEpsilon is not None
    return _homogeneousEpsilon(getattr(layer, "epsilon")) is not None


def layerEpsilonData(layer: Layer | CompiledLayer, harmonics: Harmonics) -> FactorizationData:
    if hasattr(layer, "epsilonMatrix") and hasattr(layer, "epsilonInverse"):
        expectedOrders = harmonics.orders
        if getattr(layer, "orders") != expectedOrders:
            raise ValueError(
                f"compiled layer orders {getattr(layer, 'orders')} do not match requested orders {expectedOrders}"
            )
        layerTruncation = getattr(layer, "truncation", "rectangular")
        if layerTruncation != harmonics.truncation:
            raise ValueError(
                f"compiled layer truncation {layerTruncation!r} does not match requested truncation "
                f"{harmonics.truncation!r}"
            )
        return FactorizationData(
            getattr(layer, "epsilonMatrix"),
            getattr(layer, "epsilonInverse"),
            getattr(layer, "displacementMatrices", None),
            getattr(layer, "factorization", "standard"),
            getattr(layer, "homogeneousEpsilon", None),
        )

    epsilon = getattr(layer, "epsilon")
    epsilonMatrix = epsilonConvolutionMatrix(epsilon, harmonics)
    epsilonInverse = np.linalg.inv(epsilonMatrix)
    factorizationMode = _normalizeFactorization(getattr(layer, "factorization", "auto"))
    normalField = getattr(layer, "normalField", None)
    displacementMatrices: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray] | None = None
    factorization = _analyticFactorizationName(epsilon)

    if normalField is None and _shouldAutoGenerateNormalField(epsilon, factorizationMode):
        normalField = _estimateNormalField(epsilon, harmonics.orders, harmonics.truncation)

    if _useAnalyticNormalVector(epsilon, factorizationMode):
        displacementMatrices = _analyticNormalVectorDisplacementMatrices(epsilon, harmonics)
        factorization = "analytic-normal-vector-li"
    elif normalField is not None and factorizationMode in ("auto", "normal-vector", "jones"):
        displacementMatrices = _normalVectorDisplacementMatrices(epsilon, normalField, harmonics)
        factorization = "normal-vector-li"
    elif factorizationMode in ("normal-vector", "jones"):
        raise ValueError("normal-vector factorization requires a normalField or an analytic shape with normal vectors")

    return FactorizationData(
        epsilonMatrix,
        epsilonInverse,
        displacementMatrices,
        factorization,
        _homogeneousEpsilon(epsilon),
    )


def pqMatrices(
    epsilonMatrix: ComplexArray,
    harmonics: Harmonics,
    epsilonInverse: ComplexArray,
    displacementMatrices: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray] | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    n = harmonics.count
    identity = np.eye(n, dtype=complex)
    kx = harmonics.kx
    ky = harmonics.ky

    p11 = kx[:, None] * epsilonInverse * ky[None, :]
    p12 = identity - kx[:, None] * epsilonInverse * kx[None, :]
    p21 = ky[:, None] * epsilonInverse * ky[None, :] - identity
    p22 = -ky[:, None] * epsilonInverse * kx[None, :]

    if displacementMatrices is None:
        cxx = epsilonMatrix
        cxy = np.zeros_like(epsilonMatrix)
        cyx = np.zeros_like(epsilonMatrix)
        cyy = epsilonMatrix
    else:
        cxx, cxy, cyx, cyy = displacementMatrices

    q11 = -np.diag(kx * ky) - cyx
    q12 = np.diag(kx * kx) - cyy
    q21 = cxx - np.diag(ky * ky)
    q22 = cxy + np.diag(ky * kx)

    return np.block([[p11, p12], [p21, p22]]), np.block([[q11, q12], [q21, q22]])


def incidentField(
    harmonics: Harmonics,
    eps: complex,
    sAmplitude: complex,
    pAmplitude: complex,
) -> ComplexArray:
    field = np.zeros(4 * harmonics.count, dtype=complex)
    index = zeroOrderIndex(harmonics)
    sField, pField = planeWaveFields(
        harmonics.kx[index],
        harmonics.ky[index],
        forwardKz(eps - harmonics.kx[index] ** 2 - harmonics.ky[index] ** 2),
        eps,
    )
    putOrderField(field, index, sAmplitude * sField + pAmplitude * pField)
    return field


def homogeneousBasis(harmonics: Harmonics, eps: complex, direction: int) -> ComplexArray:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    basis = np.zeros((4 * harmonics.count, 2 * harmonics.count), dtype=complex)
    kzForward = forwardKz(eps - harmonics.kx**2 - harmonics.ky**2)
    for index, (kx, ky, kzPositive) in enumerate(zip(harmonics.kx, harmonics.ky, kzForward)):
        kz = kzPositive if direction > 0 else -kzPositive
        sField, pField = planeWaveFields(kx, ky, kz, eps)
        basis[:, 2 * index] = singleOrderVector(harmonics.count, index, sField)
        basis[:, 2 * index + 1] = singleOrderVector(harmonics.count, index, pField)
    return basis


def layerModes(layer: Layer | CompiledLayer, harmonics: Harmonics) -> tuple[ComplexArray, ComplexArray]:
    factorized = layerEpsilonData(layer, harmonics)
    if factorized.homogeneousEpsilon is not None and factorized.displacementMatrices is None:
        return homogeneousScalarLayerModes(harmonics, factorized.homogeneousEpsilon)

    pMatrix, qMatrix = pqMatrices(
        factorized.epsilonMatrix,
        harmonics,
        factorized.epsilonInverse,
        factorized.displacementMatrices,
    )
    qSquared, electricModes = eig(pMatrix @ qMatrix)
    qValues = forwardKz(qSquared)
    safeQ = qValues.copy()
    safeQ[np.abs(safeQ) < 1e-13] = 1e-13

    magneticModes = qMatrix @ electricModes @ np.diag(1 / safeQ)
    vectors = np.block([[electricModes, electricModes], [magneticModes, -magneticModes]])
    qAll = np.concatenate([qValues, -qValues])
    return qAll, normalizeColumns(vectors)


def homogeneousScalarLayerModes(harmonics: Harmonics, epsilon: complex) -> tuple[ComplexArray, ComplexArray]:
    forward = homogeneousBasis(harmonics, epsilon, direction=1)
    backward = homogeneousBasis(harmonics, epsilon, direction=-1)
    kzForward = forwardKz(epsilon - harmonics.kx**2 - harmonics.ky**2)
    qForward = np.empty(2 * harmonics.count, dtype=complex)
    qForward[0::2] = kzForward
    qForward[1::2] = kzForward
    return np.concatenate([qForward, -qForward]), np.concatenate([forward, backward], axis=1)


def normalizeColumns(vectors: ComplexArray) -> ComplexArray:
    scales = np.max(np.abs(vectors), axis=0)
    scales[scales == 0] = 1.0
    return vectors / scales


def orderResults(
    harmonics: Harmonics,
    epsReflected: complex,
    epsTransmitted: complex,
    reflectedBasis: ComplexArray,
    transmittedBasis: ComplexArray,
    rAmplitudes: ComplexArray,
    tAmplitudes: ComplexArray,
    incidentFlux: float,
) -> Iterable[DiffractionOrder]:
    kzReflectedForward = forwardKz(epsReflected - harmonics.kx**2 - harmonics.ky**2)
    kzTransmittedForward = forwardKz(epsTransmitted - harmonics.kx**2 - harmonics.ky**2)
    for index, (mx, my, kx, ky) in enumerate(zip(harmonics.mx, harmonics.my, harmonics.kx, harmonics.ky)):
        reflectedOrder = np.zeros_like(rAmplitudes)
        transmittedOrder = np.zeros_like(tAmplitudes)
        reflectedOrder[2 * index : 2 * index + 2] = rAmplitudes[2 * index : 2 * index + 2]
        transmittedOrder[2 * index : 2 * index + 2] = tAmplitudes[2 * index : 2 * index + 2]
        reflectedPower = -flux(reflectedBasis @ reflectedOrder) / incidentFlux
        transmittedPower = flux(transmittedBasis @ transmittedOrder) / incidentFlux
        yield DiffractionOrder(
            mx=int(mx),
            my=int(my),
            kx=kx,
            ky=ky,
            kzReflected=-kzReflectedForward[index],
            kzTransmitted=kzTransmittedForward[index],
            reflectedPower=float(reflectedPower),
            transmittedPower=float(transmittedPower),
            reflectedPropagating=isPropagating(kzReflectedForward[index]),
            transmittedPropagating=isPropagating(kzTransmittedForward[index]),
        )


def zeroOrderIndex(harmonics: Harmonics) -> int:
    zeroIndices = np.where((harmonics.mx == 0) & (harmonics.my == 0))[0]
    if zeroIndices.size != 1:
        raise RuntimeError("zero diffraction order was not found")
    return int(zeroIndices[0])


def isPropagating(kz: complex) -> bool:
    return bool(abs(np.imag(kz)) < 1e-10 and np.real(kz) > 1e-12)


def _incidentAmplitudes(prepared: PreparedStack, sAmplitude: complex, pAmplitude: complex) -> ComplexArray:
    amplitudes = np.zeros(prepared.nPorts, dtype=complex)
    amplitudes[2 * prepared.zeroIndex] = sAmplitude
    amplitudes[2 * prepared.zeroIndex + 1] = pAmplitude
    return amplitudes


def _checkedFlux(field: ComplexArray) -> float:
    incidentFlux = flux(field)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def _result(
    prepared: PreparedStack,
    rAmplitudes: ComplexArray,
    tAmplitudes: ComplexArray,
    incidentFlux: float,
    solvedBy: str,
    layerSolutions: tuple[LayerFieldSolution, ...],
    sAmplitude: complex,
    pAmplitude: complex,
) -> RCWAResult:
    reflectedField = prepared.incidentBackward @ rAmplitudes
    transmittedField = prepared.transmissionForward @ tAmplitudes
    reflection = float(-flux(reflectedField) / incidentFlux)
    transmission = float(flux(transmittedField) / incidentFlux)
    orders = orderResults(
        harmonics=prepared.harmonics,
        epsReflected=prepared.epsIncident,
        epsTransmitted=prepared.epsTransmission,
        reflectedBasis=prepared.incidentBackward,
        transmittedBasis=prepared.transmissionForward,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        incidentFlux=incidentFlux,
    )
    return RCWAResult(
        reflection=reflection,
        transmission=transmission,
        conservation=reflection + transmission,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=tuple(orders),
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


def _layerSolutions(prepared: PreparedStack, incidentAmplitudes: ComplexArray) -> tuple[LayerFieldSolution, ...]:
    prefixes = prefixSMatrices(prepared.components, prepared.nPorts)
    suffixes = suffixSMatrices(prepared.components, prepared.nPorts)
    identity = np.eye(prepared.nPorts, dtype=complex)
    solutions = []
    for layerIndex, (layer, (qValues, modeMatrix)) in enumerate(zip(prepared.layers, prepared.layerModes)):
        boundaryIndex = 2 * layerIndex + 1
        leftNetwork = prefixes[boundaryIndex]
        rightNetwork = suffixes[boundaryIndex]
        rhs = rightNetwork.s11 @ (leftNetwork.s21 @ incidentAmplitudes)
        backwardAtLeft = np.linalg.solve(identity - rightNetwork.s11 @ leftNetwork.s22, rhs)
        forwardAtLeft = leftNetwork.s21 @ incidentAmplitudes + leftNetwork.s22 @ backwardAtLeft
        rightBoundaryIndex = boundaryIndex + 1
        leftAtRight = prefixes[rightBoundaryIndex]
        rightAtRight = suffixes[rightBoundaryIndex]
        rhsAtRight = rightAtRight.s11 @ (leftAtRight.s21 @ incidentAmplitudes)
        backwardAtRight = np.linalg.solve(identity - rightAtRight.s11 @ leftAtRight.s22, rhsAtRight)
        coefficients = np.concatenate([forwardAtLeft, backwardAtLeft])
        solutions.append(
            LayerFieldSolution(
                name=getattr(layer, "name", ""),
                thickness=float(layer.thickness),
                wavelength=prepared.wavelength,
                period=prepared.period,
                orders=prepared.orders,
                mx=prepared.harmonics.mx.copy(),
                my=prepared.harmonics.my.copy(),
                kx=prepared.harmonics.kx.copy(),
                ky=prepared.harmonics.ky.copy(),
                qValues=qValues.copy(),
                modeMatrix=modeMatrix.copy(),
                coefficients=coefficients,
                epsilonInverse=layerEpsilonData(layer, prepared.harmonics).epsilonInverse.copy(),
                backwardCoefficientsRight=backwardAtRight.copy(),
            )
        )
    return tuple(solutions)


def _layerModesTorch(layer: Layer | CompiledLayer, harmonics: Harmonics, torch: Any, device: Any) -> tuple[Any, Any]:
    factorized = _layerDataForTorch(layer, harmonics)
    if factorized.homogeneousEpsilon is not None and factorized.displacementMatrices is None:
        qValuesNp, modeMatrixNp = homogeneousScalarLayerModes(harmonics, factorized.homogeneousEpsilon)
        return _toTorchComplex(qValuesNp, torch, device), _toTorchComplex(modeMatrixNp, torch, device)

    epsilonMatrix = _toTorchComplex(factorized.epsilonMatrix, torch, device)
    if factorized.epsilonInverse is None:
        epsilonInverse = torch.linalg.inv(epsilonMatrix)
    else:
        epsilonInverse = _toTorchComplex(factorized.epsilonInverse, torch, device)
    displacementMatrices = None
    if factorized.displacementMatrices is not None:
        displacementMatrices = tuple(_toTorchComplex(matrix, torch, device) for matrix in factorized.displacementMatrices)

    pMatrix, qMatrix = _pqMatricesTorch(epsilonMatrix, harmonics, epsilonInverse, displacementMatrices, torch, device)
    qSquared, electricModes = torch.linalg.eig(pMatrix @ qMatrix)
    qValues = _forwardKzTorch(qSquared, torch)
    safeQ = qValues.clone()
    safeQ[torch.abs(safeQ) < 1e-13] = 1e-13 + 0j

    magneticModes = qMatrix @ electricModes @ torch.diag(1.0 / safeQ)
    vectors = torch.cat(
        [
            torch.cat([electricModes, electricModes], dim=1),
            torch.cat([magneticModes, -magneticModes], dim=1),
        ],
        dim=0,
    )
    qAll = torch.cat([qValues, -qValues], dim=0)
    return qAll, _normalizeColumnsTorch(vectors, torch)


def _layerDataForTorch(layer: Layer | CompiledLayer, harmonics: Harmonics) -> _TorchLayerData:
    if hasattr(layer, "epsilonMatrix") and hasattr(layer, "epsilonInverse"):
        factorized = layerEpsilonData(layer, harmonics)
        return _TorchLayerData(
            factorized.epsilonMatrix,
            factorized.epsilonInverse,
            factorized.displacementMatrices,
            factorized.homogeneousEpsilon,
        )

    epsilon = getattr(layer, "epsilon")
    factorizationMode = _normalizeFactorization(getattr(layer, "factorization", "auto"))
    normalField = getattr(layer, "normalField", None)
    needsFullFactorization = (
        _useAnalyticNormalVector(epsilon, factorizationMode)
        or factorizationMode in ("normal-vector", "jones")
        or (factorizationMode == "auto" and normalField is not None)
        or _shouldAutoGenerateNormalField(epsilon, factorizationMode)
    )
    if needsFullFactorization:
        factorized = layerEpsilonData(layer, harmonics)
        return _TorchLayerData(
            factorized.epsilonMatrix,
            factorized.epsilonInverse,
            factorized.displacementMatrices,
            factorized.homogeneousEpsilon,
        )

    epsilonMatrix = epsilonConvolutionMatrix(epsilon, harmonics)
    return _TorchLayerData(
        epsilonMatrix=epsilonMatrix,
        epsilonInverse=None,
        displacementMatrices=None,
        homogeneousEpsilon=_homogeneousEpsilon(epsilon),
    )


def _pqMatricesTorch(
    epsilonMatrix: Any,
    harmonics: Harmonics,
    epsilonInverse: Any,
    displacementMatrices: tuple[Any, Any, Any, Any] | None,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    n = harmonics.count
    identity = torch.eye(n, dtype=torch.complex128, device=device)
    kx = _toTorchComplex(harmonics.kx, torch, device)
    ky = _toTorchComplex(harmonics.ky, torch, device)

    p11 = kx[:, None] * epsilonInverse * ky[None, :]
    p12 = identity - kx[:, None] * epsilonInverse * kx[None, :]
    p21 = ky[:, None] * epsilonInverse * ky[None, :] - identity
    p22 = -ky[:, None] * epsilonInverse * kx[None, :]

    if displacementMatrices is None:
        cxx = epsilonMatrix
        cxy = torch.zeros_like(epsilonMatrix)
        cyx = torch.zeros_like(epsilonMatrix)
        cyy = epsilonMatrix
    else:
        cxx, cxy, cyx, cyy = displacementMatrices

    q11 = -torch.diag(kx * ky) - cyx
    q12 = torch.diag(kx * kx) - cyy
    q21 = cxx - torch.diag(ky * ky)
    q22 = cxy + torch.diag(ky * kx)
    pMatrix = torch.cat([torch.cat([p11, p12], dim=1), torch.cat([p21, p22], dim=1)], dim=0)
    qMatrix = torch.cat([torch.cat([q11, q12], dim=1), torch.cat([q21, q22], dim=1)], dim=0)
    return pMatrix, qMatrix


def _interfaceSMatrixTorch(
    leftForward: Any,
    leftBackward: Any,
    rightForward: Any,
    rightBackward: Any,
    torch: Any,
) -> _TorchSMatrix:
    size = leftForward.shape[1]
    matrix = torch.cat([leftBackward, -rightForward], dim=1)
    rhs = torch.cat([-leftForward, rightBackward], dim=1)
    solved = torch.linalg.solve(matrix, rhs)
    return _TorchSMatrix(
        s11=solved[:size, :size],
        s12=solved[:size, size:],
        s21=solved[size:, :size],
        s22=solved[size:, size:],
    )


def _propagationSMatrixTorch(propagation: Any, torch: Any) -> _TorchSMatrix:
    zero = torch.zeros_like(propagation)
    return _TorchSMatrix(s11=zero, s12=propagation, s21=propagation, s22=zero)


def _identitySMatrixTorch(size: int, torch: Any, device: Any) -> _TorchSMatrix:
    zero = torch.zeros((size, size), dtype=torch.complex128, device=device)
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    return _TorchSMatrix(s11=zero, s12=identity, s21=identity, s22=zero)


def _redhefferStarTorch(left: _TorchSMatrix, right: _TorchSMatrix, torch: Any, device: Any) -> _TorchSMatrix:
    size = left.s11.shape[0]
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    leftSolved = torch.linalg.solve(
        identity - right.s11 @ left.s22,
        torch.cat([right.s11, right.s12], dim=1),
    )
    rightSolved = torch.linalg.solve(
        identity - left.s22 @ right.s11,
        torch.cat([left.s22, left.s21], dim=1),
    )
    leftDenominator, leftTransmission = torch.split(leftSolved, size, dim=1)
    rightDenominator, rightTransmission = torch.split(rightSolved, size, dim=1)
    s11 = left.s11 + left.s12 @ leftDenominator @ left.s21
    s12 = left.s12 @ leftTransmission
    s21 = right.s21 @ rightTransmission
    s22 = right.s22 + right.s21 @ rightDenominator @ right.s12
    return _TorchSMatrix(s11=s11, s12=s12, s21=s21, s22=s22)


def _cascadeManyTorch(components: Sequence[_TorchSMatrix], size: int, torch: Any, device: Any) -> _TorchSMatrix:
    if not components:
        return _identitySMatrixTorch(size, torch, device)
    result = components[0]
    for component in components[1:]:
        result = _redhefferStarTorch(result, component, torch, device)
    return result


def _prefixSMatricesTorch(components: Sequence[_TorchSMatrix], size: int, torch: Any, device: Any) -> list[_TorchSMatrix]:
    prefixes = [_identitySMatrixTorch(size, torch, device)]
    current = prefixes[0]
    for component in components:
        current = _redhefferStarTorch(current, component, torch, device)
        prefixes.append(current)
    return prefixes


def _suffixSMatricesTorch(components: Sequence[_TorchSMatrix], size: int, torch: Any, device: Any) -> list[_TorchSMatrix]:
    suffixes = [_identitySMatrixTorch(size, torch, device) for _ in range(len(components) + 1)]
    current = _identitySMatrixTorch(size, torch, device)
    suffixes[len(components)] = current
    for index in range(len(components) - 1, -1, -1):
        current = _redhefferStarTorch(components[index], current, torch, device)
        suffixes[index] = current
    return suffixes


def _forwardKzTorch(values: Any, torch: Any) -> Any:
    roots = torch.sqrt(values + 0j)
    flip = (roots.imag < -1e-14) | ((torch.abs(roots.imag) <= 1e-14) & (roots.real < 0))
    return torch.where(flip, -roots, roots)


def _normalizeColumnsTorch(vectors: Any, torch: Any) -> Any:
    scales = torch.amax(torch.abs(vectors), dim=0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return vectors / scales


def _incidentAmplitudesTorch(prepared: PreparedTorchStack, sAmplitude: complex, pAmplitude: complex, torch: Any) -> Any:
    amplitudes = torch.zeros(prepared.nPorts, dtype=torch.complex128, device=prepared.total.s11.device)
    amplitudes[2 * prepared.zeroIndex] = complex(sAmplitude)
    amplitudes[2 * prepared.zeroIndex + 1] = complex(pAmplitude)
    return amplitudes


def _incidentAmplitudesNumpy(
    nPorts: int,
    zeroIndex: int,
    sAmplitude: complex,
    pAmplitude: complex,
) -> ComplexArray:
    amplitudes = np.zeros(nPorts, dtype=complex)
    amplitudes[2 * zeroIndex] = complex(sAmplitude)
    amplitudes[2 * zeroIndex + 1] = complex(pAmplitude)
    return amplitudes


def _checkedIncidentFluxTorch(prepared: PreparedTorchStack, incidentAmplitudes: ComplexArray) -> float:
    incidentFieldValue = prepared.incidentForward @ incidentAmplitudes
    incidentFlux = flux(incidentFieldValue)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def _layerSolutionsTorch(prepared: PreparedTorchStack, incidentAmplitudes: Any) -> tuple[LayerFieldSolution, ...]:
    torch = prepared.backend.xp
    device = prepared.total.s11.device
    prefixes = _prefixSMatricesTorch(prepared.components, prepared.nPorts, torch, device)
    suffixes = _suffixSMatricesTorch(prepared.components, prepared.nPorts, torch, device)
    identity = torch.eye(prepared.nPorts, dtype=torch.complex128, device=device)

    solutions = []
    for layerIndex, (layer, (qValues, modeMatrix)) in enumerate(zip(prepared.layers, prepared.layerModes)):
        boundaryIndex = 2 * layerIndex + 1
        leftNetwork = prefixes[boundaryIndex]
        rightNetwork = suffixes[boundaryIndex]
        rhs = rightNetwork.s11 @ (leftNetwork.s21 @ incidentAmplitudes)
        backwardAtLeft = torch.linalg.solve(identity - rightNetwork.s11 @ leftNetwork.s22, rhs)
        forwardAtLeft = leftNetwork.s21 @ incidentAmplitudes + leftNetwork.s22 @ backwardAtLeft
        rightBoundaryIndex = boundaryIndex + 1
        leftAtRight = prefixes[rightBoundaryIndex]
        rightAtRight = suffixes[rightBoundaryIndex]
        rhsAtRight = rightAtRight.s11 @ (leftAtRight.s21 @ incidentAmplitudes)
        backwardAtRight = torch.linalg.solve(identity - rightAtRight.s11 @ leftAtRight.s22, rhsAtRight)
        coefficients = torch.cat([forwardAtLeft, backwardAtLeft])
        solutions.append(
            LayerFieldSolution(
                name=getattr(layer, "name", ""),
                thickness=float(layer.thickness),
                wavelength=prepared.wavelength,
                period=prepared.period,
                orders=prepared.orders,
                mx=prepared.harmonics.mx.copy(),
                my=prepared.harmonics.my.copy(),
                kx=prepared.harmonics.kx.copy(),
                ky=prepared.harmonics.ky.copy(),
                qValues=prepared.backend.asnumpy(qValues).copy(),
                modeMatrix=prepared.backend.asnumpy(modeMatrix).copy(),
                coefficients=prepared.backend.asnumpy(coefficients).copy(),
                epsilonInverse=_layerEpsilonInverseNumpy(prepared, layer),
                backwardCoefficientsRight=prepared.backend.asnumpy(backwardAtRight).copy(),
            )
        )
    return tuple(solutions)


def _layerEpsilonInverseNumpy(prepared: PreparedTorchStack, layer: Layer | CompiledLayer) -> ComplexArray | None:
    epsilonInverse = layerEpsilonData(layer, prepared.harmonics).epsilonInverse
    if epsilonInverse is None:
        return None
    return prepared.backend.asnumpy(epsilonInverse).copy()


def _resultFromTorchAmplitudes(
    prepared: PreparedTorchStack,
    rAmplitudes: ComplexArray,
    tAmplitudes: ComplexArray,
    incidentFlux: float,
    solvedBy: str,
    layerSolutions: tuple[LayerFieldSolution, ...],
    sAmplitude: complex,
    pAmplitude: complex,
) -> RCWAResult:
    reflectedField = prepared.incidentBackward @ rAmplitudes
    transmittedField = prepared.transmissionForward @ tAmplitudes
    reflection = float(-flux(reflectedField) / incidentFlux)
    transmission = float(flux(transmittedField) / incidentFlux)
    orders = orderResults(
        harmonics=prepared.harmonics,
        epsReflected=prepared.epsIncident,
        epsTransmitted=prepared.epsTransmission,
        reflectedBasis=prepared.incidentBackward,
        transmittedBasis=prepared.transmissionForward,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        incidentFlux=incidentFlux,
    )
    return RCWAResult(
        reflection=reflection,
        transmission=transmission,
        conservation=reflection + transmission,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=tuple(orders),
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


def _toTorchComplex(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex128, device=device)


def _normalizeFactorization(value: str) -> str:
    normalized = str(value).lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "standard": "standard",
        "none": "standard",
        "direct": "standard",
        "normal-vector": "normal-vector",
        "normal-vector-li": "normal-vector",
        "nv": "normal-vector",
        "jones": "jones",
        "jones-li": "jones",
    }
    if normalized not in aliases:
        raise ValueError("factorization must be 'auto', 'standard', 'normal-vector', or 'jones'")
    return aliases[normalized]


def _analyticFactorizationName(epsilon: object) -> str:
    if hasattr(epsilon, "convolutionMatrix"):
        return "analytic-li"
    return "standard"


def _useAnalyticNormalVector(epsilon: object, factorizationMode: str) -> bool:
    return bool(
        hasattr(epsilon, "normalVectorMatrices")
        and hasattr(epsilon, "reciprocalConvolutionMatrix")
        and (
            factorizationMode == "jones"
            or (factorizationMode == "auto" and getattr(epsilon, "factorization", "analytic") == "jones")
        )
    )


def _shouldAutoGenerateNormalField(epsilon: object, factorizationMode: str) -> bool:
    if factorizationMode not in ("auto", "normal-vector", "jones"):
        return False
    if hasattr(epsilon, "convolutionMatrix") or np.isscalar(epsilon):
        return False
    array = np.asarray(epsilon)
    if array.ndim != 2 or array.shape == (3, 3):
        return False
    if factorizationMode == "auto":
        return _looksPiecewiseConstant(array)
    return True


def _looksPiecewiseConstant(values: object) -> bool:
    grid = np.asarray(values)
    if grid.ndim != 2 or grid.size == 0:
        return False
    if not np.all(np.isfinite(grid)):
        return False

    scale = max(1.0, float(np.max(np.abs(grid))))
    tolerance = 1e-10 * scale
    rounded = np.round(grid.real / tolerance) + 1j * np.round(grid.imag / tolerance)
    uniqueCount = np.unique(rounded).size
    return 1 < uniqueCount <= max(16, grid.size // 4)


def _estimateNormalField(
    values: object,
    orders: int | tuple[int, int] | None = None,
    truncation: str | None = None,
) -> ComplexArray:
    del orders, truncation
    grid = np.asarray(values, dtype=complex)
    if grid.ndim != 2:
        raise ValueError("normal-field estimation requires a 2D scalar grid")
    if grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("sampled epsilon grid must be non-empty")

    contrast = np.abs(grid - np.mean(grid))
    dx = 0.5 * (np.roll(contrast, -1, axis=1) - np.roll(contrast, 1, axis=1))
    dy = 0.5 * (np.roll(contrast, -1, axis=0) - np.roll(contrast, 1, axis=0))
    length = np.sqrt(dx**2 + dy**2)

    normals = np.zeros(grid.shape + (2,), dtype=float)
    active = length > 1e-12 * max(1.0, float(np.max(length)) if length.size else 1.0)
    normals[..., 0] = np.where(active, dx / np.where(active, length, 1.0), 1.0)
    normals[..., 1] = np.where(active, dy / np.where(active, length, 1.0), 0.0)
    return normals


def _analyticNormalVectorDisplacementMatrices(
    epsilon: object,
    harmonics: Harmonics,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    direct = epsilonConvolutionMatrix(epsilon, harmonics)
    inverseRule = np.linalg.inv(epsilon.reciprocalConvolutionMatrix(harmonics))
    nx, ny, tx, ty = epsilon.normalVectorMatrices(harmonics)
    return _normalVectorBlocks(direct, inverseRule, nx, ny, tx, ty)


def _normalVectorDisplacementMatrices(
    epsilon: object,
    normalField: object,
    harmonics: Harmonics,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    grid = np.asarray(epsilon, dtype=complex)
    normals = np.asarray(normalField, dtype=float)
    if grid.ndim != 2:
        raise ValueError("normal-vector factorization requires a 2D scalar epsilon grid")
    if normals.shape != grid.shape + (2,):
        raise ValueError("normalField must have shape (ny, nx, 2) matching epsilon")

    normalX = normals[..., 0]
    normalY = normals[..., 1]
    length = np.sqrt(normalX * normalX + normalY * normalY)
    safe = length > 1e-12
    normalX = np.where(safe, normalX / np.where(safe, length, 1.0), 1.0)
    normalY = np.where(safe, normalY / np.where(safe, length, 1.0), 0.0)
    tangentX = -normalY
    tangentY = normalX

    direct = epsilonConvolutionMatrix(grid, harmonics)
    inverseRule = np.linalg.inv(epsilonConvolutionMatrix(1.0 / grid, harmonics))
    nx = epsilonConvolutionMatrix(normalX, harmonics)
    ny = epsilonConvolutionMatrix(normalY, harmonics)
    tx = epsilonConvolutionMatrix(tangentX, harmonics)
    ty = epsilonConvolutionMatrix(tangentY, harmonics)
    return _normalVectorBlocks(direct, inverseRule, nx, ny, tx, ty)


def _normalVectorBlocks(
    direct: ComplexArray,
    inverseRule: ComplexArray,
    nx: ComplexArray,
    ny: ComplexArray,
    tx: ComplexArray,
    ty: ComplexArray,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    cxx = nx @ inverseRule @ nx + tx @ direct @ tx
    cxy = nx @ inverseRule @ ny + tx @ direct @ ty
    cyx = ny @ inverseRule @ nx + ty @ direct @ tx
    cyy = ny @ inverseRule @ ny + ty @ direct @ ty
    return cxx, cxy, cyx, cyy


def _homogeneousEpsilon(epsilon: object) -> complex | None:
    if hasattr(epsilon, "convolutionMatrix"):
        return None
    if np.isscalar(epsilon):
        return complex(epsilon)
    array = np.asarray(epsilon, dtype=complex)
    if array.ndim == 0:
        return complex(array.item())
    if array.ndim == 2 and array.size > 0 and np.allclose(array, array.flat[0], rtol=0.0, atol=1e-14):
        return complex(array.flat[0])
    return None


def _validateGeometry(wavelength: float, period: tuple[float, float], orders: int | tuple[int, int]) -> None:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")
    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")


def _validateIsotropicLayers(layers: Sequence[Layer | CompiledLayer], kind: str) -> None:
    for index, layer in enumerate(layers):
        if type(layer).__module__.startswith("rcwa3d_anisotropic"):
            _raiseAnisotropicPathError(kind, index)

        if hasattr(layer, "epsilon"):
            epsilon = getattr(layer, "epsilon")
            if hasattr(epsilon, "convolutionMatrix"):
                continue
            if np.isscalar(epsilon):
                continue
            epsilonArray = np.asarray(epsilon)
            if epsilonArray.ndim not in (0, 2):
                _raiseAnisotropicPathError(kind, index)


def _raiseAnisotropicPathError(kind: str, layerIndex: int) -> None:
    raise TypeError(
        f"layer {layerIndex} uses an anisotropic/tensor-like permittivity in the isotropic {kind} path; "
        "use `rcwa3d_anisotropic.solveStack` or `rcwa3d_anisotropic.compileLayers` instead."
    )


def _expandAdaptiveLayers(
    layers: Sequence[Layer | CompiledLayer | AdaptiveLayerSpec],
) -> list[Layer | CompiledLayer]:
    expanded: list[Layer | CompiledLayer] = []
    for layer in layers:
        if isinstance(layer, AdaptiveLayerSpec):
            expanded.extend(layer.toLayers())
        else:
            expanded.append(layer)
    return expanded


def _requireSMatrixMethod(method: str) -> None:
    if str(method).lower() != "smatrix":
        raise ValueError("the isotropic solver now supports only method='smatrix'")


def _backendSuffix(backend: ArrayBackend) -> str:
    if not backend.isCuda:
        raise ValueError("the isotropic solver is CUDA-only")
    return "cuda"


prepareStackSMatrix = prepareStack
