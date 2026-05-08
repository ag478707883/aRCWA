from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .backend import ArrayBackend, resolveBackend
from ._factorization import (
    homogeneousEpsilon as _homogeneousEpsilon,
    layerDataForTorch as _layerDataForTorch,
    pqMatricesTorch as _pqMatricesTorch,
    solveIdentityTorch as _solveIdentityTorch,
    toTorchComplex as _toTorchComplex,
)
from .fourier import Harmonics, flux, forwardKz, makeHarmonics, normalizeOrders, planeWaveFields, putOrderField, singleOrderVector, sqrtBranch
from .types import ComplexArray, CompiledLayer, DiffractionOrder, Layer, LayerEigTiming, LayerFieldSolution, RCWAResult
from .varrcwa import AdaptiveLayerSpec


@dataclass(frozen=True)
class _TorchSMatrix:
    s11: Any
    s12: Any
    s21: Any
    s22: Any
    isIdentity: bool = False
    isPropagation: bool = False


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
    layerEigTimings: tuple[LayerEigTiming, ...]
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


def _compileLayers(
    layers: Sequence[Layer | AdaptiveLayerSpec],
    orders: int | tuple[int, int],
    truncation: str = "circular",
    backend: str | ArrayBackend | None = "cuda",
) -> tuple[CompiledLayer, ...]:
    """Precompute Fourier convolution data for fixed orders/truncation using CUDA PyTorch."""

    expandedLayers = _expandAdaptiveLayers(layers)
    _validateIsotropicLayers(expandedLayers, kind="compile")
    arrayBackend = resolveBackend(backend)
    torch = arrayBackend.xp
    device = arrayBackend.device if arrayBackend.device is not None else torch.device("cuda")

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
        factorized = _layerDataForTorch(layer, harmonics, torch, device)
        epsilonInverse = factorized.epsilonInverse
        if epsilonInverse is None:
            epsilonInverse = _solveIdentityTorch(factorized.epsilonMatrix, torch, device)
        displacement = None
        if factorized.displacementMatrices is not None:
            displacement = tuple(arrayBackend.asnumpy(matrix).copy() for matrix in factorized.displacementMatrices)
        compiledLayers.append(
            CompiledLayer(
                thickness=layer.thickness,
                epsilonMatrix=arrayBackend.asnumpy(factorized.epsilonMatrix).copy(),
                epsilonInverse=arrayBackend.asnumpy(epsilonInverse).copy(),
                orders=normalizedOrders,
                truncation=harmonics.truncation,
                name=layer.name,
                displacementMatrices=displacement,
                factorization=factorized.factorization,
                homogeneousEpsilon=factorized.homogeneousEpsilon,
            )
        )
    return tuple(compiledLayers)


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
    profile: bool = False,
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

    incidentForward = _homogeneousBasisTorch(harmonics, epsIncident, direction=1, torch=torch, device=device)
    incidentBackward = _homogeneousBasisTorch(harmonics, epsIncident, direction=-1, torch=torch, device=device)
    transmissionForward = _homogeneousBasisTorch(harmonics, epsTransmission, direction=1, torch=torch, device=device)
    transmissionBackward = _homogeneousBasisTorch(harmonics, epsTransmission, direction=-1, torch=torch, device=device)

    modeList: list[tuple[Any, Any]] = []
    layerEigTimings: list[LayerEigTiming] = []
    for layerIndex, layer in enumerate(layers):
        modes, timing = _layerModesWithTimingTorch(
            layer,
            harmonics,
            torch,
            device,
            layerIndex=layerIndex,
            profile=profile,
        )
        modeList.append(modes)
        if timing is not None:
            layerEigTimings.append(timing)
    modes = tuple(modeList)
    regionForward: list[Any] = [incidentForward]
    regionBackward: list[Any] = [incidentBackward]
    for _qValues, modeMatrix in modes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)

    interfaces = _interfaceSMatricesTorch(regionForward, regionBackward, torch)
    components: list[_TorchSMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(interfaces[regionIndex])
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
        layerEigTimings=tuple(layerEigTimings),
        components=componentTuple,
        total=_reflectionTransmissionOnlySMatrixTorch(componentTuple, nPorts, torch, device),
        incidentForward=arrayBackend.asnumpy(incidentForward).copy(),
        incidentBackward=arrayBackend.asnumpy(incidentBackward).copy(),
        transmissionForward=arrayBackend.asnumpy(transmissionForward).copy(),
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


def evaluatePreparedBatchPowersTorch(
    prepared: PreparedTorchStack,
    excitations: Mapping[str, tuple[complex, complex]],
) -> dict[str, tuple[float, float]]:
    labels = tuple(excitations)
    if not labels:
        return {}

    torch = prepared.backend.xp
    columns = [_incidentAmplitudesTorch(prepared, *excitations[label], torch) for label in labels]
    incidentColumns = torch.column_stack(columns)
    reflected = prepared.backend.asnumpy(prepared.total.s11 @ incidentColumns)
    transmitted = prepared.backend.asnumpy(prepared.total.s21 @ incidentColumns)

    reflectedFluxes = _orderFluxesFromLocalBasis(prepared.incidentBackward, reflected)
    transmittedFluxes = _orderFluxesFromLocalBasis(prepared.transmissionForward, transmitted)
    powers: dict[str, tuple[float, float]] = {}
    for column, label in enumerate(labels):
        incidentFlux = _checkedIncidentFluxTorch(
            prepared,
            _incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, *excitations[label]),
        )
        reflection = float(np.sum(-reflectedFluxes[:, column] / incidentFlux))
        transmission = float(np.sum(transmittedFluxes[:, column] / incidentFlux))
        powers[label] = (reflection, transmission)
    return powers


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
    profile: bool = False,
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
        profile=profile,
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


def _solveBatchReducedPowersTorch(
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
) -> dict[str, tuple[float, float]]:
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
    return evaluatePreparedBatchPowersTorch(prepared, excitations)


def _reducedLayers(
    layers: Sequence[Layer | CompiledLayer],
    plan: _OrderReductionPlan,
) -> tuple[Layer | CompiledLayer, ...]:
    reduced: list[Layer | CompiledLayer] = []
    for layer in layers:
        if _isHomogeneousLayer(layer):
            reduced.append(_homogeneousEquivalentLayer(layer))
        elif _isCompiledLayer(layer):
            reduced.append(_sliceCompiledLayer(layer, plan))
        else:
            reduced.append(layer)
    return tuple(reduced)


def _homogeneousEquivalentLayer(layer: Layer | CompiledLayer) -> Layer:
    if _isCompiledLayer(layer):
        if getattr(layer, "homogeneousEpsilon", None) is None:
            raise RuntimeError("compiled layer is not homogeneous")
        return Layer(thickness=layer.thickness, epsilon=getattr(layer, "homogeneousEpsilon"), name=getattr(layer, "name", ""))
    epsilon = _homogeneousEpsilon(getattr(layer, "epsilon"))
    if epsilon is None:
        raise RuntimeError("layer is not homogeneous")
    return Layer(thickness=layer.thickness, epsilon=epsilon, name=getattr(layer, "name", ""))


def _sliceCompiledLayer(layer: CompiledLayer, plan: _OrderReductionPlan) -> CompiledLayer:
    displacement = getattr(layer, "displacementMatrices", None)
    if _isTorchTensor(getattr(layer, "epsilonMatrix")):
        index = _torchIndex(plan.keptIndices, getattr(layer, "epsilonMatrix"))
        return type(layer)(
            thickness=getattr(layer, "thickness"),
            epsilonMatrix=_torchSliceSquare(getattr(layer, "epsilonMatrix"), index),
            epsilonInverse=_torchSliceSquare(getattr(layer, "epsilonInverse"), index),
            orders=plan.reducedOrders,
            truncation=plan.fullHarmonics.truncation,
            name=getattr(layer, "name", ""),
            displacementMatrices=None
            if displacement is None
            else tuple(_torchSliceSquare(matrix, index) for matrix in displacement),
            factorization=getattr(layer, "factorization", "standard"),
            homogeneousEpsilon=getattr(layer, "homogeneousEpsilon", None),
        )

    indexer = np.ix_(plan.keptIndices, plan.keptIndices)
    return CompiledLayer(
        thickness=getattr(layer, "thickness"),
        epsilonMatrix=np.asarray(getattr(layer, "epsilonMatrix"))[indexer].copy(),
        epsilonInverse=np.asarray(getattr(layer, "epsilonInverse"))[indexer].copy(),
        orders=plan.reducedOrders,
        truncation=plan.fullHarmonics.truncation,
        name=getattr(layer, "name", ""),
        displacementMatrices=None
        if displacement is None
        else tuple(np.asarray(matrix)[indexer].copy() for matrix in displacement),
        factorization=getattr(layer, "factorization", "standard"),
        homogeneousEpsilon=getattr(layer, "homogeneousEpsilon", None),
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

    incidentBackward = _homogeneousBasis(fullHarmonics, epsIncident, direction=-1)
    transmissionForward = _homogeneousBasis(fullHarmonics, epsTransmission, direction=1)
    incident = _incidentField(fullHarmonics, epsIncident, sAmplitude, pAmplitude)
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
        layerEigTimings=reduced.layerEigTimings,
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
        _isCompiledLayer(layer)
        and (getattr(layer, "orders") != harmonics.orders or getattr(layer, "truncation") != harmonics.truncation)
        for layer in layers
    )


def _stackInvariantAlong(axis: str, layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return all(_layerInvariantAlong(axis, layer, harmonics) for layer in layers)


def _layerInvariantAlong(axis: str, layer: Layer | CompiledLayer, harmonics: Harmonics) -> bool:
    if _isHomogeneousLayer(layer):
        return True
    if _isCompiledLayer(layer):
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
    if getattr(layer, "orders") != harmonics.orders or getattr(layer, "truncation") != harmonics.truncation:
        return False
    if getattr(layer, "homogeneousEpsilon", None) is not None and getattr(layer, "displacementMatrices", None) is None:
        return True
    if axis == "y":
        uncoupled = harmonics.deltaMy != 0
    elif axis == "x":
        uncoupled = harmonics.deltaMx != 0
    else:
        raise ValueError("axis must be 'x' or 'y'")

    matrices = [getattr(layer, "epsilonMatrix"), getattr(layer, "epsilonInverse")]
    displacement = getattr(layer, "displacementMatrices", None)
    if displacement is not None:
        matrices.extend(displacement)
    scale = max(1.0, max(_matrixMaxAbs(matrix) for matrix in matrices if _matrixSize(matrix)))
    tolerance = 1e-10 * scale
    return all(_matrixMaskedMaxAbs(matrix, uncoupled) <= tolerance for matrix in matrices)


def _isHomogeneousLayer(layer: Layer | CompiledLayer) -> bool:
    if _isCompiledLayer(layer):
        return getattr(layer, "homogeneousEpsilon", None) is not None
    return _homogeneousEpsilon(getattr(layer, "epsilon")) is not None


def _isCompiledLayer(layer: object) -> bool:
    return hasattr(layer, "epsilonMatrix") and hasattr(layer, "epsilonInverse")


def _isTorchTensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device")


def _torchIndex(indices: ComplexArray, reference: Any) -> Any:
    import torch as torch_module

    return torch_module.as_tensor(indices, dtype=torch_module.long, device=reference.device)


def _torchSliceSquare(matrix: Any, index: Any) -> Any:
    return matrix.index_select(0, index).index_select(1, index).clone()


def _matrixSize(matrix: object) -> int:
    return int(matrix.numel()) if _isTorchTensor(matrix) else int(np.asarray(matrix).size)


def _matrixMaxAbs(matrix: object) -> float:
    if _isTorchTensor(matrix):
        if matrix.numel() == 0:
            return 0.0
        return float(matrix.abs().amax().detach().cpu().item())
    array = np.asarray(matrix)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _matrixMaskedMaxAbs(matrix: object, mask: ComplexArray) -> float:
    if _isTorchTensor(matrix):
        import torch

        torchMask = torch.as_tensor(mask, dtype=torch.bool, device=matrix.device)
        if not bool(torch.any(torchMask).item()):
            return 0.0
        return float(matrix[torchMask].abs().amax().detach().cpu().item())
    values = np.asarray(matrix)[mask]
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _incidentField(
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


def _homogeneousBasis(harmonics: Harmonics, eps: complex, direction: int) -> ComplexArray:
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
    reflectedFluxes = _orderFluxesFromLocalBasis(reflectedBasis, rAmplitudes)
    transmittedFluxes = _orderFluxesFromLocalBasis(transmittedBasis, tAmplitudes)
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
            reflectedPropagating=isPropagating(kzReflectedForward[index]),
            transmittedPropagating=isPropagating(kzTransmittedForward[index]),
        )


def _orderFluxesFromLocalBasis(basis: ComplexArray, amplitudes: ComplexArray) -> ComplexArray:
    amplitudeArray = np.asarray(amplitudes)
    nOrders = amplitudeArray.shape[0] // 2
    singleColumn = amplitudeArray.ndim == 1
    if singleColumn:
        amplitudeArray = amplitudeArray[:, None]

    orderIndex = np.arange(nOrders)
    sColumns = 2 * orderIndex
    pColumns = sColumns + 1

    sAmplitudes = amplitudeArray[sColumns, :]
    pAmplitudes = amplitudeArray[pColumns, :]

    sEx = basis[orderIndex, sColumns][:, None]
    pEx = basis[orderIndex, pColumns][:, None]
    sEy = basis[nOrders + orderIndex, sColumns][:, None]
    pEy = basis[nOrders + orderIndex, pColumns][:, None]
    sHx = basis[2 * nOrders + orderIndex, sColumns][:, None]
    pHx = basis[2 * nOrders + orderIndex, pColumns][:, None]
    sHy = basis[3 * nOrders + orderIndex, sColumns][:, None]
    pHy = basis[3 * nOrders + orderIndex, pColumns][:, None]

    ex = sEx * sAmplitudes + pEx * pAmplitudes
    ey = sEy * sAmplitudes + pEy * pAmplitudes
    hx = sHx * sAmplitudes + pHx * pAmplitudes
    hy = sHy * sAmplitudes + pHy * pAmplitudes
    values = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
    return values[:, 0] if singleColumn else values


def zeroOrderIndex(harmonics: Harmonics) -> int:
    zeroIndices = np.where((harmonics.mx == 0) & (harmonics.my == 0))[0]
    if zeroIndices.size != 1:
        raise RuntimeError("zero diffraction order was not found")
    return int(zeroIndices[0])


def isPropagating(kz: complex) -> bool:
    return bool(abs(np.imag(kz)) < 1e-10 and np.real(kz) > 1e-12)


def _checkedFlux(field: ComplexArray) -> float:
    incidentFlux = flux(field)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def _layerModesTorch(layer: Layer | CompiledLayer, harmonics: Harmonics, torch: Any, device: Any) -> tuple[Any, Any]:
    modes, _timing = _layerModesCoreTorch(
        layer,
        harmonics,
        torch,
        device,
        layerIndex=-1,
        profile=False,
    )
    return modes


def _layerModesWithTimingTorch(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
    *,
    layerIndex: int,
    profile: bool,
) -> tuple[tuple[Any, Any], LayerEigTiming | None]:
    return _layerModesCoreTorch(
        layer,
        harmonics,
        torch,
        device,
        layerIndex=layerIndex,
        profile=profile,
    )


def _layerModesCoreTorch(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
    *,
    layerIndex: int,
    profile: bool,
) -> tuple[tuple[Any, Any], LayerEigTiming | None]:
    factorized = _layerDataForTorch(layer, harmonics, torch, device)
    if factorized.homogeneousEpsilon is not None and factorized.displacementMatrices is None:
        modes = _homogeneousScalarLayerModesTorch(harmonics, factorized.homogeneousEpsilon, torch, device)
        timing = None
        if profile:
            timing = LayerEigTiming(
                layerIndex=layerIndex,
                name=getattr(layer, "name", ""),
                kind="homogeneous-analytic",
                matrixShape=(4, 4),
                eigTimeSeconds=0.0,
            )
        return modes, timing

    epsilonMatrix = _toTorchComplex(factorized.epsilonMatrix, torch, device)
    if factorized.epsilonInverse is None:
        epsilonInverse = torch.linalg.solve(
            epsilonMatrix,
            torch.eye(epsilonMatrix.shape[0], dtype=torch.complex128, device=device),
        )
    else:
        epsilonInverse = _toTorchComplex(factorized.epsilonInverse, torch, device)
    displacementMatrices = None
    if factorized.displacementMatrices is not None:
        displacementMatrices = tuple(_toTorchComplex(matrix, torch, device) for matrix in factorized.displacementMatrices)

    pMatrix, qMatrix = _pqMatricesTorch(epsilonMatrix, harmonics, epsilonInverse, displacementMatrices, torch, device)
    eigenMatrix = pMatrix @ qMatrix
    if profile:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        qSquared, electricModes = torch.linalg.eig(eigenMatrix)
        torch.cuda.synchronize(device)
        eigTime = time.perf_counter() - start
    else:
        qSquared, electricModes = torch.linalg.eig(eigenMatrix)
        eigTime = 0.0
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
    modes = qAll, _normalizeColumnsTorch(vectors, torch)
    timing = None
    if profile:
        timing = LayerEigTiming(
            layerIndex=layerIndex,
            name=getattr(layer, "name", ""),
            kind=factorized.factorization,
            matrixShape=tuple(eigenMatrix.shape),
            eigTimeSeconds=eigTime,
        )
    return modes, timing


def _homogeneousBasisTorch(harmonics: Harmonics, eps: complex, direction: int, torch: Any, device: Any) -> Any:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    n = harmonics.count
    basis = torch.zeros((4 * n, 2 * n), dtype=torch.complex128, device=device)
    kxValues = _toTorchComplex(harmonics.kx, torch, device)
    kyValues = _toTorchComplex(harmonics.ky, torch, device)
    kzForward = _forwardKzTorch(complex(eps) - kxValues * kxValues - kyValues * kyValues, torch)
    kz = kzForward if direction > 0 else -kzForward
    refractiveIndex = complex(sqrtBranch(eps))

    kp = torch.sqrt(kxValues * kxValues + kyValues * kyValues + 0j)
    normalIncidence = torch.abs(kp) < 1e-14
    safeKp = torch.where(normalIncidence, torch.ones_like(kp), kp)
    sx = torch.where(normalIncidence, torch.zeros_like(kp), -kyValues / safeKp)
    sy = torch.where(normalIncidence, torch.ones_like(kp), kxValues / safeKp)
    px = sy * kz / refractiveIndex
    py = -sx * kz / refractiveIndex
    hxS = -kz * sy
    hyS = kz * sx
    pz = (sx * kyValues - sy * kxValues) / refractiveIndex
    hxP = kyValues * pz - kz * py
    hyP = kz * px - kxValues * pz

    row = torch.arange(n, device=device)
    sColumn = 2 * row
    pColumn = sColumn + 1
    basis[row, sColumn] = sx
    basis[n + row, sColumn] = sy
    basis[2 * n + row, sColumn] = hxS
    basis[3 * n + row, sColumn] = hyS
    basis[row, pColumn] = px
    basis[n + row, pColumn] = py
    basis[2 * n + row, pColumn] = hxP
    basis[3 * n + row, pColumn] = hyP
    return basis


def _homogeneousScalarLayerModesTorch(harmonics: Harmonics, epsilon: complex, torch: Any, device: Any) -> tuple[Any, Any]:
    forward = _homogeneousBasisTorch(harmonics, epsilon, direction=1, torch=torch, device=device)
    backward = _homogeneousBasisTorch(harmonics, epsilon, direction=-1, torch=torch, device=device)
    kx = _toTorchComplex(harmonics.kx, torch, device)
    ky = _toTorchComplex(harmonics.ky, torch, device)
    kzForward = _forwardKzTorch(complex(epsilon) - kx * kx - ky * ky, torch)
    qForward = torch.empty(2 * harmonics.count, dtype=torch.complex128, device=device)
    qForward[0::2] = kzForward
    qForward[1::2] = kzForward
    return torch.cat([qForward, -qForward]), torch.cat([forward, backward], dim=1)


def _interfaceSMatricesTorch(
    regionForward: Sequence[Any],
    regionBackward: Sequence[Any],
    torch: Any,
) -> tuple[_TorchSMatrix, ...]:
    size = regionForward[0].shape[1]
    matrices = []
    rightHandSides = []
    for index in range(len(regionForward) - 1):
        matrices.append(torch.cat([regionBackward[index], -regionForward[index + 1]], dim=1))
        rightHandSides.append(torch.cat([-regionForward[index], regionBackward[index + 1]], dim=1))

    if not matrices:
        return ()

    solvedBatch = torch.linalg.solve(torch.stack(matrices, dim=0), torch.stack(rightHandSides, dim=0))
    return tuple(
        _TorchSMatrix(
            s11=solved[:size, :size],
            s12=solved[:size, size:],
            s21=solved[size:, :size],
            s22=solved[size:, size:],
        )
        for solved in solvedBatch
    )


def _propagationSMatrixTorch(propagation: Any, torch: Any) -> _TorchSMatrix:
    zero = torch.zeros_like(propagation)
    return _TorchSMatrix(s11=zero, s12=propagation, s21=propagation, s22=zero, isPropagation=True)


def _identitySMatrixTorch(size: int, torch: Any, device: Any) -> _TorchSMatrix:
    zero = torch.zeros((size, size), dtype=torch.complex128, device=device)
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    return _TorchSMatrix(s11=zero, s12=identity, s21=identity, s22=zero, isIdentity=True)


def _redhefferStarTorch(left: _TorchSMatrix, right: _TorchSMatrix, torch: Any, device: Any) -> _TorchSMatrix:
    if left.isIdentity:
        return right
    if right.isIdentity:
        return left
    if right.isPropagation:
        propagation = torch.diagonal(right.s21)
        return _TorchSMatrix(
            s11=left.s11,
            s12=left.s12 * propagation[None, :],
            s21=propagation[:, None] * left.s21,
            s22=propagation[:, None] * left.s22 * propagation[None, :],
        )
    if left.isPropagation:
        propagation = torch.diagonal(left.s21)
        return _TorchSMatrix(
            s11=propagation[:, None] * right.s11 * propagation[None, :],
            s12=propagation[:, None] * right.s12,
            s21=right.s21 * propagation[None, :],
            s22=right.s22,
        )

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


def _reflectionTransmissionOnlySMatrixTorch(
    components: Sequence[_TorchSMatrix],
    size: int,
    torch: Any,
    device: Any,
) -> _TorchSMatrix:
    reflection, transmission = _enhancedReflectionTransmissionTorch(components, size, torch, device)
    zero = torch.zeros_like(reflection)
    return _TorchSMatrix(s11=reflection, s12=zero.clone(), s21=transmission, s22=zero.clone())


def _enhancedReflectionTransmissionTorch(
    components: Sequence[_TorchSMatrix],
    size: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    reflection = torch.zeros((size, size), dtype=torch.complex128, device=device)
    transmission = identity.clone()
    for component in reversed(components):
        if component.isIdentity:
            continue
        if component.isPropagation:
            propagation = torch.diagonal(component.s21)
            reflection = propagation[:, None] * reflection * propagation[None, :]
            transmission = transmission * propagation[None, :]
            continue
        internalReflection = torch.linalg.solve(
            identity - reflection @ component.s22,
            reflection @ component.s21,
        )
        forward = component.s21 + component.s22 @ internalReflection
        transmission = transmission @ forward
        reflection = component.s11 + component.s12 @ internalReflection
    return reflection, transmission


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
    device = prepared.total.s11.device
    factorized = _layerDataForTorch(layer, prepared.harmonics, prepared.backend.xp, device)
    epsilonInverse = factorized.epsilonInverse
    if epsilonInverse is None:
        epsilonInverse = _solveIdentityTorch(factorized.epsilonMatrix, prepared.backend.xp, device)
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
        layerEigTimings=prepared.layerEigTimings,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


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


def _backendSuffix(backend: ArrayBackend) -> str:
    if not backend.isCuda:
        raise ValueError("the isotropic solver is CUDA-only")
    return "cuda"
