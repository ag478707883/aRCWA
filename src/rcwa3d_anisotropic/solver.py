from __future__ import annotations

import os
import time
from typing import Iterable, Mapping, Sequence
import weakref

import numpy as np

from .backend import ArrayBackend as ArrayBackend
from .backend import resolveBackend as resolveBackend
from .factorization import TensorConvolutionData
from .factorization import constantTensor as constantTensor
from .factorization import layerTensorData as layerTensorData
from .factorization import tensorConvolutionData as tensorConvolutionData
from .fourier import Harmonics as Harmonics
from .fourier import makeHarmonics as makeHarmonics
from .fourier import normalizeOrders as normalizeOrders
from .phase import flux as flux
from .phase import forwardKz as forwardKz
from .phase import planeWaveFields as planeWaveFields
from .phase import putOrderField as putOrderField
from .smatrix import (
    cascadeMany,
    matrixDiagonal,
    patternedHomogeneousInterfaceEnabled,
    prefixSMatrices,
    propagationSMatrixBidirectional,
    propagationSMatrixBidirectionalBatch,
    redhefferStar,
    reflectionTransmissionOnlySMatrix,
    solveFactored,
    solveInterfaceBlocks,
    suffixSMatrices,
)
from .types import (
    ComplexArray,
    DiffractionOrder,
    Layer,
    LayerEigTiming,
    LayerFieldSolution,
    PreparedBatchStack,
    PreparedStack,
    RCWAResult,
    StackTiming,
    TensorLike,
    AutomaticFastPathPlan,
    BackendTensorConvolutionData,
    BatchedHarmonics,
    BatchedHomogeneousLayer,
    CompiledLayer,
    PatternedHomogeneousInterfaceWork,
    SMatrix,
)


BackendTensorCacheKey = tuple[int, str, str, str]
BackendTensorCacheEntry = tuple[weakref.ReferenceType[TensorConvolutionData], "BackendTensorConvolutionData"]
BACKEND_TENSOR_DATA_CACHE: dict[BackendTensorCacheKey, BackendTensorCacheEntry] = {}


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
    backend: str | ArrayBackend | None = "cuda",
    precision: str = "complex128",
    fullTotal: bool = True,
    profile: bool = False,
) -> PreparedStack:
    prepareStart = time.perf_counter()
    validateInputs(wavelength, period, orders)
    arrayBackend = resolveBackend(backend, precision=precision)
    xp = arrayBackend.xp

    harmonics = makeHarmonics(wavelength, period, orders, epsIncident, theta, phi, truncation=truncation)
    nOrders = harmonics.count
    nPorts = 2 * nOrders
    k0 = 2 * np.pi / wavelength

    incidentForward = arrayBackend.asarray(homogeneousBasis(harmonics, epsIncident, direction=1))
    incidentBackward = arrayBackend.asarray(homogeneousBasis(harmonics, epsIncident, direction=-1))
    transmissionForward = arrayBackend.asarray(homogeneousBasis(harmonics, epsTransmission, direction=1))
    transmissionBackward = arrayBackend.asarray(homogeneousBasis(harmonics, epsTransmission, direction=-1))

    profileTimings = profileTimingsEnabled(profile)
    layerModeList: list[tuple[object, object]] = []
    layerEigTimings: list[LayerEigTiming] = []
    layerModeCache: dict[tuple[object, ...], tuple[object, object]] | None = {} if not profileTimings else None
    for layerIndex, layer in enumerate(layers):
        modeCacheKey = layerModeCacheKey(layer) if layerModeCache is not None else None
        cachedModes = layerModeCache.get(modeCacheKey) if modeCacheKey is not None else None
        if cachedModes is None:
            modes, timing = layerModesWithTiming(
                layer,
                harmonics,
                arrayBackend,
                layerIndex=layerIndex,
                collectTiming=profileTimings,
            )
            if layerModeCache is not None and modeCacheKey is not None:
                layerModeCache[modeCacheKey] = modes
        else:
            modes, timing = cachedModes, None
        layerModeList.append(modes)
        if timing is not None:
            layerEigTimings.append(timing)
    layerModes = tuple(layerModeList)
    layerHomogeneous = tuple(isHomogeneousLayer(layer) for layer in layers)
    regionForward = [incidentForward]
    regionBackward = [incidentBackward]
    for qValues, modeMatrix in layerModes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)
    regionHomogeneous = (True, *layerHomogeneous, True)

    interfaceStart = startTimedOperation(arrayBackend) if profileTimings else None
    interfaces = interfaceSMatrices(
        regionForward,
        regionBackward,
        regionHomogeneous,
        nOrders,
        arrayBackend,
    )
    conditionNumbers = (
        interfaceConditionNumbers(regionForward, regionBackward, arrayBackend) if profileTimings else ()
    )
    interfaceTime = finishTimedOperation(arrayBackend, interfaceStart) if interfaceStart is not None else 0.0
    components: list[SMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(interfaces[regionIndex])
        if regionIndex < len(layers):
            qValues = layerModes[regionIndex][0]
            qForward = qValues[:nPorts]
            qBackward = qValues[nPorts:]
            propagationForward = xp.exp(1j * qForward * k0 * layers[regionIndex].thickness)
            propagationBackward = xp.exp(-1j * qBackward * k0 * layers[regionIndex].thickness)
            components.append(propagationSMatrixBidirectional(propagationForward, propagationBackward, arrayBackend))

    componentTuple = tuple(components)
    cascadeStart = startTimedOperation(arrayBackend) if profileTimings else None
    total = (
        cascadeMany(componentTuple, nPorts, arrayBackend)
        if fullTotal
        else reflectionTransmissionOnlySMatrix(componentTuple, nPorts, arrayBackend)
    )
    cascadeTime = finishTimedOperation(arrayBackend, cascadeStart) if cascadeStart is not None else 0.0
    stackTiming = None
    if profileTimings:
        stackTiming = StackTiming(
            interfaceTimeSeconds=interfaceTime,
            cascadeTimeSeconds=cascadeTime,
            totalPrepareTimeSeconds=time.perf_counter() - prepareStart,
            interfaceConditionNumbers=conditionNumbers,
            maxInterfaceCondition=max(conditionNumbers) if conditionNumbers else None,
            stabilityWarnings=stabilityWarnings(layerEigTimings, conditionNumbers),
        )
    return PreparedStack(
        layers=tuple(layers),
        wavelength=float(wavelength),
        period=period,
        orders=tuple(normalizeOrders(orders)),
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=float(theta),
        phi=float(phi),
        truncation=harmonics.truncation,
        backend=arrayBackend.name,
        precision=backendPrecisionLabel(arrayBackend),
        harmonics=harmonics,
        components=componentTuple,
        total=total,
        layerModes=layerModes,
        incidentBackward=incidentBackward,
        transmissionForward=transmissionForward,
        zeroIndex=zeroOrderIndex(harmonics),
        layerEigTimings=tuple(layerEigTimings),
        stackTiming=stackTiming,
    )


def prepareStackSMatrixBatch(
    layers: Sequence[Layer | CompiledLayer | BatchedHomogeneousLayer],
    wavelengths: Sequence[float],
    period: tuple[float, float],
    orders: tuple[int, int],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float | Sequence[float] = 0.0,
    phi: float | Sequence[float] = 0.0,
    truncation: str = "circular",
    backend: str | ArrayBackend | None = "cuda",
    precision: str = "complex128",
) -> PreparedBatchStack:
    """Prepare reflection/transmission S-matrices for a wavelength batch."""

    values = np.asarray(tuple(wavelengths), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("wavelength batch must be a non-empty one-dimensional sequence")
    for wavelength in values:
        validateInputs(float(wavelength), period, orders)
    thetaValues = batchParameterValues(theta, values.size, "theta")
    phiValues = batchParameterValues(phi, values.size, "phi")

    arrayBackend = resolveBackend(backend, precision=precision)
    if not arrayBackend.isTorch:
        raise ValueError("batched anisotropic spectrum preparation requires the CUDA torch backend")

    harmonics = makeBatchedHarmonics(
        wavelengths=values,
        period=period,
        orders=orders,
        epsIncident=epsIncident,
        theta=thetaValues,
        phi=phiValues,
        truncation=truncation,
        backend=arrayBackend,
    )
    validateBatchedCompiledLayers(layers, harmonics)
    nOrders = harmonics.count
    nPorts = 2 * nOrders
    k0 = arrayBackend.asarray(2 * np.pi / values)

    incidentForward = homogeneousBasisBatch(harmonics, epsIncident, direction=1, backend=arrayBackend)
    incidentBackward = homogeneousBasisBatch(harmonics, epsIncident, direction=-1, backend=arrayBackend)
    transmissionForward = homogeneousBasisBatch(harmonics, epsTransmission, direction=1, backend=arrayBackend)
    transmissionBackward = homogeneousBasisBatch(harmonics, epsTransmission, direction=-1, backend=arrayBackend)

    layerModeCache: dict[tuple[object, ...], tuple[object, object]] = {}
    layerModeList = []
    for layer in layers:
        modeCacheKey = layerModeCacheKey(layer)
        cachedModes = layerModeCache.get(modeCacheKey) if modeCacheKey is not None else None
        if cachedModes is None:
            cachedModes = layerModesBatch(layer, harmonics, arrayBackend)
            if modeCacheKey is not None:
                layerModeCache[modeCacheKey] = cachedModes
        layerModeList.append(cachedModes)
    layerModes = tuple(layerModeList)
    layerHomogeneous = tuple(isHomogeneousLayer(layer) for layer in layers)
    regionForward = [incidentForward]
    regionBackward = [incidentBackward]
    for qValues, modeMatrix in layerModes:
        regionForward.append(modeMatrix[..., :, :nPorts])
        regionBackward.append(modeMatrix[..., :, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)
    regionHomogeneous = (True, *layerHomogeneous, True)

    interfaces = interfaceSMatrices(
        regionForward,
        regionBackward,
        regionHomogeneous,
        nOrders,
        arrayBackend,
    )
    components: list[SMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(interfaces[regionIndex])
        if regionIndex < len(layers):
            qValues = layerModes[regionIndex][0]
            qForward = qValues[..., :nPorts]
            qBackward = qValues[..., nPorts:]
            propagationForward = arrayBackend.xp.exp(
                1j * qForward * k0[:, None] * layers[regionIndex].thickness
            )
            propagationBackward = arrayBackend.xp.exp(
                -1j * qBackward * k0[:, None] * layers[regionIndex].thickness
            )
            components.append(propagationSMatrixBidirectionalBatch(propagationForward, propagationBackward, arrayBackend))

    total = reflectionTransmissionOnlySMatrix(tuple(components), nPorts, arrayBackend)
    return PreparedBatchStack(
        layers=tuple(layers),
        wavelengths=values.copy(),
        period=period,
        orders=tuple(normalizeOrders(orders)),
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        theta=batchStoredParameter(thetaValues),
        phi=batchStoredParameter(phiValues),
        truncation=harmonics.truncation,
        backend=arrayBackend.name,
        precision=backendPrecisionLabel(arrayBackend),
        harmonics=harmonics,
        total=total,
        incidentForward=incidentForward,
        incidentBackward=incidentBackward,
        transmissionForward=transmissionForward,
        zeroIndex=zeroOrderIndexFromOrders(harmonics.mx, harmonics.my),
    )


def evaluatePreparedStack(
    prepared: PreparedStack,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    *,
    solvedBy: str = "smatrix",
) -> RCWAResult:
    arrayBackend = resolveBackend(prepared.backend, precision=prepared.precision)
    incident = incidentField(
        harmonics=prepared.harmonics,
        eps=prepared.epsIncident,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )
    incidentFlux = checkedIncidentFlux(incident)
    incidentVector = incidentAmplitudes(prepared, sAmplitude, pAmplitude, arrayBackend)
    rAmplitudes = prepared.total.s11 @ incidentVector
    tAmplitudes = prepared.total.s21 @ incidentVector

    return result(
        harmonics=prepared.harmonics,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        reflectedBasis=arrayBackend.toNumpy(prepared.incidentBackward),
        transmittedBasis=arrayBackend.toNumpy(prepared.transmissionForward),
        rAmplitudes=arrayBackend.toNumpy(rAmplitudes),
        tAmplitudes=arrayBackend.toNumpy(tAmplitudes),
        incidentFlux=incidentFlux,
        solvedBy=solvedBy,
        layerSolutions=(),
        layerEigTimings=prepared.layerEigTimings,
        stackTiming=prepared.stackTiming,
    )


def evaluatePreparedFieldStack(
    prepared: PreparedStack,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    *,
    solvedBy: str = "smatrix-fields",
) -> RCWAResult:
    """Evaluate a full prepared stack and attach finite-layer field data."""

    arrayBackend = resolveBackend(prepared.backend, precision=prepared.precision)
    incident = incidentField(
        harmonics=prepared.harmonics,
        eps=prepared.epsIncident,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )
    incidentFlux = checkedIncidentFlux(incident)
    incidentVector = incidentAmplitudes(prepared, sAmplitude, pAmplitude, arrayBackend)
    rAmplitudes = prepared.total.s11 @ incidentVector
    tAmplitudes = prepared.total.s21 @ incidentVector
    layerSolutions = (
        smatrixLayerSolutions(
            layers=prepared.layers,
            layerModes=prepared.layerModes,
            components=prepared.components,
            incidentAmplitudes=incidentVector,
            harmonics=prepared.harmonics,
            wavelength=prepared.wavelength,
            period=prepared.period,
            orders=prepared.orders,
            backend=arrayBackend,
        )
        if prepared.layers
        else ()
    )

    return result(
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
        stackTiming=prepared.stackTiming,
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

    arrayBackend = resolveBackend(prepared.backend, precision=prepared.precision)
    incidentColumns = arrayBackend.asarray(
        np.column_stack([incidentAmplitudeVector(prepared, *excitations[label]) for label in labels])
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
        incident = incidentField(
            harmonics=prepared.harmonics,
            eps=prepared.epsIncident,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
        )
        results[label] = result(
            harmonics=prepared.harmonics,
            epsIncident=prepared.epsIncident,
            epsTransmission=prepared.epsTransmission,
            reflectedBasis=reflectedBasis,
            transmittedBasis=transmittedBasis,
            rAmplitudes=reflectedNumpy[:, column],
            tAmplitudes=transmittedNumpy[:, column],
            incidentFlux=checkedIncidentFlux(incident),
            solvedBy=solvedBy,
            layerSolutions=(),
            layerEigTimings=prepared.layerEigTimings,
            stackTiming=prepared.stackTiming,
        )
    return results


def evaluatePreparedBatchPowers(
    prepared: PreparedStack,
    excitations: Mapping[str, tuple[complex, complex]],
) -> dict[str, tuple[float, float]]:
    labels = tuple(excitations)
    if not labels:
        return {}

    arrayBackend = resolveBackend(prepared.backend, precision=prepared.precision)
    incidentColumns = arrayBackend.asarray(
        np.column_stack([incidentAmplitudeVector(prepared, *excitations[label]) for label in labels])
    )
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    if arrayBackend.isTorch:
        return evaluatePreparedBatchPowersOnBackend(
            prepared,
            labels,
            excitations,
            arrayBackend,
            reflected,
            transmitted,
        )

    reflectedBasis = arrayBackend.toNumpy(prepared.incidentBackward)
    transmittedBasis = arrayBackend.toNumpy(prepared.transmissionForward)
    reflectedNumpy = arrayBackend.toNumpy(reflected)
    transmittedNumpy = arrayBackend.toNumpy(transmitted)

    powers: dict[str, tuple[float, float]] = {}
    for column, label in enumerate(labels):
        sAmplitude, pAmplitude = excitations[label]
        incident = incidentField(
            harmonics=prepared.harmonics,
            eps=prepared.epsIncident,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
        )
        incidentFlux = checkedIncidentFlux(incident)
        reflectedFluxes = homogeneousOrderFluxes(reflectedBasis, reflectedNumpy[:, column], prepared.harmonics.count)
        transmittedFluxes = homogeneousOrderFluxes(
            transmittedBasis,
            transmittedNumpy[:, column],
            prepared.harmonics.count,
        )
        reflection = float(np.sum(-reflectedFluxes / incidentFlux))
        transmission = float(np.sum(transmittedFluxes / incidentFlux))
        powers[label] = (reflection, transmission)
    return powers


def evaluatePreparedSpectrumBatchPowers(
    prepared: PreparedBatchStack,
    excitations: Mapping[str, tuple[complex, complex]],
) -> dict[str, tuple[ComplexArray, ComplexArray]]:
    """Evaluate reflected/transmitted powers for all wavelengths on CUDA."""

    labels = tuple(excitations)
    if not labels:
        return {}

    arrayBackend = resolveBackend(prepared.backend, precision=prepared.precision)
    incidentColumns = arrayBackend.asarray(
        np.column_stack([batchedIncidentAmplitudeVector(prepared, *excitations[label]) for label in labels])
    )
    incidentColumns = incidentColumns[None, :, :].expand(prepared.harmonics.batchSize, -1, -1)
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    torch = arrayBackend.xp.torch
    incidentFluxes = torch.sum(
        homogeneousOrderFluxesBatchBackend(
            prepared.incidentForward,
            incidentColumns,
            prepared.harmonics.count,
            arrayBackend,
        ),
        dim=1,
    )
    if bool(torch.any(~torch.isfinite(incidentFluxes))) or bool(torch.any(torch.abs(incidentFluxes) < 1e-14)):
        raise ValueError("incident field has near-zero real power flux")
    reflectedFluxes = homogeneousOrderFluxesBatchBackend(
        prepared.incidentBackward,
        reflected,
        prepared.harmonics.count,
        arrayBackend,
    )
    transmittedFluxes = homogeneousOrderFluxesBatchBackend(
        prepared.transmissionForward,
        transmitted,
        prepared.harmonics.count,
        arrayBackend,
    )
    reflection = torch.sum(-reflectedFluxes / incidentFluxes[:, None, :], dim=1)
    transmission = torch.sum(transmittedFluxes / incidentFluxes[:, None, :], dim=1)

    reflectionValues = reflection.detach().cpu().numpy()
    transmissionValues = transmission.detach().cpu().numpy()
    return {
        label: (reflectionValues[:, index].copy(), transmissionValues[:, index].copy())
        for index, label in enumerate(labels)
    }


def evaluatePreparedBatchPowersOnBackend(
    prepared: PreparedStack,
    labels: tuple[str, ...],
    excitations: Mapping[str, tuple[complex, complex]],
    backend: ArrayBackend,
    reflected: object,
    transmitted: object,
) -> dict[str, tuple[float, float]]:
    """Evaluate batch powers on the CUDA backend and transfer only scalars."""

    torch = backend.xp.torch
    incidentFluxes = [
        checkedIncidentFlux(
            incidentField(
                harmonics=prepared.harmonics,
                eps=prepared.epsIncident,
                sAmplitude=excitations[label][0],
                pAmplitude=excitations[label][1],
            )
        )
        for label in labels
    ]
    incidentFlux = torch.as_tensor(incidentFluxes, dtype=backend.floatDtype, device=backend.device)

    reflectedFluxes = homogeneousOrderFluxesBackend(
        prepared.incidentBackward,
        reflected,
        prepared.harmonics.count,
        backend,
    )
    transmittedFluxes = homogeneousOrderFluxesBackend(
        prepared.transmissionForward,
        transmitted,
        prepared.harmonics.count,
        backend,
    )
    reflection = torch.sum(-reflectedFluxes / incidentFlux[None, :], dim=0)
    transmission = torch.sum(transmittedFluxes / incidentFlux[None, :], dim=0)

    reflectionValues = reflection.detach().cpu().numpy()
    transmissionValues = transmission.detach().cpu().numpy()
    return {
        label: (float(reflectionValues[index]), float(transmissionValues[index]))
        for index, label in enumerate(labels)
    }


def compileLayers(
    layers: Sequence[Layer],
    orders: int | tuple[int, int],
    truncation: str = "circular",
) -> tuple[CompiledLayer, ...]:
    """Precompute tensor convolution matrices for a fixed harmonic truncation."""

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
    compiledLayers = []
    for layer in layers:
        validateHomogeneousMuSupport(layer)
        muTensor = None if getattr(layer, "mu", None) is None else constantTensor(getattr(layer, "mu"))
        compiledLayers.append(
            CompiledLayer(
                thickness=layer.thickness,
                tensorData=tensorConvolutionData(
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
                sampleShape=getattr(layer, "sampleShape", sampleShapeFromEpsilon(layer.epsilon)),
                mu=muTensor,
            )
        )
    return tuple(compiledLayers)


def sampleShapeFromEpsilon(epsilon: object) -> tuple[int, int] | None:
    if epsilon is None or hasattr(epsilon, "convolutionMatrix"):
        return None
    if isinstance(epsilon, Mapping):
        shapes = [sampleShapeFromEpsilon(value) for value in epsilon.values()]
        shapes = [shape for shape in shapes if shape is not None]
        return shapes[0] if shapes and all(shape == shapes[0] for shape in shapes) else None
    array = np.asarray(epsilon)
    if array.ndim == 2 and array.shape != (3, 3):
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 4 and array.shape[-2:] == (3, 3):
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 4 and array.shape[:2] == (3, 3):
        return int(array.shape[2]), int(array.shape[3])
    return None


def warmBackendTensorCache(
    layers: Sequence[Layer | CompiledLayer],
    backend: str | ArrayBackend | None = "cuda",
    precision: str = "complex128",
) -> None:
    """Materialize compiled tensor convolution data on the target backend."""

    arrayBackend = resolveBackend(backend, precision=precision)
    for layer in layers:
        if isinstance(layer, CompiledLayer):
            backendTensorConvolutionData(layer.tensorData, arrayBackend)


def backendPrecisionLabel(backend: ArrayBackend) -> str:
    torch = getattr(backend.xp, "torch", None)
    if torch is not None and backend.complexDtype is torch.complex64:
        return "complex64"
    return "complex128"


def automaticFastPathPlan(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex,
    theta: float,
    phi: float,
    truncation: str,
) -> AutomaticFastPathPlan | None:
    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders == (0, 0):
        return None

    fullHarmonics = makeHarmonics(
        wavelength,
        period,
        normalizedOrders,
        epsIncident,
        theta,
        phi,
        truncation=truncation,
    )
    if hasMismatchedCompiledLayer(layers, fullHarmonics):
        return None
    if all(isHomogeneousLayer(layer) for layer in layers):
        return makeFastPathPlan("homogeneous-4x4", (0, 0), fullHarmonics)

    nx, ny = normalizedOrders
    if ny > 0 and stackInvariantAlong("y", layers, fullHarmonics):
        return makeFastPathPlan("1d-x-4x4", (nx, 0), fullHarmonics)
    if nx > 0 and stackInvariantAlong("x", layers, fullHarmonics):
        return makeFastPathPlan("1d-y-4x4", (0, ny), fullHarmonics)
    return None


def makeBatchedHarmonics(
    *,
    wavelengths: ComplexArray,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex,
    theta: ComplexArray,
    phi: ComplexArray,
    truncation: str,
    backend: ArrayBackend,
) -> BatchedHarmonics:
    reference = makeHarmonics(
        float(wavelengths[0]),
        period,
        orders,
        epsIncident,
        float(theta[0]),
        float(phi[0]),
        truncation=truncation,
    )
    torch = backend.xp.torch
    wavelengthTensor = torch.as_tensor(wavelengths, dtype=backend.complexDtype, device=backend.device)
    mx = torch.as_tensor(reference.mx, dtype=backend.complexDtype, device=backend.device)
    my = torch.as_tensor(reference.my, dtype=backend.complexDtype, device=backend.device)
    nIncident = forwardKz(epsIncident)
    thetaTensor = torch.as_tensor(theta, dtype=backend.complexDtype, device=backend.device)
    phiTensor = torch.as_tensor(phi, dtype=backend.complexDtype, device=backend.device)
    nIncidentTensor = torch.as_tensor(nIncident, dtype=backend.complexDtype, device=backend.device)
    kx0 = nIncidentTensor * torch.sin(thetaTensor) * torch.cos(phiTensor)
    ky0 = nIncidentTensor * torch.sin(thetaTensor) * torch.sin(phiTensor)
    kx = kx0[:, None] + wavelengthTensor[:, None] * mx[None, :] / period[0]
    ky = ky0[:, None] + wavelengthTensor[:, None] * my[None, :] / period[1]
    return BatchedHarmonics(
        mx=reference.mx.copy(),
        my=reference.my.copy(),
        kx=kx,
        ky=ky,
        orders=reference.orders,
        truncation=reference.truncation,
    )


def batchParameterValues(value: float | Sequence[float], batchSize: int, name: str) -> ComplexArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(batchSize, float(array), dtype=float)
    if array.shape != (batchSize,):
        raise ValueError(f"{name} batch must be scalar or have shape (batch,)")
    return array.copy()


def batchStoredParameter(values: ComplexArray) -> float:
    if values.size == 0:
        return 0.0
    first = float(values[0])
    return first if np.allclose(values, first, rtol=0.0, atol=0.0) else float("nan")


def validateBatchedCompiledLayers(
    layers: Sequence[Layer | CompiledLayer | BatchedHomogeneousLayer],
    harmonics: BatchedHarmonics,
) -> None:
    mismatch = [
        layer
        for layer in layers
        if isinstance(layer, CompiledLayer)
        and (layer.orders != harmonics.orders or layer.truncation != harmonics.truncation)
    ]
    if mismatch:
        raise ValueError("compiled layer orders or truncation do not match the requested batched harmonics")
    invalidHomogeneous = [
        layer
        for layer in layers
        if isinstance(layer, BatchedHomogeneousLayer)
        and (np.asarray(layer.tensors).shape != (harmonics.batchSize, 3, 3))
    ]
    if invalidHomogeneous:
        raise ValueError("batched homogeneous layer tensors must have shape (batch, 3, 3)")
    invalidMu = [
        layer
        for layer in layers
        if isinstance(layer, BatchedHomogeneousLayer)
        and layer.mus is not None
        and np.asarray(layer.mus).shape != (harmonics.batchSize, 3, 3)
    ]
    if invalidMu:
        raise ValueError("batched homogeneous layer mu tensors must have shape (batch, 3, 3)")


def makeFastPathPlan(
    label: str,
    reducedOrders: tuple[int, int],
    fullHarmonics: Harmonics,
) -> AutomaticFastPathPlan:
    kept = reducedHarmonicIndices(fullHarmonics, reducedOrders)
    return AutomaticFastPathPlan(
        label=label,
        reducedOrders=reducedOrders,
        fullHarmonics=fullHarmonics,
        keptIndices=kept,
    )


def reducedFastPathLayers(
    layers: Sequence[Layer | CompiledLayer],
    plan: AutomaticFastPathPlan,
) -> tuple[Layer | CompiledLayer, ...]:
    reduced: list[Layer | CompiledLayer] = []
    for layer in layers:
        if isHomogeneousLayer(layer):
            reduced.append(homogeneousEquivalentLayer(layer))
        elif isinstance(layer, CompiledLayer):
            reduced.append(sliceCompiledLayer(layer, plan))
        else:
            reduced.append(layer)
    return tuple(reduced)


def homogeneousEquivalentLayer(layer: Layer | CompiledLayer) -> Layer:
    if isinstance(layer, CompiledLayer):
        tensor = layer.tensorData.constantTensor
        if tensor is None:
            raise RuntimeError("compiled layer is not homogeneous")
        return Layer(thickness=layer.thickness, epsilon=tensor, mu=layer.mu, name=layer.name)
    tensor, muTensor = constantLayerTensors(layer)
    if tensor is None:
        raise RuntimeError("layer is not homogeneous")
    return Layer(thickness=layer.thickness, epsilon=tensor, mu=muTensor, name=getattr(layer, "name", ""))


def sliceCompiledLayer(layer: CompiledLayer, plan: AutomaticFastPathPlan) -> CompiledLayer:
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
        sampleShape=layer.sampleShape,
        mu=layer.mu,
    )


def embedReducedResult(
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

    incident = incidentField(
        harmonics=fullHarmonics,
        eps=epsIncident,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )
    return result(
        harmonics=fullHarmonics,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        reflectedBasis=homogeneousBasis(fullHarmonics, epsIncident, direction=-1),
        transmittedBasis=homogeneousBasis(fullHarmonics, epsTransmission, direction=1),
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        incidentFlux=checkedIncidentFlux(incident),
        solvedBy=solvedBy,
        layerSolutions=(),
        layerEigTimings=reduced.layerEigTimings,
        stackTiming=reduced.stackTiming,
    )


def reducedHarmonicIndices(harmonics: Harmonics, reducedOrders: tuple[int, int]) -> ComplexArray:
    nx, ny = reducedOrders
    mask = (np.abs(harmonics.mx) <= nx) & (np.abs(harmonics.my) <= ny)
    return np.flatnonzero(mask)


def stackInvariantAlong(axis: str, layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return all(layerInvariantAlong(axis, layer, harmonics) for layer in layers)


def hasMismatchedCompiledLayer(layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return any(
        isinstance(layer, CompiledLayer)
        and (layer.orders != harmonics.orders or layer.truncation != harmonics.truncation)
        for layer in layers
    )


def layerInvariantAlong(axis: str, layer: Layer | CompiledLayer, harmonics: Harmonics) -> bool:
    if isHomogeneousLayer(layer):
        return True
    if isinstance(layer, CompiledLayer):
        return compiledLayerInvariantAlong(axis, layer, harmonics)
    return rawLayerInvariantAlong(axis, layer)


def rawLayerInvariantAlong(axis: str, layer: Layer) -> bool:
    if getattr(layer, "mu", None) is not None:
        muTensor = constantTensor(getattr(layer, "mu"))
        if muTensor is None and not epsilonInvariantAlong(axis, getattr(layer, "mu")):
            return False
    epsilon = getattr(layer, "epsilon")
    if constantTensor(epsilon) is not None:
        return True
    if not epsilonInvariantAlong(axis, epsilon):
        return False
    normal = getattr(layer, "normalField", None)
    return normal is None or arrayInvariantAlong(axis, np.asarray(normal))


def epsilonInvariantAlong(axis: str, epsilon: TensorLike) -> bool:
    if isinstance(epsilon, Mapping):
        return all(epsilonInvariantAlong(axis, value) for value in epsilon.values())
    if hasattr(epsilon, "invariantAxes"):
        return axis in epsilon.invariantAxes()
    if hasattr(epsilon, "convolutionMatrix"):
        return False
    if np.isscalar(epsilon):
        return True
    array = np.asarray(epsilon)
    if array.ndim == 0 or (array.ndim == 2 and array.shape == (3, 3)):
        return True
    if array.ndim in (2, 4):
        return arrayInvariantAlong(axis, array)
    return False


def arrayInvariantAlong(axis: str, array: ComplexArray) -> bool:
    if array.ndim < 2:
        return True
    if axis == "y":
        reference = array[:1, ...]
        return bool(np.allclose(array, reference, rtol=1e-10, atol=1e-12))
    if axis == "x":
        reference = array[:, :1, ...]
        return bool(np.allclose(array, reference, rtol=1e-10, atol=1e-12))
    raise ValueError("axis must be 'x' or 'y'")


def compiledLayerInvariantAlong(axis: str, layer: CompiledLayer, harmonics: Harmonics) -> bool:
    if layer.orders != harmonics.orders or layer.truncation != harmonics.truncation:
        return False
    if layer.tensorData.constantTensor is not None:
        return True
    if layer.mu is not None and not isIdentityTensor(layer.mu):
        return False
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


def layerModes(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    backend: ArrayBackend,
) -> tuple[object, object]:
    modes, timing = layerModesWithTiming(
        layer,
        harmonics,
        backend,
        layerIndex=0,
        collectTiming=False,
    )
    return modes


def layerModesBatch(
    layer: Layer | CompiledLayer | BatchedHomogeneousLayer,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
) -> tuple[object, object]:
    if isinstance(layer, BatchedHomogeneousLayer):
        tensors = np.asarray(layer.tensors, dtype=complex)
        mus = None if layer.mus is None else np.asarray(layer.mus, dtype=complex)
        if mus is None and isBatchedScalarTensor(tensors):
            return homogeneousScalarLayerModesBatch(tensors[:, 0, 0], harmonics, backend)
        return homogeneousTensorLayerModesBatch(tensors, harmonics, backend, mu=mus)

    validateHomogeneousMuSupport(layer)
    if not isinstance(layer, CompiledLayer) and getattr(layer, "normalField", None) is None:
        tensor, muTensor = constantLayerTensors(layer)
        if tensor is not None:
            if isIdentityTensor(muTensor) and isScalarTensor(tensor):
                return homogeneousScalarLayerModesBatch(tensor[0, 0], harmonics, backend)
            return homogeneousTensorLayerModesBatch(tensor, harmonics, backend, mu=muTensor)

    factorized = layerTensorData(layer, batchedReferenceHarmonics(harmonics))
    if factorized.constantTensor is not None:
        muTensor = constantCompiledMuTensor(layer)
        if isIdentityTensor(muTensor) and isScalarTensor(factorized.constantTensor):
            return homogeneousScalarLayerModesBatch(factorized.constantTensor[0, 0], harmonics, backend)
        return homogeneousTensorLayerModesBatch(factorized.constantTensor, harmonics, backend, mu=muTensor)
    if hasNoLongitudinalCoupling(factorized):
        return transverseBlockLayerModesBatch(factorized, harmonics, backend)
    system = liFactorizedSystemMatrixBatchBackend(factorized, harmonics, backend)
    qValues, vectors = backend.eig(system)
    vectors = normalizeModes(vectors, backend)
    return splitForwardBackward(qValues, vectors, 2 * harmonics.count, backend)


def batchedReferenceHarmonics(harmonics: BatchedHarmonics) -> Harmonics:
    return Harmonics(
        mx=harmonics.mx,
        my=harmonics.my,
        kx=np.zeros(harmonics.count, dtype=complex),
        ky=np.zeros(harmonics.count, dtype=complex),
        orders=harmonics.orders,
        truncation=harmonics.truncation,
    )


def layerModeCacheKey(layer: Layer | CompiledLayer) -> tuple[object, ...] | None:
    if isinstance(layer, BatchedHomogeneousLayer):
        return None
    if isinstance(layer, CompiledLayer):
        return ("compiled", id(layer.tensorData), layer.orders, layer.truncation, complexArrayCacheKey(layer.mu) if layer.mu is not None else None)
    if getattr(layer, "normalField", None) is None:
        tensor, muTensor = constantLayerTensors(layer)
        if tensor is not None:
            return (
                "homogeneous",
                complexArrayCacheKey(tensor),
                complexArrayCacheKey(muTensor),
                getattr(layer, "factorization", "auto"),
            )
    return None


def validateHomogeneousMuSupport(layer: object) -> None:
    validateNoMagnetoelectric(layer)
    mu = getattr(layer, "mu", None)
    if mu is None:
        return
    muTensor = constantTensor(mu)
    if muTensor is None:
        raise NotImplementedError(
            "sampled or analytic mu tensors require the full Li 2003 electric-magnetic "
            "Fourier factorization; use a homogeneous constant mu tensor for now"
        )
    if isinstance(layer, CompiledLayer):
        epsilonTensor = layer.tensorData.constantTensor
    else:
        epsilonTensor = constantTensor(getattr(layer, "epsilon"))
    if epsilonTensor is None and not isIdentityTensor(muTensor):
        raise NotImplementedError(
            "patterned non-identity mu tensors require the full Li 2003 electric-magnetic "
            "Fourier factorization; only homogeneous epsilon/mu layers are implemented"
        )


def complexArrayCacheKey(value: ComplexArray) -> tuple[tuple[int, ...], tuple[complex, ...]]:
    array = np.asarray(value, dtype=complex)
    return tuple(int(size) for size in array.shape), tuple(complex(item) for item in array.reshape(-1))


def layerModesWithTiming(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    backend: ArrayBackend,
    *,
    layerIndex: int,
    collectTiming: bool,
) -> tuple[tuple[object, object], LayerEigTiming | None]:
    totalStart = startTimedOperation(backend) if collectTiming else None
    validateHomogeneousMuSupport(layer)
    if not isinstance(layer, CompiledLayer) and getattr(layer, "normalField", None) is None:
        tensor, muTensor = constantLayerTensors(layer)
        if tensor is not None:
            if isIdentityTensor(muTensor) and isScalarTensor(tensor):
                qValues, vectors, matrixShape, eigTime = homogeneousScalarLayerModes(
                    tensor[0, 0],
                    harmonics,
                    backend,
                )
            else:
                qValues, vectors, matrixShape, eigTime = homogeneousTensorLayerModesMeasured(
                    tensor,
                    harmonics,
                    backend,
                    mu=muTensor,
                    collectTiming=collectTiming,
                )
            modes = (qValues, vectors)
            return modes, layerEigTiming(
                layerIndex,
                layer,
                "homogeneous-4x4",
                matrixShape,
                eigTime,
                collectTiming,
                totalStart=totalStart,
                qValues=qValues,
                backend=backend,
            )

    factorizationStart = time.perf_counter() if collectTiming else None
    factorized = layerTensorData(layer, harmonics)
    factorizationTime = time.perf_counter() - factorizationStart if factorizationStart is not None else 0.0
    if factorized.constantTensor is not None:
        muTensor = constantCompiledMuTensor(layer)
        if isIdentityTensor(muTensor) and isScalarTensor(factorized.constantTensor):
            qValues, vectors, matrixShape, eigTime = homogeneousScalarLayerModes(
                factorized.constantTensor[0, 0],
                harmonics,
                backend,
            )
        else:
            qValues, vectors, matrixShape, eigTime = homogeneousTensorLayerModesMeasured(
                factorized.constantTensor,
                harmonics,
                backend,
                mu=muTensor,
                collectTiming=collectTiming,
            )
        modes = (qValues, vectors)
        return modes, layerEigTiming(
            layerIndex,
            layer,
            "homogeneous-4x4",
            matrixShape,
            eigTime,
            collectTiming,
            factorizationTime=factorizationTime,
            totalStart=totalStart,
            qValues=qValues,
            backend=backend,
        )
    if hasNoLongitudinalCoupling(factorized):
        timing: dict[str, object] | None = {} if collectTiming else None
        modes = transverseBlockLayerModes(factorized, harmonics, backend, timing=timing)
        if timing is not None:
            timing["factorizationTimeSeconds"] = factorizationTime
        return modes, layerEigTimingFromDict(layerIndex, layer, "transverse-2N", timing, totalStart, backend, modes[0])
    matrixStart = startTimedOperation(backend) if collectTiming else None
    system = liFactorizedSystemMatrixBackend(factorized, harmonics, backend)
    matrixBuildTime = finishTimedOperation(backend, matrixStart) if matrixStart is not None else 0.0
    start = startTimedOperation(backend) if collectTiming else None
    qValues, vectors = backend.eig(system)
    eigTime = finishTimedOperation(backend, start) if start is not None else 0.0
    vectors = normalizeModes(vectors, backend)
    modes = splitForwardBackward(qValues, vectors, 2 * harmonics.count, backend)
    return modes, layerEigTiming(
        layerIndex,
        layer,
        "full-4N",
        tuple(system.shape),
        eigTime,
        collectTiming,
        factorizationTime=factorizationTime,
        matrixBuildTime=matrixBuildTime,
        totalStart=totalStart,
        qValues=modes[0],
        backend=backend,
    )


def profileTimingsEnabled(profile: bool) -> bool:
    if profile:
        return True
    value = os.environ.get("RCWA3D_PROFILE_EIG", "").strip().lower()
    return value in ("1", "true", "yes", "on")


def layerEigTiming(
    layerIndex: int,
    layer: Layer | CompiledLayer,
    kind: str,
    matrixShape: tuple[int, ...],
    eigTime: float,
    collectTiming: bool,
    *,
    factorizationTime: float = 0.0,
    matrixBuildTime: float = 0.0,
    totalStart: float | None = None,
    qValues: object | None = None,
    backend: ArrayBackend | None = None,
) -> LayerEigTiming | None:
    if not collectTiming:
        return None
    totalTime = finishTimedOperation(backend, totalStart) if backend is not None and totalStart is not None else 0.0
    minAbsQ, safeQThreshold, nearZeroCount = qStabilityStats(qValues, backend)
    return LayerEigTiming(
        layerIndex=layerIndex,
        name=getattr(layer, "name", ""),
        kind=kind,
        matrixShape=tuple(int(value) for value in matrixShape),
        eigTimeSeconds=float(eigTime),
        factorizationTimeSeconds=float(factorizationTime),
        matrixBuildTimeSeconds=float(matrixBuildTime),
        totalTimeSeconds=float(totalTime),
        minAbsQ=minAbsQ,
        safeQThreshold=safeQThreshold,
        nearZeroModeCount=nearZeroCount,
    )


def layerEigTimingFromDict(
    layerIndex: int,
    layer: Layer | CompiledLayer,
    kind: str,
    timing: dict[str, object] | None,
    totalStart: float | None = None,
    backend: ArrayBackend | None = None,
    qValues: object | None = None,
) -> LayerEigTiming | None:
    if timing is None:
        return None
    return layerEigTiming(
        layerIndex,
        layer,
        kind,
        tuple(timing.get("matrixShape", ())),
        float(timing.get("eigTimeSeconds", 0.0)),
        True,
        factorizationTime=float(timing.get("factorizationTimeSeconds", 0.0)),
        matrixBuildTime=float(timing.get("matrixBuildTimeSeconds", 0.0)),
        totalStart=totalStart,
        qValues=qValues,
        backend=backend,
    )


def qStabilityStats(qValues: object | None, backend: ArrayBackend | None) -> tuple[float | None, float | None, int]:
    if qValues is None or backend is None:
        return None, None, 0
    torch = backend.xp.torch
    if qValues.numel() == 0:
        return float("inf"), 0.0, 0
    absQ = torch.abs(qValues)
    minAbs = float(torch.amin(absQ).detach().cpu().item())
    maxAbs = float(torch.amax(absQ).detach().cpu().item())
    threshold = 1e-12 * max(1.0, maxAbs)
    nearZero = int(torch.count_nonzero(absQ < threshold).detach().cpu().item())
    return minAbs, threshold, nearZero


def stabilityWarnings(
    layerTimings: Sequence[LayerEigTiming],
    interfaceConditionNumbers: Sequence[float],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for timing in layerTimings:
        if timing.nearZeroModeCount > 0:
            minAbs = timing.minAbsQ if timing.minAbsQ is not None else float("nan")
            threshold = timing.safeQThreshold if timing.safeQThreshold is not None else float("nan")
            warnings.append(
                f"layer {timing.layerIndex} has {timing.nearZeroModeCount} near-grazing anisotropic modes "
                f"(min |q|={minAbs:.3e}, threshold={threshold:.3e}); "
                "check harmonic order and wavelength sampling near Wood anomalies"
            )
    if interfaceConditionNumbers:
        maximum = max(interfaceConditionNumbers)
        if maximum > 1e10:
            warnings.append(
                f"max anisotropic interface condition number is {maximum:.3e}; compare nearby wavelengths "
                "or reduce material discontinuity/factorization error"
            )
    return tuple(warnings)


def startTimedOperation(backend: ArrayBackend) -> float:
    backend.synchronize()
    return time.perf_counter()


def finishTimedOperation(backend: ArrayBackend, start: float) -> float:
    backend.synchronize()
    return time.perf_counter() - start


def hasNoLongitudinalCoupling(data: TensorConvolutionData) -> bool:
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


def transverseBlockLayerModes(
    data: TensorConvolutionData,
    harmonics: Harmonics,
    backend: ArrayBackend,
    timing: dict[str, object] | None = None,
) -> tuple[object, object]:
    return transverseBlockLayerModesBackend(data, harmonics, backend, timing=timing)


def transverseBlockLayerModesBackend(
    data: TensorConvolutionData,
    harmonics: Harmonics,
    backend: ArrayBackend,
    timing: dict[str, object] | None = None,
) -> tuple[object, object]:
    xp = backend.xp
    torch = xp.torch
    n = harmonics.count
    diagonal = xp.arange(n)
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)
    cached = backendTensorConvolutionData(data, backend)
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
    start = startTimedOperation(backend) if timing is not None else None
    qSquared, electricModes = backend.eig(eigenMatrix)
    if start is not None:
        timing["eigTimeSeconds"] = finishTimedOperation(backend, start)
    qValues = forwardKzTorch(qSquared, torch)
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
    return qAll, normalizeModes(vectors, backend)


def transverseBlockLayerModesBatch(
    data: TensorConvolutionData,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
) -> tuple[object, object]:
    xp = backend.xp
    torch = xp.torch
    batchSize = harmonics.batchSize
    n = harmonics.count
    diagonal = torch.arange(n, dtype=torch.long, device=backend.device)
    kx = harmonics.kx
    ky = harmonics.ky
    cached = backendTensorConvolutionData(data, backend)
    c = cached.components
    eta = cached.etaZz
    cxx, cxy = c[0][0], c[0][1]
    cyx, cyy = c[1][0], c[1][1]

    identity = xp.eye(n, dtype=complex)
    p11 = kx[:, :, None] * eta[None, :, :] * ky[:, None, :]
    p12 = identity[None, :, :] - kx[:, :, None] * eta[None, :, :] * kx[:, None, :]
    p21 = ky[:, :, None] * eta[None, :, :] * ky[:, None, :] - identity[None, :, :]
    p22 = -ky[:, :, None] * eta[None, :, :] * kx[:, None, :]

    q11 = (-cyx).expand(batchSize, -1, -1).clone()
    q11[:, diagonal, diagonal] = q11[:, diagonal, diagonal] - kx * ky
    q12 = (-cyy).expand(batchSize, -1, -1).clone()
    q12[:, diagonal, diagonal] = q12[:, diagonal, diagonal] + kx * kx
    q21 = cxx.expand(batchSize, -1, -1).clone()
    q21[:, diagonal, diagonal] = q21[:, diagonal, diagonal] - ky * ky
    q22 = cxy.expand(batchSize, -1, -1).clone()
    q22[:, diagonal, diagonal] = q22[:, diagonal, diagonal] + ky * kx

    pMatrix = xp.empty((batchSize, 2 * n, 2 * n), dtype=complex)
    pMatrix[:, 0:n, 0:n] = p11
    pMatrix[:, 0:n, n : 2 * n] = p12
    pMatrix[:, n : 2 * n, 0:n] = p21
    pMatrix[:, n : 2 * n, n : 2 * n] = p22
    qMatrix = xp.empty((batchSize, 2 * n, 2 * n), dtype=complex)
    qMatrix[:, 0:n, 0:n] = q11
    qMatrix[:, 0:n, n : 2 * n] = q12
    qMatrix[:, n : 2 * n, 0:n] = q21
    qMatrix[:, n : 2 * n, n : 2 * n] = q22

    qSquared, electricModes = backend.eig(pMatrix @ qMatrix)
    qValues = forwardKzTorch(qSquared, torch)
    safeQ = torch.where(
        torch.abs(qValues) < 1e-13,
        torch.as_tensor(1e-13, dtype=qValues.dtype, device=qValues.device),
        qValues,
    )
    magneticModes = qMatrix @ (electricModes * (1.0 / safeQ)[:, None, :])
    vectors = xp.concatenate(
        [
            xp.concatenate([electricModes, electricModes], axis=2),
            xp.concatenate([magneticModes, -magneticModes], axis=2),
        ],
        axis=1,
    )
    qAll = xp.concatenate([qValues, -qValues], axis=1)
    return qAll, normalizeModes(vectors, backend)


def forwardKzTorch(values: object, torch: object) -> object:
    roots = torch.sqrt(values)
    flip = (torch.imag(roots) < -1e-14) | (
        (torch.abs(torch.imag(roots)) <= 1e-14) & (torch.real(roots) < 0)
    )
    return torch.where(flip, -roots, roots)


def liFactorizedSystemMatrixBackend(
    data: TensorConvolutionData,
    harmonics: Harmonics,
    backend: ArrayBackend,
) -> object:
    xp = backend.xp
    n = harmonics.count
    diagonal = xp.arange(n)
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)
    cached = backendTensorConvolutionData(data, backend)
    c = cached.components
    eta = cached.etaZz

    cxx, cxy, cxz = c[0]
    cyx, cyy, cyz = c[1]
    czx, czy = c[2][0], c[2][1]

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


def liFactorizedSystemMatrixBatchBackend(
    data: TensorConvolutionData,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
) -> object:
    xp = backend.xp
    torch = xp.torch
    batchSize = harmonics.batchSize
    n = harmonics.count
    diagonal = torch.arange(n, dtype=torch.long, device=backend.device)
    kx = harmonics.kx
    ky = harmonics.ky
    cached = backendTensorConvolutionData(data, backend)
    c = cached.components
    eta = cached.etaZz

    cxx, cxy, cxz = c[0]
    cyx, cyy, cyz = c[1]
    czx, czy = c[2][0], c[2][1]

    cxzEta = cxz @ eta
    cyzEta = cyz @ eta
    etaCzx = eta @ czx
    etaCzy = eta @ czy

    dxEx = cxx - cxzEta @ czx
    dxEy = cxy - cxzEta @ czy
    dxHx = cxzEta[None, :, :] * ky[:, None, :]
    dxHy = -cxzEta[None, :, :] * kx[:, None, :]

    dyEx = cyx - cyzEta @ czx
    dyEy = cyy - cyzEta @ czy
    dyHx = cyzEta[None, :, :] * ky[:, None, :]
    dyHy = -cyzEta[None, :, :] * kx[:, None, :]

    identity = xp.eye(n, dtype=complex)
    a11 = -kx[:, :, None] * etaCzx[None, :, :]
    a12 = -kx[:, :, None] * etaCzy[None, :, :]
    a13 = kx[:, :, None] * eta[None, :, :] * ky[:, None, :]
    a14 = identity[None, :, :] - kx[:, :, None] * eta[None, :, :] * kx[:, None, :]

    a21 = -ky[:, :, None] * etaCzx[None, :, :]
    a22 = -ky[:, :, None] * etaCzy[None, :, :]
    a23 = ky[:, :, None] * eta[None, :, :] * ky[:, None, :] - identity[None, :, :]
    a24 = -ky[:, :, None] * eta[None, :, :] * kx[:, None, :]

    a31 = (-dyEx).expand(batchSize, -1, -1).clone()
    a31[:, diagonal, diagonal] = a31[:, diagonal, diagonal] - kx * ky
    a32 = (-dyEy).expand(batchSize, -1, -1).clone()
    a32[:, diagonal, diagonal] = a32[:, diagonal, diagonal] + kx * kx
    a33 = -dyHx
    a34 = -dyHy

    a41 = dxEx.expand(batchSize, -1, -1).clone()
    a41[:, diagonal, diagonal] = a41[:, diagonal, diagonal] - ky * ky
    a42 = dxEy.expand(batchSize, -1, -1).clone()
    a42[:, diagonal, diagonal] = a42[:, diagonal, diagonal] + ky * kx
    a43 = dxHx
    a44 = dxHy

    system = xp.empty((batchSize, 4 * n, 4 * n), dtype=complex)
    system[:, 0:n, 0:n] = a11
    system[:, 0:n, n : 2 * n] = a12
    system[:, 0:n, 2 * n : 3 * n] = a13
    system[:, 0:n, 3 * n : 4 * n] = a14
    system[:, n : 2 * n, 0:n] = a21
    system[:, n : 2 * n, n : 2 * n] = a22
    system[:, n : 2 * n, 2 * n : 3 * n] = a23
    system[:, n : 2 * n, 3 * n : 4 * n] = a24
    system[:, 2 * n : 3 * n, 0:n] = a31
    system[:, 2 * n : 3 * n, n : 2 * n] = a32
    system[:, 2 * n : 3 * n, 2 * n : 3 * n] = a33
    system[:, 2 * n : 3 * n, 3 * n : 4 * n] = a34
    system[:, 3 * n : 4 * n, 0:n] = a41
    system[:, 3 * n : 4 * n, n : 2 * n] = a42
    system[:, 3 * n : 4 * n, 2 * n : 3 * n] = a43
    system[:, 3 * n : 4 * n, 3 * n : 4 * n] = a44
    return system


def backendTensorConvolutionData(
    data: TensorConvolutionData,
    backend: ArrayBackend,
) -> BackendTensorConvolutionData:
    key = (id(data), backend.name, str(backend.device), backendPrecisionLabel(backend))
    entry = BACKEND_TENSOR_DATA_CACHE.get(key)
    if entry is not None:
        dataRef, cached = entry
        if dataRef() is data:
            return cached
        BACKEND_TENSOR_DATA_CACHE.pop(key, None)

    cached = BackendTensorConvolutionData(
        components=tuple(tuple(backend.asarray(component) for component in row) for row in data.components),
        etaZz=backend.asarray(data.etaZz),
    )

    def forget(ref: weakref.ReferenceType[TensorConvolutionData], cacheKey: BackendTensorCacheKey = key) -> None:
        BACKEND_TENSOR_DATA_CACHE.pop(cacheKey, None)

    BACKEND_TENSOR_DATA_CACHE[key] = (weakref.ref(data, forget), cached)
    return cached


def isHomogeneousLayer(layer: Layer | CompiledLayer) -> bool:
    if isinstance(layer, BatchedHomogeneousLayer):
        return True
    if isinstance(layer, CompiledLayer):
        return layer.tensorData.constantTensor is not None and (layer.mu is None or constantTensor(layer.mu) is not None)
    if getattr(layer, "normalField", None) is not None:
        return False
    epsilonTensor, muTensor = constantLayerTensors(layer)
    if epsilonTensor is None:
        return False
    return muTensor is not None


def constantLayerTensors(layer: object) -> tuple[ComplexArray | None, ComplexArray | None]:
    validateNoMagnetoelectric(layer)
    epsilonTensor = constantTensor(getattr(layer, "epsilon"))
    if epsilonTensor is None:
        return None, None
    mu = getattr(layer, "mu", None)
    if mu is None:
        return epsilonTensor, identityTensor()
    muTensor = constantTensor(mu)
    if muTensor is None:
        raise NotImplementedError(
            "sampled or analytic mu tensors require the full Li 2003 electric-magnetic "
            "Fourier factorization; use a homogeneous constant mu tensor for now"
        )
    return epsilonTensor, muTensor


def constantCompiledMuTensor(layer: object) -> ComplexArray:
    mu = getattr(layer, "mu", None)
    if mu is None:
        return identityTensor()
    return np.asarray(mu, dtype=complex)


def identityTensor() -> ComplexArray:
    return np.eye(3, dtype=complex)


def isIdentityTensor(tensor: ComplexArray | None) -> bool:
    if tensor is None:
        return False
    array = np.asarray(tensor, dtype=complex)
    if array.shape != (3, 3):
        return False
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    return bool(np.max(np.abs(array - np.eye(3, dtype=complex))) <= 1e-14 * scale)


def validateNoMagnetoelectric(layer: object) -> None:
    if getattr(layer, "chi", None) is not None or getattr(layer, "xi", None) is not None:
        raise NotImplementedError(
            "magnetoelectric chi/xi coupling requires the coupled bi-anisotropic "
            "Fourier formulation; this solver currently implements chi=xi=0"
        )


def isScalarTensor(tensor: ComplexArray) -> bool:
    array = np.asarray(tensor, dtype=complex)
    diagonal = np.diag(array)
    offDiagonal = array - np.diag(diagonal)
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    return bool(np.max(np.abs(offDiagonal)) <= 1e-14 * scale and np.max(np.abs(diagonal - diagonal[0])) <= 1e-14 * scale)


def isBatchedScalarTensor(tensors: ComplexArray) -> bool:
    array = np.asarray(tensors, dtype=complex)
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        return False
    diagonal = np.stack([array[:, 0, 0], array[:, 1, 1], array[:, 2, 2]], axis=1)
    offDiagonal = array.copy()
    offDiagonal[:, 0, 0] = 0.0
    offDiagonal[:, 1, 1] = 0.0
    offDiagonal[:, 2, 2] = 0.0
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    return bool(
        np.max(np.abs(offDiagonal)) <= 1e-14 * scale
        and np.max(np.abs(diagonal - diagonal[:, :1])) <= 1e-14 * scale
    )


def homogeneousScalarLayerModes(
    epsilon: complex,
    harmonics: Harmonics,
    backend: ArrayBackend,
) -> tuple[object, object, tuple[int, ...], float]:
    nOrders = harmonics.count
    forward = backend.asarray(homogeneousBasis(harmonics, epsilon, direction=1))
    backward = backend.asarray(homogeneousBasis(harmonics, epsilon, direction=-1))
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)
    qForward = forwardKzTorch(complex(epsilon) - kx * kx - ky * ky, backend.xp.torch)
    qPairs = backend.xp.empty(2 * nOrders, dtype=complex)
    qPairs[0::2] = qForward
    qPairs[1::2] = qForward
    return backend.xp.concatenate([qPairs, -qPairs]), backend.xp.concatenate([forward, backward], axis=1), (nOrders, 2), 0.0


def homogeneousScalarLayerModesBatch(
    epsilon: complex | ComplexArray,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
) -> tuple[object, object]:
    nOrders = harmonics.count
    forward = homogeneousBasisBatch(harmonics, epsilon, direction=1, backend=backend)
    backward = homogeneousBasisBatch(harmonics, epsilon, direction=-1, backend=backend)
    epsBatch = batchedScalarEpsilon(epsilon, harmonics, backend)
    qForward = forwardKzTorch(epsBatch - harmonics.kx * harmonics.kx - harmonics.ky * harmonics.ky, backend.xp.torch)
    qPairs = backend.xp.empty((harmonics.batchSize, 2 * nOrders), dtype=complex)
    qPairs[:, 0::2] = qForward
    qPairs[:, 1::2] = qForward
    return backend.xp.concatenate([qPairs, -qPairs], axis=1), backend.xp.concatenate([forward, backward], axis=2)


def batchedScalarEpsilon(
    epsilon: complex | ComplexArray,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    if np.isscalar(epsilon):
        return torch.as_tensor(complex(epsilon), dtype=backend.complexDtype, device=backend.device)
    values = torch.as_tensor(np.asarray(epsilon, dtype=complex), dtype=backend.complexDtype, device=backend.device)
    if values.ndim == 0:
        return values
    if values.shape != (harmonics.batchSize,):
        raise ValueError("batched scalar epsilon must have shape (batch,)")
    return values[:, None]


def homogeneousTensorLayerModesMeasured(
    tensor: ComplexArray,
    harmonics: Harmonics,
    backend: ArrayBackend,
    *,
    mu: ComplexArray | None = None,
    collectTiming: bool,
) -> tuple[object, object, tuple[int, ...], float]:
    return homogeneousTensorLayerModesBackend(
        tensor,
        harmonics,
        backend,
        mu=mu,
        collectTiming=collectTiming,
    )


def homogeneousTensorLayerModesBackend(
    tensor: ComplexArray,
    harmonics: Harmonics,
    backend: ArrayBackend,
    *,
    mu: ComplexArray | None = None,
    collectTiming: bool,
) -> tuple[object, object, tuple[int, ...], float]:
    nOrders = harmonics.count
    systems = homogeneousOrderSystemMatricesBackend(tensor, harmonics, backend, mu=mu)
    start = startTimedOperation(backend) if collectTiming else None
    orderQ, orderVectors = backend.eig(systems)
    eigTime = finishTimedOperation(backend, start) if start is not None else 0.0

    xp = backend.xp
    qValues = orderQ.reshape(-1)
    vectors = xp.zeros((4 * nOrders, 4 * nOrders), dtype=complex)
    baseColumns = 4 * xp.arange(nOrders)
    orderIndices = xp.arange(nOrders)
    for localIndex in range(4):
        columns = baseColumns + localIndex
        vectors[orderIndices, columns] = orderVectors[:, 0, localIndex]
        vectors[nOrders + orderIndices, columns] = orderVectors[:, 1, localIndex]
        vectors[2 * nOrders + orderIndices, columns] = orderVectors[:, 2, localIndex]
        vectors[3 * nOrders + orderIndices, columns] = orderVectors[:, 3, localIndex]
    vectors = normalizeModes(vectors, backend)
    qValues, vectors = splitForwardBackward(qValues, vectors, 2 * nOrders, backend)
    return qValues, vectors, tuple(int(value) for value in systems.shape), eigTime


def homogeneousTensorLayerModesBatch(
    tensor: ComplexArray,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
    *,
    mu: ComplexArray | None = None,
) -> tuple[object, object]:
    nOrders = harmonics.count
    systems = homogeneousOrderSystemMatricesBatchBackend(tensor, harmonics, backend, mu=mu)
    orderQ, orderVectors = backend.eig(systems)

    xp = backend.xp
    torch = xp.torch
    qValues = orderQ.reshape(harmonics.batchSize, -1)
    vectors = xp.zeros((harmonics.batchSize, 4 * nOrders, 4 * nOrders), dtype=complex)
    baseColumns = 4 * torch.arange(nOrders, dtype=torch.long, device=backend.device)
    orderIndices = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    batchIndices = torch.arange(harmonics.batchSize, dtype=torch.long, device=backend.device)[:, None]
    for localIndex in range(4):
        columns = baseColumns + localIndex
        vectors[batchIndices, orderIndices[None, :], columns[None, :]] = orderVectors[:, :, 0, localIndex]
        vectors[batchIndices, nOrders + orderIndices[None, :], columns[None, :]] = orderVectors[:, :, 1, localIndex]
        vectors[batchIndices, 2 * nOrders + orderIndices[None, :], columns[None, :]] = orderVectors[:, :, 2, localIndex]
        vectors[batchIndices, 3 * nOrders + orderIndices[None, :], columns[None, :]] = orderVectors[:, :, 3, localIndex]
    vectors = normalizeModes(vectors, backend)
    return splitForwardBackward(qValues, vectors, 2 * nOrders, backend)


def homogeneousOrderSystemMatricesBackend(
    tensor: ComplexArray,
    harmonics: Harmonics,
    backend: ArrayBackend,
    *,
    mu: ComplexArray | None = None,
) -> object:
    xp = backend.xp
    torch = xp.torch
    tensorGpu = backend.asarray(tensor)
    muGpu = backend.asarray(identityTensor() if mu is None else mu)
    exx, exy, exz = tensorGpu[0]
    eyx, eyy, eyz = tensorGpu[1]
    ezx, ezy, ezz = tensorGpu[2]
    mxx, mxy, mxz = muGpu[0]
    myx, myy, myz = muGpu[1]
    mzx, mzy, mzz = muGpu[2]
    if bool(torch.abs(ezz) < 1e-14):
        raise ValueError("epsilon_zz is near zero in a homogeneous anisotropic layer")
    if bool(torch.abs(mzz) < 1e-14):
        raise ValueError("mu_zz is near zero in a homogeneous magnetic anisotropic layer")
    eta = 1.0 / ezz
    nu = 1.0 / mzz
    kx = backend.asarray(harmonics.kx)
    ky = backend.asarray(harmonics.ky)

    dxEx = exx - exz * eta * ezx
    dxEy = exy - exz * eta * ezy
    dxHx = exz * eta * ky
    dxHy = -exz * eta * kx

    dyEx = eyx - eyz * eta * ezx
    dyEy = eyy - eyz * eta * ezy
    dyHx = eyz * eta * ky
    dyHy = -eyz * eta * kx

    bxEx = -mxz * nu * ky
    bxEy = mxz * nu * kx
    bxHx = mxx - mxz * nu * mzx
    bxHy = mxy - mxz * nu * mzy

    byEx = -myz * nu * ky
    byEy = myz * nu * kx
    byHx = myx - myz * nu * mzx
    byHy = myy - myz * nu * mzy

    systems = xp.empty((harmonics.count, 4, 4), dtype=complex)
    systems[:, 0, 0] = -kx * eta * ezx + byEx
    systems[:, 0, 1] = -kx * eta * ezy + byEy
    systems[:, 0, 2] = kx * eta * ky + byHx
    systems[:, 0, 3] = -kx * eta * kx + byHy
    systems[:, 1, 0] = -ky * eta * ezx - bxEx
    systems[:, 1, 1] = -ky * eta * ezy - bxEy
    systems[:, 1, 2] = ky * eta * ky - bxHx
    systems[:, 1, 3] = -ky * eta * kx - bxHy
    systems[:, 2, 0] = -kx * nu * ky - dyEx
    systems[:, 2, 1] = kx * nu * kx - dyEy
    systems[:, 2, 2] = -kx * nu * mzx - dyHx
    systems[:, 2, 3] = -kx * nu * mzy - dyHy
    systems[:, 3, 0] = dxEx - ky * nu * ky
    systems[:, 3, 1] = dxEy + ky * nu * kx
    systems[:, 3, 2] = dxHx - ky * nu * mzx
    systems[:, 3, 3] = dxHy - ky * nu * mzy
    return systems


def homogeneousOrderSystemMatricesBatchBackend(
    tensor: ComplexArray,
    harmonics: BatchedHarmonics,
    backend: ArrayBackend,
    *,
    mu: ComplexArray | None = None,
) -> object:
    xp = backend.xp
    torch = xp.torch
    tensorGpu = backend.asarray(tensor)
    muGpu = backend.asarray(identityTensor() if mu is None else mu)
    if tensorGpu.ndim == 2:
        exx, exy, exz = tensorGpu[0]
        eyx, eyy, eyz = tensorGpu[1]
        ezx, ezy, ezz = tensorGpu[2]
    elif tensorGpu.ndim == 3 and tensorGpu.shape[0] == harmonics.batchSize and tensorGpu.shape[1:] == (3, 3):
        exx, exy, exz = (tensorGpu[:, 0, index][:, None] for index in range(3))
        eyx, eyy, eyz = (tensorGpu[:, 1, index][:, None] for index in range(3))
        ezx, ezy, ezz = (tensorGpu[:, 2, index][:, None] for index in range(3))
    else:
        raise ValueError("homogeneous tensor batch must be a (3, 3) tensor or a (batch, 3, 3) tensor array")
    if muGpu.ndim == 2:
        mxx, mxy, mxz = muGpu[0]
        myx, myy, myz = muGpu[1]
        mzx, mzy, mzz = muGpu[2]
    elif muGpu.ndim == 3 and muGpu.shape[0] == harmonics.batchSize and muGpu.shape[1:] == (3, 3):
        mxx, mxy, mxz = (muGpu[:, 0, index][:, None] for index in range(3))
        myx, myy, myz = (muGpu[:, 1, index][:, None] for index in range(3))
        mzx, mzy, mzz = (muGpu[:, 2, index][:, None] for index in range(3))
    else:
        raise ValueError("homogeneous mu batch must be a (3, 3) tensor or a (batch, 3, 3) tensor array")
    if bool(torch.any(torch.abs(ezz) < 1e-14)):
        raise ValueError("epsilon_zz is near zero in a homogeneous anisotropic layer")
    if bool(torch.any(torch.abs(mzz) < 1e-14)):
        raise ValueError("mu_zz is near zero in a homogeneous magnetic anisotropic layer")
    eta = 1.0 / ezz
    nu = 1.0 / mzz
    kx = harmonics.kx
    ky = harmonics.ky

    dxEx = exx - exz * eta * ezx
    dxEy = exy - exz * eta * ezy
    dxHx = exz * eta * ky
    dxHy = -exz * eta * kx

    dyEx = eyx - eyz * eta * ezx
    dyEy = eyy - eyz * eta * ezy
    dyHx = eyz * eta * ky
    dyHy = -eyz * eta * kx

    bxEx = -mxz * nu * ky
    bxEy = mxz * nu * kx
    bxHx = mxx - mxz * nu * mzx
    bxHy = mxy - mxz * nu * mzy

    byEx = -myz * nu * ky
    byEy = myz * nu * kx
    byHx = myx - myz * nu * mzx
    byHy = myy - myz * nu * mzy

    systems = xp.empty((harmonics.batchSize, harmonics.count, 4, 4), dtype=complex)
    systems[:, :, 0, 0] = -kx * eta * ezx + byEx
    systems[:, :, 0, 1] = -kx * eta * ezy + byEy
    systems[:, :, 0, 2] = kx * eta * ky + byHx
    systems[:, :, 0, 3] = -kx * eta * kx + byHy
    systems[:, :, 1, 0] = -ky * eta * ezx - bxEx
    systems[:, :, 1, 1] = -ky * eta * ezy - bxEy
    systems[:, :, 1, 2] = ky * eta * ky - bxHx
    systems[:, :, 1, 3] = -ky * eta * kx - bxHy
    systems[:, :, 2, 0] = -kx * nu * ky - dyEx
    systems[:, :, 2, 1] = kx * nu * kx - dyEy
    systems[:, :, 2, 2] = -kx * nu * mzx - dyHx
    systems[:, :, 2, 3] = -kx * nu * mzy - dyHy
    systems[:, :, 3, 0] = dxEx - ky * nu * ky
    systems[:, :, 3, 1] = dxEy + ky * nu * kx
    systems[:, :, 3, 2] = dxHx - ky * nu * mzx
    systems[:, :, 3, 3] = dxHy - ky * nu * mzy
    return systems


def splitForwardBackward(
    qValues: object,
    vectors: object,
    nForward: int,
    backend: ArrayBackend,
) -> tuple[object, object]:
    if getattr(qValues, "ndim", 1) == 2:
        return splitForwardBackwardBatch(qValues, vectors, nForward, backend)

    fluxes = modeFluxes(vectors, backend)
    indexTensor = splitForwardBackwardIndicesTorch(qValues, fluxes, nForward, backend)
    return qValues[indexTensor], vectors[:, indexTensor]


def splitForwardBackwardBatch(
    qValues: object,
    vectors: object,
    nForward: int,
    backend: ArrayBackend,
) -> tuple[object, object]:
    fluxes = modeFluxes(vectors, backend)
    indexTensor = splitForwardBackwardIndicesBatchTorch(qValues, fluxes, nForward, backend)
    torch = backend.xp.torch
    gatheredVectors = torch.gather(
        vectors,
        2,
        indexTensor[:, None, :].expand(-1, vectors.shape[1], -1),
    )
    return torch.gather(qValues, 1, indexTensor), gatheredVectors


def splitForwardBackwardIndicesTorch(
    qValues: object,
    fluxes: object,
    nForward: int,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    realFluxes = torch.real(fluxes)
    scores = forwardModeScoresTorch(qValues, fluxes, backend)
    order = torchLexsortRows(
        (
            torch.arange(qValues.shape[0], dtype=torch.float64, device=backend.device),
            torch.real(qValues).to(dtype=torch.float64),
            torch.imag(qValues).to(dtype=torch.float64),
            realFluxes.to(dtype=torch.float64),
            scores.to(dtype=torch.float64),
        ),
        backend,
    )
    forwardUnsorted = order[-nForward:]
    selected = torch.zeros(qValues.shape[0], dtype=torch.bool, device=backend.device)
    selected[forwardUnsorted] = True
    allIndices = torch.arange(qValues.shape[0], dtype=torch.long, device=backend.device)
    backwardUnsorted = allIndices[~selected]
    if int(backwardUnsorted.numel()) != nForward:
        raise RuntimeError("anisotropic layer mode split did not produce equal forward/backward spaces")
    return torch.cat(
        [
            sortModeIndicesTorch(forwardUnsorted, qValues, fluxes, forward=True, backend=backend),
            sortModeIndicesTorch(backwardUnsorted, qValues, fluxes, forward=False, backend=backend),
        ]
    )


def splitForwardBackwardIndicesBatchTorch(
    qValues: object,
    fluxes: object,
    nForward: int,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    realFluxes = torch.real(fluxes)
    scores = forwardModeScoresTorch(qValues, fluxes, backend)
    order = torchLexsortRows(
        (
            torch.arange(qValues.shape[1], dtype=torch.float64, device=backend.device)[None, :].expand(qValues.shape[0], -1),
            torch.real(qValues).to(dtype=torch.float64),
            torch.imag(qValues).to(dtype=torch.float64),
            realFluxes.to(dtype=torch.float64),
            scores.to(dtype=torch.float64),
        ),
        backend,
    )
    forwardUnsorted = order[:, -nForward:]
    selected = torch.zeros(qValues.shape, dtype=torch.bool, device=backend.device)
    selected.scatter_(1, forwardUnsorted, True)
    allIndices = torch.arange(qValues.shape[1], dtype=torch.long, device=backend.device)[None, :].expand(qValues.shape[0], -1)
    backwardUnsorted = allIndices[~selected].reshape(qValues.shape[0], -1)
    if int(backwardUnsorted.shape[1]) != nForward:
        raise RuntimeError("anisotropic layer mode split did not produce equal forward/backward spaces")
    return torch.cat(
        [
            sortModeIndicesBatchTorch(forwardUnsorted, qValues, fluxes, forward=True, backend=backend),
            sortModeIndicesBatchTorch(backwardUnsorted, qValues, fluxes, forward=False, backend=backend),
        ],
        dim=1,
    )


def forwardModeScoresTorch(qValues: object, fluxes: object, backend: ArrayBackend) -> object:
    torch = backend.xp.torch
    realFluxes = torch.real(fluxes)
    absFlux = torch.abs(realFluxes)
    absImag = torch.abs(torch.imag(qValues))
    if getattr(qValues, "ndim", 1) == 2:
        fluxScale = torch.clamp(torch.amax(absFlux, dim=1, keepdim=True), min=1.0)
        imagScale = torch.clamp(torch.amax(absImag, dim=1, keepdim=True), min=1.0)
        realScale = torch.clamp(torch.amax(torch.abs(torch.real(qValues)), dim=1, keepdim=True), min=1.0)
        rowMax = torch.amax(torch.abs(torch.where(absFlux <= 1e-9 * fluxScale, torch.imag(qValues) / imagScale, realFluxes / fluxScale)), dim=1, keepdim=True)
    else:
        fluxScale = torch.clamp(torch.amax(absFlux), min=1.0)
        imagScale = torch.clamp(torch.amax(absImag), min=1.0)
        realScale = torch.clamp(torch.amax(torch.abs(torch.real(qValues))), min=1.0)
        rowMax = torch.amax(torch.abs(torch.where(absFlux <= 1e-9 * fluxScale, torch.imag(qValues) / imagScale, realFluxes / fluxScale)))

    scores = realFluxes / fluxScale
    scores = torch.where(absFlux <= 1e-9 * fluxScale, torch.imag(qValues) / imagScale, scores)
    return torch.where(rowMax <= 1e-14, torch.real(qValues) / realScale, scores)


def sortModeIndicesTorch(
    indices: object,
    qValues: object,
    fluxes: object,
    *,
    forward: bool,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    realFluxes = torch.real(fluxes)
    sign = -1.0 if forward else 1.0
    return indices[
        torchLexsortRows(
            (
                indices.to(dtype=torch.float64),
                sign * torch.real(qValues[indices]).to(dtype=torch.float64),
                sign * torch.imag(qValues[indices]).to(dtype=torch.float64),
                sign * realFluxes[indices].to(dtype=torch.float64),
            ),
            backend,
        )
    ]


def sortModeIndicesBatchTorch(
    indices: object,
    qValues: object,
    fluxes: object,
    *,
    forward: bool,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    realFluxes = torch.real(fluxes)
    sign = -1.0 if forward else 1.0
    rows = torch.arange(indices.shape[0], dtype=torch.long, device=backend.device)[:, None]
    order = torchLexsortRows(
        (
            indices.to(dtype=torch.float64),
            sign * torch.real(torch.gather(qValues, 1, indices)).to(dtype=torch.float64),
            sign * torch.imag(torch.gather(qValues, 1, indices)).to(dtype=torch.float64),
            sign * torch.gather(realFluxes, 1, indices).to(dtype=torch.float64),
        ),
        backend,
    )
    return indices[rows, order]


def torchLexsortRows(keys: Sequence[object], backend: ArrayBackend) -> object:
    torch = backend.xp.torch
    order = torch.argsort(keys[0], dim=-1, stable=True)
    for key in keys[1:]:
        order = torch.gather(order, -1, torch.argsort(torch.gather(key, -1, order), dim=-1, stable=True))
    return order


def forwardModeIndices(qValues: ComplexArray, fluxes: ComplexArray, nForward: int) -> ComplexArray:
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


def modeFluxes(vectors: object, backend: ArrayBackend) -> object:
    xp = backend.xp
    if getattr(vectors, "ndim", 2) == 3:
        nOrders = vectors.shape[1] // 4
        ex = vectors[:, :nOrders, :]
        ey = vectors[:, nOrders : 2 * nOrders, :]
        hx = vectors[:, 2 * nOrders : 3 * nOrders, :]
        hy = vectors[:, 3 * nOrders :, :]
        return 0.5 * xp.real(xp.sum(ex * xp.conj(hy) - ey * xp.conj(hx), axis=1))

    nOrders = vectors.shape[0] // 4
    ex = vectors[:nOrders, :]
    ey = vectors[nOrders : 2 * nOrders, :]
    hx = vectors[2 * nOrders : 3 * nOrders, :]
    hy = vectors[3 * nOrders :, :]
    return 0.5 * xp.real(xp.sum(ex * xp.conj(hy) - ey * xp.conj(hx), axis=0))


def sortModes(indices: ComplexArray, qValues: ComplexArray, fluxes: ComplexArray, forward: bool) -> ComplexArray:
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


def normalizeModes(vectors: object, backend: ArrayBackend) -> object:
    xp = backend.xp
    if getattr(vectors, "ndim", 2) == 3:
        amplitudes = xp.max(xp.abs(vectors), axis=1)
        fluxes = xp.abs(modeFluxes(vectors, backend))
        fluxful = fluxes > 1e-12
        scales = xp.where(fluxful, xp.sqrt(fluxes), amplitudes)
        scales = xp.where(scales > 0, scales, 1.0)
        normalized = vectors / scales[:, None, :]

        pivotIndices = xp.argmax(xp.abs(normalized), axis=1)
        pivotValues = xp.torch.gather(normalized, 1, pivotIndices[:, None, :]).squeeze(1)
        phases = xp.where(xp.abs(pivotValues) > 0, pivotValues / xp.abs(pivotValues), 1.0 + 0.0j)
        return normalized / phases[:, None, :]

    amplitudes = xp.max(xp.abs(vectors), axis=0)
    fluxes = xp.abs(modeFluxes(vectors, backend))
    fluxful = fluxes > 1e-12
    scales = xp.where(fluxful, xp.sqrt(fluxes), amplitudes)
    scales = xp.where(scales > 0, scales, 1.0)
    normalized = vectors / scales

    pivotIndices = xp.argmax(xp.abs(normalized), axis=0)
    pivotValues = normalized[pivotIndices, xp.arange(normalized.shape[1])]
    phases = xp.where(xp.abs(pivotValues) > 0, pivotValues / xp.abs(pivotValues), 1.0 + 0.0j)
    return normalized / phases


def interfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    backend: ArrayBackend,
    *,
    homogeneousLeft: bool = False,
    homogeneousRight: bool = False,
    nOrders: int | None = None,
) -> SMatrix:
    if nOrders is not None:
        structured = structuredInterfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            nOrders,
            backend,
            homogeneousLeft=homogeneousLeft,
            homogeneousRight=homogeneousRight,
        )
        if structured is not None:
            return structured

    xp = backend.xp
    size = leftForward.shape[-1]
    matrix = xp.concatenate([leftBackward, -rightForward], axis=-1)
    rhsLeft = -leftForward
    rhsRight = rightBackward
    solved = solveInterfaceBlocks(backend, matrix, rhsLeft, rhsRight)
    return SMatrix(
        s11=solved[..., :size, :size],
        s12=solved[..., :size, size:],
        s21=solved[..., size:, :size],
        s22=solved[..., size:, size:],
    )


def interfaceSMatrices(
    regionForward: Sequence[object],
    regionBackward: Sequence[object],
    regionHomogeneous: Sequence[bool],
    nOrders: int,
    backend: ArrayBackend,
) -> tuple[SMatrix, ...]:
    xp = backend.xp
    batched = bool(regionForward and getattr(regionForward[0], "ndim", 2) == 3)
    results: list[SMatrix | None] = [None] * (len(regionForward) - 1)
    batchedMatrices = []
    batchedRhs = []
    batchedIndices: list[int] = []
    structuredWorks: list[PatternedHomogeneousInterfaceWork] = []
    structuredIndices: list[int] = []
    for index in range(len(regionForward) - 1):
        homogeneousLeft = regionHomogeneous[index]
        homogeneousRight = regionHomogeneous[index + 1]
        if homogeneousLeft and homogeneousRight:
            structured = homogeneousInterfaceSMatrix(
                regionForward[index],
                regionBackward[index],
                regionForward[index + 1],
                regionBackward[index + 1],
                nOrders,
                backend,
            )
            if structured is not None:
                results[index] = structured
                continue
        elif (
            homogeneousLeft != homogeneousRight
            and not batched
            and patternedHomogeneousInterfaceEnabled()
        ):
            work = patternedHomogeneousInterfaceWork(
                regionForward[index],
                regionBackward[index],
                regionForward[index + 1],
                regionBackward[index + 1],
                nOrders,
                backend,
                homogeneousLeft=homogeneousLeft,
            )
            if work is not None:
                structuredWorks.append(work)
                structuredIndices.append(index)
                continue

        batchedMatrices.append(xp.concatenate([regionBackward[index], -regionForward[index + 1]], axis=-1))
        batchedRhs.append(xp.concatenate([-regionForward[index], regionBackward[index + 1]], axis=-1))
        batchedIndices.append(index)

    if structuredWorks:
        solvedStructured = solvePatternedHomogeneousInterfaceWorks(structuredWorks, backend)
        for regionIndex, result in zip(structuredIndices, solvedStructured):
            results[regionIndex] = result

    if batchedMatrices:
        stackAxis = 1 if batched else 0
        matrixBatch = xp.stack(batchedMatrices, axis=stackAxis)
        rhsBatch = xp.stack(batchedRhs, axis=stackAxis)
        solvedBatch = solveInterfaceBlocks(backend, matrixBatch, rhsBatch)
        size = regionForward[0].shape[-1]
        for batchIndex, regionIndex in enumerate(batchedIndices):
            solved = solvedBatch[:, batchIndex] if batched else solvedBatch[batchIndex]
            results[regionIndex] = SMatrix(
                s11=solved[..., :size, :size],
                s12=solved[..., :size, size:],
                s21=solved[..., size:, :size],
                s22=solved[..., size:, size:],
            )

    if any(result is None for result in results):
        raise RuntimeError("batched interface solve did not fill every interface result")
    return tuple(result for result in results if result is not None)


def interfaceConditionNumbers(
    regionForward: Sequence[object],
    regionBackward: Sequence[object],
    backend: ArrayBackend,
) -> tuple[float, ...]:
    if len(regionForward) < 2:
        return ()
    torch = backend.xp.torch
    matrices = [
        torch.cat([regionBackward[index], -regionForward[index + 1]], dim=1)
        for index in range(len(regionForward) - 1)
    ]
    if not matrices:
        return ()
    values = torch.linalg.cond(torch.stack(matrices, dim=0))
    return tuple(float(value.detach().cpu().item()) for value in values)


def structuredInterfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
    *,
    homogeneousLeft: bool,
    homogeneousRight: bool,
) -> SMatrix | None:
    if homogeneousLeft and homogeneousRight:
        return homogeneousInterfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            nOrders,
            backend,
        )
    if homogeneousLeft == homogeneousRight or not patternedHomogeneousInterfaceEnabled():
        return None
    return patternedHomogeneousInterfaceSMatrix(
        leftForward,
        leftBackward,
        rightForward,
        rightBackward,
        nOrders,
        backend,
        homogeneousLeft=homogeneousLeft,
    )


def patternedHomogeneousInterfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
    *,
    homogeneousLeft: bool,
) -> SMatrix | None:
    work = patternedHomogeneousInterfaceWork(
        leftForward,
        leftBackward,
        rightForward,
        rightBackward,
        nOrders,
        backend,
        homogeneousLeft=homogeneousLeft,
    )
    if work is None:
        return None
    return solvePatternedHomogeneousInterfaceWorks([work], backend)[0]


def patternedHomogeneousInterfaceWork(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
    *,
    homogeneousLeft: bool,
) -> PatternedHomogeneousInterfaceWork | None:
    xp = backend.xp
    size = int(leftForward.shape[-1])
    rhs = xp.concatenate([-leftForward, rightBackward], axis=-1)

    if homogeneousLeft:
        factors = homogeneousBlockFactors(leftBackward, nOrders, backend)
        if factors is None:
            return None
        columns, rowIndex, nullRows, leftInverse = factors
        return PatternedHomogeneousInterfaceWork(
            homogeneousLeft=True,
            homogeneousMatrix=leftBackward,
            patternedMatrix=rightForward,
            reducedMatrix=-applyHomogeneousBlockRows(nullRows, rightForward, rowIndex),
            reducedRhs=applyHomogeneousBlockRows(nullRows, rhs, rowIndex),
            fullRhs=rhs,
            columns=columns,
            rowIndex=rowIndex,
            leftInverse=leftInverse,
            size=size,
        )

    factors = homogeneousBlockFactors(rightForward, nOrders, backend)
    if factors is None:
        return None
    columns, rowIndex, nullRows, leftInverse = factors
    return PatternedHomogeneousInterfaceWork(
        homogeneousLeft=False,
        homogeneousMatrix=rightForward,
        patternedMatrix=leftBackward,
        reducedMatrix=applyHomogeneousBlockRows(nullRows, leftBackward, rowIndex),
        reducedRhs=applyHomogeneousBlockRows(nullRows, rhs, rowIndex),
        fullRhs=rhs,
        columns=columns,
        rowIndex=rowIndex,
        leftInverse=leftInverse,
        size=size,
    )


def solvePatternedHomogeneousInterfaceWorks(
    works: Sequence[PatternedHomogeneousInterfaceWork],
    backend: ArrayBackend,
) -> tuple[SMatrix, ...]:
    if not works:
        return ()
    if len(works) == 1:
        work = works[0]
        patternedSolution = solveInterfaceBlocks(backend, work.reducedMatrix, work.reducedRhs)
        return (finishPatternedHomogeneousInterface(work, patternedSolution, backend),)

    xp = backend.xp
    batched = getattr(works[0].reducedMatrix, "ndim", 2) == 3
    sameShape = all(
        work.reducedMatrix.shape == works[0].reducedMatrix.shape
        and work.reducedRhs.shape == works[0].reducedRhs.shape
        for work in works
    )
    if sameShape:
        matrixBatch = xp.stack([work.reducedMatrix for work in works], axis=1 if batched else 0)
        rhsBatch = xp.stack([work.reducedRhs for work in works], axis=1 if batched else 0)
        solvedBatch = solveInterfaceBlocks(backend, matrixBatch, rhsBatch)
        return tuple(
            finishPatternedHomogeneousInterface(
                work,
                solvedBatch[:, index] if batched else solvedBatch[index],
                backend,
            )
            for index, work in enumerate(works)
        )

    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, PatternedHomogeneousInterfaceWork]]] = {}
    for index, work in enumerate(works):
        key = (tuple(int(value) for value in work.reducedMatrix.shape), tuple(int(value) for value in work.reducedRhs.shape))
        groups.setdefault(key, []).append((index, work))
    results: list[SMatrix | None] = [None] * len(works)
    for group in groups.values():
        if len(group) == 1:
            index, work = group[0]
            solution = solveInterfaceBlocks(backend, work.reducedMatrix, work.reducedRhs)
            results[index] = finishPatternedHomogeneousInterface(work, solution, backend)
            continue
        groupWorks = [work for ignored, work in group]
        groupBatched = getattr(groupWorks[0].reducedMatrix, "ndim", 2) == 3
        matrixBatch = xp.stack([work.reducedMatrix for work in groupWorks], axis=1 if groupBatched else 0)
        rhsBatch = xp.stack([work.reducedRhs for work in groupWorks], axis=1 if groupBatched else 0)
        solvedBatch = solveInterfaceBlocks(backend, matrixBatch, rhsBatch)
        for groupIndex, (resultIndex, work) in enumerate(group):
            solution = solvedBatch[:, groupIndex] if groupBatched else solvedBatch[groupIndex]
            results[resultIndex] = finishPatternedHomogeneousInterface(work, solution, backend)
    if any(result is None for result in results):
        raise RuntimeError("patterned-homogeneous interface batching did not fill every result")
    return tuple(
        result for result in results if result is not None
    )


def finishPatternedHomogeneousInterface(
    work: PatternedHomogeneousInterfaceWork,
    patternedSolution: object,
    backend: ArrayBackend,
) -> SMatrix:
    xp = backend.xp
    if work.homogeneousLeft:
        residual = work.fullRhs + work.patternedMatrix @ patternedSolution
        homogeneousSolution = homogeneousAmplitudesFromRows(
            work.leftInverse,
            residual,
            work.columns,
            work.rowIndex,
            work.size,
            backend,
        )
        solved = xp.concatenate([homogeneousSolution, patternedSolution], axis=-2)
    else:
        residual = work.patternedMatrix @ patternedSolution - work.fullRhs
        homogeneousSolution = homogeneousAmplitudesFromRows(
            work.leftInverse,
            residual,
            work.columns,
            work.rowIndex,
            work.size,
            backend,
        )
        solved = xp.concatenate([patternedSolution, homogeneousSolution], axis=-2)

    size = work.size
    return SMatrix(
        s11=solved[..., :size, :size],
        s12=solved[..., :size, size:],
        s21=solved[..., size:, :size],
        s22=solved[..., size:, size:],
    )


def homogeneousBlockFactors(
    matrix: object,
    nOrders: int,
    backend: ArrayBackend,
) -> tuple[object, object, object, object] | None:
    torch = backend.xp.torch
    rowIndex = fieldRowsForOrderTorch(nOrders, backend)

    if getattr(matrix, "ndim", 2) == 3:
        assignments = homogeneousColumnOrdersBatchTorch(matrix, nOrders, backend)
        if assignments is None:
            return None
        columns = columnsForOrdersBatchTorch(assignments, nOrders, backend)
        if columns is None:
            return None
        batchSize = int(matrix.shape[0])
        batchIndex = torch.arange(batchSize, dtype=torch.long, device=backend.device)[:, None, None, None]
        blocks = matrix[batchIndex, rowIndex[None, :, :, None], columns[:, :, None, :]]
    else:
        assignments = homogeneousColumnOrdersTorch(matrix, nOrders, backend)
        if assignments is None:
            return None
        columns = columnsForOrdersTorch(assignments, nOrders, backend)
        if columns is None:
            return None
        blocks = matrix[rowIndex[:, :, None], columns[:, None, :]]

    qMatrix, rMatrix = torch.linalg.qr(blocks, mode="complete")
    nullRows = torch.conj(torch.transpose(qMatrix[..., :, 2:], -2, -1))
    leftInverse = torch.linalg.solve(rMatrix[..., :2, :], torch.conj(torch.transpose(qMatrix[..., :, :2], -2, -1)))
    return columns, rowIndex, nullRows, leftInverse


def applyHomogeneousBlockRows(operator: object, matrix: object, rowIndex: object) -> object:
    if getattr(matrix, "ndim", 2) == 3:
        rows = matrix[:, rowIndex, :]
        return (operator @ rows).reshape(matrix.shape[0], -1, matrix.shape[-1])
    rows = matrix[rowIndex, :]
    return (operator @ rows).reshape(-1, matrix.shape[-1])


def homogeneousAmplitudesFromRows(
    leftInverse: object,
    values: object,
    columns: object,
    rowIndex: object,
    size: int,
    backend: ArrayBackend,
) -> object:
    xp = backend.xp
    if getattr(values, "ndim", 2) == 3:
        amplitudes = leftInverse @ values[:, rowIndex, :]
        result = xp.zeros((values.shape[0], size, values.shape[-1]), dtype=complex)
        flatColumns = columns.reshape(columns.shape[0], -1)
        flatAmplitudes = amplitudes.reshape(values.shape[0], -1, values.shape[-1])
        result.scatter_(1, flatColumns[:, :, None].expand(-1, -1, values.shape[-1]), flatAmplitudes)
        return result

    amplitudes = leftInverse @ values[rowIndex, :]
    result = xp.zeros((size, values.shape[-1]), dtype=complex)
    result[columns.reshape(-1), :] = amplitudes.reshape(-1, values.shape[-1])
    return result


def homogeneousInterfaceSMatrix(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
) -> SMatrix | None:
    if getattr(leftForward, "ndim", 2) == 3:
        return homogeneousInterfaceSMatrixBatchTorch(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            nOrders,
            backend,
        )
    return homogeneousInterfaceSMatrixTorch(
        leftForward,
        leftBackward,
        rightForward,
        rightBackward,
        nOrders,
        backend,
    )


def homogeneousInterfaceSMatrixBatchTorch(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
) -> SMatrix | None:
    leftForwardOrders = homogeneousColumnOrdersBatchTorch(leftForward, nOrders, backend)
    leftBackwardOrders = homogeneousColumnOrdersBatchTorch(leftBackward, nOrders, backend)
    rightForwardOrders = homogeneousColumnOrdersBatchTorch(rightForward, nOrders, backend)
    rightBackwardOrders = homogeneousColumnOrdersBatchTorch(rightBackward, nOrders, backend)
    if (
        leftForwardOrders is None
        or leftBackwardOrders is None
        or rightForwardOrders is None
        or rightBackwardOrders is None
    ):
        return None

    xp = backend.xp
    torch = xp.torch
    batchSize = int(leftForward.shape[0])
    size = int(leftForward.shape[-1])
    rowIndex = fieldRowsForOrderTorch(nOrders, backend)
    leftForwardColumns = columnsForOrdersBatchTorch(leftForwardOrders, nOrders, backend)
    leftBackwardColumns = columnsForOrdersBatchTorch(leftBackwardOrders, nOrders, backend)
    rightForwardColumns = columnsForOrdersBatchTorch(rightForwardOrders, nOrders, backend)
    rightBackwardColumns = columnsForOrdersBatchTorch(rightBackwardOrders, nOrders, backend)
    if (
        leftForwardColumns is None
        or leftBackwardColumns is None
        or rightForwardColumns is None
        or rightBackwardColumns is None
    ):
        return None

    batchIndex = torch.arange(batchSize, dtype=torch.long, device=backend.device)[:, None, None, None]
    rows = rowIndex[None, :, :, None]
    matrices = torch.cat(
        [
            leftBackward[batchIndex, rows, leftBackwardColumns[:, :, None, :]],
            -rightForward[batchIndex, rows, rightForwardColumns[:, :, None, :]],
        ],
        dim=3,
    )
    rightHandSides = torch.cat(
        [
            -leftForward[batchIndex, rows, leftForwardColumns[:, :, None, :]],
            rightBackward[batchIndex, rows, rightBackwardColumns[:, :, None, :]],
        ],
        dim=3,
    )
    solvedBlocks = backend.solve(matrices, rightHandSides)

    s11 = xp.zeros((batchSize, size, size), dtype=complex)
    s12 = xp.zeros((batchSize, size, size), dtype=complex)
    s21 = xp.zeros((batchSize, size, size), dtype=complex)
    s22 = xp.zeros((batchSize, size, size), dtype=complex)

    batchRows = torch.arange(batchSize, dtype=torch.long, device=backend.device)[:, None, None, None]
    leftBackwardTarget = leftBackwardColumns[:, :, :, None]
    leftForwardSource = leftForwardColumns[:, :, None, :]
    rightForwardTarget = rightForwardColumns[:, :, :, None]
    rightBackwardSource = rightBackwardColumns[:, :, None, :]
    s11[batchRows, leftBackwardTarget, leftForwardSource] = solvedBlocks[:, :, :2, :2]
    s12[batchRows, leftBackwardTarget, rightBackwardSource] = solvedBlocks[:, :, :2, 2:]
    s21[batchRows, rightForwardTarget, leftForwardSource] = solvedBlocks[:, :, 2:, :2]
    s22[batchRows, rightForwardTarget, rightBackwardSource] = solvedBlocks[:, :, 2:, 2:]

    return SMatrix(s11=s11, s12=s12, s21=s21, s22=s22)


def homogeneousInterfaceSMatrixTorch(
    leftForward: object,
    leftBackward: object,
    rightForward: object,
    rightBackward: object,
    nOrders: int,
    backend: ArrayBackend,
) -> SMatrix | None:
    if getattr(leftForward, "ndim", 2) == 3:
        return None
    leftForwardOrders = homogeneousColumnOrdersTorch(leftForward, nOrders, backend)
    leftBackwardOrders = homogeneousColumnOrdersTorch(leftBackward, nOrders, backend)
    rightForwardOrders = homogeneousColumnOrdersTorch(rightForward, nOrders, backend)
    rightBackwardOrders = homogeneousColumnOrdersTorch(rightBackward, nOrders, backend)
    if (
        leftForwardOrders is None
        or leftBackwardOrders is None
        or rightForwardOrders is None
        or rightBackwardOrders is None
    ):
        return None

    xp = backend.xp
    torch = xp.torch
    size = leftForward.shape[1]
    rowIndex = fieldRowsForOrderTorch(nOrders, backend)
    leftForwardColumns = columnsForOrdersTorch(leftForwardOrders, nOrders, backend)
    leftBackwardColumns = columnsForOrdersTorch(leftBackwardOrders, nOrders, backend)
    rightForwardColumns = columnsForOrdersTorch(rightForwardOrders, nOrders, backend)
    rightBackwardColumns = columnsForOrdersTorch(rightBackwardOrders, nOrders, backend)
    if (
        leftForwardColumns is None
        or leftBackwardColumns is None
        or rightForwardColumns is None
        or rightBackwardColumns is None
    ):
        return None

    matrices = torch.cat(
        [
            leftBackward[rowIndex[:, :, None], leftBackwardColumns[:, None, :]],
            -rightForward[rowIndex[:, :, None], rightForwardColumns[:, None, :]],
        ],
        dim=2,
    )
    rightHandSides = torch.cat(
        [
            -leftForward[rowIndex[:, :, None], leftForwardColumns[:, None, :]],
            rightBackward[rowIndex[:, :, None], rightBackwardColumns[:, None, :]],
        ],
        dim=2,
    )
    solvedBlocks = backend.solve(matrices, rightHandSides)

    s11 = xp.zeros((size, size), dtype=complex)
    s12 = xp.zeros((size, size), dtype=complex)
    s21 = xp.zeros((size, size), dtype=complex)
    s22 = xp.zeros((size, size), dtype=complex)
    s11[leftBackwardColumns[:, :, None], leftForwardColumns[:, None, :]] = solvedBlocks[:, :2, :2]
    s12[leftBackwardColumns[:, :, None], rightBackwardColumns[:, None, :]] = solvedBlocks[:, :2, 2:]
    s21[rightForwardColumns[:, :, None], leftForwardColumns[:, None, :]] = solvedBlocks[:, 2:, :2]
    s22[rightForwardColumns[:, :, None], rightBackwardColumns[:, None, :]] = solvedBlocks[:, 2:, 2:]

    return SMatrix(s11=s11, s12=s12, s21=s21, s22=s22)


def homogeneousColumnOrdersTorch(matrix: object, nOrders: int, backend: ArrayBackend) -> object | None:
    if matrix.shape[0] != 4 * nOrders:
        return None
    torch = backend.xp.torch
    magnitude = torch.abs(matrix)
    maxMagnitude = torch.amax(magnitude) if matrix.numel() else torch.as_tensor(0.0, dtype=backend.floatDtype, device=backend.device)
    tolerance = 1e-10 * torch.clamp(maxMagnitude, min=1.0)
    active = magnitude > tolerance
    if bool(torch.any(torch.sum(active, dim=0) == 0)):
        return None

    rowOrders = torch.arange(4 * nOrders, dtype=torch.long, device=backend.device) % nOrders
    sentinelHigh = torch.full((4 * nOrders, 1), nOrders, dtype=torch.long, device=backend.device)
    sentinelLow = torch.full((4 * nOrders, 1), -1, dtype=torch.long, device=backend.device)
    minOrders = torch.amin(torch.where(active, rowOrders[:, None], sentinelHigh), dim=0)
    maxOrders = torch.amax(torch.where(active, rowOrders[:, None], sentinelLow), dim=0)
    if bool(torch.any(minOrders != maxOrders)):
        return None
    return minOrders


def homogeneousColumnOrdersBatchTorch(matrix: object, nOrders: int, backend: ArrayBackend) -> object | None:
    if matrix.shape[1] != 4 * nOrders:
        return None
    torch = backend.xp.torch
    magnitude = torch.abs(matrix)
    maxMagnitude = torch.amax(magnitude) if matrix.numel() else torch.as_tensor(0.0, dtype=backend.floatDtype, device=backend.device)
    tolerance = 1e-10 * torch.clamp(maxMagnitude, min=1.0)
    active = magnitude > tolerance
    if bool(torch.any(torch.sum(active, dim=1) == 0)):
        return None

    rowOrders = torch.arange(4 * nOrders, dtype=torch.long, device=backend.device) % nOrders
    sentinelHigh = torch.full((1, 4 * nOrders, 1), nOrders, dtype=torch.long, device=backend.device)
    sentinelLow = torch.full((1, 4 * nOrders, 1), -1, dtype=torch.long, device=backend.device)
    minOrders = torch.amin(torch.where(active, rowOrders[None, :, None], sentinelHigh), dim=1)
    maxOrders = torch.amax(torch.where(active, rowOrders[None, :, None], sentinelLow), dim=1)
    if bool(torch.any(minOrders != maxOrders)):
        return None
    return minOrders


def columnsForOrdersTorch(assignments: object, nOrders: int, backend: ArrayBackend) -> object | None:
    torch = backend.xp.torch
    columns = torch.arange(assignments.shape[0], dtype=torch.long, device=backend.device)
    orderIds = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    mask = assignments[None, :] == orderIds[:, None]
    if bool(torch.any(torch.sum(mask, dim=1) != 2)):
        return None
    return columns.expand(nOrders, -1)[mask].reshape(nOrders, 2)


def columnsForOrdersBatchTorch(assignments: object, nOrders: int, backend: ArrayBackend) -> object | None:
    torch = backend.xp.torch
    batchSize, nColumns = assignments.shape
    columns = torch.arange(nColumns, dtype=torch.long, device=backend.device)
    orderIds = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    mask = assignments[:, None, :] == orderIds[None, :, None]
    if bool(torch.any(torch.sum(mask, dim=2) != 2)):
        return None
    return columns[None, None, :].expand(batchSize, nOrders, -1)[mask].reshape(batchSize, nOrders, 2)


def fieldRowsForOrderTorch(nOrders: int, backend: ArrayBackend) -> object:
    torch = backend.xp.torch
    orders = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    offsets = torch.as_tensor([0, nOrders, 2 * nOrders, 3 * nOrders], dtype=torch.long, device=backend.device)
    return orders[:, None] + offsets[None, :]


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
    nOrders = harmonics.count
    basis = np.zeros((4 * nOrders, 2 * nOrders), dtype=complex)
    kx = harmonics.kx
    ky = harmonics.ky
    kz = forwardKz(eps - kx**2 - ky**2)
    if direction < 0:
        kz = -kz

    transverse = np.sqrt(kx * kx + ky * ky + 0j)
    normal = np.abs(transverse) < 1e-14
    safeTransverse = np.where(normal, 1.0 + 0.0j, transverse)
    sx = np.where(normal, 0.0 + 0.0j, -ky / safeTransverse)
    sy = np.where(normal, 1.0 + 0.0j, kx / safeTransverse)

    refractiveIndex = np.sqrt(eps + 0j)
    if np.imag(refractiveIndex) < -1e-14 or (
        abs(np.imag(refractiveIndex)) <= 1e-14 and np.real(refractiveIndex) < 0
    ):
        refractiveIndex = -refractiveIndex

    px = sy * kz / refractiveIndex
    py = -sx * kz / refractiveIndex
    pz = (sx * ky - sy * kx) / refractiveIndex
    hsx = -kz * sy
    hsy = kz * sx
    hpx = ky * pz - kz * py
    hpy = kz * px - kx * pz

    orderIndices = np.arange(nOrders)
    sColumns = 2 * orderIndices
    pColumns = sColumns + 1
    basis[orderIndices, sColumns] = sx
    basis[nOrders + orderIndices, sColumns] = sy
    basis[2 * nOrders + orderIndices, sColumns] = hsx
    basis[3 * nOrders + orderIndices, sColumns] = hsy
    basis[orderIndices, pColumns] = px
    basis[nOrders + orderIndices, pColumns] = py
    basis[2 * nOrders + orderIndices, pColumns] = hpx
    basis[3 * nOrders + orderIndices, pColumns] = hpy
    return basis


def homogeneousBasisBatch(
    harmonics: BatchedHarmonics,
    eps: complex | ComplexArray,
    direction: int,
    backend: ArrayBackend,
) -> object:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    torch = backend.xp.torch
    nOrders = harmonics.count
    basis = backend.xp.zeros((harmonics.batchSize, 4 * nOrders, 2 * nOrders), dtype=complex)
    kx = harmonics.kx
    ky = harmonics.ky
    epsBatch = batchedScalarEpsilon(eps, harmonics, backend)
    kz = forwardKzTorch(epsBatch - kx * kx - ky * ky, torch)
    if direction < 0:
        kz = -kz

    transverse = torch.sqrt(kx * kx + ky * ky + 0j)
    normal = torch.abs(transverse) < 1e-14
    safeTransverse = torch.where(normal, torch.as_tensor(1.0 + 0.0j, dtype=kx.dtype, device=kx.device), transverse)
    sx = torch.where(normal, torch.zeros((), dtype=kx.dtype, device=kx.device), -ky / safeTransverse)
    sy = torch.where(normal, torch.ones((), dtype=kx.dtype, device=kx.device), kx / safeTransverse)

    refractiveIndex = sqrtBranchBatchTorch(epsBatch, torch)
    px = sy * kz / refractiveIndex
    py = -sx * kz / refractiveIndex
    pz = (sx * ky - sy * kx) / refractiveIndex
    hsx = -kz * sy
    hsy = kz * sx
    hpx = ky * pz - kz * py
    hpy = kz * px - kx * pz

    orderIndices = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    batchIndices = torch.arange(harmonics.batchSize, dtype=torch.long, device=backend.device)[:, None]
    sColumns = 2 * orderIndices
    pColumns = sColumns + 1
    basis[batchIndices, orderIndices[None, :], sColumns[None, :]] = sx
    basis[batchIndices, nOrders + orderIndices[None, :], sColumns[None, :]] = sy
    basis[batchIndices, 2 * nOrders + orderIndices[None, :], sColumns[None, :]] = hsx
    basis[batchIndices, 3 * nOrders + orderIndices[None, :], sColumns[None, :]] = hsy
    basis[batchIndices, orderIndices[None, :], pColumns[None, :]] = px
    basis[batchIndices, nOrders + orderIndices[None, :], pColumns[None, :]] = py
    basis[batchIndices, 2 * nOrders + orderIndices[None, :], pColumns[None, :]] = hpx
    basis[batchIndices, 3 * nOrders + orderIndices[None, :], pColumns[None, :]] = hpy
    return basis


def sqrtBranchBatchTorch(value: object, torch: object) -> object:
    root = torch.sqrt(value)
    flip = (torch.imag(root) < -1e-14) | (
        (torch.abs(torch.imag(root)) <= 1e-14) & (torch.real(root) < 0)
    )
    return torch.where(flip, -root, root)


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
    reflectedFluxes = homogeneousOrderFluxes(reflectedBasis, rAmplitudes, harmonics.count)
    transmittedFluxes = homogeneousOrderFluxes(transmittedBasis, tAmplitudes, harmonics.count)
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


def homogeneousOrderFluxes(basis: ComplexArray, amplitudes: ComplexArray, nOrders: int) -> ComplexArray:
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


def homogeneousOrderFluxesBackend(
    basis: object,
    amplitudes: object,
    nOrders: int,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    orderIndices = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    sColumns = 2 * orderIndices
    pColumns = sColumns + 1

    sAmplitudes = amplitudes[sColumns, :]
    pAmplitudes = amplitudes[pColumns, :]
    ex = basis[orderIndices, sColumns][:, None] * sAmplitudes + basis[orderIndices, pColumns][:, None] * pAmplitudes
    ey = (
        basis[nOrders + orderIndices, sColumns][:, None] * sAmplitudes
        + basis[nOrders + orderIndices, pColumns][:, None] * pAmplitudes
    )
    hx = (
        basis[2 * nOrders + orderIndices, sColumns][:, None] * sAmplitudes
        + basis[2 * nOrders + orderIndices, pColumns][:, None] * pAmplitudes
    )
    hy = (
        basis[3 * nOrders + orderIndices, sColumns][:, None] * sAmplitudes
        + basis[3 * nOrders + orderIndices, pColumns][:, None] * pAmplitudes
    )
    return 0.5 * torch.real(ex * torch.conj(hy) - ey * torch.conj(hx))


def homogeneousOrderFluxesBatchBackend(
    basis: object,
    amplitudes: object,
    nOrders: int,
    backend: ArrayBackend,
) -> object:
    torch = backend.xp.torch
    orderIndices = torch.arange(nOrders, dtype=torch.long, device=backend.device)
    sColumns = 2 * orderIndices
    pColumns = sColumns + 1

    sAmplitudes = amplitudes[:, sColumns, :]
    pAmplitudes = amplitudes[:, pColumns, :]
    ex = basis[:, orderIndices, sColumns][:, :, None] * sAmplitudes + basis[:, orderIndices, pColumns][:, :, None] * pAmplitudes
    ey = (
        basis[:, nOrders + orderIndices, sColumns][:, :, None] * sAmplitudes
        + basis[:, nOrders + orderIndices, pColumns][:, :, None] * pAmplitudes
    )
    hx = (
        basis[:, 2 * nOrders + orderIndices, sColumns][:, :, None] * sAmplitudes
        + basis[:, 2 * nOrders + orderIndices, pColumns][:, :, None] * pAmplitudes
    )
    hy = (
        basis[:, 3 * nOrders + orderIndices, sColumns][:, :, None] * sAmplitudes
        + basis[:, 3 * nOrders + orderIndices, pColumns][:, :, None] * pAmplitudes
    )
    return 0.5 * torch.real(ex * torch.conj(hy) - ey * torch.conj(hx))


def smatrixLayerSolutions(
    layers: Sequence[Layer | CompiledLayer],
    layerModes: Sequence[tuple[ComplexArray, ComplexArray]],
    components: Sequence[SMatrix],
    incidentAmplitudes: ComplexArray,
    harmonics: Harmonics,
    wavelength: float,
    period: tuple[float, float],
    orders: tuple[int, int],
    backend: ArrayBackend,
) -> tuple[LayerFieldSolution, ...]:
    xp = backend.xp
    nPorts = 2 * harmonics.count
    prefix = prefixSMatrices(components, nPorts, backend)
    suffix = suffixSMatrices(components, nPorts, backend)
    identity = xp.eye(nPorts, dtype=complex)
    layerSolutionList = []
    for layerIndex, (layer, (qValues, modeMatrix)) in enumerate(zip(layers, layerModes)):
        boundaryIndex = 2 * layerIndex + 1
        leftNetwork = prefix[boundaryIndex]
        rightNetwork = suffix[boundaryIndex]
        rhs = rightNetwork.s11 @ (leftNetwork.s21 @ incidentAmplitudes)
        backwardAtLeft = solveFactored(backend, identity - rightNetwork.s11 @ leftNetwork.s22, rhs)
        forwardAtLeft = leftNetwork.s21 @ incidentAmplitudes + leftNetwork.s22 @ backwardAtLeft
        coefficients = xp.concatenate([forwardAtLeft, backwardAtLeft])
        layerSolutionList.append(
            layerSolution(
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


def incidentAmplitudeVector(prepared: PreparedStack, sAmplitude: complex, pAmplitude: complex) -> ComplexArray:
    amplitudes = np.zeros(prepared.nPorts, dtype=complex)
    amplitudes[2 * prepared.zeroIndex] = sAmplitude
    amplitudes[2 * prepared.zeroIndex + 1] = pAmplitude
    return amplitudes


def batchedIncidentAmplitudeVector(
    prepared: PreparedBatchStack,
    sAmplitude: complex,
    pAmplitude: complex,
) -> ComplexArray:
    amplitudes = np.zeros(prepared.nPorts, dtype=complex)
    amplitudes[2 * prepared.zeroIndex] = sAmplitude
    amplitudes[2 * prepared.zeroIndex + 1] = pAmplitude
    return amplitudes


def incidentAmplitudes(
    prepared: PreparedStack,
    sAmplitude: complex,
    pAmplitude: complex,
    backend: ArrayBackend,
) -> object:
    return backend.asarray(incidentAmplitudeVector(prepared, sAmplitude, pAmplitude))


def layerSolution(
    layer: Layer | CompiledLayer,
    qValues: ComplexArray,
    modeMatrix: ComplexArray,
    coefficients: ComplexArray,
    harmonics: Harmonics,
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


def result(
    harmonics: Harmonics,
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
    stackTiming: StackTiming | None = None,
) -> RCWAResult:
    diffractionOrders = tuple(
        orderResults(
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
    reflection = float(sum(order.reflectedPower for order in diffractionOrders))
    transmission = float(sum(order.transmittedPower for order in diffractionOrders))

    return RCWAResult(
        reflection=reflection,
        transmission=transmission,
        conservation=reflection + transmission,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=diffractionOrders,
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        layerEigTimings=layerEigTimings,
        stackTiming=stackTiming,
        diagnostics=tuple(stackTiming.stabilityWarnings) if stackTiming is not None else (),
    )


def checkedIncidentFlux(field: ComplexArray) -> float:
    incidentFlux = flux(field)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def zeroOrderIndex(harmonics: Harmonics) -> int:
    return zeroOrderIndexFromOrders(harmonics.mx, harmonics.my)


def zeroOrderIndexFromOrders(mx: ComplexArray, my: ComplexArray) -> int:
    zeroIndices = np.where((mx == 0) & (my == 0))[0]
    if zeroIndices.size != 1:
        raise RuntimeError("zero diffraction order was not found")
    return int(zeroIndices[0])


def validateInputs(wavelength: float, period: tuple[float, float], orders: int | tuple[int, int]) -> None:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")
    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")


def isPropagating(kz: complex) -> bool:
    return bool(abs(np.imag(kz)) < 1e-10 and np.real(kz) > 1e-12)
