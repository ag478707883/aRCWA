from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .backend import ArrayBackend, resolveBackend
from .factorization import (
    homogeneousEpsilon as homogeneousEpsilon,
    layerDataForTorch as layerDataForTorch,
    pqMatricesTorch as pqMatricesTorch,
    solveIdentityTorch as solveIdentityTorch,
    toTorchComplex as toTorchComplex,
)
from .fourier import Harmonics, flux, forwardKz, makeHarmonics, normalizeOrders, planeWaveFields, putOrderField, singleOrderVector, sqrtBranch
from .types import ComplexArray, CompiledLayer, DiffractionOrder, Layer, LayerEigTiming, LayerFieldSolution, RCWAResult, StackTiming
from .varrcwa import AdaptiveLayerSpec


Q_RELATIVE_TOLERANCE = 256.0 * np.finfo(float).eps
INTERFACE_CONDITION_WARNING = 1e10
LOSS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TorchSMatrix:
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
    components: tuple[TorchSMatrix, ...]
    total: TorchSMatrix
    incidentForward: ComplexArray
    incidentBackward: ComplexArray
    transmissionForward: ComplexArray
    zeroIndex: int
    backend: ArrayBackend
    stackTiming: StackTiming | None = None
    columnForwardOperators: tuple[Any | None, ...] = ()

    @property
    def nPorts(self) -> int:
        return 2 * self.harmonics.count


@dataclass(frozen=True)
class OrderReductionPlan:
    label: str
    reducedOrders: tuple[int, int]
    fullHarmonics: Harmonics
    keptIndices: ComplexArray


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
    return prepareStackTorchCore(
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
        profile=profile,
        powersOnlyColumns=False,
    )


def prepareStackPowersTorch(
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
    return prepareStackTorchCore(
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
        profile=profile,
        powersOnlyColumns=True,
    )


def prepareStackTorchCore(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelength: float,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    epsIncident: complex,
    epsTransmission: complex,
    theta: float,
    phi: float,
    truncation: str,
    backend: str | ArrayBackend,
    profile: bool,
    powersOnlyColumns: bool,
) -> PreparedTorchStack:
    prepareStart = time.perf_counter()
    validateGeometry(wavelength, period, orders)
    arrayBackend = resolveBackend(backend)
    if not arrayBackend.isTorch or not arrayBackend.isCuda:
        raise ValueError("prepareStackTorch requires the CUDA backend")

    torch = arrayBackend.xp
    device = arrayBackend.device if arrayBackend.device is not None else torch.device("cuda")
    normalizedOrders = normalizeOrders(orders)
    harmonics = makeHarmonics(wavelength, period, normalizedOrders, epsIncident, theta, phi, truncation=truncation)
    nPorts = 2 * harmonics.count
    k0 = 2 * np.pi / wavelength

    incidentForward = homogeneousBasisTorch(harmonics, epsIncident, direction=1, torch=torch, device=device)
    incidentBackward = homogeneousBasisTorch(harmonics, epsIncident, direction=-1, torch=torch, device=device)
    transmissionForward = homogeneousBasisTorch(harmonics, epsTransmission, direction=1, torch=torch, device=device)
    transmissionBackward = homogeneousBasisTorch(harmonics, epsTransmission, direction=-1, torch=torch, device=device)

    modeList: list[tuple[Any, Any]] = []
    layerEigTimings: list[LayerEigTiming] = []
    for layerIndex, layer in enumerate(layers):
        modes, timing = layerModesWithTimingTorch(
            layer,
            harmonics,
            torch,
            device,
            layerIndex=layerIndex,
            profile=profile,
            k0=k0,
        )
        modeList.append(modes)
        if timing is not None:
            layerEigTimings.append(timing)
    modes = tuple(modeList)
    regionForward: list[Any] = [incidentForward]
    regionBackward: list[Any] = [incidentBackward]
    for qValues, modeMatrix in modes:
        regionForward.append(modeMatrix[:, :nPorts])
        regionBackward.append(modeMatrix[:, nPorts:])
    regionForward.append(transmissionForward)
    regionBackward.append(transmissionBackward)

    if profile:
        torch.cuda.synchronize(device)
    interfaceStart = time.perf_counter()
    interfaces = interfaceSMatricesTorch(regionForward, regionBackward, torch)
    interfaceConditionNumbers = (
        interfaceConditionNumbersTorch(regionForward, regionBackward, torch) if profile else ()
    )
    if profile:
        torch.cuda.synchronize(device)
    interfaceTime = time.perf_counter() - interfaceStart if profile else 0.0
    components: list[TorchSMatrix] = []
    for regionIndex in range(len(layers) + 1):
        components.append(interfaces[regionIndex])
        if regionIndex < len(layers):
            qForward = modes[regionIndex][0][:nPorts]
            propagation = torch.diag(torch.exp(1j * qForward * k0 * layers[regionIndex].thickness))
            components.append(propagationSMatrixTorch(propagation, torch))

    componentTuple = tuple(components)
    if profile:
        torch.cuda.synchronize(device)
    cascadeStart = time.perf_counter()
    if powersOnlyColumns:
        reflection, forwardOperators = reflectionAndForwardOperatorsTorch(componentTuple, nPorts, torch, device)
        zero = torch.zeros_like(reflection)
        total = TorchSMatrix(s11=reflection, s12=zero.clone(), s21=zero.clone(), s22=zero.clone())
        columnForwardOperators = forwardOperators
    else:
        total = reflectionTransmissionOnlySMatrixTorch(componentTuple, nPorts, torch, device)
        columnForwardOperators = ()
    if profile:
        torch.cuda.synchronize(device)
    cascadeTime = time.perf_counter() - cascadeStart if profile else 0.0
    stackTiming = None
    if profile:
        warningMessages = stabilityWarnings(layerEigTimings, interfaceConditionNumbers)
        stackTiming = StackTiming(
            interfaceTimeSeconds=interfaceTime,
            cascadeTimeSeconds=cascadeTime,
            totalPrepareTimeSeconds=time.perf_counter() - prepareStart,
            interfaceConditionNumbers=interfaceConditionNumbers,
            maxInterfaceCondition=max(interfaceConditionNumbers) if interfaceConditionNumbers else None,
            stabilityWarnings=warningMessages,
        )
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
        stackTiming=stackTiming,
        components=componentTuple,
        total=total,
        incidentForward=arrayBackend.asnumpy(incidentForward).copy(),
        incidentBackward=arrayBackend.asnumpy(incidentBackward).copy(),
        transmissionForward=arrayBackend.asnumpy(transmissionForward).copy(),
        zeroIndex=zeroOrderIndex(harmonics),
        backend=arrayBackend,
        columnForwardOperators=columnForwardOperators,
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
    incident = incidentAmplitudesTorch(prepared, sAmplitude, pAmplitude, torch)
    incidentFlux = checkedIncidentFluxTorch(
        prepared,
        incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, sAmplitude, pAmplitude),
    )
    rAmplitudes = prepared.total.s11 @ incident
    tAmplitudes = prepared.total.s21 @ incident
    return resultFromTorchAmplitudes(
        prepared=prepared,
        rAmplitudes=prepared.backend.asnumpy(rAmplitudes),
        tAmplitudes=prepared.backend.asnumpy(tAmplitudes),
        incidentFlux=incidentFlux,
        solvedBy=solvedBy,
        layerSolutions=layerSolutionsTorch(prepared, incident) if returnFields and prepared.layers else (),
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
    columns = [incidentAmplitudesTorch(prepared, *excitations[label], torch) for label in labels]
    incidentColumns = torch.column_stack(columns)
    reflected = prepared.total.s11 @ incidentColumns
    transmitted = prepared.total.s21 @ incidentColumns

    incidentColumnsNp = np.column_stack(
        [incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, *excitations[label]) for label in labels]
    )
    reflectedNp = prepared.backend.asnumpy(reflected)
    transmittedNp = prepared.backend.asnumpy(transmitted)

    results: dict[str, RCWAResult] = {}
    for column, label in enumerate(labels):
        sAmplitude, pAmplitude = excitations[label]
        incidentFlux = checkedIncidentFluxTorch(prepared, incidentColumnsNp[:, column])
        results[label] = resultFromTorchAmplitudes(
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
    columns = [incidentAmplitudesTorch(prepared, *excitations[label], torch) for label in labels]
    incidentColumns = torch.column_stack(columns)
    reflectedTorch = prepared.total.s11 @ incidentColumns
    if prepared.columnForwardOperators:
        transmittedTorch = applyForwardOperatorsTorch(prepared.columnForwardOperators, incidentColumns)
    else:
        transmittedTorch = prepared.total.s21 @ incidentColumns
    reflected = prepared.backend.asnumpy(reflectedTorch)
    transmitted = prepared.backend.asnumpy(transmittedTorch)

    reflectedFluxes = orderFluxesFromLocalBasis(prepared.incidentBackward, reflected)
    transmittedFluxes = orderFluxesFromLocalBasis(prepared.transmissionForward, transmitted)
    powers: dict[str, tuple[float, float]] = {}
    for column, label in enumerate(labels):
        incidentFlux = incidentFluxFromAmplitudes(prepared, *excitations[label])
        reflection = float(np.sum(-reflectedFluxes[:, column] / incidentFlux))
        transmission = float(np.sum(transmittedFluxes[:, column] / incidentFlux))
        powers[label] = (reflection, transmission)
    return powers


def evaluateSpectrumBatchPowersTorch(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelengths: Sequence[float],
    period: tuple[float, float],
    orders: int | tuple[int, int],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    truncation: str = "circular",
    backend: str | ArrayBackend = "cuda",
    chunkSize: int = 16,
) -> dict[str, tuple[ComplexArray, ComplexArray]]:
    values = np.asarray(tuple(wavelengths), dtype=float)
    labels = tuple(excitations)
    if values.size == 0:
        return {label: (np.empty(values.shape), np.empty(values.shape)) for label in labels}
    if chunkSize < 1:
        raise ValueError("chunkSize must be at least 1")

    arrayBackend = resolveBackend(backend)
    if not arrayBackend.isTorch or not arrayBackend.isCuda:
        raise ValueError("evaluateSpectrumBatchPowersTorch requires the CUDA backend")
    torch = arrayBackend.xp
    device = arrayBackend.device if arrayBackend.device is not None else torch.device("cuda")

    reflection = {label: np.empty(values.shape, dtype=float) for label in labels}
    transmission = {label: np.empty(values.shape, dtype=float) for label in labels}
    for start in range(0, values.size, chunkSize):
        stop = min(start + chunkSize, values.size)
        chunk = values[start:stop]
        chunkPowers = evaluateSpectrumChunkPowersTorch(
            layers=layers,
            wavelengths=chunk,
            period=period,
            orders=orders,
            excitations=excitations,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            truncation=truncation,
            torch=torch,
            device=device,
        )
        for label, (rValues, tValues) in chunkPowers.items():
            reflection[label][start:stop] = rValues
            transmission[label][start:stop] = tValues
    return {label: (reflection[label], transmission[label]) for label in labels}


def evaluateSpectrumChunkPowersTorch(
    *,
    layers: Sequence[Layer | CompiledLayer],
    wavelengths: ComplexArray,
    period: tuple[float, float],
    orders: int | tuple[int, int],
    excitations: Mapping[str, tuple[complex, complex]],
    epsIncident: complex,
    epsTransmission: complex,
    theta: float,
    phi: float,
    truncation: str,
    torch: Any,
    device: Any,
) -> dict[str, tuple[ComplexArray, ComplexArray]]:
    normalizedOrders = normalizeOrders(orders)
    harmonicsList = [
        makeHarmonics(float(wavelength), period, normalizedOrders, epsIncident, theta, phi, truncation=truncation)
        for wavelength in wavelengths
    ]
    reference = harmonicsList[0]
    if any(
        harmonics.count != reference.count
        or harmonics.orders != reference.orders
        or harmonics.truncation != reference.truncation
        or not np.array_equal(harmonics.mx, reference.mx)
        or not np.array_equal(harmonics.my, reference.my)
        for harmonics in harmonicsList[1:]
    ):
        raise RuntimeError("batched spectrum chunk has inconsistent harmonic bases")

    nPorts = 2 * reference.count
    k0Values = torch.as_tensor(2 * np.pi / wavelengths, dtype=torch.float64, device=device)
    regionForwardBatch: list[Any] = [
        torch.stack(
            [homogeneousBasisTorch(harmonics, epsIncident, direction=1, torch=torch, device=device) for harmonics in harmonicsList],
            dim=0,
        )
    ]
    regionBackwardBatch: list[Any] = [
        torch.stack(
            [homogeneousBasisTorch(harmonics, epsIncident, direction=-1, torch=torch, device=device) for harmonics in harmonicsList],
            dim=0,
        )
    ]
    layerModesBatch: list[tuple[Any, Any]] = []
    for layer in layers:
        qAll, modeMatrix = layerModesBatchTorch(layer, harmonicsList, torch, device, k0Values=k0Values)
        layerModesBatch.append((qAll, modeMatrix))
        regionForwardBatch.append(modeMatrix[:, :, :nPorts])
        regionBackwardBatch.append(modeMatrix[:, :, nPorts:])
    regionForwardBatch.append(
        torch.stack(
            [
                homogeneousBasisTorch(harmonics, epsTransmission, direction=1, torch=torch, device=device)
                for harmonics in harmonicsList
            ],
            dim=0,
        )
    )
    regionBackwardBatch.append(
        torch.stack(
            [
                homogeneousBasisTorch(harmonics, epsTransmission, direction=-1, torch=torch, device=device)
                for harmonics in harmonicsList
            ],
            dim=0,
        )
    )

    interfaces = interfaceSMatricesBatchTorch(regionForwardBatch, regionBackwardBatch, torch)
    batchSize = int(wavelengths.size)
    reflection = torch.zeros((batchSize, nPorts, nPorts), dtype=torch.complex128, device=device)
    forwardOperators: list[Any | None] = []
    identity = torch.eye(nPorts, dtype=torch.complex128, device=device).expand(batchSize, nPorts, nPorts)
    reversedOperators: list[Any | None] = []
    for componentIndex in range(2 * len(layers), -1, -1):
        if componentIndex % 2 == 0:
            component = interfaces[componentIndex // 2]
            internalReflection = torch.linalg.solve(
                identity - reflection @ component.s22,
                reflection @ component.s21,
            )
            forward = component.s21 + component.s22 @ internalReflection
            reversedOperators.append(forward)
            reflection = component.s11 + component.s12 @ internalReflection
        else:
            layerIndex = componentIndex // 2
            qForward = layerModesBatch[layerIndex][0][:, :nPorts]
            propagation = torch.exp(1j * qForward * k0Values[:, None] * layers[layerIndex].thickness)
            reflection = propagation[:, :, None] * reflection * propagation[:, None, :]
            reversedOperators.append(propagation)
    forwardOperators = list(reversed(reversedOperators))

    incidentForwardNp = regionForwardBatch[0].detach().cpu().numpy()
    incidentBackwardNp = regionBackwardBatch[0].detach().cpu().numpy()
    transmissionForwardNp = regionForwardBatch[-1].detach().cpu().numpy()
    zeroIndex = zeroOrderIndex(reference)
    results: dict[str, tuple[ComplexArray, ComplexArray]] = {}
    for label, (sAmplitude, pAmplitude) in excitations.items():
        incidentColumn = torch.zeros((batchSize, nPorts, 1), dtype=torch.complex128, device=device)
        incidentColumn[:, 2 * zeroIndex, 0] = complex(sAmplitude)
        incidentColumn[:, 2 * zeroIndex + 1, 0] = complex(pAmplitude)
        reflected = reflection @ incidentColumn
        transmitted = incidentColumn
        for forward in forwardOperators:
            if forward is None:
                continue
            if forward.ndim == 2:
                transmitted = forward[:, :, None] * transmitted
            else:
                transmitted = forward @ transmitted
        reflectedNp = reflected[:, :, 0].detach().cpu().numpy()
        transmittedNp = transmitted[:, :, 0].detach().cpu().numpy()
        rValues = np.empty(batchSize, dtype=float)
        tValues = np.empty(batchSize, dtype=float)
        incidentNp = incidentAmplitudesNumpy(nPorts, zeroIndex, sAmplitude, pAmplitude)
        for index in range(batchSize):
            incidentFlux = flux(incidentForwardNp[index] @ incidentNp)
            if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
                raise ValueError("incident field has near-zero real power flux")
            reflectedFlux = orderFluxesFromLocalBasis(incidentBackwardNp[index], reflectedNp[index])
            transmittedFlux = orderFluxesFromLocalBasis(transmissionForwardNp[index], transmittedNp[index])
            rValues[index] = float(np.sum(-reflectedFlux / incidentFlux))
            tValues[index] = float(np.sum(transmittedFlux / incidentFlux))
        results[label] = (rValues, tValues)
    return results


def automaticOrderReductionPlan(
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
) -> OrderReductionPlan | None:
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
    if hasMismatchedCompiledLayer(layers, fullHarmonics):
        return None
    if all(isHomogeneousLayer(layer) for layer in layers):
        return makeOrderReductionPlan("homogeneous", (0, 0), fullHarmonics)

    nx, ny = normalizedOrders
    if ny > 0 and stackInvariantAlong("y", layers, fullHarmonics):
        return makeOrderReductionPlan("1d-x", (nx, 0), fullHarmonics)
    if nx > 0 and stackInvariantAlong("x", layers, fullHarmonics):
        return makeOrderReductionPlan("1d-y", (0, ny), fullHarmonics)
    return None


def makeOrderReductionPlan(
    label: str,
    reducedOrders: tuple[int, int],
    fullHarmonics: Harmonics,
) -> OrderReductionPlan:
    return OrderReductionPlan(
        label=label,
        reducedOrders=reducedOrders,
        fullHarmonics=fullHarmonics,
        keptIndices=reducedHarmonicIndices(fullHarmonics, reducedOrders),
    )


def solveStackReducedTorch(
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
    plan: OrderReductionPlan,
    profile: bool = False,
) -> RCWAResult:
    reducedLayerSet = reducedLayers(layers, plan)
    prepared = prepareStackTorch(
        layers=reducedLayerSet,
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
        solvedBy=f"smatrix-{backendSuffix(backend)}",
        returnFields=False,
    )
    return embedReducedResult(
        reduced,
        fullHarmonics=plan.fullHarmonics,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        solvedBy=f"smatrix-{plan.label}-{backendSuffix(backend)}",
    )


def solveBatchReducedTorch(
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
    plan: OrderReductionPlan,
) -> dict[str, RCWAResult]:
    reducedLayerSet = reducedLayers(layers, plan)
    prepared = prepareStackTorch(
        layers=reducedLayerSet,
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
        solvedBy=f"smatrix-batch-{backendSuffix(backend)}",
    )
    return {
        label: embedReducedResult(
            result,
            fullHarmonics=plan.fullHarmonics,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            sAmplitude=excitations[label][0],
            pAmplitude=excitations[label][1],
            solvedBy=f"smatrix-batch-{plan.label}-{backendSuffix(backend)}",
        )
        for label, result in reducedResults.items()
    }


def solveBatchReducedPowersTorch(
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
    plan: OrderReductionPlan,
) -> dict[str, tuple[float, float]]:
    reducedLayerSet = reducedLayers(layers, plan)
    prepared = prepareStackPowersTorch(
        layers=reducedLayerSet,
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


def reducedLayers(
    layers: Sequence[Layer | CompiledLayer],
    plan: OrderReductionPlan,
) -> tuple[Layer | CompiledLayer, ...]:
    reduced: list[Layer | CompiledLayer] = []
    for layer in layers:
        if isHomogeneousLayer(layer):
            reduced.append(homogeneousEquivalentLayer(layer))
        elif isCompiledLayer(layer):
            reduced.append(sliceCompiledLayer(layer, plan))
        else:
            reduced.append(layer)
    return tuple(reduced)


def homogeneousEquivalentLayer(layer: Layer | CompiledLayer) -> Layer:
    if isCompiledLayer(layer):
        if getattr(layer, "homogeneousEpsilon", None) is None:
            raise RuntimeError("compiled layer is not homogeneous")
        return Layer(thickness=layer.thickness, epsilon=getattr(layer, "homogeneousEpsilon"), name=getattr(layer, "name", ""))
    epsilon = homogeneousEpsilon(getattr(layer, "epsilon"))
    if epsilon is None:
        raise RuntimeError("layer is not homogeneous")
    return Layer(thickness=layer.thickness, epsilon=epsilon, name=getattr(layer, "name", ""))


def sliceCompiledLayer(layer: CompiledLayer, plan: OrderReductionPlan) -> CompiledLayer:
    displacement = getattr(layer, "displacementMatrices", None)
    if isTorchTensor(getattr(layer, "epsilonMatrix")):
        index = torchIndex(plan.keptIndices, getattr(layer, "epsilonMatrix"))
        return type(layer)(
            thickness=getattr(layer, "thickness"),
            epsilonMatrix=torchSliceSquare(getattr(layer, "epsilonMatrix"), index),
            epsilonInverse=torchSliceSquare(getattr(layer, "epsilonInverse"), index),
            orders=plan.reducedOrders,
            truncation=plan.fullHarmonics.truncation,
            name=getattr(layer, "name", ""),
            displacementMatrices=None
            if displacement is None
            else tuple(torchSliceSquare(matrix, index) for matrix in displacement),
            factorization=getattr(layer, "factorization", "standard"),
            homogeneousEpsilon=getattr(layer, "homogeneousEpsilon", None),
            sampleShape=getattr(layer, "sampleShape", None),
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
        sampleShape=getattr(layer, "sampleShape", None),
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

    incidentBackward = homogeneousBasis(fullHarmonics, epsIncident, direction=-1)
    transmissionForward = homogeneousBasis(fullHarmonics, epsTransmission, direction=1)
    incident = incidentField(fullHarmonics, epsIncident, sAmplitude, pAmplitude)
    incidentFlux = checkedFlux(incident)
    reflection = float(-flux(incidentBackward @ rAmplitudes) / incidentFlux)
    transmission = float(flux(transmissionForward @ tAmplitudes) / incidentFlux)
    absorption = float(1.0 - reflection - transmission)
    energyError = None if reduced.energyError is None else abs(reflection + transmission - 1.0)
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
        absorption=absorption,
        energyError=energyError,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=tuple(orders),
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=(),
        layerEigTimings=reduced.layerEigTimings,
        stackTiming=reduced.stackTiming,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        powerWarning=reduced.powerWarning,
        diagnostics=tuple(reduced.diagnostics),
    )


def reducedHarmonicIndices(harmonics: Harmonics, reducedOrders: tuple[int, int]) -> ComplexArray:
    nx, ny = reducedOrders
    mask = (np.abs(harmonics.mx) <= nx) & (np.abs(harmonics.my) <= ny)
    return np.flatnonzero(mask)


def hasMismatchedCompiledLayer(layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return any(
        isCompiledLayer(layer)
        and (getattr(layer, "orders") != harmonics.orders or getattr(layer, "truncation") != harmonics.truncation)
        for layer in layers
    )


def stackInvariantAlong(axis: str, layers: Sequence[Layer | CompiledLayer], harmonics: Harmonics) -> bool:
    return all(layerInvariantAlong(axis, layer, harmonics) for layer in layers)


def layerInvariantAlong(axis: str, layer: Layer | CompiledLayer, harmonics: Harmonics) -> bool:
    if isHomogeneousLayer(layer):
        return True
    if isCompiledLayer(layer):
        return compiledLayerInvariantAlong(axis, layer, harmonics)
    return rawLayerInvariantAlong(axis, layer)


def rawLayerInvariantAlong(axis: str, layer: Layer) -> bool:
    epsilon = getattr(layer, "epsilon")
    if homogeneousEpsilon(epsilon) is not None:
        return True
    if hasattr(epsilon, "invariantAxes"):
        epsilonInvariant = axis in epsilon.invariantAxes()
    elif hasattr(epsilon, "convolutionMatrix"):
        epsilonInvariant = False
    else:
        epsilonInvariant = arrayInvariantAlong(axis, np.asarray(epsilon))
    if not epsilonInvariant:
        return False

    normal = getattr(layer, "normalField", None)
    return normal is None or arrayInvariantAlong(axis, np.asarray(normal))


def arrayInvariantAlong(axis: str, array: ComplexArray) -> bool:
    if array.ndim < 2:
        return True
    if axis == "y":
        return bool(np.allclose(array, array[:1, ...], rtol=1e-10, atol=1e-12))
    if axis == "x":
        return bool(np.allclose(array, array[:, :1, ...], rtol=1e-10, atol=1e-12))
    raise ValueError("axis must be 'x' or 'y'")


def compiledLayerInvariantAlong(axis: str, layer: CompiledLayer, harmonics: Harmonics) -> bool:
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
    scale = max(1.0, max(matrixMaxAbs(matrix) for matrix in matrices if matrixSize(matrix)))
    tolerance = 1e-10 * scale
    return all(matrixMaskedMaxAbs(matrix, uncoupled) <= tolerance for matrix in matrices)


def isHomogeneousLayer(layer: Layer | CompiledLayer) -> bool:
    if isCompiledLayer(layer):
        return getattr(layer, "homogeneousEpsilon", None) is not None
    return homogeneousEpsilon(getattr(layer, "epsilon")) is not None


def isCompiledLayer(layer: object) -> bool:
    return hasattr(layer, "epsilonMatrix") and hasattr(layer, "epsilonInverse")


def isTorchTensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device")


def torchIndex(indices: ComplexArray, reference: Any) -> Any:
    import torch as torch_module

    return torch_module.as_tensor(indices, dtype=torch_module.long, device=reference.device)


def torchSliceSquare(matrix: Any, index: Any) -> Any:
    return matrix.index_select(0, index).index_select(1, index).clone()


def matrixSize(matrix: object) -> int:
    return int(matrix.numel()) if isTorchTensor(matrix) else int(np.asarray(matrix).size)


def matrixMaxAbs(matrix: object) -> float:
    if isTorchTensor(matrix):
        if matrix.numel() == 0:
            return 0.0
        return float(matrix.abs().amax().detach().cpu().item())
    array = np.asarray(matrix)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def matrixMaskedMaxAbs(matrix: object, mask: ComplexArray) -> float:
    if isTorchTensor(matrix):
        import torch

        torchMask = torch.as_tensor(mask, dtype=torch.bool, device=matrix.device)
        if not bool(torch.any(torchMask).item()):
            return 0.0
        return float(matrix[torchMask].abs().amax().detach().cpu().item())
    values = np.asarray(matrix)[mask]
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


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
    reflectedFluxes = orderFluxesFromLocalBasis(reflectedBasis, rAmplitudes)
    transmittedFluxes = orderFluxesFromLocalBasis(transmittedBasis, tAmplitudes)
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


def orderFluxesFromLocalBasis(basis: ComplexArray, amplitudes: ComplexArray) -> ComplexArray:
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


def checkedFlux(field: ComplexArray) -> float:
    incidentFlux = flux(field)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def layerModesWithTimingTorch(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
    *,
    layerIndex: int,
    profile: bool,
    k0: float,
) -> tuple[tuple[Any, Any], LayerEigTiming | None]:
    return layerModesCoreTorch(
        layer,
        harmonics,
        torch,
        device,
        layerIndex=layerIndex,
        profile=profile,
        k0=k0,
    )


def layerModesCoreTorch(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
    *,
    layerIndex: int,
    profile: bool,
    k0: float,
) -> tuple[tuple[Any, Any], LayerEigTiming | None]:
    if profile:
        torch.cuda.synchronize(device)
    totalStart = time.perf_counter()
    factorizationStart = totalStart
    factorized = layerDataForTorch(layer, harmonics, torch, device)
    if profile:
        torch.cuda.synchronize(device)
    factorizationTime = time.perf_counter() - factorizationStart if profile else 0.0
    if factorized.homogeneousEpsilon is not None and factorized.displacementMatrices is None:
        modes = homogeneousScalarLayerModesTorch(harmonics, factorized.homogeneousEpsilon, torch, device)
        timing = None
        if profile:
            qStats = qStabilityStatsTorch(modes[0], None, harmonics, torch, k0)
            timing = LayerEigTiming(
                layerIndex=layerIndex,
                name=getattr(layer, "name", ""),
                kind="homogeneous-analytic",
                matrixShape=(4, 4),
                eigTimeSeconds=0.0,
                factorizationTimeSeconds=factorizationTime,
                totalTimeSeconds=time.perf_counter() - totalStart,
                minAbsQ=qStats[0],
                safeQThreshold=qStats[1],
                nearZeroModeCount=qStats[2],
            )
        return modes, timing

    epsilonMatrix = toTorchComplex(factorized.epsilonMatrix, torch, device)
    if profile:
        torch.cuda.synchronize(device)
    inverseStart = time.perf_counter()
    if factorized.epsilonInverse is None:
        epsilonInverse = torch.linalg.solve(
            epsilonMatrix,
            torch.eye(epsilonMatrix.shape[0], dtype=torch.complex128, device=device),
        )
    else:
        epsilonInverse = toTorchComplex(factorized.epsilonInverse, torch, device)
    displacementMatrices = None
    if factorized.displacementMatrices is not None:
        displacementMatrices = tuple(toTorchComplex(matrix, torch, device) for matrix in factorized.displacementMatrices)
    if profile:
        torch.cuda.synchronize(device)
    inverseTime = time.perf_counter() - inverseStart if profile else 0.0

    if profile:
        torch.cuda.synchronize(device)
    pqStart = time.perf_counter()
    pMatrix, qMatrix = pqMatricesTorch(epsilonMatrix, harmonics, epsilonInverse, displacementMatrices, torch, device)
    eigenMatrix = pMatrix @ qMatrix
    if profile:
        torch.cuda.synchronize(device)
    pqTime = time.perf_counter() - pqStart if profile else 0.0
    if profile:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        qSquared, electricModes = torch.linalg.eig(eigenMatrix)
        torch.cuda.synchronize(device)
        eigTime = time.perf_counter() - start
    else:
        qSquared, electricModes = torch.linalg.eig(eigenMatrix)
        eigTime = 0.0
    qValues = forwardKzTorch(qSquared, torch)
    safeQ = qValues.clone()
    safeQThreshold = safeQThresholdTorch(qValues, eigenMatrix, harmonics, torch, k0)
    safeQ[torch.abs(safeQ) < safeQThreshold] = safeQThreshold + 0j

    magneticModes = qMatrix @ electricModes @ torch.diag(1.0 / safeQ)
    vectors = torch.cat(
        [
            torch.cat([electricModes, electricModes], dim=1),
            torch.cat([magneticModes, -magneticModes], dim=1),
        ],
        dim=0,
    )
    qAll = torch.cat([qValues, -qValues], dim=0)
    modes = qAll, normalizeColumnsTorch(vectors, torch)
    timing = None
    if profile:
        minAbsQ, threshold, nearZeroCount = qStabilityStatsTorch(qValues, eigenMatrix, harmonics, torch, k0)
        timing = LayerEigTiming(
            layerIndex=layerIndex,
            name=getattr(layer, "name", ""),
            kind=factorized.factorization,
            matrixShape=tuple(eigenMatrix.shape),
            eigTimeSeconds=eigTime,
            factorizationTimeSeconds=factorizationTime,
            inverseTimeSeconds=inverseTime,
            pqTimeSeconds=pqTime,
            totalTimeSeconds=time.perf_counter() - totalStart,
            minAbsQ=minAbsQ,
            safeQThreshold=safeQThreshold,
            nearZeroModeCount=nearZeroCount,
        )
    return modes, timing


def layerModesBatchTorch(
    layer: Layer | CompiledLayer,
    harmonicsList: Sequence[Harmonics],
    torch: Any,
    device: Any,
    *,
    k0Values: Any,
) -> tuple[Any, Any]:
    first = layerDataForTorch(layer, harmonicsList[0], torch, device)
    if first.homogeneousEpsilon is not None and first.displacementMatrices is None:
        modes = [homogeneousScalarLayerModesTorch(harmonics, first.homogeneousEpsilon, torch, device) for harmonics in harmonicsList]
        return torch.stack([item[0] for item in modes], dim=0), torch.stack([item[1] for item in modes], dim=0)

    epsilonMatrices = []
    epsilonInverses = []
    displacementByLayer = []
    for harmonics in harmonicsList:
        factorized = layerDataForTorch(layer, harmonics, torch, device)
        epsilonMatrix = toTorchComplex(factorized.epsilonMatrix, torch, device)
        epsilonMatrices.append(epsilonMatrix)
        if factorized.epsilonInverse is None:
            epsilonInverses.append(
                torch.linalg.solve(
                    epsilonMatrix,
                    torch.eye(epsilonMatrix.shape[0], dtype=torch.complex128, device=device),
                )
            )
        else:
            epsilonInverses.append(toTorchComplex(factorized.epsilonInverse, torch, device))
        if factorized.displacementMatrices is None:
            displacementByLayer.append(None)
        else:
            displacementByLayer.append(
                tuple(toTorchComplex(matrix, torch, device) for matrix in factorized.displacementMatrices)
            )

    pMatrices = []
    qMatrices = []
    for harmonics, epsilonMatrix, epsilonInverse, displacementMatrices in zip(
        harmonicsList,
        epsilonMatrices,
        epsilonInverses,
        displacementByLayer,
    ):
        pMatrix, qMatrix = pqMatricesTorch(epsilonMatrix, harmonics, epsilonInverse, displacementMatrices, torch, device)
        pMatrices.append(pMatrix)
        qMatrices.append(qMatrix)

    qMatrixBatch = torch.stack(qMatrices, dim=0)
    eigenMatrix = torch.stack(pMatrices, dim=0) @ qMatrixBatch
    qSquared, electricModes = torch.linalg.eig(eigenMatrix)
    qValues = forwardKzTorch(qSquared, torch)
    safeQ = qValues.clone()
    safeQThreshold = safeQThresholdBatchTorch(qValues, eigenMatrix, harmonicsList, torch, k0Values)
    safeQ = torch.where(torch.abs(safeQ) < safeQThreshold[:, None], safeQThreshold[:, None].to(torch.complex128), safeQ)
    magneticModes = torch.matmul(qMatrixBatch, electricModes) * (1.0 / safeQ)[:, None, :]
    vectors = torch.cat(
        [
            torch.cat([electricModes, electricModes], dim=2),
            torch.cat([magneticModes, -magneticModes], dim=2),
        ],
        dim=1,
    )
    qAll = torch.cat([qValues, -qValues], dim=1)
    return qAll, normalizeColumnsBatchTorch(vectors, torch)


def homogeneousBasisTorch(harmonics: Harmonics, eps: complex, direction: int, torch: Any, device: Any) -> Any:
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    n = harmonics.count
    basis = torch.zeros((4 * n, 2 * n), dtype=torch.complex128, device=device)
    kxValues = toTorchComplex(harmonics.kx, torch, device)
    kyValues = toTorchComplex(harmonics.ky, torch, device)
    kzForward = forwardKzTorch(complex(eps) - kxValues * kxValues - kyValues * kyValues, torch)
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


def homogeneousScalarLayerModesTorch(harmonics: Harmonics, epsilon: complex, torch: Any, device: Any) -> tuple[Any, Any]:
    forward = homogeneousBasisTorch(harmonics, epsilon, direction=1, torch=torch, device=device)
    backward = homogeneousBasisTorch(harmonics, epsilon, direction=-1, torch=torch, device=device)
    kx = toTorchComplex(harmonics.kx, torch, device)
    ky = toTorchComplex(harmonics.ky, torch, device)
    kzForward = forwardKzTorch(complex(epsilon) - kx * kx - ky * ky, torch)
    qForward = torch.empty(2 * harmonics.count, dtype=torch.complex128, device=device)
    qForward[0::2] = kzForward
    qForward[1::2] = kzForward
    return torch.cat([qForward, -qForward]), torch.cat([forward, backward], dim=1)


def interfaceSMatricesTorch(
    regionForward: Sequence[Any],
    regionBackward: Sequence[Any],
    torch: Any,
) -> tuple[TorchSMatrix, ...]:
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
        TorchSMatrix(
            s11=solved[:size, :size],
            s12=solved[:size, size:],
            s21=solved[size:, :size],
            s22=solved[size:, size:],
        )
        for solved in solvedBatch
    )


def interfaceConditionNumbersTorch(
    regionForward: Sequence[Any],
    regionBackward: Sequence[Any],
    torch: Any,
) -> tuple[float, ...]:
    matrices = []
    for index in range(len(regionForward) - 1):
        matrices.append(torch.cat([regionBackward[index], -regionForward[index + 1]], dim=1))
    if not matrices:
        return ()
    values = torch.linalg.cond(torch.stack(matrices, dim=0))
    return tuple(float(value.detach().cpu().item()) for value in values)


def interfaceSMatricesBatchTorch(
    regionForward: Sequence[Any],
    regionBackward: Sequence[Any],
    torch: Any,
) -> tuple[TorchSMatrix, ...]:
    size = regionForward[0].shape[2]
    matrices = []
    rightHandSides = []
    for index in range(len(regionForward) - 1):
        matrices.append(torch.cat([regionBackward[index], -regionForward[index + 1]], dim=2))
        rightHandSides.append(torch.cat([-regionForward[index], regionBackward[index + 1]], dim=2))

    if not matrices:
        return ()

    solvedBatch = torch.linalg.solve(torch.cat(matrices, dim=0), torch.cat(rightHandSides, dim=0))
    batchSize = regionForward[0].shape[0]
    solvedBatch = solvedBatch.reshape(len(matrices), batchSize, 2 * size, 2 * size)
    return tuple(
        TorchSMatrix(
            s11=solved[:, :size, :size],
            s12=solved[:, :size, size:],
            s21=solved[:, size:, :size],
            s22=solved[:, size:, size:],
        )
        for solved in solvedBatch
    )


def propagationSMatrixTorch(propagation: Any, torch: Any) -> TorchSMatrix:
    zero = torch.zeros_like(propagation)
    return TorchSMatrix(s11=zero, s12=propagation, s21=propagation, s22=zero, isPropagation=True)


def identitySMatrixTorch(size: int, torch: Any, device: Any) -> TorchSMatrix:
    zero = torch.zeros((size, size), dtype=torch.complex128, device=device)
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    return TorchSMatrix(s11=zero, s12=identity, s21=identity, s22=zero, isIdentity=True)


def redhefferStarTorch(left: TorchSMatrix, right: TorchSMatrix, torch: Any, device: Any) -> TorchSMatrix:
    if left.isIdentity:
        return right
    if right.isIdentity:
        return left
    if right.isPropagation:
        propagation = torch.diagonal(right.s21)
        return TorchSMatrix(
            s11=left.s11,
            s12=left.s12 * propagation[None, :],
            s21=propagation[:, None] * left.s21,
            s22=propagation[:, None] * left.s22 * propagation[None, :],
        )
    if left.isPropagation:
        propagation = torch.diagonal(left.s21)
        return TorchSMatrix(
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
    return TorchSMatrix(s11=s11, s12=s12, s21=s21, s22=s22)


def reflectionTransmissionOnlySMatrixTorch(
    components: Sequence[TorchSMatrix],
    size: int,
    torch: Any,
    device: Any,
) -> TorchSMatrix:
    reflection, transmission = enhancedReflectionTransmissionTorch(components, size, torch, device)
    zero = torch.zeros_like(reflection)
    return TorchSMatrix(s11=reflection, s12=zero.clone(), s21=transmission, s22=zero.clone())


def enhancedReflectionTransmissionTorch(
    components: Sequence[TorchSMatrix],
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


def reflectionAndForwardOperatorsTorch(
    components: Sequence[TorchSMatrix],
    size: int,
    torch: Any,
    device: Any,
) -> tuple[Any, tuple[Any | None, ...]]:
    identity = torch.eye(size, dtype=torch.complex128, device=device)
    reflection = torch.zeros((size, size), dtype=torch.complex128, device=device)
    forwardOperators: list[Any | None] = [None] * len(components)
    for index in range(len(components) - 1, -1, -1):
        component = components[index]
        if component.isIdentity:
            continue
        if component.isPropagation:
            propagation = torch.diagonal(component.s21)
            reflection = propagation[:, None] * reflection * propagation[None, :]
            forwardOperators[index] = propagation
            continue
        internalReflection = torch.linalg.solve(
            identity - reflection @ component.s22,
            reflection @ component.s21,
        )
        forward = component.s21 + component.s22 @ internalReflection
        forwardOperators[index] = forward
        reflection = component.s11 + component.s12 @ internalReflection

    return reflection, tuple(forwardOperators)


def applyForwardOperatorsTorch(forwardOperators: Sequence[Any | None], incidentColumns: Any) -> Any:
    transmissionColumns = incidentColumns
    for forward in forwardOperators:
        if forward is None:
            continue
        if forward.ndim == 1:
            transmissionColumns = forward[:, None] * transmissionColumns
        else:
            transmissionColumns = forward @ transmissionColumns
    return transmissionColumns


def prefixSMatricesTorch(components: Sequence[TorchSMatrix], size: int, torch: Any, device: Any) -> list[TorchSMatrix]:
    prefixes = [identitySMatrixTorch(size, torch, device)]
    current = prefixes[0]
    for component in components:
        current = redhefferStarTorch(current, component, torch, device)
        prefixes.append(current)
    return prefixes


def suffixSMatricesTorch(components: Sequence[TorchSMatrix], size: int, torch: Any, device: Any) -> list[TorchSMatrix]:
    suffixes = [identitySMatrixTorch(size, torch, device) for ignored in range(len(components) + 1)]
    current = identitySMatrixTorch(size, torch, device)
    suffixes[len(components)] = current
    for index in range(len(components) - 1, -1, -1):
        current = redhefferStarTorch(components[index], current, torch, device)
        suffixes[index] = current
    return suffixes


def forwardKzTorch(values: Any, torch: Any) -> Any:
    roots = torch.sqrt(values + 0j)
    flip = (roots.imag < -1e-14) | ((torch.abs(roots.imag) <= 1e-14) & (roots.real < 0))
    return torch.where(flip, -roots, roots)


def safeQThresholdTorch(qValues: Any, eigenMatrix: Any | None, harmonics: Harmonics, torch: Any, k0: float) -> float:
    if eigenMatrix is None:
        matrixScale = 1.0
    else:
        matrixScale = float(torch.linalg.matrix_norm(eigenMatrix, ord="fro").detach().cpu().item())
        matrixScale = float(np.sqrt(max(matrixScale, 0.0)))
    lateralScale = max(1.0, float(np.max(np.abs(harmonics.kx))) if harmonics.count else 1.0)
    lateralScale = max(lateralScale, float(np.max(np.abs(harmonics.ky))) if harmonics.count else 1.0)
    scale = max(1.0, matrixScale, lateralScale, abs(float(k0)))
    threshold = Q_RELATIVE_TOLERANCE * max(1, qValues.numel()) * scale
    return float(max(1e-15, min(1e-8 * scale, threshold)))


def safeQThresholdBatchTorch(
    qValues: Any,
    eigenMatrix: Any,
    harmonicsList: Sequence[Harmonics],
    torch: Any,
    k0Values: Any,
) -> Any:
    matrixScale = torch.sqrt(torch.clamp(torch.linalg.matrix_norm(eigenMatrix, ord="fro", dim=(-2, -1)), min=0.0))
    lateralScales = []
    for harmonics in harmonicsList:
        lateral = max(1.0, float(np.max(np.abs(harmonics.kx))) if harmonics.count else 1.0)
        lateral = max(lateral, float(np.max(np.abs(harmonics.ky))) if harmonics.count else 1.0)
        lateralScales.append(lateral)
    lateralScale = torch.as_tensor(lateralScales, dtype=torch.float64, device=qValues.device)
    scale = torch.maximum(torch.maximum(matrixScale, lateralScale), torch.abs(k0Values))
    scale = torch.maximum(scale, torch.ones_like(matrixScale))
    threshold = Q_RELATIVE_TOLERANCE * max(1, qValues.shape[-1]) * scale
    return torch.maximum(
        torch.full_like(threshold, 1e-15),
        torch.minimum(1e-8 * scale, threshold),
    )


def qStabilityStatsTorch(
    qValues: Any,
    eigenMatrix: Any | None,
    harmonics: Harmonics,
    torch: Any,
    k0: float,
) -> tuple[float, float, int]:
    threshold = safeQThresholdTorch(qValues, eigenMatrix, harmonics, torch, k0)
    absQ = torch.abs(qValues)
    minAbsQ = float(torch.amin(absQ).detach().cpu().item()) if qValues.numel() else float("inf")
    nearZeroCount = int(torch.count_nonzero(absQ < threshold).detach().cpu().item())
    return minAbsQ, threshold, nearZeroCount


def stabilityWarnings(
    layerTimings: Sequence[LayerEigTiming],
    interfaceConditionNumbers: Sequence[float],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for timing in layerTimings:
        if timing.nearZeroModeCount > 0:
            warnings.append(
                f"layer {timing.layerIndex} has {timing.nearZeroModeCount} near-grazing modes "
                f"(min |q|={timing.minAbsQ:.3e}, threshold={timing.safeQThreshold:.3e}); "
                "check order and wavelength sampling near Wood anomalies"
            )
    if interfaceConditionNumbers:
        maximum = max(interfaceConditionNumbers)
        if maximum > INTERFACE_CONDITION_WARNING:
            warnings.append(
                f"max interface condition number is {maximum:.3e}; consider increasing orders, "
                "refining wavelength sampling, or comparing a nearby wavelength"
            )
    return tuple(warnings)


def normalizeColumnsTorch(vectors: Any, torch: Any) -> Any:
    scales = torch.amax(torch.abs(vectors), dim=0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return vectors / scales


def normalizeColumnsBatchTorch(vectors: Any, torch: Any) -> Any:
    scales = torch.amax(torch.abs(vectors), dim=1, keepdim=True)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return vectors / scales


def incidentAmplitudesTorch(prepared: PreparedTorchStack, sAmplitude: complex, pAmplitude: complex, torch: Any) -> Any:
    amplitudes = torch.zeros(prepared.nPorts, dtype=torch.complex128, device=prepared.total.s11.device)
    amplitudes[2 * prepared.zeroIndex] = complex(sAmplitude)
    amplitudes[2 * prepared.zeroIndex + 1] = complex(pAmplitude)
    return amplitudes


def incidentAmplitudesNumpy(
    nPorts: int,
    zeroIndex: int,
    sAmplitude: complex,
    pAmplitude: complex,
) -> ComplexArray:
    amplitudes = np.zeros(nPorts, dtype=complex)
    amplitudes[2 * zeroIndex] = complex(sAmplitude)
    amplitudes[2 * zeroIndex + 1] = complex(pAmplitude)
    return amplitudes


def checkedIncidentFluxTorch(prepared: PreparedTorchStack, incidentAmplitudes: ComplexArray) -> float:
    incidentFieldValue = prepared.incidentForward @ incidentAmplitudes
    incidentFlux = flux(incidentFieldValue)
    if not np.isfinite(incidentFlux) or abs(incidentFlux) < 1e-14:
        raise ValueError("incident field has near-zero real power flux")
    return incidentFlux


def incidentFluxFromAmplitudes(prepared: PreparedTorchStack, sAmplitude: complex, pAmplitude: complex) -> float:
    incidentAmplitudes = incidentAmplitudesNumpy(prepared.nPorts, prepared.zeroIndex, sAmplitude, pAmplitude)
    return checkedIncidentFluxTorch(prepared, incidentAmplitudes)


def layerSolutionsTorch(prepared: PreparedTorchStack, incidentAmplitudes: Any) -> tuple[LayerFieldSolution, ...]:
    torch = prepared.backend.xp
    device = prepared.total.s11.device
    prefixes = prefixSMatricesTorch(prepared.components, prepared.nPorts, torch, device)
    suffixes = suffixSMatricesTorch(prepared.components, prepared.nPorts, torch, device)
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
                epsilonInverse=layerEpsilonInverseNumpy(prepared, layer),
                backwardCoefficientsRight=prepared.backend.asnumpy(backwardAtRight).copy(),
            )
        )
    return tuple(solutions)


def layerEpsilonInverseNumpy(prepared: PreparedTorchStack, layer: Layer | CompiledLayer) -> ComplexArray | None:
    device = prepared.total.s11.device
    factorized = layerDataForTorch(layer, prepared.harmonics, prepared.backend.xp, device)
    epsilonInverse = factorized.epsilonInverse
    if epsilonInverse is None:
        epsilonInverse = solveIdentityTorch(factorized.epsilonMatrix, prepared.backend.xp, device)
    return prepared.backend.asnumpy(epsilonInverse).copy()


def resultFromTorchAmplitudes(
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
    absorption, energyError, powerWarning, diagnostics = powerDiagnostics(
        layers=prepared.layers,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        reflection=reflection,
        transmission=transmission,
        stackTiming=prepared.stackTiming,
    )
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
        absorption=absorption,
        energyError=energyError,
        rAmplitudes=rAmplitudes,
        tAmplitudes=tAmplitudes,
        orders=tuple(orders),
        incidentFlux=float(incidentFlux),
        solvedBy=solvedBy,
        layerSolutions=layerSolutions,
        layerEigTimings=prepared.layerEigTimings,
        stackTiming=prepared.stackTiming,
        epsIncident=prepared.epsIncident,
        epsTransmission=prepared.epsTransmission,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        powerWarning=powerWarning,
        diagnostics=diagnostics,
    )


def powerDiagnostics(
    *,
    layers: Sequence[Layer | CompiledLayer],
    epsIncident: complex,
    epsTransmission: complex,
    reflection: float,
    transmission: float,
    stackTiming: StackTiming | None,
) -> tuple[float, float | None, str | None, tuple[str, ...]]:
    absorption = float(1.0 - reflection - transmission)
    finiteLayersLossless = all(layerLossless(layer) for layer in layers)
    exteriorLossless = epsilonLossless(epsIncident) and epsilonLossless(epsTransmission)
    energyError = abs(reflection + transmission - 1.0) if finiteLayersLossless and exteriorLossless else None
    powerWarning = None
    diagnostics: list[str] = []
    if not exteriorLossless:
        powerWarning = (
            "incident or transmission half-space has complex permittivity; diffraction power "
            "normalization is reported from real Poynting flux and may need interpretation"
        )
        diagnostics.append(powerWarning)
    if stackTiming is not None:
        diagnostics.extend(stackTiming.stabilityWarnings)
    return absorption, energyError, powerWarning, tuple(diagnostics)


def layerLossless(layer: Layer | CompiledLayer) -> bool:
    if getattr(layer, "homogeneousEpsilon", None) is not None:
        return epsilonLossless(getattr(layer, "homogeneousEpsilon"))
    if hasattr(layer, "epsilonMatrix"):
        diagonal = matrixDiagonalNumpy(getattr(layer, "epsilonMatrix"))
        return arrayLossless(diagonal)
    if hasattr(layer, "epsilon"):
        epsilon = getattr(layer, "epsilon")
        if hasattr(epsilon, "background") and not epsilonLossless(getattr(epsilon, "background")):
            return False
        for attr in ("inclusion", "ring", "hole"):
            if hasattr(epsilon, attr):
                value = getattr(epsilon, attr)
                if value is not None and not epsilonLossless(value):
                    return False
        if hasattr(epsilon, "terms"):
            if not epsilonLossless(getattr(epsilon, "background", 0.0)):
                return False
            for term in getattr(epsilon, "terms"):
                # Composite terms store material contrast; direct sampling below catches
                # ordinary arrays, while analytic composites expose enough constants for
                # loss detection through background + deltas.
                if not epsilonLossless(getattr(term, "delta", 0.0)):
                    return False
            return True
        homogeneous = homogeneousEpsilon(epsilon)
        if homogeneous is not None:
            return epsilonLossless(homogeneous)
        return arrayLossless(np.asarray(epsilon))
    return True


def epsilonLossless(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=complex)
    except Exception:
        return True
    return arrayLossless(array)


def arrayLossless(array: ComplexArray) -> bool:
    if array.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(array))))
    return bool(np.max(np.abs(np.imag(array))) <= LOSS_TOLERANCE * scale)


def matrixDiagonalNumpy(matrix: object) -> ComplexArray:
    if isTorchTensor(matrix):
        return matrix.detach().diagonal().cpu().numpy()
    return np.asarray(matrix).diagonal()


def validateGeometry(wavelength: float, period: tuple[float, float], orders: int | tuple[int, int]) -> None:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if period[0] <= 0 or period[1] <= 0:
        raise ValueError("period values must be positive")
    normalizedOrders = normalizeOrders(orders)
    if normalizedOrders[0] < 0 or normalizedOrders[1] < 0:
        raise ValueError("orders must be non-negative")


def validateIsotropicLayers(layers: Sequence[Layer | CompiledLayer], kind: str) -> None:
    for index, layer in enumerate(layers):
        if type(layer).__module__.startswith("rcwa3d_anisotropic"):
            raiseAnisotropicPathError(kind, index)

        if hasattr(layer, "epsilon"):
            epsilon = getattr(layer, "epsilon")
            if hasattr(epsilon, "convolutionMatrix"):
                continue
            if np.isscalar(epsilon):
                continue
            epsilonArray = np.asarray(epsilon)
            if epsilonArray.ndim not in (0, 2):
                raiseAnisotropicPathError(kind, index)


def raiseAnisotropicPathError(kind: str, layerIndex: int) -> None:
    raise TypeError(
        f"layer {layerIndex} uses an anisotropic/tensor-like permittivity in the isotropic {kind} path; "
        "use `rcwa3d_anisotropic.RCWASimulation` instead."
    )


def expandAdaptiveLayers(
    layers: Sequence[Layer | CompiledLayer | AdaptiveLayerSpec],
) -> list[Layer | CompiledLayer]:
    expanded: list[Layer | CompiledLayer] = []
    for layer in layers:
        if isinstance(layer, AdaptiveLayerSpec):
            expanded.extend(layer.toLayers())
        else:
            expanded.append(layer)
    return expanded


def backendSuffix(backend: ArrayBackend) -> str:
    if not backend.isCuda:
        raise ValueError("the isotropic solver is CUDA-only")
    return "cuda"
