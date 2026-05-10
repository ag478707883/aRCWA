from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from threading import Lock
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union
import warnings

import numpy as np

from .backend import resolveBackend
from .constitutive import splitConstitutiveInput
from .factorization import constantTensor
from .solver import (
    BatchedHomogeneousLayer,
    CompiledLayer,
    Layer,
    PreparedStack,
    RCWAResult,
    automaticFastPathPlan,
    embedReducedResult,
    reducedFastPathLayers,
    evaluatePreparedBatch,
    evaluatePreparedBatchPowers,
    evaluatePreparedFieldStack,
    evaluatePreparedSpectrumBatchPowers,
    evaluatePreparedStack,
    compileLayers,
    prepareStackSMatrix,
    prepareStackSMatrixBatch,
    warmBackendTensorCache,
)
from .fourier import normalizeOrders


ComplexArray = np.ndarray
TensorSource = Union[ComplexArray, complex, Callable[[float], ComplexArray]]
LayerInput = Any
Polarization = Union[Literal["TE", "TM", "s", "p"], tuple[complex, complex]]
ExcitationMap = Mapping[str, tuple[complex, complex]]
SpectrumParallel = Literal["auto", "serial", "thread", "process"]


@dataclass(frozen=True)
class LayerSpec:
    """A homogeneous layer whose material may depend on wavelength."""

    thickness: float
    epsilon: TensorSource
    name: str = ""
    mu: TensorSource | None = None
    chi: TensorSource | None = None
    xi: TensorSource | None = None

    def __post_init__(self) -> None:
        epsilon, mu, chi, xi = splitConstitutiveInput(self.epsilon, self.mu, self.chi, self.xi)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "xi", xi)

    @property
    def isStatic(self) -> bool:
        return not any(callable(value) for value in (self.epsilon, self.mu, self.chi, self.xi))

    def at(self, wavelength: float) -> Layer:
        epsilon = materialTensorAt(self.epsilon, wavelength, name=self.name, tensorName="permittivity")
        mu = materialTensorAt(self.mu, wavelength, name=self.name, tensorName="permeability") if self.mu is not None else None
        chi = materialTensorAt(self.chi, wavelength, name=self.name, tensorName="magnetoelectric chi") if self.chi is not None else None
        xi = materialTensorAt(self.xi, wavelength, name=self.name, tensorName="magnetoelectric xi") if self.xi is not None else None
        return Layer(thickness=self.thickness, epsilon=epsilon, name=self.name, mu=mu, chi=chi, xi=xi)


def homogeneousLayer(
    thickness: float,
    epsilon: TensorSource,
    name: str = "",
    *,
    mu: TensorSource | None = None,
    chi: TensorSource | None = None,
    xi: TensorSource | None = None,
) -> Layer | LayerSpec:
    """Return a static or wavelength-dependent homogeneous layer.

    Wavelength-dependent material callbacks are user supplied and must return
    one ``(3, 3)`` tensor for the requested wavelength.  Isotropic dispersive
    materials should return ``value * np.eye(3)``.
    """

    if any(callable(value) for value in (epsilon, mu, chi, xi)):
        return LayerSpec(thickness=thickness, epsilon=epsilon, name=name, mu=mu, chi=chi, xi=xi)
    return Layer(thickness=thickness, epsilon=epsilon, name=name, mu=mu, chi=chi, xi=xi)


def materialTensorAt(value: TensorSource, wavelength: float, *, name: str = "", tensorName: str = "material") -> object:
    if not callable(value):
        return value
    return materialTensor(value(wavelength), name=name, tensorName=tensorName)


def materialTensor(value: object, *, name: str = "", tensorName: str = "permittivity") -> ComplexArray:
    tensor = np.asarray(value, dtype=complex)
    if tensor.shape != (3, 3):
        label = f" for {name!r}" if name else ""
        raise ValueError(
            f"wavelength-dependent material{label} must return a (3, 3) {tensorName} tensor; "
            "wrap isotropic values as value * np.eye(3, dtype=complex)"
        )
    return tensor


@dataclass(frozen=True)
class SimulationConfig:
    """Shared solver settings for geometry-only example variants."""

    period: tuple[float, float]
    orders: int | tuple[int, int]
    truncation: Literal["circular", "rectangular"] = "circular"
    backend: str = "cuda"
    precision: Literal["complex128", "complex64", "mixed"] = "complex128"
    epsIncident: complex = 1.0
    epsTransmission: complex = 1.0
    precompile: bool = True
    cacheModes: bool = True
    workers: int = 1


def buildSimulation(config: SimulationConfig, layers: Sequence[LayerInput]) -> "RCWASimulation":
    """Build an anisotropic simulation from shared settings and geometry layers."""

    return RCWASimulation(
        period=config.period,
        layers=layers,
        orders=config.orders,
        truncation=config.truncation,
        backend=config.backend,
        precision=config.precision,
        epsIncident=config.epsIncident,
        epsTransmission=config.epsTransmission,
        precompile=config.precompile,
        cacheModes=config.cacheModes,
        workers=config.workers,
    )


def solveSpectrum(
    config: SimulationConfig,
    layers: Sequence[LayerInput],
    wavelengths: Iterable[float],
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarizations: Sequence[Polarization] = ("TE", "TM"),
    excitations: ExcitationMap | None = None,
    bidirectional: bool = True,
    workers: int | None = None,
) -> dict[str, dict[str, ComplexArray] | ComplexArray]:
    """Solve a spectrum while keeping geometry separate from solver settings."""

    return buildSimulation(config, layers).spectrum(
        wavelengths,
        theta=theta,
        phi=phi,
        polarizations=polarizations,
        excitations=excitations,
        bidirectional=bidirectional,
        workers=config.workers if workers is None else workers,
    )


@dataclass
class RCWASimulation:
    """Small high-level wrapper around the anisotropic RCWA solver.

    Static layers are precompiled once. Wavelength-dependent layers are built
    only at solve time. Use user-defined callables returning ``(3, 3)``
    material tensors when a homogeneous layer changes with wavelength, and
    ordinary ``Layer`` objects for fixed geometry.

    The CUDA S-matrix route is the only solver path.  CUDA is required for
    solves; ``backend="auto"`` resolves to the same PyTorch CUDA backend and
    never falls back to CPU.
    """

    period: tuple[float, float]
    layers: Sequence[LayerInput]
    orders: int | tuple[int, int] = 5
    truncation: Literal["circular", "rectangular"] = "circular"
    backend: str = "cuda"
    precision: Literal["complex128", "complex64", "mixed"] = "complex128"
    epsIncident: complex = 1.0
    epsTransmission: complex = 1.0
    precompile: bool = True
    workers: int = 1
    cacheModes: bool = True
    cacheSize: int = 10
    preparedCache: OrderedDict[tuple, PreparedStack] = field(default_factory=OrderedDict, init=False, repr=False)
    cacheLock: Lock = field(default_factory=Lock, init=False, repr=False)
    samplingWarningKeys: set[tuple[object, ...]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend = resolveBackend(self.backend, precision=self.precision).name
        self.precision = normalizePrecisionLabel(self.precision)
        self.layers = tuple(self.prepareLayer(layer) for layer in self.layers)
        if self.precompile:
            warmBackendTensorCache(self.layers, self.backend, precision=self.precision)

    def solve(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarization: Polarization = "TE",
        profile: bool = False,
    ) -> RCWAResult:
        """Solve one scattering state and return only reflection/transmission data."""

        sAmplitude, pAmplitude = polarizationAmplitudes(polarization)
        return self.solveExcitation(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            profile=profile,
        )

    def solveExcitation(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        sAmplitude: complex = 1.0,
        pAmplitude: complex = 0.0,
        profile: bool = False,
    ) -> RCWAResult:
        """Solve one custom s/p incident excitation for scattering data only."""

        layers = self.layersAt(wavelength)
        reduced = self.reducedSolve(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=layers,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            profile=profile,
        )
        if reduced is not None:
            return reduced

        prepared = self.preparedStack(
            wavelength,
            theta=theta,
            phi=phi,
            layers=layers,
            fullTotal=False,
            profile=profile,
        )
        return evaluatePreparedStack(
            prepared,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=f"smatrix-{prepared.backend}",
        )

    def solveFields(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarization: Polarization = "TE",
        profile: bool = False,
    ) -> RCWAResult:
        """Solve one field state and include finite-layer modal coefficients.

        This is intentionally separate from ``solve``/``spectrum``: field
        reconstruction needs the full cascaded S-matrix, while spectra only
        need reflection/transmission blocks.
        """

        sAmplitude, pAmplitude = polarizationAmplitudes(polarization)
        return self.solveFieldExcitation(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            profile=profile,
        )

    def solveFieldExcitation(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        sAmplitude: complex = 1.0,
        pAmplitude: complex = 0.0,
        profile: bool = False,
    ) -> RCWAResult:
        """Solve one custom s/p incident excitation with layer field data."""

        layers = self.layersAt(wavelength)
        prepared = self.preparedStack(
            wavelength,
            theta=theta,
            phi=phi,
            layers=layers,
            fullTotal=True,
            profile=profile,
        )
        return evaluatePreparedFieldStack(
            prepared,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=f"smatrix-fields-{prepared.backend}",
        )

    def solveExcitations(
        self,
        wavelength: float,
        excitations: ExcitationMap,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> dict[str, RCWAResult]:
        """Solve several s/p incident excitations while reusing one prepared stack."""

        excitationMap = {str(label): (complex(values[0]), complex(values[1])) for label, values in excitations.items()}
        if not excitationMap:
            return {}

        layers = self.layersAt(wavelength)
        reduced = self.reducedBatchSolve(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=layers,
            excitations=excitationMap,
        )
        if reduced is not None:
            return reduced

        prepared = self.preparedStack(
            wavelength,
            theta=theta,
            phi=phi,
            layers=layers,
            fullTotal=False,
        )
        return evaluatePreparedBatch(
            prepared,
            excitationMap,
            solvedBy=f"smatrix-batch-{prepared.backend}",
        )

    def absorption(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarization: Polarization = "TE",
    ) -> float:
        result = self.solve(wavelength, theta=theta, phi=phi, polarization=polarization)
        return float(np.real_if_close(1.0 - result.reflection - result.transmission))

    def bidirectionalAbsorption(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float = 0.0,
        polarization: Polarization = "TE",
    ) -> tuple[float, float]:
        forward = self.absorption(wavelength, theta=theta, phi=phi, polarization=polarization)
        backward = self.absorption(wavelength, theta=-theta, phi=phi, polarization=polarization)
        return forward, backward

    def spectrum(
        self,
        wavelengths: Iterable[float],
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarizations: Sequence[Polarization] = ("TE", "TM"),
        excitations: ExcitationMap | None = None,
        bidirectional: bool = True,
        workers: int | None = None,
        parallel: SpectrumParallel = "auto",
    ) -> dict[str, dict[str, ComplexArray] | ComplexArray]:
        values = np.asarray(tuple(wavelengths), dtype=float)
        spectra: dict[str, dict[str, ComplexArray] | ComplexArray] = {"wavelengths": values}
        excitationMap = spectrumExcitations(polarizations, excitations)
        labels = tuple(excitationMap)
        forward = {label: np.empty(values.shape, dtype=float) for label in labels}
        backward = {label: np.full(values.shape, np.nan, dtype=float) for label in labels}

        batched = self.spectrumBatchPowers(
            values,
            theta=theta,
            phi=phi,
            excitations=excitationMap,
            bidirectional=bidirectional,
        )
        if batched is not None:
            for label in labels:
                forward[label][:] = batched["forward"][label]
                if bidirectional:
                    backward[label][:] = batched["backward"][label]
            for label in labels:
                entry: dict[str, ComplexArray] = {"absorptivity": forward[label]}
                if bidirectional:
                    entry["emissivity"] = backward[label]
                    entry["nonreciprocity"] = np.abs(forward[label] - backward[label])
                spectra[label] = entry
            return spectra

        requestedWorkers = self.workers if workers is None else workers
        if requestedWorkers < 1:
            raise ValueError("workers must be at least 1")
        ignored, workerCount = spectrumParallelPlan(requestedWorkers, parallel, self.backend)
        usePreparedCache = values.size == 1

        def solvePoint(item: tuple[int, float]) -> tuple[int, dict[str, tuple[float, float]]]:
            index, wavelength = item
            return index, self.spectrumPoint(
                float(wavelength),
                theta=theta,
                phi=phi,
                excitations=excitationMap,
                bidirectional=bidirectional,
                usePreparedCache=usePreparedCache,
            )

        items = list(enumerate(values))
        if workerCount == 1 or len(items) <= 1:
            pointResults = map(solvePoint, items)
        else:
            executor = ThreadPoolExecutor(max_workers=workerCount)
            pointResults = executor.map(solvePoint, items)

        try:
            for index, point in pointResults:
                for label, (forwardValue, backwardValue) in point.items():
                    forward[label][index] = forwardValue
                    backward[label][index] = backwardValue
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True)

        for label in labels:
            entry: dict[str, ComplexArray] = {"absorptivity": forward[label]}
            if bidirectional:
                entry["emissivity"] = backward[label]
                entry["nonreciprocity"] = np.abs(forward[label] - backward[label])
            spectra[label] = entry
        return spectra

    def spectrumPoint(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        bidirectional: bool,
        usePreparedCache: bool,
    ) -> dict[str, tuple[float, float]]:
        layers = self.layersAt(wavelength)
        pointPreparedCache: dict[tuple[float, float], dict[str, tuple[float, float]]] = {}
        forwardPowers = self.solveBatchPowers(
            wavelength,
            layers=layers,
            theta=theta,
            phi=phi,
            excitations=excitations,
            usePreparedCache=usePreparedCache,
            pointPreparedCache=pointPreparedCache,
        )
        if bidirectional:
            backwardPowers = self.solveBatchPowers(
                wavelength,
                layers=layers,
                theta=-theta,
                phi=phi,
                excitations=excitations,
                usePreparedCache=usePreparedCache,
                pointPreparedCache=pointPreparedCache,
            )
        else:
            backwardPowers = {}
        return {
            label: (
                absorptionFromPowers(*forwardPowers[label]),
                absorptionFromPowers(*backwardPowers[label]) if bidirectional else np.nan,
            )
            for label in excitations
        }

    def spectrumBatchPowers(
        self,
        wavelengths: ComplexArray,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        bidirectional: bool,
    ) -> dict[str, dict[str, ComplexArray]] | None:
        if wavelengths.size <= 1 or not self.canUseBatchedSpectrum(wavelengths):
            return None

        referenceLayers = tuple(self.layers) if any(not isinstance(layer, (Layer, CompiledLayer)) for layer in self.layers) else self.layersAt(float(wavelengths[0]))
        chunkSize = self.spectrumBatchChunkSize(referenceLayers)
        combineBidirectionalAngles = bidirectional and chunkSize > 1
        if combineBidirectionalAngles:
            chunkSize = max(1, chunkSize // 2)
        forward = {label: np.empty(wavelengths.shape, dtype=float) for label in excitations}
        backward = {label: np.empty(wavelengths.shape, dtype=float) for label in excitations}

        for start in range(0, wavelengths.size, chunkSize):
            stop = min(start + chunkSize, wavelengths.size)
            chunk = wavelengths[start:stop]
            layers = self.batchedLayers(chunk)
            if layers is None:
                return None

            if combineBidirectionalAngles:
                combinedChunk = np.concatenate([chunk, chunk])
                combinedLayers = repeatBatchedLayersForAngles(layers, 2)
                angleBatch = np.concatenate(
                    [
                        np.full(chunk.shape, float(theta), dtype=float),
                        np.full(chunk.shape, -float(theta), dtype=float),
                    ]
                )
                phiBatch = np.full(combinedChunk.shape, float(phi), dtype=float)
                combinedPowers = self.prepareSpectrumBatchPowers(
                    combinedChunk,
                    layers=combinedLayers,
                    theta=angleBatch,
                    phi=phiBatch,
                    excitations=excitations,
                )
                width = chunk.size
                for label, (reflection, transmission) in combinedPowers.items():
                    absorptivity = 1.0 - reflection - transmission
                    forward[label][start:stop] = absorptivity[:width]
                    backward[label][start:stop] = absorptivity[width:]
            else:
                forwardPowers = self.prepareSpectrumBatchPowers(
                    chunk,
                    layers=layers,
                    theta=theta,
                    phi=phi,
                    excitations=excitations,
                )
                for label, (reflection, transmission) in forwardPowers.items():
                    forward[label][start:stop] = 1.0 - reflection - transmission
                if bidirectional:
                    backwardPowers = self.prepareSpectrumBatchPowers(
                        chunk,
                        layers=layers,
                        theta=-theta,
                        phi=phi,
                        excitations=excitations,
                    )
                    for label, (reflection, transmission) in backwardPowers.items():
                        backward[label][start:stop] = 1.0 - reflection - transmission

            invalid = invalidSpectrumMask(forward, backward, start, stop, tuple(excitations), bidirectional)
            if np.any(invalid):
                badWavelengths = ", ".join(f"{float(value):.12g}" for value in chunk[invalid])
                raise FloatingPointError(
                    "anisotropic batched spectrum produced non-finite powers "
                    f"for wavelength(s): {badWavelengths}"
                )

        return {"forward": forward, "backward": backward}

    def prepareSpectrumBatchPowers(
        self,
        wavelengths: ComplexArray,
        *,
        layers: Sequence[Layer | CompiledLayer | BatchedHomogeneousLayer],
        theta: float | ComplexArray,
        phi: float | ComplexArray,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, tuple[ComplexArray, ComplexArray]]:
        prepared = prepareStackSMatrixBatch(
            layers=layers,
            wavelengths=wavelengths,
            period=self.period,
            orders=self.orders,
            truncation=self.truncation,
            backend=self.backend,
            precision=self.precision,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
        )
        return evaluatePreparedSpectrumBatchPowers(prepared, excitations)

    def canUseBatchedSpectrum(self, wavelengths: ComplexArray) -> bool:
        if self.backend != "cuda":
            return False
        if wavelengths.size < 2:
            return False
        if self.hasOnlyStaticLayerInputs() and self.hasReducedSpectrumFastPath(float(wavelengths[0])):
            return False
        return True

    def hasOnlyStaticLayerInputs(self) -> bool:
        return all(isinstance(layer, (Layer, CompiledLayer)) for layer in self.layers)

    def hasReducedSpectrumFastPath(self, wavelength: float) -> bool:
        layers = tuple(layersAt(tuple(self.layers), wavelength))
        forward = automaticFastPathPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=0.0,
            phi=0.0,
            truncation=self.truncation,
        )
        return forward is not None

    def spectrumBatchChunkSize(self, layers: Sequence[LayerInput]) -> int:
        configured = os.environ.get("RCWA3D_ANISOTROPIC_SPECTRUM_BATCH", "").strip()
        if configured:
            return max(1, int(configured))

        orderX, orderY = normalizeOrders(self.orders)
        if self.truncation == "circular":
            entries = [
                (mx, my)
                for my in range(-orderY, orderY + 1)
                for mx in range(-orderX, orderX + 1)
                if insideCircularOrder(mx, my, orderX, orderY)
            ]
            nOrders = max(1, len(entries))
        else:
            nOrders = (2 * orderX + 1) * (2 * orderY + 1)

        hasFullLayer = any(not isSimpleBatchLayer(layer) for layer in layers)
        matrixSize = (4 if hasFullLayer else 2) * nOrders
        bytesPerPoint = max(1, matrixSize * matrixSize * 16 * 12)
        budget = 512 * 1024 * 1024
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("anisotropic CUDA batched spectrum requires torch.cuda to be available")
        freeBytes, totalBytes = torch.cuda.mem_get_info()
        budget = max(budget, int(0.20 * freeBytes))
        return max(1, min(64, budget // bytesPerPoint))

    def batchedLayers(
        self,
        wavelengths: ComplexArray,
        *,
        warn: bool = True,
    ) -> tuple[Layer | CompiledLayer | BatchedHomogeneousLayer, ...] | None:
        perPoint = []
        for wavelength in wavelengths:
            layers = layersAt(tuple(self.layers), float(wavelength))
            if warn:
                self.warnIfUndersampled(layers)
            perPoint.append(layers)

        if not perPoint:
            return ()
        layerCount = len(perPoint[0])
        if any(len(layers) != layerCount for layers in perPoint):
            return None

        batched: list[Layer | CompiledLayer | BatchedHomogeneousLayer] = []
        for layerIndex in range(layerCount):
            column = [layers[layerIndex] for layers in perPoint]
            first = column[0]
            if sameStaticBatchLayer(column):
                batched.append(first)
                continue
            homogeneous = batchedHomogeneousLayer(column)
            if homogeneous is None:
                return None
            batched.append(homogeneous)
        return tuple(batched)

    def solveBatchPowers(
        self,
        wavelength: float,
        *,
        layers: Sequence[Layer | CompiledLayer] | None = None,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        usePreparedCache: bool = True,
        pointPreparedCache: dict[tuple[float, float], dict[str, tuple[float, float]]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        if not excitations:
            return {}

        pointKey = (float(theta), float(phi))
        if pointPreparedCache is not None:
            cached = pointPreparedCache.get(pointKey)
            if cached is not None and all(label in cached for label in excitations):
                return {label: cached[label] for label in excitations}

        materializedLayers = tuple(self.layersAt(wavelength) if layers is None else layers)
        reduced = self.reducedBatchPowersSolve(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=materializedLayers,
            excitations=excitations,
            useCache=usePreparedCache,
        )
        if reduced is not None:
            if pointPreparedCache is not None:
                pointPreparedCache[pointKey] = dict(reduced)
            return reduced

        prepared = self.preparedStack(
            wavelength,
            theta=theta,
            phi=phi,
            layers=materializedLayers,
            fullTotal=False,
            useCache=usePreparedCache,
        )
        powers = evaluatePreparedBatchPowers(prepared, excitations)
        if pointPreparedCache is not None:
            pointPreparedCache[pointKey] = dict(powers)
        return powers

    def preparedStack(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        layers: Sequence[Layer | CompiledLayer] | None = None,
        orders: int | tuple[int, int] | None = None,
        fullTotal: bool = True,
        profile: bool = False,
        useCache: bool = True,
        cacheLayerKey: tuple[object, ...] | None = None,
    ) -> PreparedStack:
        materializedLayers = tuple(self.layersAt(wavelength) if layers is None else layers)
        preparedOrders = self.orders if orders is None else orders
        cacheKey = (
            float(wavelength),
            float(theta),
            float(phi),
            complex(self.epsIncident),
            complex(self.epsTransmission),
            tuple(self.period),
            preparedOrders,
            self.truncation,
            self.backend,
            self.precision,
            bool(fullTotal),
            layersCacheKey(materializedLayers) if cacheLayerKey is None else cacheLayerKey,
        )
        if self.cacheModes and not profile and useCache:
            with self.cacheLock:
                cached = self.preparedCache.get(cacheKey)
                if cached is not None:
                    self.preparedCache.move_to_end(cacheKey)
                    return cached

        prepared = prepareStackSMatrix(
            layers=materializedLayers,
            wavelength=wavelength,
            period=self.period,
            orders=preparedOrders,
            truncation=self.truncation,
            backend=self.backend,
            precision=self.precision,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            fullTotal=fullTotal,
            profile=profile,
        )
        if self.cacheModes and not profile and useCache:
            with self.cacheLock:
                self.preparedCache[cacheKey] = prepared
                while len(self.preparedCache) > self.cacheSize:
                    self.preparedCache.popitem(last=False)
        return prepared

    def reducedSolve(
        self,
        *,
        wavelength: float,
        theta: float,
        phi: float,
        layers: Sequence[Layer | CompiledLayer],
        sAmplitude: complex,
        pAmplitude: complex,
        profile: bool,
    ) -> RCWAResult | None:
        plan = automaticFastPathPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
        )
        if plan is None:
            return None
        reducedLayers = reducedFastPathLayers(layers, plan)
        reducedCacheKey = reducedCacheLayerKey(layers, plan)
        prepared = self.preparedStack(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=reducedLayers,
            orders=plan.reducedOrders,
            fullTotal=False,
            profile=profile,
            cacheLayerKey=reducedCacheKey,
        )
        solvedBy = f"smatrix-{plan.label}-{prepared.backend}"
        reduced = evaluatePreparedStack(
            prepared,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=solvedBy,
        )
        return embedReducedResult(
            reduced,
            fullHarmonics=plan.fullHarmonics,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=solvedBy,
        )

    def reducedBatchSolve(
        self,
        *,
        wavelength: float,
        theta: float,
        phi: float,
        layers: Sequence[Layer | CompiledLayer],
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult] | None:
        plan = automaticFastPathPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
        )
        if plan is None:
            return None
        reducedLayers = reducedFastPathLayers(layers, plan)
        reducedCacheKey = reducedCacheLayerKey(layers, plan)
        prepared = self.preparedStack(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=reducedLayers,
            orders=plan.reducedOrders,
            fullTotal=False,
            cacheLayerKey=reducedCacheKey,
        )
        solvedBy = f"smatrix-batch-{plan.label}-{prepared.backend}"
        reducedResults = evaluatePreparedBatch(prepared, excitations, solvedBy=solvedBy)
        return {
            label: embedReducedResult(
                reduced,
                fullHarmonics=plan.fullHarmonics,
                epsIncident=self.epsIncident,
                epsTransmission=self.epsTransmission,
                sAmplitude=excitations[label][0],
                pAmplitude=excitations[label][1],
                solvedBy=solvedBy,
            )
            for label, reduced in reducedResults.items()
        }

    def reducedBatchPowersSolve(
        self,
        *,
        wavelength: float,
        theta: float,
        phi: float,
        layers: Sequence[Layer | CompiledLayer],
        excitations: dict[str, tuple[complex, complex]],
        useCache: bool = True,
    ) -> dict[str, tuple[float, float]] | None:
        plan = automaticFastPathPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
        )
        if plan is None:
            return None
        reducedLayers = reducedFastPathLayers(layers, plan)
        reducedCacheKey = reducedCacheLayerKey(layers, plan)
        prepared = self.preparedStack(
            wavelength=wavelength,
            theta=theta,
            phi=phi,
            layers=reducedLayers,
            orders=plan.reducedOrders,
            fullTotal=False,
            useCache=useCache,
            cacheLayerKey=reducedCacheKey,
        )
        return evaluatePreparedBatchPowers(prepared, excitations)

    def prepareLayer(self, layer: LayerInput) -> LayerInput:
        if not self.precompile:
            return layer
        if isinstance(layer, CompiledLayer):
            return layer
        if isinstance(layer, Layer):
            if (
                layer.normalField is None
                and constantTensor(layer.epsilon) is not None
                and (layer.mu is None or constantTensor(layer.mu) is not None)
            ):
                return layer
            return compileLayers([layer], orders=self.orders, truncation=self.truncation)[0]
        if isinstance(layer, LayerSpec) and layer.isStatic:
            staticLayer = layer.at(1.0)
            if (
                staticLayer.normalField is None
                and constantTensor(staticLayer.epsilon) is not None
                and (staticLayer.mu is None or constantTensor(staticLayer.mu) is not None)
            ):
                return staticLayer
            return compileLayers([staticLayer], orders=self.orders, truncation=self.truncation)[0]
        return layer

    def layersAt(self, wavelength: float) -> list[Layer | CompiledLayer]:
        layers = layersAt(tuple(self.layers), wavelength)
        self.warnIfUndersampled(layers)
        return layers

    def warnIfUndersampled(self, layers: Sequence[Layer | CompiledLayer]) -> None:
        orderX, orderY = normalizeOrders(self.orders)
        minimumRecommended = 8 * (2 * max(orderX, orderY) + 1)
        if minimumRecommended <= 8:
            return
        for index, layer in enumerate(layers):
            shape = getattr(layer, "sampleShape", None)
            if shape is None:
                shape = sampleShape(layer)
            if shape is None:
                continue
            if min(shape) < minimumRecommended:
                key = (index, tuple(shape), self.orders, self.truncation)
                with self.cacheLock:
                    if key in self.samplingWarningKeys:
                        continue
                    self.samplingWarningKeys.add(key)
                warnings.warn(
                    f"layer {index} sampled grid {shape} is low for orders={self.orders}; "
                    f"use min(shape) >= {minimumRecommended} or increase samples to reduce staircasing error",
                    RuntimeWarning,
                    stacklevel=3,
                )


def polarizationAmplitudes(polarization: Polarization) -> tuple[complex, complex]:
    if not isinstance(polarization, str):
        sAmplitude, pAmplitude = polarization
        return complex(sAmplitude), complex(pAmplitude)

    value = str(polarization).upper()
    if value in ("TE", "S"):
        return 1.0, 0.0
    if value in ("TM", "P"):
        return 0.0, 1.0
    raise ValueError("polarization must be 'TE', 'TM', 's', or 'p'")


def polarizationLabel(polarization: Polarization) -> str:
    if not isinstance(polarization, str):
        raise ValueError("custom spectrum excitations must be passed with labels through excitations={...}")

    value = str(polarization).upper()
    if value == "S":
        return "TE"
    if value == "P":
        return "TM"
    return value


def spectrumExcitations(
    polarizations: Sequence[Polarization],
    excitations: ExcitationMap | None,
) -> dict[str, tuple[complex, complex]]:
    result = {
        polarizationLabel(polarization): polarizationAmplitudes(polarization)
        for polarization in polarizations
    }
    if excitations is not None:
        for label, amplitudes in excitations.items():
            sAmplitude, pAmplitude = amplitudes
            result[str(label)] = (complex(sAmplitude), complex(pAmplitude))
    if not result:
        raise ValueError("spectrum requires at least one polarization or custom excitation")
    return result


def absorptionFromPowers(reflection: float, transmission: float) -> float:
    return float(np.real_if_close(1.0 - reflection - transmission))


def invalidSpectrumMask(
    forward: Mapping[str, ComplexArray],
    backward: Mapping[str, ComplexArray],
    start: int,
    stop: int,
    labels: Sequence[str],
    bidirectional: bool,
) -> ComplexArray:
    mask = np.zeros(stop - start, dtype=bool)
    for label in labels:
        mask |= ~np.isfinite(forward[label][start:stop])
        if bidirectional:
            mask |= ~np.isfinite(backward[label][start:stop])
    return mask


def normalizePrecisionLabel(value: str) -> str:
    normalized = str(value).lower().replace("-", "").replace("_", "")
    aliases = {
        "complex128": "complex128",
        "128": "complex128",
        "double": "complex128",
        "float64": "complex128",
        "complex64": "complex64",
        "64": "complex64",
        "single": "complex64",
        "float32": "complex64",
        "mixed": "mixed",
    }
    if normalized not in aliases:
        raise ValueError("precision must be 'complex128', 'complex64', or 'mixed'")
    return aliases[normalized]


def layersCacheKey(layers: Sequence[Layer | CompiledLayer]) -> tuple[tuple[object, ...], ...]:
    return tuple(layerCacheKey(layer) for layer in layers)


def reducedCacheLayerKey(
    layers: Sequence[Layer | CompiledLayer],
    plan: object,
) -> tuple[object, ...]:
    return (
        "reduced",
        getattr(plan, "label"),
        getattr(plan, "reducedOrders"),
        tuple(int(index) for index in getattr(plan, "keptIndices")),
        layersCacheKey(layers),
    )


def layerCacheKey(layer: Layer | CompiledLayer) -> tuple[object, ...]:
    if isinstance(layer, CompiledLayer):
        return ("compiled", id(layer), complexArrayCacheKey(layer.mu) if getattr(layer, "mu", None) is not None else None)

    if layer.normalField is None:
        tensor = constantTensor(layer.epsilon)
        muTensor = identityTensor() if getattr(layer, "mu", None) is None else constantTensor(layer.mu)
        if tensor is not None and muTensor is not None:
            return (
                "homogeneous",
                float(layer.thickness),
                complexArrayCacheKey(tensor),
                complexArrayCacheKey(muTensor),
                getattr(layer, "factorization", "auto"),
            )
    return ("layer", id(layer))


def complexArrayCacheKey(value: ComplexArray) -> tuple[tuple[int, ...], tuple[complex, ...]]:
    array = np.asarray(value, dtype=complex)
    return tuple(int(size) for size in array.shape), tuple(complex(item) for item in array.reshape(-1))


def identityTensor() -> ComplexArray:
    return np.eye(3, dtype=complex)


def spectrumParallelPlan(
    requestedWorkers: int,
    parallel: SpectrumParallel,
    backend: str,
) -> tuple[str, int]:
    """Protect dense LAPACK solves from Python thread oversubscription.

    The anisotropic path spends most solve time in dense CUDA linear algebra.
    Running several wavelength points concurrently usually makes them fight for
    the same GPU resources, so ``auto`` stays serial.
    """

    if requestedWorkers <= 1:
        return "serial", requestedWorkers

    value = parallel.lower()
    if value not in ("auto", "serial", "thread", "process"):
        raise ValueError("parallel must be 'auto', 'serial', 'thread', or 'process'")
    if value == "serial":
        return "serial", 1
    if value == "process":
        raise ValueError("process spectrum parallelism is not supported with the CUDA-only anisotropic solver")
    if value == "thread":
        return "thread", requestedWorkers
    if os.environ.get("RCWA3D_ALLOW_THREADED_ANISOTROPIC_SPECTRUM") == "1":
        return "thread", requestedWorkers
    return "serial", 1


def layersAt(items: tuple[LayerInput, ...], wavelength: float) -> list[Layer | CompiledLayer]:
    layers: list[Layer | CompiledLayer] = []
    for item in items:
        if isinstance(item, (Layer, CompiledLayer)):
            layers.append(item)
        elif isinstance(item, LayerSpec):
            layers.append(item.at(wavelength))
        elif callable(item):
            produced = item(wavelength)
            if isinstance(produced, (Layer, CompiledLayer)):
                layers.append(produced)
            else:
                layers.extend(produced)
        else:
            raise TypeError(f"unsupported layer input {type(item)!r}")
    return layers


def sameStaticBatchLayer(layers: Sequence[Layer | CompiledLayer]) -> bool:
    first = layers[0]
    if isinstance(first, CompiledLayer):
        return all(layer is first for layer in layers)
    if not isinstance(first, Layer):
        return False
    return all(layer is first for layer in layers)


def batchedHomogeneousLayer(layers: Sequence[Layer | CompiledLayer]) -> BatchedHomogeneousLayer | None:
    tensors: list[ComplexArray] = []
    mus: list[ComplexArray] = []
    firstThickness = float(layers[0].thickness)
    firstName = getattr(layers[0], "name", "")
    for layer in layers:
        if abs(float(layer.thickness) - firstThickness) > 1e-14 * max(1.0, abs(firstThickness)):
            return None
        if isinstance(layer, CompiledLayer):
            tensor = layer.tensorData.constantTensor
            muTensor = layer.mu
        elif getattr(layer, "normalField", None) is None:
            tensor = constantTensor(layer.epsilon)
            muTensor = identityTensor() if layer.mu is None else constantTensor(layer.mu)
        else:
            return None
        if tensor is None or muTensor is None:
            return None
        tensors.append(np.asarray(tensor, dtype=complex))
        mus.append(np.asarray(muTensor, dtype=complex))
    muStack = np.stack(mus, axis=0)
    return BatchedHomogeneousLayer(
        thickness=firstThickness,
        tensors=np.stack(tensors, axis=0),
        name=str(firstName),
        mus=None if isBatchedIdentityTensor(muStack) else muStack,
    )


def repeatBatchedLayersForAngles(
    layers: Sequence[Layer | CompiledLayer | BatchedHomogeneousLayer],
    repeats: int,
) -> tuple[Layer | CompiledLayer | BatchedHomogeneousLayer, ...]:
    repeated: list[Layer | CompiledLayer | BatchedHomogeneousLayer] = []
    for layer in layers:
        if isinstance(layer, BatchedHomogeneousLayer):
            repeated.append(
                BatchedHomogeneousLayer(
                    thickness=layer.thickness,
                    tensors=np.concatenate([np.asarray(layer.tensors, dtype=complex)] * repeats, axis=0),
                    name=layer.name,
                    mus=(
                        None
                        if layer.mus is None
                        else np.concatenate([np.asarray(layer.mus, dtype=complex)] * repeats, axis=0)
                    ),
                )
            )
        else:
            repeated.append(layer)
    return tuple(repeated)


def sampleShape(layer: object) -> tuple[int, int] | None:
    epsilon = getattr(layer, "epsilon", None)
    return sampleShapeFromEpsilon(epsilon)


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


def insideCircularOrder(mx: int, my: int, nx: int, ny: int) -> bool:
    if nx == 0 and ny == 0:
        return mx == 0 and my == 0
    if nx == 0:
        return mx == 0 and abs(my) <= ny
    if ny == 0:
        return my == 0 and abs(mx) <= nx
    return (mx / nx) ** 2 + (my / ny) ** 2 <= 1.0 + 1e-12


def isSimpleBatchLayer(layer: Layer | CompiledLayer) -> bool:
    if isinstance(layer, LayerSpec):
        return True
    if callable(layer):
        return False
    if isinstance(layer, CompiledLayer):
        tensor = layer.tensorData.constantTensor
        if tensor is None:
            return False
        muTensor = identityTensor() if layer.mu is None else layer.mu
    elif getattr(layer, "normalField", None) is None:
        tensor = constantTensor(layer.epsilon)
        if tensor is None:
            return False
        muTensor = identityTensor() if layer.mu is None else constantTensor(layer.mu)
    else:
        return False
    if muTensor is None or not isIdentityTensor(muTensor):
        return False
    diagonal = np.diag(np.asarray(tensor, dtype=complex))
    offDiagonal = tensor - np.diag(diagonal)
    scale = max(1.0, float(np.max(np.abs(tensor))) if tensor.size else 1.0)
    return bool(np.max(np.abs(offDiagonal)) <= 1e-14 * scale and np.max(np.abs(diagonal - diagonal[0])) <= 1e-14 * scale)


def isIdentityTensor(tensor: ComplexArray) -> bool:
    array = np.asarray(tensor, dtype=complex)
    if array.shape != (3, 3):
        return False
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    return bool(np.max(np.abs(array - np.eye(3, dtype=complex))) <= 1e-14 * scale)


def isBatchedIdentityTensor(tensors: ComplexArray) -> bool:
    array = np.asarray(tensors, dtype=complex)
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        return False
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 1.0)
    return bool(np.max(np.abs(array - np.eye(3, dtype=complex)[None, :, :])) <= 1e-14 * scale)
