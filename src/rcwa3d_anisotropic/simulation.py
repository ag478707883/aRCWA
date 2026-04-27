from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from threading import Lock
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union

import numpy as np

from .backend import resolveBackend
from .factorization import constantTensor
from .solver import (
    CompiledLayer,
    Layer,
    PreparedStack,
    RCWAResult,
    compileLayers,
    prepareStackSMatrix,
    solveStack,
    solveStackBatch,
)


ComplexArray = np.ndarray
EpsilonSource = Any
LayerFactory = Callable[[float], Any]
LayerInput = Any
Polarization = Union[Literal["TE", "TM", "s", "p"], tuple[complex, complex]]
ExcitationMap = Mapping[str, tuple[complex, complex]]
SpectrumParallel = Literal["auto", "serial", "thread", "process"]


@dataclass(frozen=True)
class _SpectrumWorkerState:
    period: tuple[float, float]
    layers: tuple[LayerInput, ...]
    orders: int | tuple[int, int]
    truncation: str
    backend: str
    epsIncident: complex
    epsTransmission: complex


_PROCESS_SPECTRUM_STATE: _SpectrumWorkerState | None = None


@dataclass(frozen=True)
class LayerSpec:
    """A homogeneous layer whose material may depend on wavelength."""

    thickness: float
    epsilon: EpsilonSource
    name: str = ""

    @property
    def isStatic(self) -> bool:
        return not callable(self.epsilon)

    def at(self, wavelength: float) -> Layer:
        epsilon = self.epsilon(wavelength) if callable(self.epsilon) else self.epsilon
        return Layer(thickness=self.thickness, epsilon=epsilon, name=self.name)


def homogeneousLayer(thickness: float, epsilon: EpsilonSource, name: str = "") -> Layer | LayerSpec:
    """Return a static or wavelength-dependent homogeneous layer."""

    if callable(epsilon):
        return LayerSpec(thickness=thickness, epsilon=epsilon, name=name)
    return Layer(thickness=thickness, epsilon=epsilon, name=name)


@dataclass
class RCWASimulation:
    """Small high-level wrapper around the anisotropic RCWA solver.

    Static layers are precompiled once. Wavelength-dependent layers are built
    only at solve time. Use callables when a material tensor changes with
    wavelength and ordinary ``Layer`` objects for fixed geometry.

    ``method="smatrix"`` is the default because it is the most numerically
    stable path for anisotropic multilayers, and it is now the only public
    solve method.  CUDA is required for solves; ``backend="auto"`` resolves to
    the same PyTorch CUDA backend and never falls back to CPU.
    """

    period: tuple[float, float]
    layers: Sequence[LayerInput]
    orders: int | tuple[int, int] = 5
    truncation: Literal["circular", "rectangular"] = "circular"
    backend: str = "cuda"
    epsIncident: complex = 1.0
    epsTransmission: complex = 1.0
    precompile: bool = True
    method: Literal["smatrix"] = "smatrix"
    workers: int = 1
    cacheModes: bool = True
    cacheSize: int = 10
    _preparedCache: OrderedDict[tuple, PreparedStack] = field(default_factory=OrderedDict, init=False, repr=False)
    _cacheLock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method != "smatrix":
            raise ValueError("RCWASimulation now supports only method='smatrix'")
        self.backend = resolveBackend(self.backend).name
        self.layers = tuple(self._prepareLayer(layer) for layer in self.layers)

    def solve(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarization: Polarization = "TE",
        returnFields: bool = False,
    ) -> RCWAResult:
        sAmplitude, pAmplitude = _polarizationAmplitudes(polarization)
        return solveStack(
            layers=self._layersAt(wavelength),
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            truncation=self.truncation,
            backend=self.backend,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=returnFields,
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
        excitationMap = _spectrumExcitations(polarizations, excitations)
        labels = tuple(excitationMap)
        forward = {label: np.empty(values.shape, dtype=float) for label in labels}
        backward = {label: np.full(values.shape, np.nan, dtype=float) for label in labels}

        requestedWorkers = self.workers if workers is None else workers
        if requestedWorkers < 1:
            raise ValueError("workers must be at least 1")
        parallelMode, workerCount = _spectrumParallelPlan(requestedWorkers, parallel, self.backend)

        def solvePoint(item: tuple[int, float]) -> tuple[int, dict[str, tuple[float, float]]]:
            index, wavelength = item
            return index, self._spectrumPoint(
                float(wavelength),
                theta=theta,
                phi=phi,
                excitations=excitationMap,
                bidirectional=bidirectional,
            )

        items = list(enumerate(values))
        if workerCount == 1 or len(items) <= 1:
            pointResults = map(solvePoint, items)
        elif parallelMode == "process":
            state = _SpectrumWorkerState(
                period=self.period,
                layers=tuple(self.layers),
                orders=self.orders,
                truncation=self.truncation,
                backend=self.backend,
                epsIncident=self.epsIncident,
                epsTransmission=self.epsTransmission,
            )
            executor = ProcessPoolExecutor(
                max_workers=workerCount,
                initializer=_initializeSpectrumWorker,
                initargs=(state,),
            )
            pointResults = executor.map(
                _processSpectrumPoint,
                (
                    (index, float(wavelength), theta, phi, excitations, bidirectional)
                    for index, wavelength in items
                ),
            )
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

    def _spectrumPoint(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        bidirectional: bool,
    ) -> dict[str, tuple[float, float]]:
        layers = self._layersAt(wavelength)
        forwardResults = self._solveBatch(
            wavelength,
            layers=layers,
            theta=theta,
            phi=phi,
            excitations=excitations,
        )
        if bidirectional:
            backwardResults = self._solveBatch(
                wavelength,
                layers=layers,
                theta=-theta,
                phi=phi,
                excitations=excitations,
            )
        else:
            backwardResults = {}
        return {
            label: (
                _absorptionFromResult(forwardResults[label]),
                _absorptionFromResult(backwardResults[label]) if bidirectional else np.nan,
            )
            for label in excitations
        }

    def _solveBatch(
        self,
        wavelength: float,
        *,
        layers: Sequence[Layer | CompiledLayer] | None = None,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult]:
        return solveStackBatch(
            layers=self._layersAt(wavelength) if layers is None else layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            truncation=self.truncation,
            backend=self.backend,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            excitations=excitations,
        )

    def _preparedStack(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        layers: Sequence[Layer | CompiledLayer] | None = None,
    ) -> PreparedStack:
        materializedLayers = tuple(self._layersAt(wavelength) if layers is None else layers)
        cacheKey = (
            float(wavelength),
            float(theta),
            float(phi),
            complex(self.epsIncident),
            complex(self.epsTransmission),
            tuple(self.period),
            self.orders,
            self.truncation,
            self.backend,
            tuple((type(layer).__name__, id(layer)) for layer in materializedLayers),
        )
        if self.cacheModes:
            with self._cacheLock:
                cached = self._preparedCache.get(cacheKey)
                if cached is not None:
                    self._preparedCache.move_to_end(cacheKey)
                    return cached

        prepared = prepareStackSMatrix(
            layers=materializedLayers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            truncation=self.truncation,
            backend=self.backend,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
        )
        if self.cacheModes:
            with self._cacheLock:
                self._preparedCache[cacheKey] = prepared
                while len(self._preparedCache) > self.cacheSize:
                    self._preparedCache.popitem(last=False)
        return prepared

    def _prepareLayer(self, layer: LayerInput) -> LayerInput:
        if not self.precompile:
            return layer
        if isinstance(layer, CompiledLayer):
            return layer
        if isinstance(layer, Layer):
            if layer.normalField is None and constantTensor(layer.epsilon) is not None:
                return layer
            return compileLayers([layer], orders=self.orders, truncation=self.truncation)[0]
        if isinstance(layer, LayerSpec) and layer.isStatic:
            staticLayer = layer.at(1.0)
            if staticLayer.normalField is None and constantTensor(staticLayer.epsilon) is not None:
                return staticLayer
            return compileLayers([staticLayer], orders=self.orders, truncation=self.truncation)[0]
        return layer

    def _layersAt(self, wavelength: float) -> list[Layer | CompiledLayer]:
        return _layersAt(tuple(self.layers), wavelength)


def _polarizationAmplitudes(polarization: Polarization) -> tuple[complex, complex]:
    if not isinstance(polarization, str):
        sAmplitude, pAmplitude = polarization
        return complex(sAmplitude), complex(pAmplitude)

    value = str(polarization).upper()
    if value in ("TE", "S"):
        return 1.0, 0.0
    if value in ("TM", "P"):
        return 0.0, 1.0
    raise ValueError("polarization must be 'TE', 'TM', 's', or 'p'")


def _polarizationLabel(polarization: Polarization) -> str:
    if not isinstance(polarization, str):
        raise ValueError("custom spectrum excitations must be passed with labels through excitations={...}")

    value = str(polarization).upper()
    if value == "S":
        return "TE"
    if value == "P":
        return "TM"
    return value


def _spectrumExcitations(
    polarizations: Sequence[Polarization],
    excitations: ExcitationMap | None,
) -> dict[str, tuple[complex, complex]]:
    result = {
        _polarizationLabel(polarization): _polarizationAmplitudes(polarization)
        for polarization in polarizations
    }
    if excitations is not None:
        for label, amplitudes in excitations.items():
            sAmplitude, pAmplitude = amplitudes
            result[str(label)] = (complex(sAmplitude), complex(pAmplitude))
    if not result:
        raise ValueError("spectrum requires at least one polarization or custom excitation")
    return result


def _absorptionFromResult(result: RCWAResult) -> float:
    return float(np.real_if_close(1.0 - result.reflection - result.transmission))


def _spectrumParallelPlan(
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


def _layersAt(items: tuple[LayerInput, ...], wavelength: float) -> list[Layer | CompiledLayer]:
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


def _initializeSpectrumWorker(state: _SpectrumWorkerState) -> None:
    global _PROCESS_SPECTRUM_STATE
    _PROCESS_SPECTRUM_STATE = state


def _processSpectrumPoint(
    payload: tuple[
        int,
        float,
        float,
        float,
        dict[str, tuple[complex, complex]],
        bool,
    ],
) -> tuple[int, dict[str, tuple[float, float]]]:
    if _PROCESS_SPECTRUM_STATE is None:
        raise RuntimeError("spectrum worker was not initialized")
    index, wavelength, theta, phi, excitations, bidirectional = payload
    state = _PROCESS_SPECTRUM_STATE
    layers = _layersAt(state.layers, wavelength)
    forwardResults = solveStackBatch(
        layers=layers,
        wavelength=wavelength,
        period=state.period,
        orders=state.orders,
        truncation=state.truncation,
        backend=state.backend,
        epsIncident=state.epsIncident,
        epsTransmission=state.epsTransmission,
        theta=theta,
        phi=phi,
        excitations=excitations,
    )
    if bidirectional:
        backwardResults = solveStackBatch(
            layers=layers,
            wavelength=wavelength,
            period=state.period,
            orders=state.orders,
            truncation=state.truncation,
            backend=state.backend,
            epsIncident=state.epsIncident,
            epsTransmission=state.epsTransmission,
            theta=-theta,
            phi=phi,
            excitations=excitations,
        )
    else:
        backwardResults = {}
    return index, {
        label: (
            _absorptionFromResult(forwardResults[label]),
            _absorptionFromResult(backwardResults[label]) if bidirectional else np.nan,
        )
        for label in excitations
    }
