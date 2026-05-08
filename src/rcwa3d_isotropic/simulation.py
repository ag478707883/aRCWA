from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

from .backend import resolveBackend
from .builder import PatternLayer
from .fourier import makeHarmonics, normalizeOrders
from .solver import (
    PreparedTorchStack,
    _automaticOrderReductionPlan,
    _expandAdaptiveLayers,
    _solveBatchReducedTorch,
    _solveStackReducedTorch,
    _validateIsotropicLayers,
    evaluatePreparedBatchTorch,
    evaluatePreparedStackTorch,
    prepareStackTorch,
)
from ._factorization import layerDataForTorch, solveIdentityTorch, toTorchComplex
from .types import CompiledLayer, Layer, RCWAResult
from .varrcwa import AdaptiveLayerSpec


ComplexArray = np.ndarray
EpsilonSource = Any
LayerInput = Any
Polarization = Literal["TE", "TM", "s", "p"]


@dataclass(frozen=True)
class LayerSpec:
    """A homogeneous isotropic layer whose epsilon may depend on wavelength."""

    thickness: float
    epsilon: EpsilonSource
    name: str = ""
    factorization: str = "auto"

    @property
    def isStatic(self) -> bool:
        return not callable(self.epsilon)

    def at(self, wavelength: float) -> Layer:
        epsilon = self.epsilon(wavelength) if callable(self.epsilon) else self.epsilon
        return Layer(thickness=self.thickness, epsilon=epsilon, name=self.name, factorization=self.factorization)


@dataclass(frozen=True)
class _TorchCompiledLayer:
    thickness: float
    epsilonMatrix: Any
    epsilonInverse: Any
    orders: tuple[int, int]
    truncation: str = "rectangular"
    name: str = ""
    displacementMatrices: tuple[Any, Any, Any, Any] | None = None
    factorization: str = "standard"
    homogeneousEpsilon: complex | None = None


@dataclass
class RCWASimulation:
    """High-level isotropic RCWA model with mutable layer building and mode cache."""

    period: tuple[float, float]
    layers: Sequence[LayerInput] = ()
    orders: int | tuple[int, int] = 3
    truncation: Literal["rectangular", "circular"] = "circular"
    epsIncident: complex = 1.0
    epsTransmission: complex = 1.0
    precompile: bool = True
    method: Literal["smatrix"] = "smatrix"
    backend: str = "cuda"
    workers: int = 1
    cacheModes: bool = True
    cacheSize: int = 10
    _compiledLayerCache: dict[tuple, Any] = field(default_factory=dict, init=False)
    _torchLayerCache: dict[tuple, Any] = field(default_factory=dict, init=False)
    _preparedCache: OrderedDict[tuple, Any] = field(default_factory=OrderedDict, init=False)
    _cacheLock: Lock = field(default_factory=Lock, init=False)

    def __post_init__(self) -> None:
        self.layers = list(self.layers)
        if self.method != "smatrix":
            raise ValueError("RCWASimulation now supports only method='smatrix'")
        resolved = resolveBackend(self.backend)
        self.backend = resolved.name

    def addLayer(
        self,
        layer: LayerInput | None = None,
        *,
        thickness: float | None = None,
        epsilon: EpsilonSource | None = None,
        material: EpsilonSource | None = None,
        shape: tuple[int, int] | None = None,
        name: str = "",
        factorization: str = "auto",
    ) -> LayerInput:
        """Add a layer and return it.

        Pass an existing geometry/helper layer as ``layer`` to append it
        directly.  Pass ``shape=(ny, nx)`` to create a mutable patterned layer,
        then call methods such as ``.circle(...)`` or ``.rectangle(...)`` on
        the returned object.
        """

        if layer is not None:
            self.layers.append(layer)
            self.clearCache()
            return layer
        if thickness is None:
            raise ValueError("addLayer requires thickness when no layer object is supplied")

        value = epsilon if epsilon is not None else material
        if value is None:
            raise ValueError("addLayer requires epsilon or material")

        if shape is not None:
            patterned = PatternLayer(
                period=self.period,
                thickness=thickness,
                background=value,
                shape=shape,
                name=name,
                factorization=factorization,
            )
            self.layers.append(patterned)
            self.clearCache()
            return patterned

        homogeneous = LayerSpec(thickness=thickness, epsilon=value, name=name, factorization=factorization)
        self.layers.append(homogeneous)
        self.clearCache()
        return homogeneous

    def clearCache(self) -> None:
        with self._cacheLock:
            self._compiledLayerCache.clear()
            self._torchLayerCache.clear()
            self._preparedCache.clear()

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
        return self.solveExcitation(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=returnFields,
        )

    def solveExcitation(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        sAmplitude: complex = 1.0,
        pAmplitude: complex = 0.0,
        returnFields: bool = False,
    ) -> RCWAResult:
        """Solve one custom s/p incident excitation using the cached Torch S-matrix."""

        reduced = self._reducedSolve(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=returnFields,
        )
        if reduced is not None:
            return reduced
        prepared = self._preparedTorchStack(wavelength, theta=theta, phi=phi)
        return evaluatePreparedStackTorch(
            prepared,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=f"{self.method}-{_torchBackendLabel(self.backend)}",
            returnFields=returnFields,
        )

    def spectrum(
        self,
        wavelengths: Iterable[float],
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarizations: Sequence[Polarization] = ("TE", "TM"),
        workers: int | None = None,
    ) -> dict[str, dict[str, ComplexArray] | ComplexArray]:
        values = np.asarray(tuple(wavelengths), dtype=float)
        excitations = {
            _polarizationLabel(polarization): _polarizationAmplitudes(polarization)
            for polarization in polarizations
        }
        labels = tuple(excitations)
        reflection = {label: np.empty(values.shape, dtype=float) for label in labels}
        transmission = {label: np.empty(values.shape, dtype=float) for label in labels}
        conservation = {label: np.empty(values.shape, dtype=float) for label in labels}

        workerCount = self.workers if workers is None else workers
        if workerCount < 1:
            raise ValueError("workers must be at least 1")

        def solvePoint(item: tuple[int, float]) -> tuple[int, dict[str, RCWAResult]]:
            index, wavelength = item
            return index, self._solveBatch(float(wavelength), theta=theta, phi=phi, excitations=excitations)

        items = list(enumerate(values))
        if workerCount == 1 or len(items) <= 1:
            pointResults = map(solvePoint, items)
        else:
            executor = ThreadPoolExecutor(max_workers=workerCount)
            pointResults = executor.map(solvePoint, items)

        try:
            for index, point in pointResults:
                for label, result in point.items():
                    reflection[label][index] = result.reflection
                    transmission[label][index] = result.transmission
                    conservation[label][index] = result.conservation
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True)

        spectra: dict[str, dict[str, ComplexArray] | ComplexArray] = {"wavelengths": values}
        for label in labels:
            spectra[label] = {
                "reflection": reflection[label],
                "transmission": transmission[label],
                "conservation": conservation[label],
            }
        return spectra

    def solveExcitations(
        self,
        wavelength: float,
        excitations: dict[str, tuple[complex, complex]],
        *,
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> dict[str, RCWAResult]:
        """Solve several custom s/p excitations while reusing the cached Torch S-matrix."""

        return self._solveBatch(float(wavelength), theta=theta, phi=phi, excitations=dict(excitations))

    def _solveBatch(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult]:
        reduced = self._reducedBatchSolve(wavelength, theta=theta, phi=phi, excitations=excitations)
        if reduced is not None:
            return reduced
        prepared = self._preparedTorchStack(wavelength, theta=theta, phi=phi)
        return evaluatePreparedBatchTorch(
            prepared,
            excitations,
            solvedBy=f"{self.method}-batch-{_torchBackendLabel(self.backend)}",
        )

    def _reducedSolve(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        sAmplitude: complex,
        pAmplitude: complex,
        returnFields: bool,
    ) -> RCWAResult | None:
        layers, _layerKey = self._layersAtWithKey(wavelength)
        plan = _automaticOrderReductionPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
            returnFields=returnFields,
        )
        if plan is None:
            return None
        return _solveStackReducedTorch(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            truncation=self.truncation,
            backend=resolveBackend(self.backend),
            plan=plan,
        )

    def _reducedBatchSolve(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult] | None:
        layers, _layerKey = self._layersAtWithKey(wavelength)
        plan = _automaticOrderReductionPlan(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            epsIncident=self.epsIncident,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
            returnFields=False,
        )
        if plan is None:
            return None
        return _solveBatchReducedTorch(
            layers=layers,
            wavelength=wavelength,
            period=self.period,
            excitations=excitations,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
            backend=resolveBackend(self.backend),
            plan=plan,
        )

    def _preparedTorchStack(self, wavelength: float, *, theta: float, phi: float) -> PreparedTorchStack:
        layers, layerKey = self._layersAtWithKey(wavelength)
        cacheKey = (
            "torch",
            float(wavelength),
            float(theta),
            float(phi),
            complex(self.epsIncident),
            complex(self.epsTransmission),
            tuple(self.period),
            self.orders,
            self.truncation,
            self.backend,
            layerKey,
        )
        if self.cacheModes:
            with self._cacheLock:
                cached = self._preparedCache.get(cacheKey)
                if cached is not None:
                    self._preparedCache.move_to_end(cacheKey)
                    return cached

        torchLayers = self._torchLayers(layers)
        prepared = prepareStackTorch(
            layers=torchLayers,
            wavelength=wavelength,
            period=self.period,
            orders=self.orders,
            truncation=self.truncation,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            backend=self.backend,
        )
        if self.cacheModes:
            with self._cacheLock:
                self._preparedCache[cacheKey] = prepared
                while len(self._preparedCache) > self.cacheSize:
                    self._preparedCache.popitem(last=False)
        return prepared

    def _torchLayers(self, layers: Sequence[Layer | CompiledLayer | _TorchCompiledLayer]) -> list[Layer | CompiledLayer | _TorchCompiledLayer]:
        return [self._torchLayer(layer) for layer in layers]

    def _torchLayer(self, layer: Layer | CompiledLayer | _TorchCompiledLayer) -> Layer | CompiledLayer | _TorchCompiledLayer:
        if isinstance(layer, _TorchCompiledLayer):
            return layer
        if not hasattr(layer, "epsilonMatrix"):
            return layer
        if getattr(layer, "homogeneousEpsilon", None) is not None and getattr(layer, "displacementMatrices", None) is None:
            return layer

        cacheKey = (id(layer), self.backend)
        with self._cacheLock:
            cached = self._torchLayerCache.get(cacheKey)
            if cached is not None:
                return cached

        backend = resolveBackend(self.backend)
        torch = backend.xp
        device = backend.device if backend.device is not None else torch.device("cuda")
        displacementMatrices = getattr(layer, "displacementMatrices", None)
        torchLayer = _TorchCompiledLayer(
            thickness=float(layer.thickness),
            epsilonMatrix=_asTorchComplex(getattr(layer, "epsilonMatrix"), torch, device),
            epsilonInverse=_asTorchComplex(getattr(layer, "epsilonInverse"), torch, device),
            orders=getattr(layer, "orders"),
            truncation=getattr(layer, "truncation", "rectangular"),
            name=getattr(layer, "name", ""),
            displacementMatrices=None
            if displacementMatrices is None
            else tuple(_asTorchComplex(matrix, torch, device) for matrix in displacementMatrices),
            factorization=getattr(layer, "factorization", "standard"),
            homogeneousEpsilon=getattr(layer, "homogeneousEpsilon", None),
        )
        with self._cacheLock:
            cached = self._torchLayerCache.get(cacheKey)
            if cached is not None:
                return cached
            self._torchLayerCache[cacheKey] = torchLayer
        return torchLayer

    def _layersAt(self, wavelength: float) -> list[Layer | CompiledLayer | _TorchCompiledLayer]:
        return self._layersAtWithKey(wavelength)[0]

    def _layersAtWithKey(self, wavelength: float) -> tuple[list[Layer | CompiledLayer | _TorchCompiledLayer], tuple]:
        _validateIsotropicLayers(_expandAdaptiveLayers(self.layers), kind="solve")
        layers: list[Layer | CompiledLayer | _TorchCompiledLayer] = []
        keys = []
        for item in self.layers:
            materialized, key = self._materializeLayer(item, wavelength)
            if isinstance(materialized, list):
                layers.extend(materialized)
            else:
                layers.append(materialized)
            keys.append(key)
        _validateIsotropicLayers(_expandAdaptiveLayers(layers), kind="solve")
        return layers, tuple(keys)

    def _materializeLayer(self, item: LayerInput, wavelength: float) -> tuple[Layer | CompiledLayer | _TorchCompiledLayer | list, tuple]:
        if isinstance(item, CompiledLayer):
            return item, ("compiled", id(item), item.thickness, item.orders, item.truncation)
        if isinstance(item, Layer):
            return self._compileIfRequested(item, ("layer", id(item)))
        if isinstance(item, LayerSpec):
            layer = item.at(wavelength)
            key = ("layerspec", id(item), None if item.isStatic else float(wavelength))
            return self._compileIfRequested(layer, key)
        if isinstance(item, AdaptiveLayerSpec):
            layers = []
            keys = []
            for index, adaptiveLayer in enumerate(item.toLayers()):
                materialized, key = self._compileIfRequested(adaptiveLayer, ("adaptive", id(item), index))
                layers.append(materialized)
                keys.append(key)
            return layers, ("adaptive", id(item), tuple(keys))
        if isinstance(item, PatternLayer):
            layer = item.toLayer()
            return self._compileIfRequested(layer, ("pattern", id(item), item.version))
        if callable(item):
            produced = item(wavelength)
            if isinstance(produced, (Layer, CompiledLayer)):
                return self._materializeLayer(produced, wavelength)
            layers = []
            keys = []
            for producedLayer in produced:
                layer, key = self._materializeLayer(producedLayer, wavelength)
                layers.append(layer)
                keys.append(key)
            return layers, ("factory", id(item), float(wavelength), tuple(keys))
        raise TypeError(f"unsupported layer input {type(item)!r}")

    def _compileIfRequested(self, layer: Layer, key: tuple) -> tuple[Layer | _TorchCompiledLayer, tuple]:
        if not self.precompile:
            return layer, key
        compileKey = (key, self.orders, self.truncation)
        with self._cacheLock:
            cached = self._compiledLayerCache.get(compileKey)
            if cached is not None:
                return cached, ("compiled-cache", compileKey)

        compiled = self._compileLayerTorch(layer)
        with self._cacheLock:
            cached = self._compiledLayerCache.get(compileKey)
            if cached is not None:
                return cached, ("compiled-cache", compileKey)
            self._compiledLayerCache[compileKey] = compiled
        return compiled, ("compiled-cache", compileKey)

    def _compileLayerTorch(self, layer: Layer) -> _TorchCompiledLayer:
        backend = resolveBackend(self.backend)
        torch = backend.xp
        device = backend.device if backend.device is not None else torch.device("cuda")
        orders = normalizeOrders(self.orders)
        harmonics = makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=orders,
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation=self.truncation,
        )
        factorized = layerDataForTorch(layer, harmonics, torch, device)
        epsilonInverse = factorized.epsilonInverse
        if epsilonInverse is None:
            epsilonInverse = solveIdentityTorch(factorized.epsilonMatrix, torch, device)
        return _TorchCompiledLayer(
            thickness=float(layer.thickness),
            epsilonMatrix=toTorchComplex(factorized.epsilonMatrix, torch, device),
            epsilonInverse=toTorchComplex(epsilonInverse, torch, device),
            orders=orders,
            truncation=harmonics.truncation,
            name=getattr(layer, "name", ""),
            displacementMatrices=None
            if factorized.displacementMatrices is None
            else tuple(toTorchComplex(matrix, torch, device) for matrix in factorized.displacementMatrices),
            factorization=factorized.factorization,
            homogeneousEpsilon=factorized.homogeneousEpsilon,
        )


def _polarizationAmplitudes(polarization: Polarization) -> tuple[complex, complex]:
    value = str(polarization).upper()
    if value in ("TE", "S"):
        return 1.0, 0.0
    if value in ("TM", "P"):
        return 0.0, 1.0
    raise ValueError("polarization must be 'TE', 'TM', 's', or 'p'")


def _polarizationLabel(polarization: Polarization) -> str:
    value = str(polarization).upper()
    if value == "S":
        return "TE"
    if value == "P":
        return "TM"
    return value


def _torchBackendLabel(backend: str) -> str:
    if backend != "cuda":
        raise ValueError("the isotropic solver is CUDA-only")
    return "cuda"


def _asTorchComplex(value: Any, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex128, device=device)
