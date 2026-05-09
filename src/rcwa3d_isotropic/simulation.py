from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable, Literal, Sequence
import warnings

import numpy as np

from .backend import resolveBackend
from .builder import PatternLayer
from .fourier import makeHarmonics, normalizeOrders
from .solver import (
    PreparedTorchStack,
    automaticOrderReductionPlan,
    expandAdaptiveLayers,
    solveBatchReducedPowersTorch,
    solveBatchReducedTorch,
    solveStackReducedTorch,
    validateIsotropicLayers,
    layerLossless,
    evaluatePreparedBatchTorch,
    evaluatePreparedBatchPowersTorch,
    evaluateSpectrumBatchPowersTorch,
    evaluatePreparedStackTorch,
    prepareStackPowersTorch,
    prepareStackTorch,
)
from .factorization import layerDataForTorch, solveIdentityTorch, toTorchComplex
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
        return Layer(
            thickness=self.thickness,
            epsilon=epsilon,
            name=self.name,
            factorization=self.factorization,
            sampleShape=sampleShapeFromEpsilon(epsilon),
        )


@dataclass(frozen=True)
class TorchCompiledLayer:
    thickness: float
    epsilonMatrix: Any
    epsilonInverse: Any
    orders: tuple[int, int]
    truncation: str = "rectangular"
    name: str = ""
    displacementMatrices: tuple[Any, Any, Any, Any] | None = None
    factorization: str = "standard"
    homogeneousEpsilon: complex | None = None
    sampleShape: tuple[int, int] | None = None


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
    compiledLayerCache: dict[tuple, Any] = field(default_factory=dict, init=False)
    torchLayerCache: dict[tuple, Any] = field(default_factory=dict, init=False)
    preparedCache: OrderedDict[tuple, Any] = field(default_factory=OrderedDict, init=False)
    samplingWarningKeys: set[tuple] = field(default_factory=set, init=False)
    cacheLock: Lock = field(default_factory=Lock, init=False)

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
        with self.cacheLock:
            self.compiledLayerCache.clear()
            self.torchLayerCache.clear()
            self.preparedCache.clear()

    def solve(
        self,
        wavelength: float,
        *,
        theta: float = 0.0,
        phi: float = 0.0,
        polarization: Polarization = "TE",
        returnFields: bool = False,
        profile: bool = False,
    ) -> RCWAResult:
        sAmplitude, pAmplitude = polarizationAmplitudes(polarization)
        return self.solveExcitation(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=returnFields,
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
        returnFields: bool = False,
        profile: bool = False,
    ) -> RCWAResult:
        """Solve one custom s/p incident excitation using the cached Torch S-matrix."""

        reduced = self.reducedSolve(
            wavelength,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=returnFields,
            profile=profile,
        )
        if reduced is not None:
            return reduced
        prepared = self.preparedTorchStack(wavelength, theta=theta, phi=phi, profile=profile)
        return evaluatePreparedStackTorch(
            prepared,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            solvedBy=f"{self.method}-{torchBackendLabel(self.backend)}",
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
            polarizationLabel(polarization): polarizationAmplitudes(polarization)
            for polarization in polarizations
        }
        labels = tuple(excitations)
        reflection = {label: np.empty(values.shape, dtype=float) for label in labels}
        transmission = {label: np.empty(values.shape, dtype=float) for label in labels}
        conservation = {label: np.empty(values.shape, dtype=float) for label in labels}
        absorption = {label: np.empty(values.shape, dtype=float) for label in labels}
        energyError = {label: np.empty(values.shape, dtype=float) for label in labels}

        workerCount = self.workers if workers is None else workers
        if workerCount < 1:
            raise ValueError("workers must be at least 1")
        batched = self.spectrumBatchPowers(values, theta=theta, phi=phi, excitations=excitations, workers=workerCount)
        if batched is not None:
            spectra: dict[str, dict[str, ComplexArray] | ComplexArray] = {"wavelengths": values}
            batchedEnergyError = self.spectrumEnergyError(values, batched)
            for label, (reflectionValues, transmissionValues) in batched.items():
                conservationValues = reflectionValues + transmissionValues
                spectra[label] = {
                    "reflection": reflectionValues,
                    "transmission": transmissionValues,
                    "conservation": conservationValues,
                    "absorption": 1.0 - conservationValues,
                    "energyError": batchedEnergyError[label],
                }
            return spectra
        usePreparedCache = len(values) == 1

        def solvePoint(item: tuple[int, float]) -> tuple[int, dict[str, tuple[float, float]], bool]:
            index, wavelength = item
            powers, stackLossless = self.solveBatchPowers(
                float(wavelength),
                theta=theta,
                phi=phi,
                excitations=excitations,
                usePreparedCache=usePreparedCache,
            )
            return index, powers, stackLossless

        items = list(enumerate(values))
        if workerCount == 1 or len(items) <= 1:
            pointResults = map(solvePoint, items)
        else:
            executor = ThreadPoolExecutor(max_workers=workerCount)
            pointResults = executor.map(solvePoint, items)

        try:
            for index, point, stackLossless in pointResults:
                for label, (reflectionValue, transmissionValue) in point.items():
                    reflection[label][index] = reflectionValue
                    transmission[label][index] = transmissionValue
                    conservation[label][index] = reflectionValue + transmissionValue
                    absorption[label][index] = 1.0 - conservation[label][index]
                    energyError[label][index] = (
                        abs(conservation[label][index] - 1.0) if stackLossless else float("nan")
                    )
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True)

        spectra: dict[str, dict[str, ComplexArray] | ComplexArray] = {"wavelengths": values}
        for label in labels:
            spectra[label] = {
                "reflection": reflection[label],
                "transmission": transmission[label],
                "conservation": conservation[label],
                "absorption": absorption[label],
                "energyError": energyError[label],
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

        return self.solveBatch(float(wavelength), theta=theta, phi=phi, excitations=dict(excitations))

    def solveBatch(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult]:
        reduced = self.reducedBatchSolve(wavelength, theta=theta, phi=phi, excitations=excitations)
        if reduced is not None:
            return reduced
        prepared = self.preparedTorchStack(wavelength, theta=theta, phi=phi)
        return evaluatePreparedBatchTorch(
            prepared,
            excitations,
            solvedBy=f"{self.method}-batch-{torchBackendLabel(self.backend)}",
        )

    def solveBatchPowers(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        usePreparedCache: bool = True,
    ) -> tuple[dict[str, tuple[float, float]], bool]:
        reduced = self.reducedBatchPowersSolve(wavelength, theta=theta, phi=phi, excitations=excitations)
        if reduced is not None:
            return reduced
        prepared = self.preparedTorchStack(
            wavelength,
            theta=theta,
            phi=phi,
            profile=False,
            useCache=usePreparedCache,
            powersOnly=True,
        )
        return evaluatePreparedBatchPowersTorch(prepared, excitations), self.layersLossless(prepared.layers)

    def spectrumBatchPowers(
        self,
        values: ComplexArray,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
        workers: int,
    ) -> dict[str, tuple[ComplexArray, ComplexArray]] | None:
        if workers != 1 or values.size <= 1:
            return None
        if self.estimatedRectangularHarmonics() > 121:
            return None
        layers, layerKey = self.layersAtWithKey(float(values[0]))
        if not self.layersStaticForSpectrum(values, layers):
            return None
        if (
            automaticOrderReductionPlan(
                layers=layers,
                wavelength=float(values[0]),
                period=self.period,
                orders=self.orders,
                epsIncident=self.epsIncident,
                theta=theta,
                phi=phi,
                truncation=self.truncation,
                returnFields=False,
            )
            is not None
        ):
            return None
        torchLayers = self.torchLayers(layers)
        chunkSize = self.spectrumChunkSize()
        return evaluateSpectrumBatchPowersTorch(
            layers=torchLayers,
            wavelengths=values,
            period=self.period,
            orders=self.orders,
            excitations=excitations,
            epsIncident=self.epsIncident,
            epsTransmission=self.epsTransmission,
            theta=theta,
            phi=phi,
            truncation=self.truncation,
            backend=resolveBackend(self.backend),
            chunkSize=chunkSize,
        )

    def layersStaticForSpectrum(
        self,
        values: ComplexArray,
        firstLayers: Sequence[Layer | CompiledLayer | TorchCompiledLayer],
    ) -> bool:
        if values.size <= 1:
            return True
        otherLayers, firstKey = self.layersAtWithKey(float(values[0]))
        del otherLayers
        for wavelength in values[1:]:
            layers, key = self.layersAtWithKey(float(wavelength))
            if key != firstKey or len(layers) != len(firstLayers):
                return False
        return True

    def spectrumChunkSize(self) -> int:
        estimatedOrders = self.estimatedRectangularHarmonics()
        if estimatedOrders <= 64:
            return 32
        return 16

    def spectrumEnergyError(
        self,
        values: ComplexArray,
        powers: dict[str, tuple[ComplexArray, ComplexArray]],
    ) -> dict[str, ComplexArray]:
        lossless = np.asarray([self.stackLosslessAt(float(wavelength)) for wavelength in values], dtype=bool)
        result: dict[str, ComplexArray] = {}
        for label, (reflectionValues, transmissionValues) in powers.items():
            conservation = reflectionValues + transmissionValues
            errors = np.full(values.shape, np.nan, dtype=float)
            errors[lossless] = np.abs(conservation[lossless] - 1.0)
            result[label] = errors
        return result

    def energyErrorForWavelength(self, wavelength: float, conservation: float) -> float:
        if not self.stackLosslessAt(wavelength):
            return float("nan")
        return float(abs(conservation - 1.0))

    def stackLosslessAt(self, wavelength: float) -> bool:
        layers, key = self.layersAtWithKeyNoWarnings(wavelength)
        return self.layersLossless(layers)

    def layersLossless(self, layers: Sequence[Layer | CompiledLayer | TorchCompiledLayer]) -> bool:
        if not epsilonLossless(self.epsIncident) or not epsilonLossless(self.epsTransmission):
            return False
        return all(layerLossless(layer) for layer in layers)

    def estimatedRectangularHarmonics(self) -> int:
        orderX, orderY = normalizeOrders(self.orders)
        return max(1, (2 * orderX + 1) * (2 * orderY + 1))

    def reducedSolve(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        sAmplitude: complex,
        pAmplitude: complex,
        returnFields: bool,
        profile: bool,
    ) -> RCWAResult | None:
        layers, layerKey = self.layersAtWithKey(wavelength)
        plan = automaticOrderReductionPlan(
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
        return solveStackReducedTorch(
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
            profile=profile,
        )

    def reducedBatchSolve(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> dict[str, RCWAResult] | None:
        layers, layerKey = self.layersAtWithKey(wavelength)
        plan = automaticOrderReductionPlan(
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
        return solveBatchReducedTorch(
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

    def reducedBatchPowersSolve(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        excitations: dict[str, tuple[complex, complex]],
    ) -> tuple[dict[str, tuple[float, float]], bool] | None:
        layers, layerKey = self.layersAtWithKey(wavelength)
        plan = automaticOrderReductionPlan(
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
        return (
            solveBatchReducedPowersTorch(
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
            ),
            self.layersLossless(layers),
        )

    def preparedTorchStack(
        self,
        wavelength: float,
        *,
        theta: float,
        phi: float,
        profile: bool = False,
        useCache: bool = True,
        powersOnly: bool = False,
    ) -> PreparedTorchStack:
        layers, layerKey = self.layersAtWithKey(wavelength)
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
            bool(profile),
            bool(powersOnly),
            layerKey,
        )
        if self.cacheModes and not profile and useCache:
            with self.cacheLock:
                cached = self.preparedCache.get(cacheKey)
                if cached is not None:
                    self.preparedCache.move_to_end(cacheKey)
                    return cached

        torchLayers = self.torchLayers(layers)
        prepare = prepareStackPowersTorch if powersOnly else prepareStackTorch
        prepared = prepare(
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
            profile=profile,
        )
        if self.cacheModes and not profile and useCache:
            with self.cacheLock:
                self.preparedCache[cacheKey] = prepared
                while len(self.preparedCache) > self.cacheSize:
                    self.preparedCache.popitem(last=False)
        return prepared

    def torchLayers(self, layers: Sequence[Layer | CompiledLayer | TorchCompiledLayer]) -> list[Layer | CompiledLayer | TorchCompiledLayer]:
        return [self.torchLayer(layer) for layer in layers]

    def torchLayer(self, layer: Layer | CompiledLayer | TorchCompiledLayer) -> Layer | CompiledLayer | TorchCompiledLayer:
        if isinstance(layer, TorchCompiledLayer):
            return layer
        if not hasattr(layer, "epsilonMatrix"):
            return layer
        if getattr(layer, "homogeneousEpsilon", None) is not None and getattr(layer, "displacementMatrices", None) is None:
            return layer

        cacheKey = (id(layer), self.backend)
        with self.cacheLock:
            cached = self.torchLayerCache.get(cacheKey)
            if cached is not None:
                return cached

        backend = resolveBackend(self.backend)
        torch = backend.xp
        device = backend.device if backend.device is not None else torch.device("cuda")
        displacementMatrices = getattr(layer, "displacementMatrices", None)
        torchLayer = TorchCompiledLayer(
            thickness=float(layer.thickness),
            epsilonMatrix=asTorchComplex(getattr(layer, "epsilonMatrix"), torch, device),
            epsilonInverse=asTorchComplex(getattr(layer, "epsilonInverse"), torch, device),
            orders=getattr(layer, "orders"),
            truncation=getattr(layer, "truncation", "rectangular"),
            name=getattr(layer, "name", ""),
            displacementMatrices=None
            if displacementMatrices is None
            else tuple(asTorchComplex(matrix, torch, device) for matrix in displacementMatrices),
            factorization=getattr(layer, "factorization", "standard"),
            homogeneousEpsilon=getattr(layer, "homogeneousEpsilon", None),
            sampleShape=getattr(layer, "sampleShape", None),
        )
        with self.cacheLock:
            cached = self.torchLayerCache.get(cacheKey)
            if cached is not None:
                return cached
            self.torchLayerCache[cacheKey] = torchLayer
        return torchLayer

    def layersAt(self, wavelength: float) -> list[Layer | CompiledLayer | TorchCompiledLayer]:
        return self.layersAtWithKey(wavelength)[0]

    def layersAtWithKey(self, wavelength: float) -> tuple[list[Layer | CompiledLayer | TorchCompiledLayer], tuple]:
        layers, keys = self.layersAtWithKeyNoWarnings(wavelength)
        validateIsotropicLayers(expandAdaptiveLayers(layers), kind="solve")
        self.warnIfUndersampled(layers)
        return layers, keys

    def layersAtWithKeyNoWarnings(self, wavelength: float) -> tuple[list[Layer | CompiledLayer | TorchCompiledLayer], tuple]:
        validateIsotropicLayers(expandAdaptiveLayers(self.layers), kind="solve")
        layers: list[Layer | CompiledLayer | TorchCompiledLayer] = []
        keys = []
        for item in self.layers:
            materialized, key = self.materializeLayer(item, wavelength)
            if isinstance(materialized, list):
                layers.extend(materialized)
            else:
                layers.append(materialized)
            keys.append(key)
        return layers, tuple(keys)

    def materializeLayer(self, item: LayerInput, wavelength: float) -> tuple[Layer | CompiledLayer | TorchCompiledLayer | list, tuple]:
        if isinstance(item, CompiledLayer):
            return item, ("compiled", id(item), item.thickness, item.orders, item.truncation)
        if isinstance(item, Layer):
            return self.compileIfRequested(item, ("layer", id(item)))
        if isinstance(item, LayerSpec):
            layer = item.at(wavelength)
            key = ("layerspec", id(item), None if item.isStatic else float(wavelength))
            return self.compileIfRequested(layer, key)
        if isinstance(item, AdaptiveLayerSpec):
            layers = []
            keys = []
            for index, adaptiveLayer in enumerate(item.toLayers()):
                materialized, key = self.compileIfRequested(adaptiveLayer, ("adaptive", id(item), index))
                layers.append(materialized)
                keys.append(key)
            return layers, ("adaptive", id(item), tuple(keys))
        if isinstance(item, PatternLayer):
            layer = item.toLayer()
            return self.compileIfRequested(layer, ("pattern", id(item), item.version))
        if callable(item):
            produced = item(wavelength)
            if isinstance(produced, (Layer, CompiledLayer)):
                return self.materializeLayer(produced, wavelength)
            layers = []
            keys = []
            for producedLayer in produced:
                layer, key = self.materializeLayer(producedLayer, wavelength)
                layers.append(layer)
                keys.append(key)
            return layers, ("factory", id(item), float(wavelength), tuple(keys))
        raise TypeError(f"unsupported layer input {type(item)!r}")

    def compileIfRequested(self, layer: Layer, key: tuple) -> tuple[Layer | TorchCompiledLayer, tuple]:
        if not self.precompile:
            return layer, key
        compileKey = (key, self.orders, self.truncation)
        with self.cacheLock:
            cached = self.compiledLayerCache.get(compileKey)
            if cached is not None:
                return cached, ("compiled-cache", compileKey)

        compiled = self.compileLayerTorch(layer)
        with self.cacheLock:
            cached = self.compiledLayerCache.get(compileKey)
            if cached is not None:
                return cached, ("compiled-cache", compileKey)
            self.compiledLayerCache[compileKey] = compiled
        return compiled, ("compiled-cache", compileKey)

    def compileLayerTorch(self, layer: Layer) -> TorchCompiledLayer:
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
        return TorchCompiledLayer(
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
            sampleShape=getattr(layer, "sampleShape", sampleShape(layer)),
        )

    def warnIfUndersampled(self, layers: Sequence[Layer | CompiledLayer | TorchCompiledLayer]) -> None:
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
                    f"use analytic geometry or min(shape) >= {minimumRecommended} to reduce staircasing error",
                    RuntimeWarning,
                    stacklevel=3,
                )


def polarizationAmplitudes(polarization: Polarization) -> tuple[complex, complex]:
    value = str(polarization).upper()
    if value in ("TE", "S"):
        return 1.0, 0.0
    if value in ("TM", "P"):
        return 0.0, 1.0
    raise ValueError("polarization must be 'TE', 'TM', 's', or 'p'")


def polarizationLabel(polarization: Polarization) -> str:
    value = str(polarization).upper()
    if value == "S":
        return "TE"
    if value == "P":
        return "TM"
    return value


def torchBackendLabel(backend: str) -> str:
    if backend != "cuda":
        raise ValueError("the isotropic solver is CUDA-only")
    return "cuda"


def sampleShape(layer: object) -> tuple[int, int] | None:
    epsilon = getattr(layer, "epsilon", None)
    return sampleShapeFromEpsilon(epsilon)


def sampleShapeFromEpsilon(epsilon: object) -> tuple[int, int] | None:
    if epsilon is None or hasattr(epsilon, "convolutionMatrix"):
        return None
    array = np.asarray(epsilon)
    if array.ndim == 2 and array.shape != (3, 3):
        return int(array.shape[0]), int(array.shape[1])
    return None


def epsilonLossless(value: object) -> bool:
    array = np.asarray(value, dtype=complex)
    if array.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(array))))
    return bool(np.max(np.abs(np.imag(array))) <= 1e-12 * scale)


def asTorchComplex(value: Any, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex128, device=device)
