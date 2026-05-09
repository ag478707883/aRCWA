from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .builder import LayerStack, PatternLayer
from .geometry import (
    SampledPattern,
    circularPostLayer,
    ellipticalPostLayer,
    polygonPostLayer,
    rectangularHollowPostLayer,
    rectangularPostLayer,
    stack,
)
from .simulation import LayerSpec, Polarization, RCWASimulation, homogeneousLayer
from .solver import Layer, RCWAResult


ComplexArray = np.ndarray
Material = Any
LayerInput = Any
Shape = Literal["circle", "ellipse", "rectangle"]
ExcitationMap = Mapping[str, tuple[complex, complex]]


@dataclass
class AnisotropicRCWA:
    """Readable project-style interface for tensor-material RCWA models.

    A custom structure can be written as a short sequence of layer additions:

    ```python
    model = rcwa.AnisotropicRCWA(period=(5.33, 5.33), order=3)
    model.add_cylinder(height=5.94, radius=1.065, material=3.48**2)
    model.add_uniform(height=1.962, material=inas_tensor)
    model.add_uniform(height=0.50, material=ag_tensor)
    spectrum = model.spectrum(np.linspace(11.7, 12.0, 101), theta_deg=33)
    ```

    For a more explicit layer-first build style, create the layer first and
    then draw patterns on it:

    ```python
    top = model.add_patterned_layer(height=5.94, background=1.0, samples=96)
    top.circle(radius=1.065, material=3.48**2)
    ```

    Static materials may be scalars, 3x3 tensors, or sampled tensor grids.
    Wavelength-dependent material callables must return one 3x3 tensor.
    Length units are arbitrary but must be consistent.
    The stable CUDA S-matrix route is the only solver path exposed through
    this interface.
    """

    period: tuple[float, float]
    order: int | tuple[int, int] = 3
    truncation: Literal["circular", "rectangular"] = "circular"
    incident: complex = 1.0
    transmission: complex = 1.0
    backend: str = "cuda"
    precision: Literal["complex128", "complex64", "mixed"] = "complex128"
    samples: tuple[int, int] = (128, 128)
    precompile: bool = True
    workers: int = 1
    layers: list[LayerInput] = field(default_factory=list)
    simulationCache: RCWASimulation | None = field(default=None, init=False, repr=False)
    simulationKey: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def add_uniform(self, height: float, material: Material, *, name: str = "") -> "AnisotropicRCWA":
        """Append a homogeneous layer."""

        self.layers.append(homogeneousLayer(height, material, name=name))
        return self

    def add_layer(self, layer: LayerInput | Sequence[LayerInput]) -> "AnisotropicRCWA":
        """Append an existing layer or a stack returned by a geometry helper."""

        if isinstance(layer, LayerStack):
            self.layers.extend(layer.toLayers())
            return self
        if isinstance(layer, (Layer, LayerSpec, PatternLayer)) or callable(layer):
            self.layers.append(layer)
            return self
        self.layers.extend(layer)
        return self

    def geometry_stack(self, *, samples: int | tuple[int, int] | None = None) -> LayerStack:
        """Create an editable 3D layer stack using the project's period."""

        return LayerStack(period=self.period, shape=normalizeSamples(samples, self.samples))

    def add_patterned_layer(
        self,
        *,
        height: float,
        background: Material = 1.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "patterned layer",
    ) -> PatternLayer:
        """Append a mutable layer, then draw patterns on it afterwards.

        Example
        -------
        ```python
        layer = model.add_patterned_layer(height=5.94, background=1.0, samples=96)
        layer.circle(radius=1.065, material=3.48**2)
        ```
        """

        patterned = PatternLayer(
            period=self.period,
            thickness=height,
            background=background,
            shape=normalizeSamples(samples, self.samples),
            name=name,
            factorization=factorization,
        )
        self.layers.append(patterned)
        return patterned

    def add_cylinder(
        self,
        *,
        height: float,
        radius: float,
        material: Material,
        background: Material = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "cylindrical grating",
    ) -> "AnisotropicRCWA":
        """Append a sampled circular post layer."""

        self.layers.append(
            circularPostLayer(
                period=self.period,
                thickness=height,
                background=background,
                post=material,
                radius=radius,
                shape=normalizeSamples(samples, self.samples),
                center=center,
                factorization=factorization,
                name=name,
            )
        )
        return self

    def add_rectangle(
        self,
        *,
        height: float,
        size: tuple[float, float],
        material: Material,
        background: Material = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
        angle_deg: float = 0.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "rectangular grating",
    ) -> "AnisotropicRCWA":
        """Append a sampled rectangular post layer."""

        self.layers.append(
            rectangularPostLayer(
                period=self.period,
                thickness=height,
                background=background,
                post=material,
                size=size,
                center=center,
                angle=np.deg2rad(angle_deg),
                shape=normalizeSamples(samples, self.samples),
                factorization=factorization,
                name=name,
            )
        )
        return self

    def add_ellipse(
        self,
        *,
        height: float,
        radii: tuple[float, float],
        material: Material,
        background: Material = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
        angle_deg: float = 0.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "elliptical grating",
    ) -> "AnisotropicRCWA":
        """Append an elliptical post layer."""

        self.layers.append(
            ellipticalPostLayer(
                period=self.period,
                thickness=height,
                background=background,
                post=material,
                radii=radii,
                center=center,
                angle=np.deg2rad(angle_deg),
                shape=normalizeSamples(samples, self.samples),
                factorization=factorization,
                name=name,
            )
        )
        return self

    def add_rectangular_hole_array(
        self,
        *,
        height: float,
        size: tuple[float, float],
        hole_radius: float,
        material: Material,
        background: Material = 1.0,
        hole_material: Material | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        angle_deg: float = 0.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "rectangular hollow grating",
    ) -> "AnisotropicRCWA":
        """Append a rectangular post with a circular through-hole."""

        self.layers.append(
            rectangularHollowPostLayer(
                period=self.period,
                thickness=height,
                background=background,
                post=material,
                size=size,
                holeRadius=hole_radius,
                holeMaterial=hole_material,
                center=center,
                angle=np.deg2rad(angle_deg),
                shape=normalizeSamples(samples, self.samples),
                factorization=factorization,
                name=name,
            )
        )
        return self

    def add_polygon(
        self,
        *,
        height: float,
        vertices: Iterable[tuple[float, float]],
        material: Material,
        background: Material = 1.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "polygon grating",
    ) -> "AnisotropicRCWA":
        """Append a sampled polygon post layer."""

        self.layers.append(
            polygonPostLayer(
                period=self.period,
                thickness=height,
                background=background,
                post=material,
                vertices=vertices,
                shape=normalizeSamples(samples, self.samples),
                factorization=factorization,
                name=name,
            )
        )
        return self

    def pattern(
        self,
        *,
        background: Material = 1.0,
        samples: int | tuple[int, int] | None = None,
        name: str = "",
    ) -> SampledPattern:
        """Create a sampled unit-cell pattern for custom drawing."""

        return SampledPattern(period=self.period, shape=normalizeSamples(samples, self.samples), background=background, name=name)

    def add_pattern(self, pattern: SampledPattern, *, height: float, name: str | None = None) -> "AnisotropicRCWA":
        """Append a user-drawn sampled pattern."""

        self.layers.append(pattern.toLayer(height, name=name))
        return self

    def add_taper(
        self,
        *,
        height: float,
        bottom_size: float | tuple[float, float],
        top_size: float | tuple[float, float],
        material: Material,
        background: Material = 1.0,
        shape: Shape = "rectangle",
        slices: int = 20,
        angle_deg: float = 0.0,
        samples: int | tuple[int, int] | None = None,
        factorization: Literal["auto", "standard", "normal-vector"] = "auto",
        name: str = "taper",
    ) -> "AnisotropicRCWA":
        """Append a taper approximated by many constant cross-section layers."""

        self.layers.extend(
            stack(
                sliced_taper(
                    period=self.period,
                    height=height,
                    background=background,
                    material=material,
                    bottomSize=bottom_size,
                    topSize=top_size,
                    kind=shape,
                    slices=slices,
                    angle=np.deg2rad(angle_deg),
                    samples=normalizeSamples(samples, self.samples),
                    factorization=factorization,
                    name=name,
                )
            )
        )
        return self

    def simulation(self) -> RCWASimulation:
        """Materialize the lower-level simulation object."""

        key = (
            tuple(self.period),
            self.order,
            self.truncation,
            complex(self.incident),
            complex(self.transmission),
            self.backend,
            self.precision,
            self.precompile,
            self.workers,
            tuple(layer_key(layer) for layer in self.layers),
        )
        if self.simulationCache is not None and self.simulationKey == key:
            return self.simulationCache

        self.simulationCache = RCWASimulation(
            period=self.period,
            layers=tuple(materialize_layer(layer) for layer in self.layers),
            orders=self.order,
            truncation=self.truncation,
            backend=self.backend,
            precision=self.precision,
            epsIncident=self.incident,
            epsTransmission=self.transmission,
            precompile=self.precompile,
            workers=self.workers,
        )
        self.simulationKey = key
        return self.simulationCache

    def solve(
        self,
        wavelength: float,
        *,
        theta_deg: float = 0.0,
        phi_deg: float = 0.0,
        theta: float | None = None,
        phi: float | None = None,
        polarization: Polarization = "TE",
    ) -> RCWAResult:
        """Solve one wavelength for reflection/transmission data."""

        return self.simulation().solve(
            wavelength,
            theta=angle(theta, theta_deg),
            phi=angle(phi, phi_deg),
            polarization=polarization,
        )

    def solve_fields(
        self,
        wavelength: float,
        *,
        theta_deg: float = 0.0,
        phi_deg: float = 0.0,
        theta: float | None = None,
        phi: float | None = None,
        polarization: Polarization = "TE",
    ) -> RCWAResult:
        """Solve one wavelength and include finite-layer field data."""

        return self.simulation().solveFields(
            wavelength,
            theta=angle(theta, theta_deg),
            phi=angle(phi, phi_deg),
            polarization=polarization,
        )

    def absorption(
        self,
        wavelength: float,
        *,
        theta_deg: float = 0.0,
        phi_deg: float = 0.0,
        theta: float | None = None,
        phi: float | None = None,
        polarization: Polarization = "TE",
    ) -> float:
        """Return 1 - reflection - transmission for one wavelength."""

        return self.simulation().absorption(
            wavelength,
            theta=angle(theta, theta_deg),
            phi=angle(phi, phi_deg),
            polarization=polarization,
        )

    def spectrum(
        self,
        wavelengths: Iterable[float],
        *,
        theta_deg: float = 0.0,
        phi_deg: float = 0.0,
        theta: float | None = None,
        phi: float | None = None,
        polarizations: Sequence[Polarization] = ("TE", "TM"),
        excitations: ExcitationMap | None = None,
        bidirectional: bool = True,
        workers: int | None = None,
    ) -> dict[str, dict[str, ComplexArray] | ComplexArray]:
        """Solve an absorption spectrum for one or more polarizations."""

        return self.simulation().spectrum(
            wavelengths,
            theta=angle(theta, theta_deg),
            phi=angle(phi, phi_deg),
            polarizations=polarizations,
            excitations=excitations,
            bidirectional=bidirectional,
            workers=workers,
        )


Project = AnisotropicRCWA


def normalizeSamples(value: int | tuple[int, int] | None, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError("samples must be an int or a two-item tuple")
    return int(value[0]), int(value[1])


def angle(radians: float | None, degrees: float) -> float:
    if radians is not None:
        return float(radians)
    return float(np.deg2rad(degrees))


def materialize_layer(layer: LayerInput) -> LayerInput:
    if isinstance(layer, PatternLayer):
        return layer.toLayer()
    return layer


def layer_key(layer: LayerInput) -> tuple[Any, ...]:
    if isinstance(layer, PatternLayer):
        return ("pattern", id(layer), layer.version)
    return (type(layer).__name__, id(layer))


def sliced_taper(**kwargs: Any) -> list[Layer]:
    from .geometry import slicedTaperStack

    return slicedTaperStack(
        period=kwargs["period"],
        height=kwargs["height"],
        background=kwargs["background"],
        post=kwargs["material"],
        bottomSize=kwargs["bottomSize"],
        topSize=kwargs["topSize"],
        kind=kwargs["kind"],
        slices=kwargs["slices"],
        angle=kwargs["angle"],
        shape=kwargs["samples"],
        factorization=kwargs["factorization"],
        name=kwargs["name"],
    )
