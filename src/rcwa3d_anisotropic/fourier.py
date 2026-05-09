from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from .phase import sqrtBranch


ComplexArray = np.ndarray
OrderSpec = Union[int, tuple[int, int]]


@dataclass(frozen=True)
class Harmonics:
    mx: ComplexArray
    my: ComplexArray
    kx: ComplexArray
    ky: ComplexArray
    orders: tuple[int, int]
    truncation: str = "rectangular"

    @property
    def count(self) -> int:
        return int(self.mx.size)

    @cached_property
    def deltaMx(self) -> ComplexArray:
        return self.mx[:, None] - self.mx[None, :]

    @cached_property
    def deltaMy(self) -> ComplexArray:
        return self.my[:, None] - self.my[None, :]


@dataclass(frozen=True)
class FourierConvolutionPlan:
    coeffY: ComplexArray
    coeffX: ComplexArray
    phase: ComplexArray


def makeHarmonics(
    wavelength: float,
    period: tuple[float, float],
    orders: OrderSpec,
    epsIncident: complex,
    theta: float,
    phi: float,
    truncation: str = "circular",
) -> Harmonics:
    """Create Fourier order indices and normalized in-plane wave vectors."""

    nx, ny = normalizeOrders(orders)
    entries = harmonicEntries(nx, ny, truncation)

    mxValues = np.array([item[0] for item in entries], dtype=int)
    myValues = np.array([item[1] for item in entries], dtype=int)
    nIncident = sqrtBranch(epsIncident)
    kx0 = nIncident * np.sin(theta) * np.cos(phi)
    ky0 = nIncident * np.sin(theta) * np.sin(phi)
    kx = kx0 + mxValues * wavelength / period[0]
    ky = ky0 + myValues * wavelength / period[1]
    return Harmonics(
        mx=mxValues,
        my=myValues,
        kx=kx.astype(complex),
        ky=ky.astype(complex),
        orders=(nx, ny),
        truncation=normalizeTruncation(truncation),
    )


def normalizeOrders(orders: OrderSpec) -> tuple[int, int]:
    if isinstance(orders, int):
        return int(orders), int(orders)
    if len(orders) != 2:
        raise ValueError("orders must be an int or a two-item tuple")
    return int(orders[0]), int(orders[1])


def epsilonConvolutionMatrix(epsilon: complex | ArrayLike, harmonics: Harmonics) -> ComplexArray:
    """Build the Fourier convolution matrix for sampled scalar permittivity."""

    return epsilonConvolutionMatrices((epsilon,), harmonics)[0]


def epsilonConvolutionMatrices(
    values: Sequence[complex | ArrayLike],
    harmonics: Harmonics,
) -> tuple[ComplexArray, ...]:
    """Build several Fourier convolution matrices, batching FFTs by grid shape."""

    results: list[ComplexArray | None] = [None] * len(values)
    batchedByShape: dict[tuple[int, int], list[tuple[int, ComplexArray]]] = {}

    for index, epsilon in enumerate(values):
        if hasattr(epsilon, "convolutionMatrix"):
            results[index] = np.asarray(epsilon.convolutionMatrix(harmonics), dtype=complex)
            continue

        if np.isscalar(epsilon):
            results[index] = scalarConvolutionMatrix(complex(epsilon), harmonics.count)
            continue

        grid = np.asarray(epsilon, dtype=complex)
        if grid.ndim == 0:
            results[index] = scalarConvolutionMatrix(complex(grid.item()), harmonics.count)
            continue
        if grid.ndim != 2:
            raise ValueError("sampled epsilon must be a 2D array with shape (ny, nx)")
        if grid.shape[0] < 1 or grid.shape[1] < 1:
            raise ValueError("sampled epsilon grid must be non-empty")
        if np.all(grid == grid.flat[0]):
            results[index] = scalarConvolutionMatrix(complex(grid.flat[0]), harmonics.count)
            continue
        if harmonics.count == 1:
            results[index] = np.array([[np.mean(grid)]], dtype=complex)
            continue

        shape = (int(grid.shape[0]), int(grid.shape[1]))
        batchedByShape.setdefault(shape, []).append((index, grid))

    for shape, indexedGrids in batchedByShape.items():
        ny, nx = shape
        plan = convolutionPlan(harmonics, shape)
        stack = np.stack([grid for ignored, grid in indexedGrids], axis=0)
        coeffs = np.fft.fft2(stack, axes=(-2, -1)) / (ny * nx)
        matrices = coeffs[:, plan.coeffY, plan.coeffX] * plan.phase[None, :, :]
        for matrixIndex, (resultIndex, ignored) in enumerate(indexedGrids):
            results[resultIndex] = matrices[matrixIndex]

    if any(matrix is None for matrix in results):
        raise RuntimeError("internal Fourier convolution batching did not fill every result")
    return tuple(matrix for matrix in results if matrix is not None)


def scalarConvolutionMatrix(value: complex, size: int) -> ComplexArray:
    return value * np.eye(size, dtype=complex)


def convolutionPlan(harmonics: Harmonics, shape: tuple[int, int]) -> FourierConvolutionPlan:
    return cachedConvolutionPlan(
        tuple(int(value) for value in harmonics.mx),
        tuple(int(value) for value in harmonics.my),
        int(shape[0]),
        int(shape[1]),
    )


@lru_cache(maxsize=128)
def cachedConvolutionPlan(
    mxValues: tuple[int, ...],
    myValues: tuple[int, ...],
    ny: int,
    nx: int,
) -> FourierConvolutionPlan:
    mx = np.asarray(mxValues, dtype=int)
    my = np.asarray(myValues, dtype=int)
    dmx = mx[:, None] - mx[None, :]
    dmy = my[:, None] - my[None, :]
    phase = np.exp(1j * np.pi * (dmx + dmy)) * np.exp(-1j * np.pi * (dmx / nx + dmy / ny))
    return FourierConvolutionPlan(dmy % ny, dmx % nx, phase)


def harmonicEntries(nx: int, ny: int, truncation: str) -> list[tuple[int, int]]:
    if nx < 0 or ny < 0:
        raise ValueError("orders must be non-negative")
    truncation = normalizeTruncation(truncation)
    entries: list[tuple[int, int]] = []
    for my in range(-ny, ny + 1):
        for mx in range(-nx, nx + 1):
            if truncation == "circular" and not insideCircularDomain(mx, my, nx, ny):
                continue
            entries.append((mx, my))
    if not entries:
        entries.append((0, 0))
    return entries


def insideCircularDomain(mx: int, my: int, nx: int, ny: int) -> bool:
    if nx == 0 and ny == 0:
        return mx == 0 and my == 0
    if nx == 0:
        return mx == 0 and abs(my) <= ny
    if ny == 0:
        return my == 0 and abs(mx) <= nx
    return (mx / nx) ** 2 + (my / ny) ** 2 <= 1.0 + 1e-12


def normalizeTruncation(truncation: str) -> str:
    value = truncation.lower().replace("_", "-")
    aliases = {
        "rect": "rectangular",
        "rectangle": "rectangular",
        "rectangular": "rectangular",
        "square": "rectangular",
        "circ": "circular",
        "circle": "circular",
        "circular": "circular",
    }
    if value not in aliases:
        raise ValueError("truncation must be 'rectangular' or 'circular'")
    return aliases[value]
