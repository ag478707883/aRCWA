from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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
    entries = _harmonicEntries(nx, ny, truncation)

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
        truncation=_normalizeTruncation(truncation),
    )


def normalizeOrders(orders: OrderSpec) -> tuple[int, int]:
    if isinstance(orders, int):
        return int(orders), int(orders)
    if len(orders) != 2:
        raise ValueError("orders must be an int or a two-item tuple")
    return int(orders[0]), int(orders[1])


def epsilonConvolutionMatrix(epsilon: complex | ArrayLike, harmonics: Harmonics) -> ComplexArray:
    """Build the Fourier convolution matrix for sampled scalar permittivity."""

    if hasattr(epsilon, "convolutionMatrix"):
        return np.asarray(epsilon.convolutionMatrix(harmonics), dtype=complex)

    if np.isscalar(epsilon):
        return complex(epsilon) * np.eye(harmonics.count, dtype=complex)

    grid = np.asarray(epsilon, dtype=complex)
    if grid.ndim == 0:
        return complex(grid.item()) * np.eye(harmonics.count, dtype=complex)
    if grid.ndim != 2:
        raise ValueError("sampled epsilon must be a 2D array with shape (ny, nx)")
    if grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("sampled epsilon grid must be non-empty")

    ny, nx = grid.shape
    coeffs = np.fft.fft2(grid) / grid.size
    dmx = harmonics.mx[:, None] - harmonics.mx[None, :]
    dmy = harmonics.my[:, None] - harmonics.my[None, :]
    # Pattern2D samples are cell-centered on [-period/2, period/2); FFT bins start at index zero.
    centeredCellPhase = np.exp(1j * np.pi * (dmx + dmy)) * np.exp(-1j * np.pi * (dmx / nx + dmy / ny))
    return coeffs[dmy % ny, dmx % nx] * centeredCellPhase


def _harmonicEntries(nx: int, ny: int, truncation: str) -> list[tuple[int, int]]:
    if nx < 0 or ny < 0:
        raise ValueError("orders must be non-negative")
    truncation = _normalizeTruncation(truncation)
    entries: list[tuple[int, int]] = []
    for my in range(-ny, ny + 1):
        for mx in range(-nx, nx + 1):
            if truncation == "circular" and not _insideCircularDomain(mx, my, nx, ny):
                continue
            entries.append((mx, my))
    if not entries:
        entries.append((0, 0))
    return entries


def _insideCircularDomain(mx: int, my: int, nx: int, ny: int) -> bool:
    if nx == 0 and ny == 0:
        return mx == 0 and my == 0
    if nx == 0:
        return mx == 0 and abs(my) <= ny
    if ny == 0:
        return my == 0 and abs(mx) <= nx
    return (mx / nx) ** 2 + (my / ny) ** 2 <= 1.0 + 1e-12


def _normalizeTruncation(truncation: str) -> str:
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
