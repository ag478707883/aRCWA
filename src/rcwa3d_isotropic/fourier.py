from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike


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
    if np.all(grid == grid.flat[0]):
        return complex(grid.flat[0]) * np.eye(harmonics.count, dtype=complex)
    if harmonics.count == 1:
        return np.array([[np.mean(grid)]], dtype=complex)

    ny, nx = grid.shape
    coeffs = np.fft.fft2(grid) / grid.size
    dmx = harmonics.deltaMx
    dmy = harmonics.deltaMy
    # Pattern2D samples are cell-centered on [-period/2, period/2); FFT bins start at index zero.
    centeredCellPhase = np.exp(1j * np.pi * (dmx + dmy)) * np.exp(-1j * np.pi * (dmx / nx + dmy / ny))
    return coeffs[dmy % ny, dmx % nx] * centeredCellPhase


def epsilonConvolutionMatrixTorch(
    epsilon: complex | ArrayLike,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
) -> Any:
    """Build a Fourier convolution matrix directly on a PyTorch device."""

    if hasattr(epsilon, "convolutionMatrixTorch"):
        return epsilon.convolutionMatrixTorch(harmonics, torch, device).to(device=device, dtype=torch.complex128)
    if hasattr(epsilon, "convolutionMatrix"):
        return _asTorchComplex(epsilon.convolutionMatrix(harmonics), torch, device)

    if _torchOrPythonScalar(epsilon, torch):
        return complex(epsilon) * torch.eye(harmonics.count, dtype=torch.complex128, device=device)

    grid = _asTorchComplex(epsilon, torch, device)
    if grid.ndim == 0:
        return grid.reshape(())[()] * torch.eye(harmonics.count, dtype=torch.complex128, device=device)
    if grid.ndim != 2:
        raise ValueError("sampled epsilon must be a 2D array with shape (ny, nx)")
    if grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("sampled epsilon grid must be non-empty")
    if bool(torch.all(grid == grid.reshape(-1)[0]).item()):
        return grid.reshape(-1)[0] * torch.eye(harmonics.count, dtype=torch.complex128, device=device)
    if harmonics.count == 1:
        return torch.mean(grid).reshape(1, 1).to(torch.complex128)

    ny, nx = grid.shape
    coeffs = torch.fft.fft2(grid) / grid.numel()
    dmx = _asTorchLong(harmonics.deltaMx, torch, device)
    dmy = _asTorchLong(harmonics.deltaMy, torch, device)
    dmxFloat = dmx.to(torch.float64)
    dmyFloat = dmy.to(torch.float64)
    pi = torch.as_tensor(np.pi, dtype=torch.float64, device=device)
    centeredCellPhase = torch.exp(1j * pi * (dmxFloat + dmyFloat)) * torch.exp(
        -1j * pi * (dmxFloat / nx + dmyFloat / ny)
    )
    return coeffs[torch.remainder(dmy, ny), torch.remainder(dmx, nx)] * centeredCellPhase


def _asTorchComplex(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex128, device=device)


def _asTorchLong(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(value), dtype=torch.long, device=device)


def _torchOrPythonScalar(value: object, torch: Any) -> bool:
    return np.isscalar(value) or (isinstance(value, torch.Tensor) and value.ndim == 0)


def sqrtBranch(value: complex) -> complex:
    """Square-root branch with non-negative imaginary part."""

    root = np.sqrt(value + 0j)
    if np.imag(root) < -1e-14:
        root = -root
    if abs(np.imag(root)) <= 1e-14 and np.real(root) < 0:
        root = -root
    return complex(root)


def forwardKz(kzSquared: complex | ComplexArray) -> ComplexArray:
    """Forward-going kz branch for normalized wave vectors."""

    roots = np.sqrt(np.asarray(kzSquared, dtype=complex) + 0j)
    flip = (np.imag(roots) < -1e-14) | ((np.abs(np.imag(roots)) <= 1e-14) & (np.real(roots) < 0))
    roots = np.where(flip, -roots, roots)
    return roots


def planeWaveFields(kx: complex, ky: complex, kz: complex, eps: complex) -> tuple[ComplexArray, ComplexArray]:
    """Return tangential [Ex, Ey, Hx, Hy] fields for s and p polarizations."""

    kp = np.sqrt(kx * kx + ky * ky + 0j)
    if abs(kp) < 1e-14:
        s = np.array([0.0 + 0j, 1.0 + 0j, 0.0 + 0j])
    else:
        s = np.array([-ky / kp, kx / kp, 0.0 + 0j], dtype=complex)

    kVector = np.array([kx, ky, kz], dtype=complex)
    refractiveIndex = sqrtBranch(eps)
    p = np.cross(s, kVector) / refractiveIndex

    hS = np.cross(kVector, s)
    hP = np.cross(kVector, p)
    sField = np.array([s[0], s[1], hS[0], hS[1]], dtype=complex)
    pField = np.array([p[0], p[1], hP[0], hP[1]], dtype=complex)
    return sField, pField


def putOrderField(target: ComplexArray, orderIndex: int, values: ComplexArray) -> None:
    nOrders = target.size // 4
    target[orderIndex] = values[0]
    target[nOrders + orderIndex] = values[1]
    target[2 * nOrders + orderIndex] = values[2]
    target[3 * nOrders + orderIndex] = values[3]


def singleOrderVector(nOrders: int, orderIndex: int, values: ComplexArray) -> ComplexArray:
    vector = np.zeros(4 * nOrders, dtype=complex)
    putOrderField(vector, orderIndex, values)
    return vector


def flux(field: ComplexArray) -> float:
    """Real z-directed Poynting flux for tangential Fourier coefficients."""

    nOrders = field.size // 4
    ex = field[:nOrders]
    ey = field[nOrders : 2 * nOrders]
    hx = field[2 * nOrders : 3 * nOrders]
    hy = field[3 * nOrders :]
    return float(0.5 * np.real(np.sum(ex * np.conj(hy) - ey * np.conj(hx))))
