from __future__ import annotations

import numpy as np


ComplexArray = np.ndarray


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
