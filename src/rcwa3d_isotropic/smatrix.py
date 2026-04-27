from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ComplexArray = np.ndarray


@dataclass(frozen=True)
class SMatrix:
    s11: ComplexArray
    s12: ComplexArray
    s21: ComplexArray
    s22: ComplexArray


def identitySMatrix(size: int) -> SMatrix:
    zero = np.zeros((size, size), dtype=complex)
    identity = np.eye(size, dtype=complex)
    return SMatrix(s11=zero.copy(), s12=identity.copy(), s21=identity.copy(), s22=zero.copy())


def interfaceSMatrix(
    leftForward: ComplexArray,
    leftBackward: ComplexArray,
    rightForward: ComplexArray,
    rightBackward: ComplexArray,
) -> SMatrix:
    """Scattering matrix for one modal interface."""

    size = leftForward.shape[1]
    matrix = np.concatenate([leftBackward, -rightForward], axis=1)
    rhsLeft = -leftForward
    rhsRight = rightBackward
    solved = np.linalg.solve(matrix, np.concatenate([rhsLeft, rhsRight], axis=1))
    return SMatrix(
        s11=solved[:size, :size],
        s12=solved[:size, size:],
        s21=solved[size:, :size],
        s22=solved[size:, size:],
    )


def propagationSMatrix(propagation: ComplexArray) -> SMatrix:
    zero = np.zeros_like(propagation)
    return SMatrix(s11=zero.copy(), s12=propagation.copy(), s21=propagation.copy(), s22=zero.copy())


def redhefferStar(left: SMatrix, right: SMatrix) -> SMatrix:
    """Cascade two scattering matrices from left to right."""

    size = left.s11.shape[0]
    identity = np.eye(size, dtype=complex)
    leftDenominator = np.linalg.solve(identity - right.s11 @ left.s22, right.s11)
    rightDenominator = np.linalg.solve(identity - left.s22 @ right.s11, left.s22)
    s11 = left.s11 + left.s12 @ leftDenominator @ left.s21
    s12 = left.s12 @ np.linalg.solve(identity - right.s11 @ left.s22, right.s12)
    s21 = right.s21 @ np.linalg.solve(identity - left.s22 @ right.s11, left.s21)
    s22 = right.s22 + right.s21 @ rightDenominator @ right.s12
    return SMatrix(s11=s11, s12=s12, s21=s21, s22=s22)


def cascadeMany(components: Sequence[SMatrix], size: int) -> SMatrix:
    if not components:
        return identitySMatrix(size)
    result = components[0]
    for component in components[1:]:
        result = redhefferStar(result, component)
    return result


def prefixSMatrices(components: Sequence[SMatrix], size: int) -> list[SMatrix]:
    prefixes = [identitySMatrix(size)]
    current = prefixes[0]
    for component in components:
        current = redhefferStar(current, component)
        prefixes.append(current)
    return prefixes


def suffixSMatrices(components: Sequence[SMatrix], size: int) -> list[SMatrix]:
    suffixes = [identitySMatrix(size) for _ in range(len(components) + 1)]
    current = identitySMatrix(size)
    suffixes[len(components)] = current
    for index in range(len(components) - 1, -1, -1):
        current = redhefferStar(components[index], current)
        suffixes[index] = current
    return suffixes
