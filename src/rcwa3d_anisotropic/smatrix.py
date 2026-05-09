from __future__ import annotations

import os
from typing import Sequence

from .backend import ArrayBackend as ArrayBackend
from .types import SMatrix


def propagationSMatrixBidirectional(forward: object, backward: object, backend: ArrayBackend) -> SMatrix:
    xp = backend.xp
    forwardDiagonal = matrixDiagonal(forward)
    backwardDiagonal = matrixDiagonal(backward)
    zero = xp.zeros((forwardDiagonal.shape[0], forwardDiagonal.shape[0]), dtype=complex)
    return SMatrix(
        s11=zero,
        s12=backend.copy(backwardDiagonal),
        s21=backend.copy(forwardDiagonal),
        s22=backend.copy(zero),
        isPropagation=True,
    )


def propagationSMatrixBidirectionalBatch(forward: object, backward: object, backend: ArrayBackend) -> SMatrix:
    xp = backend.xp
    zero = xp.zeros((forward.shape[0], forward.shape[1], forward.shape[1]), dtype=complex)
    return SMatrix(
        s11=zero,
        s12=backend.copy(backward),
        s21=backend.copy(forward),
        s22=backend.copy(zero),
        isPropagation=True,
    )


def identitySMatrix(size: int, backend: ArrayBackend) -> SMatrix:
    xp = backend.xp
    zero = xp.zeros((size, size), dtype=complex)
    identity = xp.eye(size, dtype=complex)
    return SMatrix(
        s11=backend.copy(zero),
        s12=backend.copy(identity),
        s21=backend.copy(identity),
        s22=backend.copy(zero),
        isIdentity=True,
    )


def redhefferStar(left: SMatrix, right: SMatrix, backend: ArrayBackend) -> SMatrix:
    if left.isIdentity:
        return right
    if right.isIdentity:
        return left
    if right.isPropagation:
        forward = propagationDiagonal(right, right.s21)
        backward = propagationDiagonal(right, right.s12)
        if getattr(forward, "ndim", 1) == 2:
            return SMatrix(
                s11=left.s11,
                s12=left.s12 * backward[:, None, :],
                s21=forward[:, :, None] * left.s21,
                s22=forward[:, :, None] * left.s22 * backward[:, None, :],
            )
        return SMatrix(
            s11=left.s11,
            s12=left.s12 * backward[None, :],
            s21=forward[:, None] * left.s21,
            s22=forward[:, None] * left.s22 * backward[None, :],
        )
    if left.isPropagation:
        forward = propagationDiagonal(left, left.s21)
        backward = propagationDiagonal(left, left.s12)
        if getattr(forward, "ndim", 1) == 2:
            return SMatrix(
                s11=backward[:, :, None] * right.s11 * forward[:, None, :],
                s12=backward[:, :, None] * right.s12,
                s21=right.s21 * forward[:, None, :],
                s22=right.s22,
            )
        return SMatrix(
            s11=backward[:, None] * right.s11 * forward[None, :],
            s12=backward[:, None] * right.s12,
            s21=right.s21 * forward[None, :],
            s22=right.s22,
        )

    size = left.s11.shape[-1]
    identity = identityLikeSMatrixBlock(left.s11, size, backend)
    leftFactor = identity - right.s11 @ left.s22
    rightFactor = identity - left.s22 @ right.s11
    leftSolved = solveFactoredBlocks(backend, leftFactor, right.s11, right.s12)
    rightSolved = solveFactoredBlocks(backend, rightFactor, left.s22, left.s21)
    leftDenominator = leftSolved[:, :size]
    leftTransmission = leftSolved[:, size:]
    rightDenominator = rightSolved[:, :size]
    rightTransmission = rightSolved[:, size:]
    return SMatrix(
        s11=left.s11 + left.s12 @ leftDenominator @ left.s21,
        s12=left.s12 @ leftTransmission,
        s21=right.s21 @ rightTransmission,
        s22=right.s22 + right.s21 @ rightDenominator @ right.s12,
    )


def matrixDiagonal(matrix: object) -> object:
    if getattr(matrix, "ndim", 2) == 1:
        return matrix
    return matrix.diagonal(dim1=-2, dim2=-1)


def propagationDiagonal(component: SMatrix, value: object) -> object:
    if getattr(component.s11, "ndim", 2) == 3 and getattr(value, "ndim", 1) == 2:
        return value
    return matrixDiagonal(value)


def identityLikeSMatrixBlock(reference: object, size: int, backend: ArrayBackend) -> object:
    identity = backend.xp.eye(size, dtype=complex)
    if getattr(reference, "ndim", 2) == 3:
        return identity[None, :, :].expand(reference.shape[0], -1, -1).clone()
    return identity


def solveFactored(backend: ArrayBackend, matrix: object, rhs: object) -> object:
    return backend.solveFactored(backend.factor(matrix), rhs)


def solveFactoredBlocks(backend: ArrayBackend, matrix: object, *rhsBlocks: object) -> object:
    if len(rhsBlocks) == 1:
        return solveFactored(backend, matrix, rhsBlocks[0])
    rhs = backend.xp.concatenate(rhsBlocks, axis=-1)
    return backend.solveFactored(backend.factor(matrix), rhs)


def solveInterfaceBlocks(backend: ArrayBackend, matrix: object, *rhsBlocks: object) -> object:
    if len(rhsBlocks) == 1:
        rhs = rhsBlocks[0]
    else:
        rhs = backend.xp.concatenate(rhsBlocks, axis=-1)

    method = interfaceSolveMethod()
    if method == "solve":
        return backend.solve(matrix, rhs)
    if method == "lu":
        return backend.solveFactored(backend.factor(matrix), rhs)
    raise ValueError("RCWA3D_INTERFACE_SOLVER must be 'lu' or 'solve'")


def interfaceSolveMethod() -> str:
    return os.environ.get("RCWA3D_INTERFACE_SOLVER", "lu").strip().lower()


def patternedHomogeneousInterfaceEnabled() -> bool:
    value = os.environ.get("RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def cascadeMany(components: Sequence[SMatrix], size: int, backend: ArrayBackend) -> SMatrix:
    active = [component for component in components if not component.isIdentity]
    if not active:
        return identitySMatrix(size, backend)
    while len(active) > 1:
        nextLevel: list[SMatrix] = []
        for index in range(0, len(active) - 1, 2):
            nextLevel.append(redhefferStar(active[index], active[index + 1], backend))
        if len(active) % 2:
            nextLevel.append(active[-1])
        active = nextLevel
    return active[0]


def reflectionTransmissionOnlySMatrix(
    components: Sequence[SMatrix],
    size: int,
    backend: ArrayBackend,
) -> SMatrix:
    reflection, transmission = enhancedReflectionTransmission(components, size, backend)
    zero = backend.xp.zeros_like(reflection)
    return SMatrix(
        s11=reflection,
        s12=backend.copy(zero),
        s21=transmission,
        s22=backend.copy(zero),
    )


def enhancedReflectionTransmission(
    components: Sequence[SMatrix],
    size: int,
    backend: ArrayBackend,
) -> tuple[object, object]:
    xp = backend.xp
    firstBlock = next(
        (
            block
            for component in components
            for block in (component.s11, component.s12, component.s21, component.s22)
            if getattr(block, "ndim", 2) == 3
        ),
        None,
    )
    if firstBlock is None:
        identity = xp.eye(size, dtype=complex)
        reflection = xp.zeros((size, size), dtype=complex)
        transmission = backend.copy(identity)
    else:
        batchSize = int(firstBlock.shape[0])
        identity = xp.eye(size, dtype=complex)[None, :, :].expand(batchSize, -1, -1).clone()
        reflection = xp.zeros((batchSize, size, size), dtype=complex)
        transmission = backend.copy(identity)
    for component in reversed(components):
        if component.isIdentity:
            continue
        if component.isPropagation:
            forward = propagationDiagonal(component, component.s21)
            backward = propagationDiagonal(component, component.s12)
            if getattr(forward, "ndim", 1) == 2:
                reflection = backward[:, :, None] * reflection * forward[:, None, :]
                transmission = transmission * forward[:, None, :]
            else:
                reflection = backward[:, None] * reflection * forward[None, :]
                transmission = transmission * forward[None, :]
            continue
        internalReflection = solveFactored(
            backend,
            identity - reflection @ component.s22,
            reflection @ component.s21,
        )
        forward = component.s21 + component.s22 @ internalReflection
        transmission = transmission @ forward
        reflection = component.s11 + component.s12 @ internalReflection
    return reflection, transmission


def prefixSMatrices(components: Sequence[SMatrix], size: int, backend: ArrayBackend) -> list[SMatrix]:
    prefixes = [identitySMatrix(size, backend)]
    current = prefixes[0]
    for component in components:
        current = redhefferStar(current, component, backend)
        prefixes.append(current)
    return prefixes


def suffixSMatrices(components: Sequence[SMatrix], size: int, backend: ArrayBackend) -> list[SMatrix]:
    suffixes = [identitySMatrix(size, backend) for ignored in range(len(components) + 1)]
    current = identitySMatrix(size, backend)
    suffixes[len(components)] = current
    for index in range(len(components) - 1, -1, -1):
        current = redhefferStar(components[index], current, backend)
        suffixes[index] = current
    return suffixes
