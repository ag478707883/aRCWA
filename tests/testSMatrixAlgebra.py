import unittest

import numpy as np

import bootstrap  # noqa: F401
import rcwa3d_anisotropic.solver as solver
from rcwa3d_anisotropic.backend import resolveBackend


def makeBackend():
    return resolveBackend("cuda")


def toBackend(smatrix: solver.SMatrix, backend) -> solver.SMatrix:
    return solver.SMatrix(
        s11=backend.asarray(smatrix.s11),
        s12=backend.asarray(smatrix.s12),
        s21=backend.asarray(smatrix.s21),
        s22=backend.asarray(smatrix.s22),
        isIdentity=smatrix.isIdentity,
        isPropagation=smatrix.isPropagation,
    )


def toNumpy(smatrix: solver.SMatrix, backend) -> solver.SMatrix:
    return solver.SMatrix(
        s11=backend.toNumpy(smatrix.s11),
        s12=backend.toNumpy(smatrix.s12),
        s21=backend.toNumpy(smatrix.s21),
        s22=backend.toNumpy(smatrix.s22),
        isIdentity=smatrix.isIdentity,
        isPropagation=smatrix.isPropagation,
    )


def randomSMatrix(rng: np.random.Generator, size: int, scale: float = 0.08) -> solver.SMatrix:
    def block() -> np.ndarray:
        return scale * (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size)))

    return solver.SMatrix(s11=block(), s12=np.eye(size) + block(), s21=np.eye(size) + block(), s22=block())


def referenceStar(left: solver.SMatrix, right: solver.SMatrix) -> solver.SMatrix:
    size = left.s11.shape[0]
    identity = np.eye(size, dtype=complex)
    return solver.SMatrix(
        s11=left.s11 + left.s12 @ np.linalg.solve(identity - right.s11 @ left.s22, right.s11 @ left.s21),
        s12=left.s12 @ np.linalg.solve(identity - right.s11 @ left.s22, right.s12),
        s21=right.s21 @ np.linalg.solve(identity - left.s22 @ right.s11, left.s21),
        s22=right.s22 + right.s21 @ np.linalg.solve(identity - left.s22 @ right.s11, left.s22 @ right.s12),
    )


class SMatrixAlgebraTests(unittest.TestCase):
    def testRedhefferStarMatchesBlockFormula(self) -> None:
        rng = np.random.default_rng(8)
        backend = makeBackend()
        left = randomSMatrix(rng, size=5)
        right = randomSMatrix(rng, size=5)

        actual = toNumpy(solver.redhefferStar(toBackend(left, backend), toBackend(right, backend), backend), backend)
        expected = referenceStar(left, right)

        np.testing.assert_allclose(actual.s11, expected.s11, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s12, expected.s12, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s21, expected.s21, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s22, expected.s22, rtol=1e-12, atol=1e-12)

    def testPropagationShortcutsMatchGeneralStarProduct(self) -> None:
        rng = np.random.default_rng(13)
        backend = makeBackend()
        middle = randomSMatrix(rng, size=6)
        forward = np.diag(np.exp(1j * rng.uniform(-0.2, 0.2, 6)))
        backward = np.diag(np.exp(1j * rng.uniform(-0.2, 0.2, 6)))
        propagation = solver.propagationSMatrixBidirectional(backend.asarray(forward), backend.asarray(backward), backend)
        dense_propagation = solver.SMatrix(
            s11=backend.toNumpy(propagation.s11),
            s12=np.diag(backend.toNumpy(solver.matrixDiagonal(propagation.s12))),
            s21=np.diag(backend.toNumpy(solver.matrixDiagonal(propagation.s21))),
            s22=backend.toNumpy(propagation.s22),
        )

        shortcut_right = toNumpy(solver.redhefferStar(toBackend(middle, backend), propagation, backend), backend)
        expected_right = referenceStar(middle, dense_propagation)
        shortcut_left = toNumpy(solver.redhefferStar(propagation, toBackend(middle, backend), backend), backend)
        expected_left = referenceStar(dense_propagation, middle)

        for actual, expected in ((shortcut_right, expected_right), (shortcut_left, expected_left)):
            np.testing.assert_allclose(actual.s11, expected.s11, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s12, expected.s12, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s21, expected.s21, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s22, expected.s22, rtol=1e-12, atol=1e-12)

    def testEnhancedReflectionTransmissionMatchesFullCascade(self) -> None:
        rng = np.random.default_rng(21)
        backend = makeBackend()
        size = 4
        q_forward = np.diag(np.exp(1j * rng.uniform(-0.5, 0.5, size) - rng.uniform(0.0, 0.2, size)))
        q_backward = np.diag(np.exp(1j * rng.uniform(-0.5, 0.5, size) - rng.uniform(0.0, 0.2, size)))
        components = (
            randomSMatrix(rng, size=size),
            solver.propagationSMatrixBidirectional(backend.asarray(q_forward), backend.asarray(q_backward), backend),
            randomSMatrix(rng, size=size),
            randomSMatrix(rng, size=size),
        )
        backend_components = tuple(
            component if component.isPropagation else toBackend(component, backend)
            for component in components
        )

        full = toNumpy(solver.cascadeMany(backend_components, size, backend), backend)
        reduced = toNumpy(solver.reflectionTransmissionOnlySMatrix(backend_components, size, backend), backend)

        np.testing.assert_allclose(reduced.s11, full.s11, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reduced.s21, full.s21, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
