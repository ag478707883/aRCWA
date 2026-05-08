import unittest

import numpy as np

import bootstrap  # noqa: F401
import rcwa3d_anisotropic.solver as solver


def random_smatrix(rng: np.random.Generator, size: int, scale: float = 0.08) -> solver._SMatrix:
    def block() -> np.ndarray:
        return scale * (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size)))

    return solver._SMatrix(s11=block(), s12=np.eye(size) + block(), s21=np.eye(size) + block(), s22=block())


def reference_star(left: solver._SMatrix, right: solver._SMatrix) -> solver._SMatrix:
    size = left.s11.shape[0]
    identity = np.eye(size, dtype=complex)
    return solver._SMatrix(
        s11=left.s11 + left.s12 @ np.linalg.solve(identity - right.s11 @ left.s22, right.s11 @ left.s21),
        s12=left.s12 @ np.linalg.solve(identity - right.s11 @ left.s22, right.s12),
        s21=right.s21 @ np.linalg.solve(identity - left.s22 @ right.s11, left.s21),
        s22=right.s22 + right.s21 @ np.linalg.solve(identity - left.s22 @ right.s11, left.s22 @ right.s12),
    )


class SMatrixAlgebraTests(unittest.TestCase):
    def testRedhefferStarMatchesBlockFormula(self) -> None:
        rng = np.random.default_rng(8)
        left = random_smatrix(rng, size=5)
        right = random_smatrix(rng, size=5)

        actual = solver._redhefferStar(left, right, solver._CPU_BACKEND)
        expected = reference_star(left, right)

        np.testing.assert_allclose(actual.s11, expected.s11, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s12, expected.s12, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s21, expected.s21, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.s22, expected.s22, rtol=1e-12, atol=1e-12)

    def testPropagationShortcutsMatchGeneralStarProduct(self) -> None:
        rng = np.random.default_rng(13)
        middle = random_smatrix(rng, size=6)
        forward = np.diag(np.exp(1j * rng.uniform(-0.2, 0.2, 6)))
        backward = np.diag(np.exp(1j * rng.uniform(-0.2, 0.2, 6)))
        propagation = solver._propagationSMatrixBidirectional(forward, backward, solver._CPU_BACKEND)
        dense_propagation = solver._SMatrix(
            s11=propagation.s11,
            s12=propagation.s12,
            s21=propagation.s21,
            s22=propagation.s22,
        )

        shortcut_right = solver._redhefferStar(middle, propagation, solver._CPU_BACKEND)
        expected_right = reference_star(middle, dense_propagation)
        shortcut_left = solver._redhefferStar(propagation, middle, solver._CPU_BACKEND)
        expected_left = reference_star(dense_propagation, middle)

        for actual, expected in ((shortcut_right, expected_right), (shortcut_left, expected_left)):
            np.testing.assert_allclose(actual.s11, expected.s11, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s12, expected.s12, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s21, expected.s21, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.s22, expected.s22, rtol=1e-12, atol=1e-12)

    def testEnhancedReflectionTransmissionMatchesFullCascade(self) -> None:
        rng = np.random.default_rng(21)
        size = 4
        q_forward = np.diag(np.exp(1j * rng.uniform(-0.5, 0.5, size) - rng.uniform(0.0, 0.2, size)))
        q_backward = np.diag(np.exp(1j * rng.uniform(-0.5, 0.5, size) - rng.uniform(0.0, 0.2, size)))
        components = (
            random_smatrix(rng, size=size),
            solver._propagationSMatrixBidirectional(q_forward, q_backward, solver._CPU_BACKEND),
            random_smatrix(rng, size=size),
            random_smatrix(rng, size=size),
        )

        full = solver._cascadeMany(components, size, solver._CPU_BACKEND)
        reduced = solver._reflectionTransmissionOnlySMatrix(components, size, solver._CPU_BACKEND)

        np.testing.assert_allclose(reduced.s11, full.s11, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(reduced.s21, full.s21, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
