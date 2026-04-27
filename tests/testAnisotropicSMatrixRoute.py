import unittest

import numpy as np

import bootstrap  # noqa: F401
import rcwa3d_anisotropic as rcwa


def anisotropic_case() -> dict[str, object]:
    """A thin real xz-coupled tensor film for the stable S-matrix route."""

    return {
        "layers": [
            rcwa.Layer(
                thickness=0.018,
                epsilon=rcwa.xzTensor(2.2, 2.4, 2.1, 0.04, 0.04),
                name="thin xz tensor film",
            )
        ],
        "wavelength": 1.05,
        "period": (0.9, 1.1),
        "orders": (1, 1),
        "epsIncident": 1.0,
        "epsTransmission": 1.0,
        "theta": np.deg2rad(3.0),
        "phi": np.deg2rad(11.0),
        "sAmplitude": 0.0,
        "pAmplitude": 1.0,
        "truncation": "circular",
    }


class AnisotropicSMatrixRouteTests(unittest.TestCase):
    def testPublicSolveRoutesDirectlyToStableSMatrix(self) -> None:
        result = rcwa.solveStack(**anisotropic_case())

        self.assertAlmostEqual(result.conservation, 1.0, places=12)
        self.assertTrue(result.solvedBy.endswith("-cuda"))
        self.assertTrue(result.solvedBy.startswith("smatrix"))

    def testUnsupportedMethodsAreRejectedAtPublicEntrypoints(self) -> None:
        common = anisotropic_case()
        batch_common = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}

        for method in ("etm", "global", "expm"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
                    rcwa.solveStack(**common, method=method)
                with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
                    rcwa.solveStackBatch(
                        **batch_common,
                        excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
                        method=method,
                    )

    def testBatchMatchesSingleSolves(self) -> None:
        common = anisotropic_case()
        direct_common = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}
        excitations = {"TE": (1.0, 0.0), "TM": (0.0, 1.0)}

        batch = rcwa.solveStackBatch(**direct_common, excitations=excitations)
        self.assertEqual(set(batch), set(excitations))

        for label, (s_amplitude, p_amplitude) in excitations.items():
            single = rcwa.solveStack(
                **direct_common,
                sAmplitude=s_amplitude,
                pAmplitude=p_amplitude,
            )
            self.assertAlmostEqual(batch[label].reflection, single.reflection, places=11)
            self.assertAlmostEqual(batch[label].transmission, single.transmission, places=11)


if __name__ == "__main__":
    unittest.main()
