import unittest

import numpy as np

import bootstrap  # noqa: F401
import rcwa3d_anisotropic as rcwa


def solveAnisotropic(
    *,
    layers,
    wavelength,
    period,
    orders,
    epsIncident=1.0,
    epsTransmission=1.0,
    theta=0.0,
    phi=0.0,
    sAmplitude=1.0,
    pAmplitude=0.0,
    truncation="circular",
    backend="cuda",
):
    simulation = rcwa.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveExcitation(
        wavelength,
        theta=theta,
        phi=phi,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
    )


def solveAnisotropicBatch(
    *,
    layers,
    wavelength,
    period,
    orders,
    excitations,
    epsIncident=1.0,
    epsTransmission=1.0,
    theta=0.0,
    phi=0.0,
    truncation="circular",
    backend="cuda",
):
    simulation = rcwa.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveExcitations(wavelength, excitations, theta=theta, phi=phi)


def solveAnisotropicBatchPowers(
    *,
    layers,
    wavelength,
    period,
    orders,
    excitations,
    epsIncident=1.0,
    epsTransmission=1.0,
    theta=0.0,
    phi=0.0,
    truncation="circular",
    backend="cuda",
):
    simulation = rcwa.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveBatchPowers(wavelength, theta=theta, phi=phi, excitations=excitations)


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
    def testSimulationSolveRoutesDirectlyToStableSMatrix(self) -> None:
        result = solveAnisotropic(**anisotropic_case())

        self.assertAlmostEqual(result.conservation, 1.0, places=12)
        self.assertTrue(result.solvedBy.endswith("-cuda"))
        self.assertTrue(result.solvedBy.startswith("smatrix"))

    def testRemovedLowLevelEntrypointsAreNotPublic(self) -> None:
        self.assertFalse(hasattr(rcwa, "solveStack"))
        self.assertFalse(hasattr(rcwa, "solveStackBatch"))
        self.assertFalse(hasattr(rcwa, "solveStackBatchPowers"))
        self.assertFalse(hasattr(rcwa, "compileLayers"))
        self.assertFalse(hasattr(rcwa, "CompiledLayer"))

    def testMethodSelectorIsRemovedFromPublicSimulationEntrypoints(self) -> None:
        common = anisotropic_case()
        batch_common = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}

        with self.assertRaises(TypeError):
            solveAnisotropic(**common, method="global")
        with self.assertRaises(TypeError):
            solveAnisotropicBatch(
                **batch_common,
                excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
                method="global",
            )
        with self.assertRaises(TypeError):
            rcwa.RCWASimulation(
                period=(1.0, 1.0),
                layers=[],
                method="global",
            )

    def testBatchMatchesSingleSolves(self) -> None:
        common = anisotropic_case()
        direct_common = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}
        excitations = {"TE": (1.0, 0.0), "TM": (0.0, 1.0)}

        batch = solveAnisotropicBatch(**direct_common, excitations=excitations)
        self.assertEqual(set(batch), set(excitations))

        for label, (s_amplitude, p_amplitude) in excitations.items():
            single = solveAnisotropic(
                **direct_common,
                sAmplitude=s_amplitude,
                pAmplitude=p_amplitude,
            )
            self.assertAlmostEqual(batch[label].reflection, single.reflection, places=11)
            self.assertAlmostEqual(batch[label].transmission, single.transmission, places=11)

    def testEmptyBatchReturnsImmediately(self) -> None:
        common = anisotropic_case()
        direct_common = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}

        self.assertEqual(solveAnisotropicBatch(**direct_common, excitations={}), {})
        self.assertEqual(solveAnisotropicBatchPowers(**direct_common, excitations={}), {})


if __name__ == "__main__":
    unittest.main()
