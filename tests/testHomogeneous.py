import math
import unittest

import numpy as np

import bootstrap  # noqa: F401
from rcwa3d_isotropic import Layer, RCWASimulation


def slabReflectance(n0: float, n1: float, n2: float, wavelength: float, thickness: float) -> float:
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    phase = 2 * math.pi * n1 * thickness / wavelength
    numerator = r01 + r12 * complex(math.cos(2 * phase), math.sin(2 * phase))
    denominator = 1 + r01 * r12 * complex(math.cos(2 * phase), math.sin(2 * phase))
    return abs(numerator / denominator) ** 2


class HomogeneousTests(unittest.TestCase):
    def testNoLayerMatchesFresnelInterface(self) -> None:
        result = _solveIsotropic(
            layers=[],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(0, 0),
            epsIncident=1.0,
            epsTransmission=2.25,
        )
        expectedReflection = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        expectedTransmission = 1.0 - expectedReflection
        self.assertAlmostEqual(result.reflection, expectedReflection, places=12)
        self.assertAlmostEqual(result.transmission, expectedTransmission, places=12)
        self.assertAlmostEqual(result.conservation, 1.0, places=12)

    def testUniformSlabMatchesTransferMatrix(self) -> None:
        wavelength = 1.0
        thickness = 0.30
        result = _solveIsotropic(
            layers=[Layer(thickness=thickness, epsilon=2.25)],
            wavelength=wavelength,
            period=(1.0, 1.0),
            orders=(0, 0),
            epsIncident=1.0,
            epsTransmission=1.0,
        )
        expectedReflection = slabReflectance(1.0, 1.5, 1.0, wavelength, thickness)
        self.assertAlmostEqual(result.reflection, expectedReflection, places=10)
        self.assertAlmostEqual(result.conservation, 1.0, places=10)

    def testObliqueInterfaceMatchesFresnelForSAndP(self) -> None:
        theta = math.radians(30)
        n0 = 1.0
        n2 = 1.5
        cosIncident = math.cos(theta)
        sinTransmitted = n0 * math.sin(theta) / n2
        cosTransmitted = math.sqrt(1 - sinTransmitted * sinTransmitted)
        expected = {
            "s": abs((n0 * cosIncident - n2 * cosTransmitted) / (n0 * cosIncident + n2 * cosTransmitted)) ** 2,
            "p": abs((n2 * cosIncident - n0 * cosTransmitted) / (n2 * cosIncident + n0 * cosTransmitted)) ** 2,
        }

        for polarization in ("s", "p"):
            with self.subTest(polarization=polarization):
                result = _solveIsotropic(
                    layers=[],
                    wavelength=1.0,
                    period=(1.0, 1.0),
                    orders=(1, 1),
                    epsIncident=n0 * n0,
                    epsTransmission=n2 * n2,
                    theta=theta,
                    sAmplitude=1.0 if polarization == "s" else 0.0,
                    pAmplitude=1.0 if polarization == "p" else 0.0,
                )
                self.assertAlmostEqual(result.reflection, expected[polarization], places=12)
                self.assertAlmostEqual(result.conservation, 1.0, places=12)

    def testSampledUniformLayerDoesNotCreateSpuriousDiffraction(self) -> None:
        result = _solveIsotropic(
            layers=[Layer(thickness=0.2, epsilon=np.full((24, 20), 2.25))],
            wavelength=1.0,
            period=(0.7, 0.8),
            orders=(1, 1),
            epsIncident=1.0,
            epsTransmission=1.0,
        )
        offZeroPower = sum(
            order.reflectedPower + order.transmittedPower
            for order in result.orders
            if (order.mx, order.my) != (0, 0)
        )
        self.assertAlmostEqual(offZeroPower, 0.0, places=10)
        self.assertAlmostEqual(result.conservation, 1.0, places=10)


def _solveIsotropic(
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
    returnFields=False,
    truncation="circular",
    backend="cuda",
):
    simulation = RCWASimulation(
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
        returnFields=returnFields,
    )


if __name__ == "__main__":
    unittest.main()
