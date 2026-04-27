import unittest

import numpy as np

import bootstrap  # noqa: F401
from rcwa3d_isotropic.analytic import AnalyticDisk, AnalyticRectangle
from rcwa3d_isotropic.fourier import makeHarmonics
from rcwa3d_isotropic import (
    AIR,
    Layer,
    LayerFieldSolution,
    LayerStack,
    SI1550,
    Pattern2D,
    RCWASimulation,
    compileLayers,
    fieldComponentsXy,
    fieldComponentsXz,
    stackFieldComponentsXz,
    stackFieldSliceXz,
    fieldSliceXy,
    fieldSliceXz,
    homogeneousFourierFields,
    incidentFieldIntensities,
    layerFourierFields,
    reconstructFourierGrid,
    stackFieldComponentsXy,
    stackFieldSliceXy,
    solveStack,
)


class GeometryFieldTests(unittest.TestCase):
    def testPatternSupportsRectangularPeriods(self) -> None:
        pattern = Pattern2D(period=(0.8, 0.45), shape=(45, 80), background=AIR)
        pattern.ellipse(radii=(0.18, 0.08), material=SI1550, angle=np.deg2rad(20))
        self.assertEqual(pattern.epsilon.shape, (45, 80))
        self.assertGreater(np.max(np.real(pattern.epsilon)), 1.0)
        self.assertAlmostEqual(pattern.period[0], 0.8)
        self.assertAlmostEqual(pattern.period[1], 0.45)

    def testSampledConvolutionUsesCenteredPhysicalCoordinates(self) -> None:
        period = (1.0, 1.0)
        size = (0.30, 0.20)
        center = (0.10, -0.05)
        background = 1.0
        inclusion = 3.0

        pattern = Pattern2D(period=period, shape=(160, 200), background=background)
        pattern.rectangle(size=size, center=center, material=inclusion, useNormal=False)
        sampledLayer = compileLayers(
            [pattern.toLayer(0.1, factorization="standard")],
            orders=(2, 2),
            truncation="rectangular",
        )[0]

        harmonics = makeHarmonics(
            wavelength=1.0,
            period=period,
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="rectangular",
        )
        analytic = AnalyticRectangle(
            period=period,
            size=size,
            center=center,
            background=background,
            inclusion=inclusion,
        ).convolutionMatrix(harmonics)

        np.testing.assert_allclose(sampledLayer.epsilonMatrix, analytic, rtol=8e-3, atol=8e-3)

    def testSampledCircleConvolutionUsesCenteredPhysicalCoordinates(self) -> None:
        period = (1.0, 1.0)
        radius = 0.18
        center = (0.13, -0.09)
        background = 1.0
        inclusion = 3.0

        pattern = Pattern2D(period=period, shape=(240, 240), background=background)
        pattern.circle(radius=radius, center=center, material=inclusion, useNormal=False)
        sampledLayer = compileLayers(
            [pattern.toLayer(0.1, factorization="standard")],
            orders=(2, 2),
            truncation="rectangular",
        )[0]

        harmonics = makeHarmonics(
            wavelength=1.0,
            period=period,
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="rectangular",
        )
        analytic = AnalyticDisk(
            period=period,
            radius=radius,
            center=center,
            background=background,
            inclusion=inclusion,
        ).convolutionMatrix(harmonics)

        np.testing.assert_allclose(sampledLayer.epsilonMatrix, analytic, rtol=8e-3, atol=8e-3)

    def testLayerStackPatternsFullLayersByXyzRegions(self) -> None:
        i3 = np.eye(3, dtype=complex)
        epsAirTensor = 1.0 * i3
        epsSiTensor = 12.0 * i3
        epsAuTensor = (-40.0 + 3.0j) * i3

        geometry = LayerStack(period=(3.5, 1.0), shape=(8, 64))
        geometry.addLayer(0.7, epsAirTensor, name="air layer patterned with silicon", factorization="standard")
        geometry.addLayer(0.1, epsAuTensor, name="gold film", factorization="standard")
        geometry.setMaterial(epsSiTensor, x=(-0.2, 0.2), y=(-0.5, 0.5), z=(0.0, 0.7))

        layers = geometry.toLayers()
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].epsilon.shape, (8, 64))
        self.assertTrue(np.any(np.isclose(layers[0].epsilon, 12.0)))
        self.assertTrue(np.any(np.isclose(layers[0].epsilon, 1.0)))
        self.assertTrue(np.allclose(layers[1].epsilon, -40.0 + 3.0j))

    def testLayerStackSplitsLayersForPartialZPatterning(self) -> None:
        geometry = LayerStack(period=(1.0, 1.0), shape=(4, 4))
        geometry.addLayer(1.0, 1.0, name="editable layer", factorization="standard")
        geometry.setMaterial(2.25, z=(0.25, 0.75))

        layers = geometry.toLayers()
        self.assertEqual([round(layer.thickness, 8) for layer in layers], [0.25, 0.5, 0.25])
        self.assertTrue(np.allclose(layers[0].epsilon, 1.0))
        self.assertTrue(np.allclose(layers[1].epsilon, 2.25))
        self.assertTrue(np.allclose(layers[2].epsilon, 1.0))

    def testLayerStackAddsSlicedThreeDimensionalBodies(self) -> None:
        cone = LayerStack(period=(1.0, 1.0), shape=(32, 32))
        cone.addLayer(1.0, 1.0, name="cone host", factorization="standard")
        cone.addCone(2.25, z=(0.0, 1.0), topRadius=0.05, bottomRadius=0.36, slices=4)

        coneLayers = cone.toLayers()
        self.assertEqual(len(coneLayers), 4)
        counts = [int(np.count_nonzero(np.isclose(layer.epsilon, 2.25))) for layer in coneLayers]
        self.assertTrue(all(left <= right for left, right in zip(counts, counts[1:])))
        self.assertGreater(counts[-1], counts[0])

        wave = LayerStack(period=(1.0, 1.0), shape=(24, 24))
        wave.addLayer(0.6, 1.0, name="wave host", factorization="standard")
        wave.addWaveBody(3.0, baseZ=0.0, meanHeight=0.30, amplitude=0.20, axis="x", slices=5)

        waveLayers = wave.toLayers()
        self.assertEqual(len(waveLayers), 6)
        self.assertTrue(any(np.any(np.isclose(layer.epsilon, 3.0)) for layer in waveLayers))
        self.assertTrue(np.allclose(waveLayers[-1].epsilon, 1.0))

    def testLayerStackRejectsNonScalarTensorsInIsotropicGeometry(self) -> None:
        geometry = LayerStack(period=(1.0, 1.0), shape=(4, 4))
        with self.assertRaisesRegex(ValueError, "scalar 3x3"):
            geometry.addLayer(0.1, np.diag([1.0, 2.0, 3.0]), factorization="standard")

    def testFieldSliceFromSingleLayerResult(self) -> None:
        pattern = Pattern2D(period=(0.7, 0.5), shape=(32, 36), background=AIR)
        pattern.rectangle(size=(0.22, 0.16), material=SI1550)
        layer = pattern.toLayer(thickness=0.12)
        layers = compileLayers([layer], orders=(1, 1))
        result = solveStack(
            layers=layers,
            wavelength=1.1,
            period=pattern.period,
            orders=(1, 1),
            epsIncident=AIR.epsilon(),
            epsTransmission=AIR.epsilon(),
            returnFields=True,
        )
        x, y, ex = fieldSliceXy(result, layerIndex=0, z=0.06, component="Ex", shape=(21, 25))
        self.assertEqual(ex.shape, (21, 25))
        self.assertEqual(x.shape, (21, 25))
        self.assertEqual(y.shape, (21, 25))
        self.assertTrue(np.all(np.isfinite(ex)))

        fields = layerFourierFields(result.layerSolutions[0], z=0.06)
        self.assertEqual(set(fields), {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"})
        for values in fields.values():
            self.assertTrue(np.all(np.isfinite(values)))

        _, _, eIntensity = fieldSliceXy(result, layerIndex=0, z=0.06, component="EIntensity", shape=(21, 25))
        self.assertEqual(eIntensity.shape, (21, 25))
        self.assertTrue(np.all(np.isfinite(eIntensity)))
        self.assertTrue(np.all(eIntensity >= -1e-12))

        _, _, componentMaps = fieldComponentsXy(result, layerIndex=0, z=0.06, shape=(21, 25))
        self.assertIn("Ez", componentMaps)
        self.assertIn("EIntensity", componentMaps)
        stackXyX, stackXyY, stackXyMaps = stackFieldComponentsXy(result, z=0.06, shape=(21, 25))
        np.testing.assert_allclose(stackXyX, x, atol=1e-12)
        np.testing.assert_allclose(stackXyY, y, atol=1e-12)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "EIntensity", "HIntensity"):
            np.testing.assert_allclose(stackXyMaps[component], componentMaps[component], rtol=1e-12, atol=1e-12)
        _, _, stackXyIntensity = stackFieldSliceXy(result, z=0.06, component="EIntensity", shape=(21, 25))
        np.testing.assert_allclose(stackXyIntensity, componentMaps["EIntensity"], rtol=1e-12, atol=1e-12)

        xzX, xzZ, xzEx = fieldSliceXz(result, layerIndex=0, y=0.0, component="Ex", shape=(17, 19))
        self.assertEqual(xzEx.shape, (17, 19))
        self.assertEqual(xzX.shape, (17, 19))
        self.assertEqual(xzZ.shape, (17, 19))
        self.assertTrue(np.all(np.isfinite(xzEx)))

        stackX, stackZ, stackMaps = stackFieldComponentsXz(
            result,
            y=0.0,
            zSpan=(0.0, layer.thickness),
            shape=(17, 19),
        )
        self.assertEqual(stackMaps["EIntensity"].shape, (17, 19))
        self.assertEqual(stackX.shape, (17, 19))
        self.assertEqual(stackZ.shape, (17, 19))
        self.assertTrue(np.all(np.isfinite(stackMaps["Ez"])))
        self.assertTrue(np.all(stackMaps["EIntensity"] >= -1e-12))
        self.assertLess(np.max(np.abs(stackMaps["Ex"] - xzEx)), 1e-10)
        _, _, layerMaps = fieldComponentsXz(result, layerIndex=0, y=0.0, shape=(17, 19))
        for component in (
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "EMagnitude",
            "HMagnitude",
            "EIntensity",
            "HIntensity",
            "ESelfNormalizedMagnitude",
            "HSelfNormalizedMagnitude",
        ):
            np.testing.assert_allclose(stackMaps[component], layerMaps[component], rtol=1e-10, atol=1e-10)

        defaultX, defaultZ, defaultIntensity = stackFieldSliceXz(result, component="EIntensity", shape=(23, 17))
        self.assertEqual(defaultIntensity.shape, (23, 17))
        self.assertLess(float(defaultZ.min()), 0.0)
        self.assertGreater(float(defaultZ.max()), layer.thickness)
        self.assertAlmostEqual(float(defaultX.min()), -pattern.period[0] / 2)
        self.assertAlmostEqual(float(defaultX.max()), pattern.period[0] / 2)

    def testMagneticIntensityMatchesAnalyticPlaneWave(self) -> None:
        layer = Layer(thickness=0.25, epsilon=1.0, name="matched air layer")
        layers = compileLayers([layer], orders=0, truncation="circular")
        result = solveStack(
            layers=layers,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=0,
            epsIncident=1.0,
            epsTransmission=1.0,
            sAmplitude=0.0,
            pAmplitude=1.0,
            returnFields=True,
            truncation="circular",
        )

        _x, _z, maps = fieldComponentsXz(result, layerIndex=0, shape=(5, 5))

        np.testing.assert_allclose(maps["Hx"], 0.0, atol=1e-12)
        np.testing.assert_allclose(maps["Hz"], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(maps["Hy"]) ** 2, 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HMagnitude"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HIntensity"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HNormalizedMagnitude"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HNormalizedIntensity"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HSelfNormalizedMagnitude"], 1.0, atol=1e-12)

        _, _, hOverH0 = fieldSliceXz(result, layerIndex=0, component="H/H0", shape=(5, 5))
        _, _, hOverH0Squared = fieldSliceXz(result, layerIndex=0, component="|H/H0|^2", shape=(5, 5))
        np.testing.assert_allclose(hOverH0, maps["HNormalizedMagnitude"], atol=1e-12)
        np.testing.assert_allclose(hOverH0Squared, maps["HNormalizedIntensity"], atol=1e-12)

    def testLayerFourierFieldsUsesStableBackwardRightCoefficients(self) -> None:
        layer = LayerFieldSolution(
            name="synthetic thick evanescent layer",
            thickness=1.0,
            wavelength=2.0 * np.pi,
            period=(1.0, 1.0),
            orders=(0, 0),
            mx=np.array([0]),
            my=np.array([0]),
            kx=np.array([0.0 + 0j]),
            ky=np.array([0.0 + 0j]),
            qValues=np.array([1000j, 1000j, -1000j, -1000j]),
            modeMatrix=np.eye(4, dtype=complex),
            coefficients=np.zeros(4, dtype=complex),
            epsilonInverse=np.eye(1, dtype=complex),
            backwardCoefficientsRight=np.array([1.0 + 0j, 0.0 + 0j]),
        )

        left = layerFourierFields(layer, z=0.0)
        right = layerFourierFields(layer, z=layer.thickness)

        for values in (*left.values(), *right.values()):
            self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(left["Hx"], 0.0, atol=1e-300)
        np.testing.assert_allclose(right["Hx"], 1.0, atol=1e-12)

    def testIncidentNormalizedIntensityUsesActualIncidentMedium(self) -> None:
        epsilon = 2.25
        layer = Layer(thickness=0.2, epsilon=epsilon, name="matched glass layer")
        layers = compileLayers([layer], orders=0, truncation="circular")
        result = solveStack(
            layers=layers,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=0,
            epsIncident=epsilon,
            epsTransmission=epsilon,
            sAmplitude=1.0,
            pAmplitude=0.0,
            returnFields=True,
            truncation="circular",
        )

        incident = incidentFieldIntensities(result)
        _x, _z, maps = fieldComponentsXz(result, layerIndex=0, shape=(5, 5))

        self.assertAlmostEqual(incident["EIntensity"], 1.0, places=12)
        self.assertAlmostEqual(incident["HIntensity"], epsilon, places=12)
        np.testing.assert_allclose(maps["EIntensity"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HIntensity"], epsilon, atol=1e-12)
        np.testing.assert_allclose(maps["EMagnitude"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HMagnitude"], np.sqrt(epsilon), atol=1e-12)
        np.testing.assert_allclose(maps["ENormalizedMagnitude"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HNormalizedMagnitude"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["ENormalizedIntensity"], 1.0, atol=1e-12)
        np.testing.assert_allclose(maps["HNormalizedIntensity"], 1.0, atol=1e-12)

    def testSMatrixFieldsAreReusableOnSmallProblem(self) -> None:
        pattern = Pattern2D(period=(0.7, 0.5), shape=(32, 36), background=AIR)
        pattern.rectangle(size=(0.22, 0.16), material=SI1550)
        layer = pattern.toLayer(thickness=0.12)
        layers = compileLayers([layer], orders=(1, 1))
        common = dict(
            layers=layers,
            wavelength=1.1,
            period=pattern.period,
            orders=(1, 1),
            epsIncident=AIR.epsilon(),
            epsTransmission=AIR.epsilon(),
            returnFields=True,
        )

        sMatrixResult = solveStack(**common, method="smatrix")
        self.assertTrue(np.isfinite(sMatrixResult.reflection))
        self.assertTrue(np.isfinite(sMatrixResult.transmission))

        _, _, exSMatrix = fieldSliceXy(sMatrixResult, layerIndex=0, z=0.06, component="Ex", shape=(21, 25))
        self.assertTrue(np.all(np.isfinite(exSMatrix)))

        _, _, ezSMatrix = fieldSliceXy(sMatrixResult, layerIndex=0, z=0.06, component="Ez", shape=(21, 25))
        self.assertTrue(np.all(np.isfinite(ezSMatrix)))

        _, _, intensitySMatrix = fieldSliceXy(
            sMatrixResult,
            layerIndex=0,
            z=0.06,
            component="EIntensity",
            shape=(21, 25),
        )
        self.assertTrue(np.all(intensitySMatrix >= -1e-12))

    def testSimulationReturnFieldsConvertsCudaLayerDataToNumpy(self) -> None:
        layer = Pattern2D(period=(0.7, 0.5), shape=(32, 36), background=AIR)
        layer.rectangle(size=(0.22, 0.16), material=SI1550)
        simulation = RCWASimulation(period=layer.period, orders=(1, 1), truncation="circular")
        simulation.addLayer(layer.toLayer(thickness=0.12))

        result = simulation.solve(1.1, polarization="TM", returnFields=True)

        self.assertEqual(len(result.layerSolutions), 1)
        self.assertIsInstance(result.layerSolutions[0].epsilonInverse, np.ndarray)
        self.assertTrue(np.all(np.isfinite(result.layerSolutions[0].epsilonInverse)))

    def testTransmissionRegionStackFieldMatchesHomogeneousReconstruction(self) -> None:
        period = (0.7, 0.5)
        thickness = 0.12
        wavelength = 1.1
        distance = 0.08
        pattern = Pattern2D(period=period, shape=(32, 36), background=AIR)
        pattern.circle(radius=0.11, material=SI1550)
        layers = compileLayers([pattern.toLayer(thickness=thickness)], orders=(1, 1))
        result = solveStack(
            layers=layers,
            wavelength=wavelength,
            period=period,
            orders=(1, 1),
            epsIncident=AIR.epsilon(),
            epsTransmission=AIR.epsilon(),
            sAmplitude=0.0,
            pAmplitude=1.0,
            returnFields=True,
        )
        x = np.linspace(-period[0] / 2, period[0] / 2, 23)
        _x, _z, maps = stackFieldComponentsXz(
            result,
            y=0.0,
            xSpan=(-period[0] / 2, period[0] / 2),
            zSpan=(thickness + distance, thickness + distance),
            shape=(1, x.size),
        )
        layer = result.layerSolutions[0]
        kz = np.array([order.kzTransmitted for order in result.orders], dtype=complex)
        fourierFields = homogeneousFourierFields(
            layer.kx,
            layer.ky,
            kz,
            result.tAmplitudes[0::2],
            result.tAmplitudes[1::2],
            AIR.epsilon(),
            distance,
            wavelength,
        )
        expected = reconstructFourierGrid(fourierFields["Ex"], layer.kx, layer.ky, wavelength, x, 0.0)

        np.testing.assert_allclose(maps["Ex"][0], expected, rtol=1e-12, atol=1e-12)

    def testTransmissionRegionStackXyMatchesHomogeneousReconstruction(self) -> None:
        period = (0.7, 0.5)
        thickness = 0.12
        wavelength = 1.1
        distance = 0.08
        pattern = Pattern2D(period=period, shape=(32, 36), background=AIR)
        pattern.circle(radius=0.11, material=SI1550)
        layers = compileLayers([pattern.toLayer(thickness=thickness)], orders=(1, 1))
        result = solveStack(
            layers=layers,
            wavelength=wavelength,
            period=period,
            orders=(1, 1),
            epsIncident=AIR.epsilon(),
            epsTransmission=AIR.epsilon(),
            sAmplitude=0.0,
            pAmplitude=1.0,
            returnFields=True,
        )
        x, y, maps = stackFieldComponentsXy(result, z=thickness + distance, shape=(19, 23))
        layer = result.layerSolutions[0]
        kz = np.array([order.kzTransmitted for order in result.orders], dtype=complex)
        fourierFields = homogeneousFourierFields(
            layer.kx,
            layer.ky,
            kz,
            result.tAmplitudes[0::2],
            result.tAmplitudes[1::2],
            AIR.epsilon(),
            distance,
            wavelength,
        )

        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            expected = reconstructFourierGrid(fourierFields[component], layer.kx, layer.ky, wavelength, x, y)
            np.testing.assert_allclose(maps[component], expected, rtol=1e-12, atol=1e-12)

        expectedIntensity = np.real(
            np.abs(maps["Ex"]) ** 2 + np.abs(maps["Ey"]) ** 2 + np.abs(maps["Ez"]) ** 2
        )
        np.testing.assert_allclose(maps["EIntensity"], expectedIntensity, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
