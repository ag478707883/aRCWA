import importlib.util
from pathlib import Path
import unittest

import numpy as np

import bootstrap  # noqa: F401
import rcwa3d_anisotropic as anisotropic
import rcwa3d_anisotropic.simulation as anisotropic_simulation
import rcwa3d_anisotropic.solver as anisotropic_solver
import rcwa3d_isotropic as isotropic
import rcwa3d_isotropic.solver as isotropic_solver
from rcwa3d_anisotropic.backend import resolveBackend as resolveAnisotropicBackend
from rcwa3d_isotropic.fourier import makeHarmonics
from rcwa3d_isotropic import Layer
from rcwa3d_isotropic.solver import _compileLayers as _compileIsotropic


def _torchCudaAvailable() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


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
    method="smatrix",
    truncation="circular",
    backend="cuda",
):
    if method != "smatrix":
        raise ValueError("RCWASimulation now supports only method='smatrix'")
    simulation = isotropic.RCWASimulation(
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


def _solveIsotropicBatch(
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
    method="smatrix",
    truncation="circular",
    backend="cuda",
):
    if method != "smatrix":
        raise ValueError("RCWASimulation now supports only method='smatrix'")
    simulation = isotropic.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveExcitations(wavelength, excitations, theta=theta, phi=phi)


class SolverPathTests(unittest.TestCase):
    def testFourierTruncationSupportsRectangularAndCircularDomains(self) -> None:
        rectangular = makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="rectangular",
        )
        circular = makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="circular",
        )

        self.assertEqual(rectangular.count, 25)
        self.assertEqual(circular.count, 13)
        self.assertIn((0, 0), set(zip(circular.mx, circular.my)))
        self.assertNotIn((2, 2), set(zip(circular.mx, circular.my)))

    def testSingleIntegerOrderMeansSameXAndYOrder(self) -> None:
        layer = isotropic.Layer(thickness=0.1, epsilon=2.25)
        common = dict(
            layers=[layer],
            wavelength=1.0,
            period=(1.0, 1.0),
            epsIncident=1.0,
            epsTransmission=1.0,
        )

        scalarOrder = _solveIsotropic(**common, orders=1)
        tupleOrder = _solveIsotropic(**common, orders=(1, 1))

        self.assertEqual(scalarOrder.orders[0].mx, tupleOrder.orders[0].mx)
        self.assertAlmostEqual(scalarOrder.reflection, tupleOrder.reflection, places=12)
        self.assertAlmostEqual(scalarOrder.transmission, tupleOrder.transmission, places=12)

    def testIsotropicDefaultTruncationIsCircular(self) -> None:
        result = _solveIsotropic(
            layers=[isotropic.Layer(thickness=0.1, epsilon=2.25)],
            wavelength=0.8,
            period=(1.0, 1.0),
            orders=1,
        )
        self.assertEqual(len(result.orders), 5)

    def testIsotropicBackendResolverRejectsCpuBackends(self) -> None:
        for backend in ("cpu", "numpy", "torch-cpu"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "CUDA-only"):
                    isotropic.resolveBackend(backend)

    def testIsotropicBackendResolverRejectsRemovedJaxPath(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend must"):
            isotropic.resolveBackend("jax")

    def testIsotropicAutoBackendRequiresCuda(self) -> None:
        if not _torchCudaAvailable():
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA-enabled torch"):
                isotropic.resolveBackend("auto")
            return

        backend = isotropic.resolveBackend("auto")
        self.assertEqual(backend.name, "cuda")
        self.assertTrue(backend.isCuda)

    def testIsotropicTorchCpuBackendIsRejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            isotropic.resolveBackend("torch-cpu")

    @unittest.skipUnless(_torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicCudaAliasesMatchCanonicalBackendForSolveAndBatch(self) -> None:
        epsilon = np.ones((20, 20), dtype=complex)
        epsilon[5:15, 7:13] = 2.25
        layer = isotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="auto")
        common = dict(
            layers=[layer],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(2, 2),
            epsIncident=1.0,
            epsTransmission=1.0,
            theta=np.deg2rad(7.0),
            phi=np.deg2rad(9.0),
            truncation="circular",
        )

        cuda = _solveIsotropic(**common, sAmplitude=0.6 + 0.2j, pAmplitude=0.15 - 0.3j, backend="cuda")

        for alias in ("gpu", "torch", "auto"):
            aliasResult = _solveIsotropic(
                **common,
                sAmplitude=0.6 + 0.2j,
                pAmplitude=0.15 - 0.3j,
                backend=alias,
            )
            self.assertAlmostEqual(aliasResult.reflection, cuda.reflection, places=9)
            self.assertAlmostEqual(aliasResult.transmission, cuda.transmission, places=9)
            self.assertAlmostEqual(aliasResult.conservation, cuda.conservation, places=9)
            np.testing.assert_allclose(aliasResult.rAmplitudes, cuda.rAmplitudes, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(aliasResult.tAmplitudes, cuda.tAmplitudes, rtol=1e-8, atol=1e-9)

        excitations = {"TE": (1.0, 0.0), "TM": (0.0, 1.0), "Mixed": (0.7 + 0.1j, -0.2 + 0.15j)}
        cudaBatch = _solveIsotropicBatch(**common, excitations=excitations, method="smatrix", backend="cuda")
        aliasBatch = _solveIsotropicBatch(**common, excitations=excitations, method="smatrix", backend="gpu")

        for label in excitations:
            self.assertAlmostEqual(aliasBatch[label].reflection, cudaBatch[label].reflection, places=9)
            self.assertAlmostEqual(aliasBatch[label].transmission, cudaBatch[label].transmission, places=9)
            np.testing.assert_allclose(aliasBatch[label].rAmplitudes, cudaBatch[label].rAmplitudes, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(aliasBatch[label].tAmplitudes, cudaBatch[label].tAmplitudes, rtol=1e-8, atol=1e-9)

    @unittest.skipUnless(_torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicCudaSimulationKeepsCoreMatricesOnGpu(self) -> None:
        epsilon = np.ones((22, 24), dtype=complex)
        epsilon[6:16, 8:17] = 2.7
        layer = isotropic.Layer(thickness=0.06, epsilon=epsilon, factorization="standard")
        simulation = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(2, 2),
            truncation="circular",
            backend="cuda",
        )

        result = simulation.solve(0.95, theta=np.deg2rad(4.0), phi=np.deg2rad(13.0), polarization="TE")

        self.assertEqual(result.solvedBy, "smatrix-cuda")
        self.assertEqual(len(simulation._preparedCache), 1)
        prepared = next(iter(simulation._preparedCache.values()))
        self.assertTrue(prepared.backend.isCuda)
        self.assertEqual(prepared.total.s11.device.type, "cuda")
        self.assertEqual(prepared.components[0].s11.device.type, "cuda")
        self.assertEqual(prepared.components[-1].s21.device.type, "cuda")

        self.assertEqual(len(simulation._torchLayerCache), 0)
        self.assertEqual(len(simulation._compiledLayerCache), 1)
        torchLayer = next(iter(simulation._compiledLayerCache.values()))
        self.assertEqual(torchLayer.epsilonMatrix.device.type, "cuda")
        self.assertEqual(torchLayer.epsilonInverse.device.type, "cuda")

    @unittest.skipUnless(_torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicTorchPrepareBuildsRawLayerDataOnGpu(self) -> None:
        epsilon = np.ones((18, 20), dtype=complex)
        epsilon[4:14, 6:13] = 2.4
        layer = isotropic.Layer(thickness=0.06, epsilon=epsilon, factorization="auto")
        prepared = isotropic_solver.prepareStackTorch(
            layers=[layer],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
        )

        self.assertEqual(prepared.total.s11.device.type, "cuda")
        qValues, modeMatrix = prepared.layerModes[0]
        self.assertEqual(qValues.device.type, "cuda")
        self.assertEqual(modeMatrix.device.type, "cuda")
        factorized = isotropic_solver._layerDataForTorch(
            layer,
            prepared.harmonics,
            prepared.backend.xp,
            prepared.total.s11.device,
        )
        self.assertEqual(factorized.epsilonMatrix.device.type, "cuda")
        self.assertEqual(factorized.epsilonInverse.device.type, "cuda")
        self.assertEqual(factorized.factorization, "normal-vector-li")
        self.assertTrue(all(matrix.device.type == "cuda" for matrix in factorized.displacementMatrices))

    def testAnisotropicCudaBackendEigProducesStableEigenpairs(self) -> None:
        backend = resolveAnisotropicBackend("cuda")
        matrix = np.array(
            [
                [2.0 + 0.2j, 0.1, -0.3j],
                [0.05, -1.3 + 0.1j, 0.2],
                [0.12j, -0.15, 0.8 - 0.05j],
            ],
            dtype=complex,
        )
        eigenvalues, eigenvectors = backend.eig(backend.asarray(matrix))
        eigenvalues = backend.toNumpy(eigenvalues)
        eigenvectors = backend.toNumpy(eigenvectors)
        residual = matrix @ eigenvectors - eigenvectors * eigenvalues[np.newaxis, :]
        self.assertLess(np.linalg.norm(residual), 1e-8 * max(1.0, np.linalg.norm(matrix)))

        for alias in (None, "auto", "gpu", "torch", "torch-cuda"):
            with self.subTest(alias=alias):
                resolved = resolveAnisotropicBackend(alias)
                self.assertEqual(resolved.name, "cuda")
                self.assertTrue(resolved.isCuda)

        for alias in ("cpu", "numpy", "torch-cpu"):
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "CUDA-only"):
                    resolveAnisotropicBackend(alias)

    def testHomogeneousInterfaceUsesSameScatteringMatrixAsDenseSolve(self) -> None:
        harmonics = anisotropic_solver._makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            epsIncident=1.0,
            theta=np.deg2rad(5),
            phi=np.deg2rad(8),
            truncation="circular",
        )
        leftForward = anisotropic_solver._homogeneousBasis(harmonics, 1.0, direction=1)
        leftBackward = anisotropic_solver._homogeneousBasis(harmonics, 1.0, direction=-1)
        rightQ, rightModes = anisotropic_solver._homogeneousTensorLayerModes(
            anisotropic.xzTensor(2.25, 2.1, 2.3, 0.05, 0.04),
            harmonics,
        )
        del rightQ
        rightForward = rightModes[:, : 2 * harmonics.count]
        rightBackward = rightModes[:, 2 * harmonics.count :]
        backend = anisotropic_solver._CPU_BACKEND

        dense = anisotropic_solver._interfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            backend,
        )
        block = anisotropic_solver._interfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            backend,
            homogeneousLeft=True,
            homogeneousRight=True,
            nOrders=harmonics.count,
        )

        np.testing.assert_allclose(block.s11, dense.s11, atol=1e-11)
        np.testing.assert_allclose(block.s12, dense.s12, atol=1e-11)
        np.testing.assert_allclose(block.s21, dense.s21, atol=1e-11)
        np.testing.assert_allclose(block.s22, dense.s22, atol=1e-11)

    def testIsotropicLegacySolveFunctionsAreNotPublic(self) -> None:
        self.assertFalse(hasattr(isotropic, "solveStack"))
        self.assertFalse(hasattr(isotropic, "solveStackBatch"))
        self.assertFalse(hasattr(isotropic, "compileLayers"))

    def testIsotropicPathRejectsTensorLikeLayerInputs(self) -> None:
        tensor = np.zeros((6, 6, 3, 3), dtype=complex)
        for index in range(3):
            tensor[..., index, index] = 2.25
        layer = anisotropic.Layer(thickness=0.2, epsilon=tensor, name="tensor layer")

        with self.assertRaisesRegex(TypeError, "rcwa3d_anisotropic.solveStack"):
            _solveIsotropic(
                layers=[layer],
                wavelength=1.0,
                period=(0.8, 0.8),
                orders=(1, 1),
                epsIncident=1.0,
                epsTransmission=1.0,
            )

        with self.assertRaisesRegex(TypeError, "rcwa3d_anisotropic.compileLayers"):
            _compileIsotropic([layer], orders=(1, 1))

    def testAnisotropicScalarLayerMatchesIsotropicSolver(self) -> None:
        common = dict(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(0, 0),
            epsIncident=1.0,
            epsTransmission=1.0,
            pAmplitude=1.0,
            sAmplitude=0.0,
        )

        isotropicResult = _solveIsotropic(layers=[isotropic.Layer(thickness=0.30, epsilon=2.25)], **common)
        anisotropicResult = anisotropic.solveStack(layers=[anisotropic.Layer(thickness=0.30, epsilon=2.25)], **common)

        self.assertAlmostEqual(anisotropicResult.reflection, isotropicResult.reflection, places=12)
        self.assertAlmostEqual(anisotropicResult.transmission, isotropicResult.transmission, places=12)
        self.assertAlmostEqual(anisotropicResult.conservation, 1.0, places=12)

    def testAnisotropicPublicSolveRejectsNonSMatrixMethods(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.02,
            background=1.0,
            post=1.6,
            size=(0.20, 0.20),
            shape=(12, 12),
            factorization="normal-vector",
        )
        common = dict(
            layers=[layer],
            wavelength=1.2,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            theta=np.deg2rad(4),
            pAmplitude=1.0,
            sAmplitude=0.0,
        )

        smatrix = anisotropic.solveStack(**common, method="smatrix")
        batchCommon = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}
        self.assertEqual(smatrix.solvedBy, "smatrix-cuda")
        for method in ("etm", "global", "expm"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
                    anisotropic.solveStack(**common, method=method)
                with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
                    anisotropic.solveStackBatch(
                        **batchCommon,
                        method=method,
                        excitations={"TM": (0.0, 1.0)},
                    )

    def testAnisotropicSimulationDefaultsToStableCudaSMatrix(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.02,
            background=1.0,
            post=1.6,
            size=(0.20, 0.20),
            shape=(12, 12),
            factorization="normal-vector",
        )
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(simulation.method, "smatrix")
        self.assertEqual(simulation.backend, "cuda")

        solve_kwargs = dict(
            wavelength=1.2,
            theta=np.deg2rad(4),
            polarization="TM",
        )

        smatrix = simulation.solve(**solve_kwargs)
        self.assertTrue(smatrix.solvedBy.endswith("-cuda"))

        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            anisotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=[layer],
                orders=(1, 1),
                truncation="circular",
                method="global",
            )

    def testAnisotropicProjectUsesStableCudaSMatrixOnly(self) -> None:
        project = anisotropic.Project(period=(1.0, 1.0), order=(0, 0), truncation="circular")
        project.add_uniform(0.05, anisotropic.xzTensor(2.2, 2.2, 2.2, 0.03, 0.03), name="tensor film")

        self.assertEqual(project.simulation().method, "smatrix")
        self.assertEqual(project.simulation().backend, "cuda")
        result = project.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            anisotropic.Project(period=(1.0, 1.0), order=(0, 0), method="global").simulation()

    def testAnisotropicPatternLayerSupportsLayerFirstPatternConstruction(self) -> None:
        layer = anisotropic.PatternLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            shape=(16, 16),
            factorization="auto",
            name="layer-first builder",
        )
        layer.circle(radius=0.2, material=2.25)

        materialized = layer.toLayer()
        self.assertEqual(np.asarray(materialized.epsilon).shape, (16, 16))
        self.assertEqual(np.asarray(materialized.normalField).shape, (16, 16, 2))

        result = anisotropic.solveStack(
            layers=[materialized],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAnisotropicProjectSupportsAddingPatternAfterCreatingLayer(self) -> None:
        project = anisotropic.Project(period=(1.0, 1.0), order=(1, 1), truncation="circular", samples=(16, 16))
        patterned = project.add_patterned_layer(
            height=0.08,
            background=1.0,
            samples=(16, 16),
            factorization="auto",
            name="patterned top layer",
        )
        patterned.circle(radius=0.2, material=2.25)
        project.add_uniform(0.05, anisotropic.xzTensor(2.1, 2.1, 2.1, 0.03, 0.03), name="tensor film")

        result = project.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAnisotropicXzZxHomogeneousLayerUsesLiSchurComplement(self) -> None:
        tensor = np.array(
            [
                [2.25, 0.0, 0.4],
                [0.0, 2.25, 0.0],
                [0.4, 0.0, 2.25],
            ],
            dtype=complex,
        )
        effectiveP = 2.25 - 0.4 * 0.4 / 2.25
        common = dict(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(0, 0),
            epsIncident=1.0,
            epsTransmission=1.0,
            pAmplitude=1.0,
            sAmplitude=0.0,
        )

        anisotropicResult = anisotropic.solveStack(
            layers=[anisotropic.Layer(thickness=0.30, epsilon=tensor)],
            **common,
        )
        effectiveScalarResult = _solveIsotropic(
            layers=[isotropic.Layer(thickness=0.30, epsilon=effectiveP)],
            **common,
        )

        self.assertAlmostEqual(anisotropicResult.reflection, effectiveScalarResult.reflection, places=12)
        self.assertAlmostEqual(anisotropicResult.transmission, effectiveScalarResult.transmission, places=12)
        self.assertAlmostEqual(anisotropicResult.conservation, 1.0, places=12)

    def testAnisotropicCompiledTensorLayerMatchesDirectTensorLayer(self) -> None:
        tensor = np.zeros((8, 6, 3, 3), dtype=complex)
        tensor[..., 0, 0] = 2.25
        tensor[..., 1, 1] = 2.25
        tensor[..., 2, 2] = 2.25
        tensor[..., 0, 2] = 0.15
        tensor[..., 2, 0] = 0.15
        layer = anisotropic.Layer(thickness=0.2, epsilon=tensor, name="sampled tensor layer")
        compiled = anisotropic.compileLayers([layer], orders=(1, 1))
        common = dict(
            wavelength=1.0,
            period=(0.8, 0.8),
            orders=(1, 1),
            epsIncident=1.0,
            epsTransmission=1.0,
        )

        directResult = anisotropic.solveStack(layers=[layer], **common)
        compiledResult = anisotropic.solveStack(layers=compiled, **common)

        self.assertAlmostEqual(compiledResult.reflection, directResult.reflection, places=12)
        self.assertAlmostEqual(compiledResult.transmission, directResult.transmission, places=12)

    def testAnisotropicCompiledLayerRejectsMismatchedFourierTruncation(self) -> None:
        layer = anisotropic.Layer(thickness=0.2, epsilon=2.25)
        compiled = anisotropic.compileLayers([layer], orders=(1, 1), truncation="circular")

        with self.assertRaisesRegex(ValueError, "truncation"):
            anisotropic.solveStack(
                layers=compiled,
                wavelength=1.0,
                period=(1.0, 1.0),
                orders=(1, 1),
                truncation="rectangular",
            )

    def testAnisotropicHomogeneousStackUsesReducedFourByFourFastPath(self) -> None:
        layer = anisotropic.Layer(thickness=0.2, epsilon=2.25)
        common = dict(
            layers=[layer],
            wavelength=1.0,
            period=(0.8, 0.9),
            orders=(2, 2),
            theta=np.deg2rad(11),
            phi=np.deg2rad(17),
            sAmplitude=1.0,
            pAmplitude=0.0,
        )

        fast = anisotropic.solveStack(**common)
        full = anisotropic.solveStack(**common, returnFields=True)

        self.assertEqual(fast.solvedBy, "smatrix-homogeneous-4x4-cuda")
        self.assertEqual(fast.rAmplitudes.shape, full.rAmplitudes.shape)
        self.assertAlmostEqual(fast.reflection, full.reflection, places=12)
        self.assertAlmostEqual(fast.transmission, full.transmission, places=12)
        offZeroPower = sum(
            order.reflectedPower + order.transmittedPower
            for order in fast.orders
            if (order.mx, order.my) != (0, 0)
        )
        self.assertAlmostEqual(offZeroPower, 0.0, places=12)

    def testAnisotropicOneDimensionalGratingUsesReducedFastPath(self) -> None:
        grid = np.tile(np.r_[np.full(12, 2.25), np.ones(12)], (20, 1))
        layer = anisotropic.Layer(thickness=0.08, epsilon=grid, factorization="standard")
        common = dict(
            layers=[layer],
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(2, 2),
            theta=np.deg2rad(7),
            phi=np.deg2rad(12),
            sAmplitude=1.0,
            pAmplitude=0.0,
            truncation="rectangular",
        )

        fast = anisotropic.solveStack(**common)
        full = anisotropic.solveStack(**common, returnFields=True)

        self.assertEqual(fast.solvedBy, "smatrix-1d-x-4x4-cuda")
        self.assertEqual(fast.rAmplitudes.shape, full.rAmplitudes.shape)
        self.assertAlmostEqual(fast.reflection, full.reflection, places=12)
        self.assertAlmostEqual(fast.transmission, full.transmission, places=12)
        offLinePower = sum(
            order.reflectedPower + order.transmittedPower
            for order in fast.orders
            if order.my != 0
        )
        self.assertAlmostEqual(offLinePower, 0.0, places=12)

    def testAnisotropicNormalVectorLiPathIsAvailableForScalarGratings(self) -> None:
        gridY, gridX = 12, 14
        yIndex, xIndex = np.mgrid[0:gridY, 0:gridX]
        xx = (xIndex + 0.5) / gridX - 0.5
        yy = (yIndex + 0.5) / gridY - 0.5
        epsilon = np.ones((gridY, gridX), dtype=complex)
        epsilon[xx**2 + yy**2 <= 0.18**2] = 2.25
        radius = np.sqrt(xx**2 + yy**2)
        safeRadius = np.where(radius > 1e-12, radius, 1.0)
        normalField = np.zeros(epsilon.shape + (2,), dtype=float)
        normalField[..., 0] = np.where(radius > 1e-12, xx / safeRadius, 1.0)
        normalField[..., 1] = np.where(radius > 1e-12, yy / safeRadius, 0.0)

        compiled = anisotropic.compileLayers(
            [anisotropic.Layer(thickness=0.1, epsilon=epsilon, normalField=normalField)],
            orders=(1, 1),
        )
        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

        result = anisotropic.solveStack(
            layers=compiled,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            epsIncident=1.0,
            epsTransmission=1.0,
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAnisotropicGeneratedVectorFieldWorksForPiecewiseConstantScalarGrid(self) -> None:
        epsilon = np.ones((20, 20), dtype=complex)
        epsilon[5:15, 7:13] = 2.25
        compiled = anisotropic.compileLayers(
            [anisotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="normal-vector")],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

        result = anisotropic.solveStack(
            layers=compiled,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAnisotropicSampledGeometryPreservesStandardFactorization(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.35, 0.25),
            shape=(18, 18),
            factorization="standard",
        )
        compiled = anisotropic.compileLayers([layer], orders=(1, 1), truncation="circular")
        self.assertEqual(compiled[0].tensorData.factorization, "z-li")

    def testIsotropicAdvancedPathsAreIndependentAndReusable(self) -> None:
        layer = isotropic.rectangularHollowPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.5, 0.5),
            holeRadius=0.11,
            shape=(18, 18),
            factorization="auto",
        )
        compiled = _compileIsotropic([layer], orders=(1, 1), truncation="circular")
        self.assertEqual(compiled[0].factorization, "normal-vector-li")

        common = dict(
            layers=compiled,
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            theta=np.deg2rad(5),
        )
        smatrix = _solveIsotropic(**common, method="smatrix", sAmplitude=1.0, pAmplitude=0.0)
        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            isotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=compiled,
                orders=(1, 1),
                truncation="circular",
                method="etm",
            )

        batch = _solveIsotropicBatch(
            **common,
            excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
        )
        sequentialTm = _solveIsotropic(**common, sAmplitude=0.0, pAmplitude=1.0)
        self.assertAlmostEqual(batch["TE"].reflection, smatrix.reflection, places=12)
        self.assertAlmostEqual(batch["TM"].transmission, sequentialTm.transmission, places=12)
        self.assertEqual(batch["TE"].solvedBy, "smatrix-batch-cuda")

        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            isotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=compiled,
                orders=(1, 1),
                truncation="circular",
                method="etm",
            )

    def testIsotropicAutoFactorizationGeneratesVectorFieldForPiecewiseConstantGrid(self) -> None:
        epsilon = np.ones((18, 18), dtype=complex)
        epsilon[4:14, 6:12] = 2.25
        compiled = _compileIsotropic(
            [isotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="auto")],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(compiled[0].factorization, "normal-vector-li")

        result = _solveIsotropic(
            layers=compiled,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testIsotropicStandardFactorizationSkipsGeneratedVectorField(self) -> None:
        epsilon = np.ones((18, 18), dtype=complex)
        epsilon[4:14, 6:12] = 2.25
        compiled = _compileIsotropic(
            [isotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="standard")],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(compiled[0].factorization, "standard")

    def testIsotropicAnalyticGeometryAndHomogeneousFastPathMetadata(self) -> None:
        disk = isotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            radius=0.2,
            analytic=True,
        )
        compiledDisk = _compileIsotropic([disk], orders=(2, 2), truncation="circular")
        self.assertEqual(compiledDisk[0].factorization, "analytic-normal-vector-li")
        expectedAverage = 1.0 + (2.25 - 1.0) * np.pi * 0.2**2
        self.assertAlmostEqual(compiledDisk[0].epsilonMatrix[0, 0], expectedAverage, places=12)

        rectangle = isotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.0,
            size=(0.3, 0.2),
            analytic=True,
        )
        ellipse = isotropic.ellipticalPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.0,
            radii=(0.2, 0.1),
            analytic=True,
        )
        annulus = isotropic.annularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            ring=2.0,
            innerRadius=0.08,
            outerRadius=0.18,
            analytic=True,
        )
        compiledShapes = _compileIsotropic([rectangle, ellipse, annulus], orders=(1, 1), truncation="circular")
        self.assertTrue(all(layer.factorization == "analytic-normal-vector-li" for layer in compiledShapes))
        self.assertTrue(all(layer.displacementMatrices is not None for layer in compiledShapes))

        standardRectangle = isotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.0,
            size=(0.3, 0.2),
            analytic=True,
            factorization="standard",
        )
        compiledStandard = _compileIsotropic([standardRectangle], orders=(1, 1), truncation="circular")
        self.assertEqual(compiledStandard[0].factorization, "analytic-li")
        self.assertIsNone(compiledStandard[0].displacementMatrices)

        uniform = _compileIsotropic([isotropic.Layer(thickness=0.1, epsilon=np.full((8, 8), 2.25))], orders=1)
        self.assertAlmostEqual(uniform[0].homogeneousEpsilon, 2.25)

    def testIsotropicOneDimensionalGratingUsesReducedCudaPath(self) -> None:
        grid = np.tile(np.r_[np.full(8, 2.25), np.ones(8)], (12, 1))
        layer = isotropic.Layer(thickness=0.06, epsilon=grid, factorization="standard")
        common = dict(
            layers=[layer],
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(2, 2),
            theta=np.deg2rad(6),
            phi=np.deg2rad(15),
            sAmplitude=1.0,
            pAmplitude=0.0,
            truncation="rectangular",
        )

        reduced = _solveIsotropic(**common)
        full = _solveIsotropic(**common, returnFields=True)

        self.assertEqual(reduced.solvedBy, "smatrix-1d-x-cuda")
        self.assertEqual(reduced.rAmplitudes.shape, full.rAmplitudes.shape)
        self.assertAlmostEqual(reduced.reflection, full.reflection, places=10)
        self.assertAlmostEqual(reduced.transmission, full.transmission, places=10)
        offLinePower = sum(
            order.reflectedPower + order.transmittedPower
            for order in reduced.orders
            if order.my != 0
        )
        self.assertAlmostEqual(offLinePower, 0.0, places=12)

        simulation = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(2, 2),
            truncation="rectangular",
        )
        simulated = simulation.solve(1.1, theta=np.deg2rad(6), phi=np.deg2rad(15), polarization="TE")
        self.assertEqual(simulated.solvedBy, "smatrix-1d-x-cuda")
        self.assertAlmostEqual(simulated.reflection, reduced.reflection, places=10)

        analyticStripe = isotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.45, 1.0),
            analytic=True,
            jonesResolution=32,
        )
        compiledStripe = _compileIsotropic([analyticStripe], orders=(2, 2), truncation="rectangular")
        stripeResult = _solveIsotropic(
            layers=compiledStripe,
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(2, 2),
            truncation="rectangular",
        )
        self.assertEqual(stripeResult.solvedBy, "smatrix-1d-x-cuda")

    def testIsotropicSweepOrdersReportsScatteringAndFieldDeltas(self) -> None:
        layer = isotropic.Layer(thickness=0.12, epsilon=2.25, factorization="standard")
        report = isotropic.sweepOrders(
            layers=[layer],
            wavelength=1.0,
            period=(1.0, 1.0),
            maxOrder=1,
            fieldComponent="EIntensity",
            fieldPlane="xz",
            fieldShape=(5, 5),
            rtTolerance=1e-9,
            fieldTolerance=1e-9,
        )

        self.assertEqual(len(report.points), 2)
        self.assertIsNone(report.points[0].deltaReflection)
        self.assertIsNotNone(report.points[1].deltaTransmission)
        self.assertIsNotNone(report.points[1].fieldRelativeDelta)
        self.assertIn("orders", report.table())

    def testIsotropicAnalyticCompositeGeometryUsesExactAreaFraction(self) -> None:
        layer = isotropic.rectangularHollowPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=3.0,
            size=(0.5, 0.4),
            holeRadius=0.1,
            analytic=True,
        )
        compiled = _compileIsotropic([layer], orders=(1, 1), truncation="circular")
        expected = 1.0 + (3.0 - 1.0) * 0.5 * 0.4 + (1.0 - 3.0) * np.pi * 0.1**2
        self.assertEqual(compiled[0].factorization, "analytic-li")
        self.assertAlmostEqual(compiled[0].epsilonMatrix[0, 0], expected, places=12)

        cross = isotropic.crossPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.0,
            armLengths=(0.7, 0.5),
            armWidths=(0.2, 0.3),
            analytic=True,
        )
        compiledCross = _compileIsotropic([cross], orders=(1, 1), truncation="circular")
        expectedCross = 1.0 + (2.0 - 1.0) * (0.7 * 0.2 + 0.3 * 0.5 - 0.3 * 0.2)
        self.assertAlmostEqual(compiledCross[0].epsilonMatrix[0, 0], expectedCross, places=12)

    def testAdaptiveZStackBuildsUsableHighOrderSlices(self) -> None:
        adaptive = isotropic.AdaptiveLayerSpec(
            thickness=0.2,
            layerAt=lambda z: isotropic.Layer(thickness=0.0, epsilon=2.0 + z),
            tolerance=1e-5,
            maxDepth=4,
            quadratureOrder=4,
            name="graded",
        )
        layers = adaptive.toLayers()
        self.assertGreaterEqual(len(layers), 1)
        self.assertAlmostEqual(sum(layer.thickness for layer in layers), 0.2)
        result = _solveIsotropic(
            layers=[adaptive],
            wavelength=0.9,
            period=(1.0, 1.0),
            orders=(0, 0),
        )
        self.assertTrue(np.isfinite(result.reflection))

    def testIsotropicHighLevelSimulationUsesBatchSpectrum(self) -> None:
        layer = isotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            radius=0.18,
            analytic=True,
        )
        simulation = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
        )
        result = simulation.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(result.transmission))

        spectrum = simulation.spectrum([1.0, 1.1], polarizations=("TE", "TM"), workers=1)
        self.assertEqual(spectrum["TE"]["reflection"].shape, (2,))
        self.assertEqual(spectrum["TM"]["transmission"].shape, (2,))

    def testIsotropicSimulationBuilderAddLayerAndPreparedCache(self) -> None:
        simulation = isotropic.RCWASimulation(period=(1.0, 1.0), orders=(1, 1), truncation="circular")
        patterned = simulation.addLayer(
            thickness=0.05,
            epsilon=1.0,
            shape=(16, 16),
            name="mutable sampled layer",
        )
        self.assertIsInstance(patterned, isotropic.PatternLayer)
        patterned.circle(radius=0.18, material=2.25)

        geometryLayer = isotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.04,
            background=1.0,
            post=2.0,
            size=(0.2, 0.2),
            analytic=True,
        )
        simulation.addLayer(geometryLayer)

        first = simulation.solve(1.0, polarization="TE")
        second = simulation.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(first.reflection))
        self.assertTrue(np.isfinite(second.transmission))
        self.assertEqual(len(simulation._preparedCache), 1)

        patterned.rectangle(size=(0.1, 0.1), material=1.5)
        updated = simulation.solve(1.0, polarization="TE")
        self.assertTrue(np.isfinite(updated.reflection))
        self.assertEqual(len(simulation._preparedCache), 2)

    def testAnalyticDiskFourierCoefficientsUseExactAreaFraction(self) -> None:
        disk = anisotropic.AnalyticDisk(
            period=(1.0, 1.0),
            radius=0.2,
            background=1.0,
            inclusion=4.0,
            factorization="analytic",
        )
        compiled = anisotropic.compileLayers([anisotropic.Layer(thickness=0.1, epsilon=disk)], orders=(2, 2))
        zeroOrder = np.where(
            (compiled[0].tensorData.components[0][0].diagonal() != 0)
        )[0]
        del zeroOrder
        harmonics = makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="circular",
        )
        matrix = anisotropic.analyticDiskConvolution(disk, harmonics)
        expectedAverage = 1.0 + (4.0 - 1.0) * np.pi * 0.2**2

        self.assertAlmostEqual(matrix[0, 0], expectedAverage, places=12)
        self.assertEqual(compiled[0].tensorData.factorization, "analytic-li")

    def testAnalyticDiskJonesFactorizationSolvesFiniteResult(self) -> None:
        disk = anisotropic.AnalyticDisk(
            period=(1.0, 1.0),
            radius=0.2,
            background=1.0,
            inclusion=2.25,
            factorization="jones",
            jonesResolution=64,
        )
        compiled = anisotropic.compileLayers(
            [anisotropic.Layer(thickness=0.1, epsilon=disk)],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(compiled[0].tensorData.factorization, "jones-li")

        result = anisotropic.solveStack(
            layers=compiled,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAutoFactorizationUsesJonesLiForAnalyticDisk(self) -> None:
        layer = anisotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.1,
            background=1.0,
            post=2.25,
            radius=0.2,
            analytic=True,
            factorization="auto",
        )
        compiled = anisotropic.compileLayers([layer], orders=(1, 1), truncation="circular")

        self.assertEqual(compiled[0].tensorData.factorization, "jones-li")

    def testCommercialStyleAnalyticRectangleNeedsOnlyOrders(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.35, 0.25),
            factorization="auto",
            jonesResolution=48,
        )
        self.assertIsInstance(layer.epsilon, anisotropic.AnalyticRectangle)

        compiled = anisotropic.compileLayers([layer], orders=(2, 2), truncation="rectangular")
        self.assertEqual(compiled[0].tensorData.factorization, "jones-li")

        harmonics = makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="rectangular",
        )
        matrix = anisotropic.analyticRectangleConvolution(layer.epsilon, harmonics)
        expectedAverage = 1.0 + (2.25 - 1.0) * 0.35 * 0.25
        self.assertAlmostEqual(matrix[0, 0], expectedAverage, places=12)

    def testAnalyticRectangleSolvesWithCircularAndRectangularTruncation(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.30, 0.20),
            angle=np.deg2rad(15),
            jonesResolution=40,
        )

        for truncation in ("circular", "rectangular"):
            with self.subTest(truncation=truncation):
                result = anisotropic.solveStack(
                    layers=[layer],
                    wavelength=1.1,
                    period=(1.0, 1.0),
                    orders=(1, 1),
                    truncation=truncation,
                    theta=np.deg2rad(5.0),
                    pAmplitude=1.0,
                    sAmplitude=0.0,
                )
                self.assertTrue(np.isfinite(result.reflection))
                self.assertTrue(np.isfinite(result.transmission))

    def testAutoFactorizationUsesNormalVectorForSampledScalarBoundaries(self) -> None:
        layer = anisotropic.rectangularHollowPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            size=(0.5, 0.5),
            holeRadius=0.12,
            shape=(16, 16),
            factorization="auto",
        )
        compiled = anisotropic.compileLayers([layer], orders=(1, 1), truncation="circular")

        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

    def testXzTensorHelpersSupportCouplingTwistAndFullRotation(self) -> None:
        angle = np.deg2rad(30)
        gyrotropy = 0.2j
        tensor = anisotropic.gyrotropicXzTensor(
            epsilonParallel=2.5,
            epsilonY=3.1,
            gyrotropy=gyrotropy,
            twist=angle,
            twistMode="coupling",
        )
        self.assertAlmostEqual(tensor[0, 2], gyrotropy * np.cos(angle))
        self.assertAlmostEqual(tensor[1, 2], gyrotropy * np.sin(angle))
        self.assertAlmostEqual(tensor[2, 0], -gyrotropy * np.cos(angle))
        self.assertAlmostEqual(tensor[2, 1], -gyrotropy * np.sin(angle))
        self.assertAlmostEqual(tensor[0, 1], 0.0)

        rotated = anisotropic.xzTensor(2.0, 4.0, 3.0, 0.1, 0.1, twist=angle, twistMode="tensor")
        self.assertGreater(abs(rotated[0, 1]), 0.1)

    def testHighLevelSimulationWrapsStaticGeometryAndDynamicXzLayer(self) -> None:
        post = anisotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            radius=0.2,
            analytic=True,
            factorization="standard",
        )

        def xzLayer(wavelength: float) -> np.ndarray:
            del wavelength
            return anisotropic.xzTensor(2.25, 2.25, 2.25, 0.05, 0.05)

        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[post, anisotropic.homogeneousLayer(0.05, xzLayer)],
            orders=(1, 1),
            truncation="circular",
        )
        result = simulation.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

        spectrum = simulation.spectrum([1.0, 1.1], theta=0.0, polarizations=("TE",), bidirectional=True)
        self.assertEqual(spectrum["TE"]["absorptivity"].shape, (2,))
        self.assertEqual(spectrum["TE"]["emissivity"].shape, (2,))

    def testBidirectionalSpectrumReusesDynamicLayersWithinOneWavelength(self) -> None:
        calls: list[float] = []

        def dispersiveLayer(wavelength: float) -> anisotropic.Layer:
            calls.append(float(wavelength))
            return anisotropic.Layer(
                thickness=0.05,
                epsilon=anisotropic.xzTensor(2.25 + 0.1 * wavelength, 2.2, 2.15, 0.03, 0.03),
                name="dynamic tensor film",
            )

        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[dispersiveLayer],
            orders=(0, 0),
            truncation="circular",
            workers=1,
        )

        simulation.spectrum([1.0, 1.1, 1.2], theta=np.deg2rad(6), polarizations=("TE",), bidirectional=True, workers=1)
        self.assertEqual(calls, [1.0, 1.1, 1.2])

    def testAnisotropicProjectInterfaceMatchesSimulation(self) -> None:
        project = anisotropic.Project(period=(1.0, 1.0), order=(1, 1), truncation="circular", samples=(16, 16))
        project.add_rectangle(
            height=0.05,
            size=(0.30, 0.22),
            material=2.25,
            background=1.0,
            angle_deg=10.0,
            name="readable rectangle",
        )
        project.add_uniform(0.04, anisotropic.xzTensor(2.1, 2.2, 2.3, 0.03, 0.02), name="tensor film")

        thetaDeg = 5.0
        projectResult = project.solve(1.05, theta_deg=thetaDeg, polarization="TE")
        simulationResult = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=tuple(project.layers),
            orders=(1, 1),
            truncation="circular",
        ).solve(1.05, theta=np.deg2rad(thetaDeg), polarization="TE")

        self.assertAlmostEqual(projectResult.reflection, simulationResult.reflection, places=12)
        self.assertAlmostEqual(projectResult.transmission, simulationResult.transmission, places=12)

        spectrum = project.spectrum([1.0, 1.1], theta_deg=thetaDeg, polarizations=("TE",), bidirectional=False)
        self.assertEqual(spectrum["TE"]["absorptivity"].shape, (2,))
        self.assertNotIn("emissivity", spectrum["TE"])

        self.assertIs(project.simulation(), project.simulation())

    def testAnisotropicBatchSolveMatchesSequentialPolarizations(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.35, 0.25),
            shape=(18, 18),
            factorization="normal-vector",
        )
        common = dict(
            layers=[layer],
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(1, 1),
            theta=np.deg2rad(8),
            truncation="circular",
        )

        batch = anisotropic.solveStackBatch(
            **common,
            excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
        )
        sequentialTE = anisotropic.solveStack(**common, sAmplitude=1.0, pAmplitude=0.0)
        sequentialTM = anisotropic.solveStack(**common, sAmplitude=0.0, pAmplitude=1.0)

        self.assertAlmostEqual(batch["TE"].reflection, sequentialTE.reflection, places=12)
        self.assertAlmostEqual(batch["TE"].transmission, sequentialTE.transmission, places=12)
        self.assertAlmostEqual(batch["TM"].reflection, sequentialTM.reflection, places=12)
        self.assertAlmostEqual(batch["TM"].transmission, sequentialTM.transmission, places=12)
        self.assertEqual(batch["TE"].solvedBy, "smatrix-batch-cuda")

    def testAnisotropicLayerEigProfileReportsMatrixShapesAndTimes(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            size=(0.30, 0.25),
            shape=(12, 12),
            factorization="normal-vector",
            name="profiled post",
        )
        result = anisotropic.solveStack(
            layers=[layer, anisotropic.Layer(thickness=0.03, epsilon=2.1, name="profiled film")],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            theta=np.deg2rad(4.0),
            truncation="circular",
            profile=True,
        )

        self.assertEqual(len(result.layerEigTimings), 2)
        self.assertEqual(result.layerEigTimings[0].name, "profiled post")
        self.assertEqual(result.layerEigTimings[1].matrixShape[-2:], (4, 4))
        self.assertTrue(all(entry.eigTimeSeconds >= 0.0 for entry in result.layerEigTimings))

    def testAnisotropicSpectrumCombinesPolarizationsAndCustomExcitations(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                anisotropic.rectangularPostLayer(
                    period=(1.0, 1.0),
                    thickness=0.05,
                    background=1.0,
                    post=2.25,
                    size=(0.30, 0.30),
                    shape=(12, 12),
                    factorization="normal-vector",
                )
            ],
            orders=(1, 1),
            truncation="circular",
        )

        spectrum = simulation.spectrum(
            [1.0, 1.1],
            theta=np.deg2rad(5.0),
            polarizations=("TE", "TM"),
            excitations={"custom": (0.7 + 0.1j, -0.2 + 0.05j)},
            bidirectional=False,
            workers=1,
        )

        for label in ("TE", "TM", "custom"):
            self.assertEqual(spectrum[label]["absorptivity"].shape, (2,))
            self.assertTrue(np.all(np.isfinite(spectrum[label]["absorptivity"])))

    def testSpectrumWorkersMatchSerialSpectrum(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                anisotropic.rectangularPostLayer(
                    period=(1.0, 1.0),
                    thickness=0.05,
                    background=1.0,
                    post=2.25,
                    size=(0.30, 0.30),
                    shape=(16, 16),
                    factorization="normal-vector",
                )
            ],
            orders=(1, 1),
            truncation="circular",
            workers=2,
        )

        serial = simulation.spectrum([1.0, 1.1, 1.2], theta=np.deg2rad(5), workers=1)
        parallel = simulation.spectrum([1.0, 1.1, 1.2], theta=np.deg2rad(5), workers=2)

        np.testing.assert_allclose(parallel["TE"]["absorptivity"], serial["TE"]["absorptivity"], atol=1e-12)
        np.testing.assert_allclose(parallel["TE"]["emissivity"], serial["TE"]["emissivity"], atol=1e-12)
        np.testing.assert_allclose(parallel["TM"]["absorptivity"], serial["TM"]["absorptivity"], atol=1e-12)
        np.testing.assert_allclose(parallel["TM"]["emissivity"], serial["TM"]["emissivity"], atol=1e-12)

    def testAnisotropicSpectrumParallelPlanKeepsAutoSerialAndRejectsProcessMode(self) -> None:
        self.assertEqual(anisotropic_simulation._spectrumParallelPlan(4, "auto", "cuda"), ("serial", 1))
        self.assertEqual(anisotropic_simulation._spectrumParallelPlan(4, "thread", "cuda"), ("thread", 4))
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            anisotropic_simulation._spectrumParallelPlan(4, "process", "cuda")

    def testAnisotropicPreparedCacheReusesPreparedStack(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[anisotropic.Layer(thickness=0.05, epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04))],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        first = simulation._preparedStack(1.0, theta=np.deg2rad(5), phi=0.0)
        second = simulation._preparedStack(1.0, theta=np.deg2rad(5), phi=0.0)
        third = simulation._preparedStack(1.0, theta=np.deg2rad(6), phi=0.0)

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(simulation._preparedCache), 2)

    def testSampledTensorGeometryAndTaperHelpersBuildValidLayers(self) -> None:
        tensor = anisotropic.xzTensor(2.25, 2.25, 2.25, 0.1, 0.1)
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=tensor,
            size=(0.3, 0.2),
            shape=(16, 16),
        )
        self.assertEqual(np.asarray(layer.epsilon).shape, (16, 16, 3, 3))

        hollow = anisotropic.rectangularHollowPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            size=(0.5, 0.5),
            holeRadius=0.12,
            shape=(16, 16),
            factorization="normal-vector",
        )
        self.assertEqual(np.asarray(hollow.epsilon).shape, (16, 16))
        self.assertEqual(np.asarray(hollow.normalField).shape, (16, 16, 2))

        taper = anisotropic.slicedTaperStack(
            period=(1.0, 1.0),
            height=0.2,
            background=1.0,
            post=2.25,
            bottomSize=(0.4, 0.4),
            topSize=(0.1, 0.1),
            kind="rectangle",
            slices=3,
            shape=(12, 12),
        )
        self.assertEqual(len(taper), 3)

        stack = anisotropic.LayerStack(period=(1.0, 1.0), shape=(18, 18))
        stack.addLayer(0.3, 1.0, name="3D host")
        stack.addPyramid(
            tensor,
            z=(0.0, 0.3),
            topSize=0.08,
            bottomSize=(0.46, 0.46),
            slices=3,
        )
        stack.addPolygonPrism(
            3.1,
            z=(0.1, 0.2),
            vertices=((-0.2, -0.1), (0.18, -0.08), (0.0, 0.22)),
        )
        layers = stack.toLayers()
        self.assertEqual(len(layers), 3)
        self.assertEqual(np.asarray(layers[0].epsilon).shape, (18, 18, 3, 3))
        pyramidCounts = [int(np.count_nonzero(np.isclose(np.asarray(layer.epsilon)[..., 0, 0], 2.25))) for layer in layers]
        self.assertGreater(pyramidCounts[-1], pyramidCounts[0])
        self.assertTrue(np.any(np.isclose(np.asarray(layers[1].epsilon)[..., 0, 0], 3.1)))

    def testShi2025ExampleRunsSmallStableCudaSpectrum(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "examples" / "anisotropic_example" / "shi2025GeInAsHollowArray.py"
        spec = importlib.util.spec_from_file_location("shi2025GeInAsHollowArray", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.ORDER = 1
        module.GRID = 16
        module.WORKERS = 1
        module.SHOW = False
        module.WAVELENGTHS = np.array([17.30, 17.34])
        module.POINTS = module.WAVELENGTHS.size

        tensor = module.inas_tensor(17.7)
        self.assertLess(np.imag(tensor[0, 2]), 0.0)
        self.assertGreater(np.imag(tensor[2, 0]), 0.0)

        simulation = module.make_simulation()
        self.assertEqual(simulation.backend, "cuda")
        spectrum = simulation.spectrum(
            module.WAVELENGTHS,
            theta=module.THETA,
            phi=module.PHI,
            polarizations=("TE", "TM"),
            bidirectional=True,
            workers=module.WORKERS,
        )

        for polarization in ("TE", "TM"):
            for key in ("absorptivity", "emissivity", "nonreciprocity"):
                self.assertEqual(spectrum[polarization][key].shape, module.WAVELENGTHS.shape)
                self.assertTrue(np.all(np.isfinite(spectrum[polarization][key])))


if __name__ == "__main__":
    unittest.main()
