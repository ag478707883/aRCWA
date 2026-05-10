import importlib.util
import os
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


def torchCudaAvailable() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def solveIsotropic(
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


def solveIsotropicBatch(
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
    profile=False,
):
    simulation = anisotropic.RCWASimulation(
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
        profile=profile,
    )


def solveAnisotropicFields(
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
    profile=False,
):
    simulation = anisotropic.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveFieldExcitation(
        wavelength,
        theta=theta,
        phi=phi,
        sAmplitude=sAmplitude,
        pAmplitude=pAmplitude,
        profile=profile,
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
    profile=False,
):
    del profile
    simulation = anisotropic.RCWASimulation(
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
    profile=False,
):
    del profile
    simulation = anisotropic.RCWASimulation(
        period=period,
        layers=layers,
        orders=orders,
        truncation=truncation,
        epsIncident=epsIncident,
        epsTransmission=epsTransmission,
        backend=backend,
    )
    return simulation.solveBatchPowers(wavelength, theta=theta, phi=phi, excitations=excitations)


def compileAnisotropicLayersForTest(layers, *, orders, truncation="circular"):
    return anisotropic_solver.compileLayers(layers, orders=orders, truncation=truncation)


def isotropicLayerMetadataForTest(
    layer,
    *,
    orders,
    truncation="circular",
):
    simulation = isotropic.RCWASimulation(
        period=(1.0, 1.0),
        orders=orders,
        truncation=truncation,
        backend="cuda",
    )
    return simulation.compileLayerTorch(layer)


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

        scalarOrder = solveIsotropic(**common, orders=1)
        tupleOrder = solveIsotropic(**common, orders=(1, 1))

        self.assertEqual(scalarOrder.orders[0].mx, tupleOrder.orders[0].mx)
        self.assertAlmostEqual(scalarOrder.reflection, tupleOrder.reflection, places=12)
        self.assertAlmostEqual(scalarOrder.transmission, tupleOrder.transmission, places=12)

    def testIsotropicDefaultTruncationIsCircular(self) -> None:
        result = solveIsotropic(
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
        if not torchCudaAvailable():
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA-enabled torch"):
                isotropic.resolveBackend("auto")
            return

        backend = isotropic.resolveBackend("auto")
        self.assertEqual(backend.name, "cuda")
        self.assertTrue(backend.isCuda)

    def testIsotropicTorchCpuBackendIsRejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            isotropic.resolveBackend("torch-cpu")

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
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

        cuda = solveIsotropic(**common, sAmplitude=0.6 + 0.2j, pAmplitude=0.15 - 0.3j, backend="cuda")

        for alias in ("gpu", "torch", "auto"):
            aliasResult = solveIsotropic(
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
        cudaBatch = solveIsotropicBatch(**common, excitations=excitations, method="smatrix", backend="cuda")
        aliasBatch = solveIsotropicBatch(**common, excitations=excitations, method="smatrix", backend="gpu")

        for label in excitations:
            self.assertAlmostEqual(aliasBatch[label].reflection, cudaBatch[label].reflection, places=9)
            self.assertAlmostEqual(aliasBatch[label].transmission, cudaBatch[label].transmission, places=9)
            np.testing.assert_allclose(aliasBatch[label].rAmplitudes, cudaBatch[label].rAmplitudes, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(aliasBatch[label].tAmplitudes, cudaBatch[label].tAmplitudes, rtol=1e-8, atol=1e-9)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
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
        self.assertEqual(len(simulation.preparedCache), 1)
        prepared = next(iter(simulation.preparedCache.values()))
        self.assertTrue(prepared.backend.isCuda)
        self.assertEqual(prepared.total.s11.device.type, "cuda")
        self.assertEqual(prepared.components[0].s11.device.type, "cuda")
        self.assertEqual(prepared.components[-1].s21.device.type, "cuda")

        self.assertEqual(len(simulation.torchLayerCache), 0)
        self.assertEqual(len(simulation.compiledLayerCache), 1)
        torchLayer = next(iter(simulation.compiledLayerCache.values()))
        self.assertEqual(torchLayer.epsilonMatrix.device.type, "cuda")
        self.assertEqual(torchLayer.epsilonInverse.device.type, "cuda")

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
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
        factorized = isotropic_solver.layerDataForTorch(
            layer,
            prepared.harmonics,
            prepared.backend.xp,
            prepared.total.s11.device,
        )
        self.assertEqual(factorized.epsilonMatrix.device.type, "cuda")
        self.assertEqual(factorized.epsilonInverse.device.type, "cuda")
        self.assertEqual(factorized.factorization, "normal-vector-li")
        self.assertTrue(all(matrix.device.type == "cuda" for matrix in factorized.displacementMatrices))

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicLayerEigProfileReportsMatrixShapesAndTimes(self) -> None:
        layer = isotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            size=(0.30, 0.24),
            shape=(16, 16),
            factorization="standard",
            name="profiled rectangle",
        )
        simulation = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer, isotropic.Layer(thickness=0.03, epsilon=2.1, name="profiled slab")],
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
        )

        result = simulation.solve(1.0, polarization="TM", profile=True)

        self.assertEqual(len(result.layerEigTimings), 2)
        self.assertEqual(result.layerEigTimings[0].name, "profiled rectangle")
        self.assertEqual(result.layerEigTimings[0].matrixShape[-2:], (10, 10))
        self.assertEqual(result.layerEigTimings[1].kind, "homogeneous-analytic")
        self.assertTrue(all(entry.eigTimeSeconds >= 0.0 for entry in result.layerEigTimings))
        self.assertTrue(result.layerEigTimings[0].factorizationTimeSeconds >= 0.0)
        self.assertTrue(result.layerEigTimings[0].inverseTimeSeconds >= 0.0)
        self.assertTrue(result.layerEigTimings[0].pqTimeSeconds >= 0.0)
        self.assertTrue(result.layerEigTimings[0].totalTimeSeconds >= result.layerEigTimings[0].eigTimeSeconds)
        self.assertIsNotNone(result.stackTiming)
        self.assertTrue(result.stackTiming.interfaceTimeSeconds >= 0.0)
        self.assertTrue(result.stackTiming.cascadeTimeSeconds >= 0.0)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicPreparedTotalMatchesFullCascadeReflectionTransmission(self) -> None:
        epsilon = np.ones((20, 22), dtype=complex)
        epsilon[5:15, 7:16] = 2.6
        layer = isotropic.Layer(thickness=0.07, epsilon=epsilon, factorization="standard")
        common = dict(
            layers=[layer],
            wavelength=0.92,
            period=(1.0, 1.0),
            orders=(1, 1),
            epsIncident=1.0,
            epsTransmission=1.0,
            theta=np.deg2rad(6.0),
            phi=np.deg2rad(11.0),
            truncation="circular",
            backend="cuda",
        )

        prepared = isotropic_solver.prepareStackTorch(**common)
        referenceTotal = isotropic_solver.prefixSMatricesTorch(
            prepared.components,
            prepared.nPorts,
            prepared.backend.xp,
            prepared.total.s11.device,
        )[-1]
        full = isotropic_solver.PreparedTorchStack(
            layers=prepared.layers,
            wavelength=prepared.wavelength,
            period=prepared.period,
            orders=prepared.orders,
            epsIncident=prepared.epsIncident,
            epsTransmission=prepared.epsTransmission,
            truncation=prepared.truncation,
            harmonics=prepared.harmonics,
            layerModes=prepared.layerModes,
            layerEigTimings=prepared.layerEigTimings,
            components=prepared.components,
            total=referenceTotal,
            incidentForward=prepared.incidentForward,
            incidentBackward=prepared.incidentBackward,
            transmissionForward=prepared.transmissionForward,
            zeroIndex=prepared.zeroIndex,
            backend=prepared.backend,
            stackTiming=prepared.stackTiming,
        )
        excitations = ((1.0, 0.0), (0.0, 1.0), (0.6 + 0.1j, -0.2 + 0.3j))

        for sAmplitude, pAmplitude in excitations:
            with self.subTest(sAmplitude=sAmplitude, pAmplitude=pAmplitude):
                fullResult = isotropic_solver.evaluatePreparedStackTorch(
                    full,
                    sAmplitude=sAmplitude,
                    pAmplitude=pAmplitude,
                )
                preparedResult = isotropic_solver.evaluatePreparedStackTorch(
                    prepared,
                    sAmplitude=sAmplitude,
                    pAmplitude=pAmplitude,
                )

                self.assertAlmostEqual(preparedResult.reflection, fullResult.reflection, places=10)
                self.assertAlmostEqual(preparedResult.transmission, fullResult.transmission, places=10)
                np.testing.assert_allclose(preparedResult.rAmplitudes, fullResult.rAmplitudes, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(preparedResult.tAmplitudes, fullResult.tAmplitudes, rtol=1e-9, atol=1e-10)

        powersPrepared = isotropic_solver.prepareStackPowersTorch(**common)
        powers = isotropic_solver.evaluatePreparedBatchPowersTorch(
            powersPrepared,
            {"TE": (1.0, 0.0), "TM": (0.0, 1.0), "custom": (0.6 + 0.1j, -0.2 + 0.3j)},
        )
        for label, (sAmplitude, pAmplitude) in {
            "TE": (1.0, 0.0),
            "TM": (0.0, 1.0),
            "custom": (0.6 + 0.1j, -0.2 + 0.3j),
        }.items():
            with self.subTest(label=label):
                result = isotropic_solver.evaluatePreparedStackTorch(
                    prepared,
                    sAmplitude=sAmplitude,
                    pAmplitude=pAmplitude,
                )
                self.assertAlmostEqual(powers[label][0], result.reflection, places=10)
                self.assertAlmostEqual(powers[label][1], result.transmission, places=10)

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

        previous = os.environ.get("RCWA3D_CHECK_CUDA_EIG_FINITE")
        try:
            os.environ["RCWA3D_CHECK_CUDA_EIG_FINITE"] = "1"
            checkedEigenvalues, checkedEigenvectors = backend.eig(backend.asarray(matrix))
            checkedEigenvalues = backend.toNumpy(checkedEigenvalues)
            checkedEigenvectors = backend.toNumpy(checkedEigenvectors)
            checkedResidual = matrix @ checkedEigenvectors - checkedEigenvectors * checkedEigenvalues[np.newaxis, :]
            self.assertLess(np.linalg.norm(checkedResidual), 1e-8 * max(1.0, np.linalg.norm(matrix)))
        finally:
            if previous is None:
                os.environ.pop("RCWA3D_CHECK_CUDA_EIG_FINITE", None)
            else:
                os.environ["RCWA3D_CHECK_CUDA_EIG_FINITE"] = previous

        for alias in (None, "auto", "gpu", "torch", "torch-cuda"):
            with self.subTest(alias=alias):
                resolved = resolveAnisotropicBackend(alias)
                self.assertEqual(resolved.name, "cuda")
                self.assertTrue(resolved.isCuda)

        fast = resolveAnisotropicBackend("cuda", precision="complex64")
        self.assertEqual(str(fast.complexDtype), "torch.complex64")
        mixed = resolveAnisotropicBackend("cuda", precision="mixed")
        self.assertEqual(str(mixed.complexDtype), "torch.complex64")

        for alias in ("cpu", "numpy", "torch-cpu"):
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "CUDA-only"):
                    resolveAnisotropicBackend(alias)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicHomogeneousInterfaceCudaBlockSolveMatchesDenseSolve(self) -> None:
        harmonics = anisotropic_solver.makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            epsIncident=1.0,
            theta=np.deg2rad(5),
            phi=np.deg2rad(8),
            truncation="circular",
        )
        backend = resolveAnisotropicBackend("cuda")
        leftForward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 1.0, direction=1))
        leftBackward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 1.0, direction=-1))
        rightQ, rightModes, matrixShape, eigTime = anisotropic_solver.homogeneousTensorLayerModesMeasured(
            anisotropic.xzTensor(2.25, 2.1, 2.3, 0.05, 0.04),
            harmonics,
            backend,
            collectTiming=False,
        )
        rightForward = rightModes[:, : 2 * harmonics.count]
        rightBackward = rightModes[:, 2 * harmonics.count :]

        dense = anisotropic_solver.interfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            backend,
        )
        block = anisotropic_solver.homogeneousInterfaceSMatrix(
            leftForward,
            leftBackward,
            rightForward,
            rightBackward,
            harmonics.count,
            backend,
        )

        self.assertIsNotNone(block)
        assert block is not None
        for name in ("s11", "s12", "s21", "s22"):
            self.assertEqual(getattr(block, name).device.type, "cuda")
            np.testing.assert_allclose(
                backend.toNumpy(getattr(block, name)),
                backend.toNumpy(getattr(dense, name)),
                rtol=1e-10,
                atol=1e-11,
            )

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicHomogeneousInterfaceSolveFailurePropagates(self) -> None:
        harmonics = anisotropic_solver.makeHarmonics(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="circular",
        )
        backend = resolveAnisotropicBackend("cuda")
        leftForward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 1.0, direction=1))
        leftBackward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 1.0, direction=-1))
        rightForward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 2.25, direction=1))
        rightBackward = backend.asarray(anisotropic_solver.homogeneousBasis(harmonics, 2.25, direction=-1))

        class FailingSolveBackend:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.xp = wrapped.xp
                self.device = wrapped.device
                self.floatDtype = wrapped.floatDtype
                self.complexDtype = wrapped.complexDtype

            def solve(self, matrix, rhs):
                raise RuntimeError("forced homogeneous interface solve failure")

        with self.assertRaisesRegex(RuntimeError, "forced homogeneous interface solve failure"):
            anisotropic_solver.homogeneousInterfaceSMatrix(
                leftForward,
                leftBackward,
                rightForward,
                rightBackward,
                harmonics.count,
                FailingSolveBackend(backend),
            )

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicModeSplitCudaMatchesReferenceOrdering(self) -> None:
        backend = resolveAnisotropicBackend("cuda")
        rng = np.random.default_rng(34)
        q = rng.standard_normal(12) + 1j * rng.standard_normal(12)
        fluxes = rng.standard_normal(12)
        qTensor = backend.asarray(q)
        fluxTensor = backend.asarray(fluxes)

        forward = anisotropic_solver.forwardModeIndices(q, fluxes, 6)
        forwardSet = set(int(index) for index in forward)
        backward = np.array([index for index in range(q.size) if index not in forwardSet], dtype=int)
        expected = np.concatenate(
            [
                anisotropic_solver.sortModes(forward, q, fluxes, forward=True),
                anisotropic_solver.sortModes(backward, q, fluxes, forward=False),
            ]
        )
        actual = backend.toNumpy(
            anisotropic_solver.splitForwardBackwardIndicesTorch(qTensor, fluxTensor, 6, backend)
        )
        np.testing.assert_array_equal(actual, expected)

        qBatch = np.stack([q, q.conjugate() + 0.07j])
        fluxBatch = np.stack([fluxes, -0.8 * fluxes])
        actualBatch = backend.toNumpy(
            anisotropic_solver.splitForwardBackwardIndicesBatchTorch(
                backend.asarray(qBatch),
                backend.asarray(fluxBatch),
                6,
                backend,
            )
        )
        expectedBatch = []
        for qRow, fluxRow in zip(qBatch, fluxBatch):
            forward = anisotropic_solver.forwardModeIndices(qRow, fluxRow, 6)
            forwardSet = set(int(index) for index in forward)
            backward = np.array([index for index in range(qRow.size) if index not in forwardSet], dtype=int)
            expectedBatch.append(
                np.concatenate(
                    [
                        anisotropic_solver.sortModes(forward, qRow, fluxRow, forward=True),
                        anisotropic_solver.sortModes(backward, qRow, fluxRow, forward=False),
                    ]
                )
            )
        np.testing.assert_array_equal(actualBatch, np.asarray(expectedBatch))

    def testLegacyLowLevelSolveFunctionsAreNotPublic(self) -> None:
        self.assertFalse(hasattr(isotropic, "solveStack"))
        self.assertFalse(hasattr(isotropic, "solveStackBatch"))
        self.assertFalse(hasattr(isotropic, "compileLayers"))
        self.assertFalse(hasattr(anisotropic, "solveStack"))
        self.assertFalse(hasattr(anisotropic, "solveStackBatch"))
        self.assertFalse(hasattr(anisotropic, "solveStackBatchPowers"))
        self.assertFalse(hasattr(anisotropic, "compileLayers"))
        self.assertFalse(hasattr(anisotropic, "CompiledLayer"))
        self.assertFalse(hasattr(anisotropic, "TensorConvolutionData"))
        self.assertFalse(hasattr(anisotropic, "tensorConvolutionData"))
        self.assertFalse(hasattr(anisotropic, "liFactorizedSystemMatrix"))

    def testIsotropicPathRejectsTensorLikeLayerInputs(self) -> None:
        tensor = np.zeros((6, 6, 3, 3), dtype=complex)
        for index in range(3):
            tensor[..., index, index] = 2.25
        layer = anisotropic.Layer(thickness=0.2, epsilon=tensor, name="tensor layer")

        with self.assertRaisesRegex(TypeError, "rcwa3d_anisotropic.RCWASimulation"):
            solveIsotropic(
                layers=[layer],
                wavelength=1.0,
                period=(0.8, 0.8),
                orders=(1, 1),
                epsIncident=1.0,
                epsTransmission=1.0,
            )

        with self.assertRaisesRegex(TypeError, "rcwa3d_anisotropic.RCWASimulation"):
            isotropic.RCWASimulation(period=(0.8, 0.8), layers=[layer], orders=(1, 1)).solve(1.0)

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

        isotropicResult = solveIsotropic(layers=[isotropic.Layer(thickness=0.30, epsilon=2.25)], **common)
        anisotropicResult = solveAnisotropic(layers=[anisotropic.Layer(thickness=0.30, epsilon=2.25)], **common)

        self.assertAlmostEqual(anisotropicResult.reflection, isotropicResult.reflection, places=12)
        self.assertAlmostEqual(anisotropicResult.transmission, isotropicResult.transmission, places=12)
        self.assertAlmostEqual(anisotropicResult.conservation, 1.0, places=12)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicMuIdentityMatchesExistingHomogeneousTensorPath(self) -> None:
        epsilon = anisotropic.xzTensor(2.2, 2.1, 2.4, 0.08, 0.03)
        common = dict(
            wavelength=1.03,
            period=(0.9, 0.8),
            orders=(1, 1),
            truncation="circular",
            theta=np.deg2rad(7.0),
            phi=np.deg2rad(18.0),
            pAmplitude=1.0,
            sAmplitude=0.3,
            profile=True,
        )

        legacy = solveAnisotropic(layers=[anisotropic.Layer(thickness=0.12, epsilon=epsilon)], **common)
        explicitMu = solveAnisotropic(
            layers=[anisotropic.Layer(thickness=0.12, epsilon=epsilon, mu=np.eye(3, dtype=complex))],
            **common,
        )

        self.assertAlmostEqual(explicitMu.reflection, legacy.reflection, places=11)
        self.assertAlmostEqual(explicitMu.transmission, legacy.transmission, places=11)
        np.testing.assert_allclose(explicitMu.rAmplitudes, legacy.rAmplitudes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(explicitMu.tAmplitudes, legacy.tAmplitudes, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicHomogeneousMagneticLayerChangesImpedanceAndConservesPower(self) -> None:
        common = dict(
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(0, 0),
            epsIncident=1.0,
            epsTransmission=1.0,
            sAmplitude=1.0,
            pAmplitude=0.0,
        )
        empty = solveAnisotropic(layers=[anisotropic.Layer(thickness=0.23, epsilon=1.0)], **common)
        magnetic = solveAnisotropic(
            layers=[anisotropic.Layer(thickness=0.23, epsilon=1.0, mu=4.0 * np.eye(3, dtype=complex))],
            **common,
        )

        self.assertGreater(magnetic.reflection, empty.reflection + 1e-4)
        self.assertAlmostEqual(magnetic.conservation, 1.0, places=10)
        self.assertAlmostEqual(empty.conservation, 1.0, places=10)

    def testAnisotropicConstitutiveInputRejectsUnimplementedBiAnisotropicCoupling(self) -> None:
        layer = anisotropic.homogeneousLayer(
            0.1,
            anisotropic.constitutiveTensors(
                2.25 * np.eye(3, dtype=complex),
                mu=np.eye(3, dtype=complex),
                chi=0.01 * np.eye(3, dtype=complex),
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "magnetoelectric chi/xi"):
            compileAnisotropicLayersForTest([layer], orders=(0, 0))

    def testAnisotropicPatternedNonIdentityMuIsExplicitlyUnsupported(self) -> None:
        epsilon = np.ones((8, 8), dtype=complex)
        epsilon[2:6, 2:6] = 2.25
        layer = anisotropic.Layer(thickness=0.04, epsilon=epsilon, mu=1.1 * np.eye(3, dtype=complex))

        with self.assertRaisesRegex(NotImplementedError, "patterned non-identity mu"):
            compileAnisotropicLayersForTest([layer], orders=(1, 1))

    def testAnisotropicDynamicMaterialCallbacksMustReturnTensor(self) -> None:
        valid = anisotropic.homogeneousLayer(
            0.1,
            lambda wavelength: (2.25 + 0.01j * wavelength) * np.eye(3, dtype=complex),
            name="valid dynamic tensor",
        )
        self.assertEqual(valid.at(1.0).epsilon.shape, (3, 3))

        invalid = anisotropic.homogeneousLayer(
            0.1,
            lambda wavelength: 2.25 + 0.01j * wavelength,
            name="invalid dynamic scalar",
        )
        with self.assertRaisesRegex(ValueError, r"must return a \(3, 3\) permittivity tensor"):
            invalid.at(1.0)

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

        smatrix = solveAnisotropic(**common)
        batchCommon = {key: value for key, value in common.items() if key not in ("sAmplitude", "pAmplitude")}
        self.assertEqual(smatrix.solvedBy, "smatrix-cuda")
        batch = solveAnisotropicBatch(**batchCommon, excitations={"TM": (0.0, 1.0)})
        self.assertEqual(batch["TM"].solvedBy, "smatrix-batch-cuda")

        with self.assertRaises(TypeError):
            solveAnisotropic(**common, method="global")
        with self.assertRaises(TypeError):
            solveAnisotropicBatch(**batchCommon, method="global", excitations={"TM": (0.0, 1.0)})

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
        self.assertFalse(hasattr(simulation, "method"))
        self.assertEqual(simulation.backend, "cuda")

        solve_kwargs = dict(
            wavelength=1.2,
            theta=np.deg2rad(4),
            polarization="TM",
        )

        smatrix = simulation.solve(**solve_kwargs)
        self.assertTrue(smatrix.solvedBy.endswith("-cuda"))

        with self.assertRaises(TypeError):
            anisotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=[layer],
                orders=(1, 1),
                truncation="circular",
                method="global",
            )

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicPreparedStackKeepsCoreMatricesOnGpu(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                anisotropic.Layer(
                    thickness=0.05,
                    epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04),
                    name="homogeneous tensor film",
                )
            ],
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
        )

        result = simulation.solve(1.0, theta=np.deg2rad(4.0), polarization="TM")

        self.assertTrue(result.solvedBy.endswith("-cuda"))
        prepared = next(iter(simulation.preparedCache.values()))
        self.assertEqual(prepared.total.s11.device.type, "cuda")
        self.assertEqual(prepared.components[0].s11.device.type, "cuda")
        self.assertEqual(prepared.components[1].s21.device.type, "cuda")
        qValues, modeMatrix = prepared.layerModes[0]
        self.assertEqual(qValues.device.type, "cuda")
        self.assertEqual(modeMatrix.device.type, "cuda")

    def testAnisotropicProjectUsesStableCudaSMatrixOnly(self) -> None:
        project = anisotropic.Project(period=(1.0, 1.0), order=(0, 0), truncation="circular")
        project.add_uniform(0.05, anisotropic.xzTensor(2.2, 2.2, 2.2, 0.03, 0.03), name="tensor film")

        self.assertFalse(hasattr(project, "method"))
        self.assertFalse(hasattr(project.simulation(), "method"))
        self.assertEqual(project.simulation().backend, "cuda")
        result = project.solve(1.0, polarization="TM")
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

        with self.assertRaises(TypeError):
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

        result = solveAnisotropic(
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

        anisotropicResult = solveAnisotropic(
            layers=[anisotropic.Layer(thickness=0.30, epsilon=tensor)],
            **common,
        )
        effectiveScalarResult = solveIsotropic(
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
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1))
        common = dict(
            wavelength=1.0,
            period=(0.8, 0.8),
            orders=(1, 1),
            epsIncident=1.0,
            epsTransmission=1.0,
        )

        directResult = solveAnisotropic(layers=[layer], **common)
        compiledResult = solveAnisotropic(layers=compiled, **common)

        self.assertAlmostEqual(compiledResult.reflection, directResult.reflection, places=12)
        self.assertAlmostEqual(compiledResult.transmission, directResult.transmission, places=12)

    def testAnisotropicCompiledLayerRejectsMismatchedFourierTruncation(self) -> None:
        layer = anisotropic.Layer(thickness=0.2, epsilon=2.25)
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")

        with self.assertRaisesRegex(ValueError, "truncation"):
            solveAnisotropic(
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

        fast = solveAnisotropic(**common)
        full = solveAnisotropicFields(**common)

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

        fast = solveAnisotropic(**common)
        full = solveAnisotropicFields(**common)

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

        compiled = compileAnisotropicLayersForTest(
            [anisotropic.Layer(thickness=0.1, epsilon=epsilon, normalField=normalField)],
            orders=(1, 1),
        )
        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

        result = solveAnisotropic(
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
        compiled = compileAnisotropicLayersForTest(
            [anisotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="normal-vector")],
            orders=(1, 1),
            truncation="circular",
        )
        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

        result = solveAnisotropic(
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
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")
        self.assertEqual(compiled[0].tensorData.factorization, "z-li")

    def testAnisotropicAnalyticRectangleConvolutionMatchesSampledLimit(self) -> None:
        period = (1.0, 0.8)
        size = (0.34, 0.22)
        center = (0.11, -0.07)
        angle = np.deg2rad(17.0)
        background = 1.0
        post = 3.2

        harmonics = anisotropic_solver.makeHarmonics(
            wavelength=1.0,
            period=period,
            orders=(2, 2),
            epsIncident=1.0,
            theta=0.0,
            phi=0.0,
            truncation="rectangular",
        )
        analyticLayer = anisotropic.rectangularPostLayer(
            period=period,
            thickness=0.08,
            background=background,
            post=post,
            size=size,
            center=center,
            angle=angle,
            factorization="standard",
            analytic=True,
        )
        sampledLayer = anisotropic.rectangularPostLayer(
            period=period,
            thickness=0.08,
            background=background,
            post=post,
            size=size,
            center=center,
            angle=angle,
            shape=(420, 500),
            factorization="standard",
        )

        analyticMatrix = anisotropic_solver.tensorConvolutionData(
            analyticLayer.epsilon,
            harmonics,
            factorization="standard",
        ).components[0][0]
        sampledMatrix = anisotropic_solver.tensorConvolutionData(
            sampledLayer.epsilon,
            harmonics,
            factorization="standard",
        ).components[0][0]

        self.assertIsNone(anisotropic_solver.sampleShapeFromEpsilon(analyticLayer.epsilon))
        np.testing.assert_allclose(sampledMatrix, analyticMatrix, rtol=1.5e-2, atol=1.5e-2)

    def testAnisotropicAnalyticNormalVectorLiPathIsAvailableForScalarPosts(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.36, 0.28),
            factorization="auto",
            analytic=True,
            normalVectorResolution=48,
        )
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")
        self.assertEqual(compiled[0].tensorData.factorization, "analytic-normal-vector-li")
        self.assertIsNone(compiled[0].sampleShape)

        result = solveAnisotropic(
            layers=compiled,
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testAnisotropicAnalyticDiskAndEllipseLayersCompile(self) -> None:
        layers = [
            anisotropic.circularPostLayer(
                period=(1.0, 1.0),
                thickness=0.05,
                background=1.0,
                post=2.25,
                radius=0.18,
                analytic=True,
                normalVectorResolution=40,
            ),
            anisotropic.ellipticalPostLayer(
                period=(1.0, 0.9),
                thickness=0.05,
                background=1.0,
                post=2.4,
                radii=(0.21, 0.12),
                angle=np.deg2rad(23.0),
                analytic=True,
                normalVectorResolution=40,
            ),
        ]

        for layer in layers:
            with self.subTest(layer=layer.name):
                compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")
                self.assertEqual(compiled[0].tensorData.factorization, "analytic-normal-vector-li")
                self.assertIsNone(compiled[0].sampleShape)

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
        compiled = isotropicLayerMetadataForTest(layer, orders=(1, 1), truncation="circular")
        self.assertEqual(compiled.factorization, "normal-vector-li")

        common = dict(
            layers=[layer],
            wavelength=1.1,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            theta=np.deg2rad(5),
        )
        smatrix = solveIsotropic(**common, method="smatrix", sAmplitude=1.0, pAmplitude=0.0)
        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            isotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=[layer],
                orders=(1, 1),
                truncation="circular",
                method="etm",
            )

        batch = solveIsotropicBatch(
            **common,
            excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
        )
        sequentialTm = solveIsotropic(**common, sAmplitude=0.0, pAmplitude=1.0)
        self.assertAlmostEqual(batch["TE"].reflection, smatrix.reflection, places=12)
        self.assertAlmostEqual(batch["TM"].transmission, sequentialTm.transmission, places=12)
        self.assertEqual(batch["TE"].solvedBy, "smatrix-batch-cuda")

        with self.assertRaisesRegex(ValueError, "only method='smatrix'"):
            isotropic.RCWASimulation(
                period=(1.0, 1.0),
                layers=[layer],
                orders=(1, 1),
                truncation="circular",
                method="etm",
            )

    def testIsotropicAutoFactorizationGeneratesVectorFieldForPiecewiseConstantGrid(self) -> None:
        epsilon = np.ones((18, 18), dtype=complex)
        epsilon[4:14, 6:12] = 2.25
        layer = isotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="auto")
        compiled = isotropicLayerMetadataForTest(layer, orders=(1, 1), truncation="circular")
        self.assertEqual(compiled.factorization, "normal-vector-li")

        result = solveIsotropic(
            layers=[layer],
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
        layer = isotropic.Layer(thickness=0.08, epsilon=epsilon, factorization="standard")
        compiled = isotropicLayerMetadataForTest(layer, orders=(1, 1), truncation="circular")
        self.assertEqual(compiled.factorization, "standard")

    def testIsotropicAnalyticGeometryAndHomogeneousFastPathMetadata(self) -> None:
        disk = isotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            radius=0.2,
            analytic=True,
        )
        compiledDisk = isotropicLayerMetadataForTest(disk, orders=(2, 2), truncation="circular")
        self.assertEqual(compiledDisk.factorization, "analytic-normal-vector-li")
        expectedAverage = 1.0 + (2.25 - 1.0) * np.pi * 0.2**2
        self.assertAlmostEqual(compiledDisk.epsilonMatrix[0, 0].item(), expectedAverage, places=12)

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
        compiledShapes = [
            isotropicLayerMetadataForTest(layer, orders=(1, 1), truncation="circular")
            for layer in (rectangle, ellipse, annulus)
        ]
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
        compiledStandard = isotropicLayerMetadataForTest(standardRectangle, orders=(1, 1), truncation="circular")
        self.assertEqual(compiledStandard.factorization, "analytic-li")
        self.assertIsNone(compiledStandard.displacementMatrices)

        uniform = isotropicLayerMetadataForTest(isotropic.Layer(thickness=0.1, epsilon=np.full((8, 8), 2.25)), orders=1)
        self.assertAlmostEqual(uniform.homogeneousEpsilon, 2.25)

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

        reduced = solveIsotropic(**common)
        full = solveIsotropic(**common, returnFields=True)

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
        stripeResult = solveIsotropic(
            layers=[analyticStripe],
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
        compiled = isotropicLayerMetadataForTest(layer, orders=(1, 1), truncation="circular")
        expected = 1.0 + (3.0 - 1.0) * 0.5 * 0.4 + (1.0 - 3.0) * np.pi * 0.1**2
        self.assertEqual(compiled.factorization, "analytic-li")
        self.assertAlmostEqual(compiled.epsilonMatrix[0, 0].item(), expected, places=12)

        cross = isotropic.crossPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.0,
            armLengths=(0.7, 0.5),
            armWidths=(0.2, 0.3),
            analytic=True,
        )
        compiledCross = isotropicLayerMetadataForTest(cross, orders=(1, 1), truncation="circular")
        expectedCross = 1.0 + (2.0 - 1.0) * (0.7 * 0.2 + 0.3 * 0.5 - 0.3 * 0.2)
        self.assertAlmostEqual(compiledCross.epsilonMatrix[0, 0].item(), expectedCross, places=12)

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
        result = solveIsotropic(
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

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testIsotropicBatchedSpectrumMatchesPointwiseSpectrum(self) -> None:
        layer = isotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.25,
            radius=0.18,
            analytic=True,
            factorization="standard",
        )
        wavelengths = np.array([0.95, 1.0, 1.05])
        batched = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
            precompile=True,
            cacheModes=False,
            workers=1,
        ).spectrum(wavelengths, polarizations=("TE", "TM"), workers=1)
        pointwise = isotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
            precompile=True,
            cacheModes=False,
            workers=2,
        ).spectrum(wavelengths, polarizations=("TE", "TM"), workers=2)

        for label in ("TE", "TM"):
            np.testing.assert_allclose(batched[label]["reflection"], pointwise[label]["reflection"], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(
                batched[label]["transmission"],
                pointwise[label]["transmission"],
                rtol=1e-9,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                batched[label]["absorption"],
                pointwise[label]["absorption"],
                rtol=1e-9,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                batched[label]["energyError"],
                pointwise[label]["energyError"],
                rtol=1e-9,
                atol=1e-10,
            )

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
        self.assertEqual(len(simulation.preparedCache), 1)

        patterned.rectangle(size=(0.1, 0.1), material=1.5)
        updated = simulation.solve(1.0, polarization="TE")
        self.assertTrue(np.isfinite(updated.reflection))
        self.assertEqual(len(simulation.preparedCache), 2)

    def testAutoFactorizationUsesNormalVectorForSampledDisk(self) -> None:
        layer = anisotropic.circularPostLayer(
            period=(1.0, 1.0),
            thickness=0.1,
            background=1.0,
            post=2.25,
            radius=0.2,
            shape=(32, 32),
            factorization="auto",
        )
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")

        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

    def testSampledRectangleNeedsShapeAndSolvesFiniteResult(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.35, 0.25),
            shape=(24, 20),
            factorization="auto",
        )
        self.assertEqual(layer.epsilon.shape, (24, 20))

        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="rectangular")
        self.assertEqual(compiled[0].tensorData.factorization, "normal-vector-li")

        result = solveAnisotropic(
            layers=compiled,
            wavelength=1.05,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="rectangular",
            theta=np.deg2rad(4.0),
        )
        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))

    def testSampledRectangleSolvesWithCircularAndRectangularTruncation(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.30, 0.20),
            angle=np.deg2rad(15),
            shape=(24, 24),
        )

        for truncation in ("circular", "rectangular"):
            with self.subTest(truncation=truncation):
                result = solveAnisotropic(
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
        compiled = compileAnisotropicLayersForTest([layer], orders=(1, 1), truncation="circular")

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
            shape=(24, 24),
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

        batch = solveAnisotropicBatch(
            **common,
            excitations={"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
        )
        sequentialTE = solveAnisotropic(**common, sAmplitude=1.0, pAmplitude=0.0)
        sequentialTM = solveAnisotropic(**common, sAmplitude=0.0, pAmplitude=1.0)

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
        result = solveAnisotropic(
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
        self.assertEqual(result.layerEigTimings[1].matrixShape[-1], 2)
        self.assertEqual(result.layerEigTimings[1].eigTimeSeconds, 0.0)
        self.assertTrue(all(entry.eigTimeSeconds >= 0.0 for entry in result.layerEigTimings))
        self.assertTrue(all(entry.totalTimeSeconds >= entry.eigTimeSeconds for entry in result.layerEigTimings))
        self.assertTrue(result.layerEigTimings[0].factorizationTimeSeconds >= 0.0)
        self.assertTrue(result.layerEigTimings[0].matrixBuildTimeSeconds >= 0.0)
        self.assertIsNotNone(result.layerEigTimings[0].minAbsQ)
        self.assertIsNotNone(result.layerEigTimings[0].safeQThreshold)
        self.assertIsNotNone(result.stackTiming)
        assert result.stackTiming is not None
        self.assertTrue(result.stackTiming.interfaceTimeSeconds >= 0.0)
        self.assertTrue(result.stackTiming.cascadeTimeSeconds >= 0.0)
        self.assertTrue(result.stackTiming.totalPrepareTimeSeconds >= result.stackTiming.cascadeTimeSeconds)
        self.assertEqual(len(result.stackTiming.interfaceConditionNumbers), 3)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicOrderTenHomogeneousStackUsesFastCudaPaths(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                anisotropic.Layer(thickness=0.03, epsilon=2.25, name="scalar film"),
                anisotropic.Layer(
                    thickness=0.04,
                    epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.03, 0.03),
                    name="tensor film",
                ),
                anisotropic.Layer(thickness=0.02, epsilon=1.8, name="scalar cap"),
            ],
            orders=10,
            truncation="circular",
            backend="cuda",
            cacheModes=False,
        )

        result = simulation.solve(1.0, theta=np.deg2rad(4.0), phi=np.deg2rad(7.0), polarization="TM", profile=True)

        self.assertTrue(np.isfinite(result.reflection))
        self.assertTrue(np.isfinite(result.transmission))
        self.assertEqual(result.layerEigTimings[0].kind, "homogeneous-4x4")
        self.assertEqual(result.layerEigTimings[0].matrixShape[-1], 2)
        self.assertEqual(result.layerEigTimings[0].eigTimeSeconds, 0.0)
        self.assertEqual(result.layerEigTimings[2].eigTimeSeconds, 0.0)
        self.assertIsNotNone(result.stackTiming)
        assert result.stackTiming is not None
        self.assertEqual(len(result.stackTiming.interfaceConditionNumbers), 4)

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

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicBatchedSpectrumMatchesSequentialSolves(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.08,
            background=1.0,
            post=2.25,
            size=(0.35, 0.45),
            shape=(24, 24),
            factorization="standard",
        )
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            workers=2,
            cacheModes=False,
        )
        wavelengths = np.linspace(0.9, 1.1, 4)

        spectrum = simulation.spectrum(
            wavelengths,
            theta=np.deg2rad(6.0),
            phi=np.deg2rad(11.0),
            polarizations=("TE", "TM"),
            bidirectional=True,
            workers=2,
        )
        expectedForward = []
        expectedBackward = []
        for wavelength in wavelengths:
            forward = simulation.solve(wavelength, theta=np.deg2rad(6.0), phi=np.deg2rad(11.0), polarization="TE")
            backward = simulation.solve(wavelength, theta=-np.deg2rad(6.0), phi=np.deg2rad(11.0), polarization="TE")
            expectedForward.append(1.0 - forward.reflection - forward.transmission)
            expectedBackward.append(1.0 - backward.reflection - backward.transmission)

        np.testing.assert_allclose(spectrum["TE"]["absorptivity"], expectedForward, atol=1e-11)
        np.testing.assert_allclose(spectrum["TE"]["emissivity"], expectedBackward, atol=1e-11)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicBatchedSpectrumSupportsDynamicHomogeneousLayers(self) -> None:
        calls: list[float] = []
        post = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.32, 0.28),
            shape=(20, 20),
            factorization="standard",
        )

        def dispersiveTensor(wavelength: float) -> np.ndarray:
            calls.append(float(wavelength))
            return anisotropic.xzTensor(
                2.1 + 0.08 * wavelength,
                2.0 + 0.03 * wavelength,
                1.9 + 0.02 * wavelength,
                0.02 + 0.01j,
                0.02 - 0.01j,
            )

        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                post,
                anisotropic.homogeneousLayer(0.04, dispersiveTensor, name="dynamic homogeneous tensor"),
                anisotropic.homogeneousLayer(
                    0.03,
                    lambda wavelength: (1.7 + 0.05j * wavelength) * np.eye(3, dtype=complex),
                    name="dynamic isotropic tensor",
                ),
            ],
            orders=(1, 1),
            truncation="circular",
            workers=1,
            cacheModes=False,
        )
        wavelengths = np.array([0.92, 1.03, 1.14])
        spectrum = simulation.spectrum(
            wavelengths,
            theta=np.deg2rad(4.0),
            phi=np.deg2rad(9.0),
            polarizations=("TE", "TM"),
            bidirectional=True,
            workers=1,
        )
        spectrumCalls = calls.copy()
        expectedForward = []
        expectedBackward = []
        for wavelength in wavelengths:
            forward = simulation.solve(wavelength, theta=np.deg2rad(4.0), phi=np.deg2rad(9.0), polarization="TE")
            backward = simulation.solve(wavelength, theta=-np.deg2rad(4.0), phi=np.deg2rad(9.0), polarization="TE")
            expectedForward.append(1.0 - forward.reflection - forward.transmission)
            expectedBackward.append(1.0 - backward.reflection - backward.transmission)

        np.testing.assert_allclose(spectrum["TE"]["absorptivity"], expectedForward, atol=1e-10)
        np.testing.assert_allclose(spectrum["TE"]["emissivity"], expectedBackward, atol=1e-10)
        self.assertEqual(spectrumCalls, list(wavelengths))

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicPatternedHomogeneousInterfaceMatchesDenseSolve(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.07,
            background=1.0,
            post=anisotropic.xzTensor(2.2, 2.0, 2.3, 0.04, 0.04),
            size=(0.34, 0.26),
            shape=(24, 24),
            factorization="standard",
        )
        layers = [
            anisotropic.homogeneousLayer(0.03, anisotropic.xzTensor(1.8, 1.7, 1.9, 0.02, 0.02)),
            layer,
            anisotropic.homogeneousLayer(0.04, anisotropic.xzTensor(2.1, 2.0, 2.2, 0.03, 0.03)),
        ]
        common = dict(
            period=(1.0, 1.0),
            layers=layers,
            orders=(1, 1),
            truncation="circular",
            workers=1,
            cacheModes=False,
        )
        previous = os.environ.get("RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE")
        try:
            os.environ.pop("RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE", None)
            defaultFast = anisotropic.RCWASimulation(**common).solve(
                1.05,
                theta=np.deg2rad(5.0),
                phi=np.deg2rad(8.0),
                polarization="TM",
            )

            os.environ["RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE"] = "1"
            fast = anisotropic.RCWASimulation(**common).solve(
                1.05,
                theta=np.deg2rad(5.0),
                phi=np.deg2rad(8.0),
                polarization="TM",
            )
            fastSpectrum = anisotropic.RCWASimulation(**common).spectrum(
                np.array([0.96, 1.04, 1.12]),
                theta=np.deg2rad(5.0),
                phi=np.deg2rad(8.0),
                polarizations=("TE", "TM"),
                bidirectional=True,
                workers=1,
            )

            os.environ["RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE"] = "0"
            dense = anisotropic.RCWASimulation(**common).solve(
                1.05,
                theta=np.deg2rad(5.0),
                phi=np.deg2rad(8.0),
                polarization="TM",
            )
            denseSpectrum = anisotropic.RCWASimulation(**common).spectrum(
                np.array([0.96, 1.04, 1.12]),
                theta=np.deg2rad(5.0),
                phi=np.deg2rad(8.0),
                polarizations=("TE", "TM"),
                bidirectional=True,
                workers=1,
            )
        finally:
            if previous is None:
                os.environ.pop("RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE", None)
            else:
                os.environ["RCWA3D_PATTERNED_HOMOGENEOUS_INTERFACE"] = previous

        self.assertAlmostEqual(defaultFast.reflection, fast.reflection, places=12)
        self.assertAlmostEqual(defaultFast.transmission, fast.transmission, places=12)
        self.assertAlmostEqual(fast.reflection, dense.reflection, places=10)
        self.assertAlmostEqual(fast.transmission, dense.transmission, places=10)
        np.testing.assert_allclose(fast.rAmplitudes, dense.rAmplitudes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(fast.tAmplitudes, dense.tAmplitudes, rtol=1e-10, atol=1e-10)
        for label in ("TE", "TM"):
            np.testing.assert_allclose(
                fastSpectrum[label]["absorptivity"],
                denseSpectrum[label]["absorptivity"],
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                fastSpectrum[label]["emissivity"],
                denseSpectrum[label]["emissivity"],
                rtol=1e-10,
                atol=1e-10,
            )

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicComplex64SpectrumTracksComplex128(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.06,
            background=1.0,
            post=2.25,
            size=(0.30, 0.25),
            shape=(20, 20),
            factorization="standard",
        )
        common = dict(
            period=(1.0, 1.0),
            layers=[
                layer,
                anisotropic.homogeneousLayer(
                    0.04,
                    lambda wavelength: anisotropic.xzTensor(2.1 + 0.05 * wavelength, 2.0, 1.9, 0.02, 0.02),
                ),
            ],
            orders=(1, 1),
            truncation="circular",
            workers=1,
            cacheModes=False,
        )
        wavelengths = np.array([0.95, 1.05, 1.15])
        reference = anisotropic.RCWASimulation(**common, precision="complex128").spectrum(
            wavelengths,
            theta=np.deg2rad(5.0),
            phi=np.deg2rad(7.0),
            polarizations=("TE", "TM"),
            bidirectional=True,
            workers=1,
        )
        fast = anisotropic.RCWASimulation(**common, precision="complex64").spectrum(
            wavelengths,
            theta=np.deg2rad(5.0),
            phi=np.deg2rad(7.0),
            polarizations=("TE", "TM"),
            bidirectional=True,
            workers=1,
        )

        np.testing.assert_allclose(fast["TE"]["absorptivity"], reference["TE"]["absorptivity"], rtol=2e-4, atol=3e-4)
        np.testing.assert_allclose(fast["TM"]["emissivity"], reference["TM"]["emissivity"], rtol=2e-4, atol=3e-4)

        mixed = anisotropic.RCWASimulation(**common, precision="mixed").spectrum(
            wavelengths,
            theta=np.deg2rad(5.0),
            phi=np.deg2rad(7.0),
            polarizations=("TE",),
            bidirectional=False,
            workers=1,
        )
        np.testing.assert_allclose(mixed["TE"]["absorptivity"], fast["TE"]["absorptivity"], rtol=0.0, atol=1e-7)

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicMixedSpectrumPropagatesBatchFailure(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.2,
            size=(0.30, 0.25),
            shape=(18, 18),
            factorization="standard",
        )
        common = dict(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            workers=1,
            cacheModes=False,
        )
        wavelengths = np.array([0.96, 1.04])

        original = anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers

        def failMixedBatch(self, *args, **kwargs):
            if self.precision == "mixed":
                raise RuntimeError("forced mixed batch failure")
            return original(self, *args, **kwargs)

        try:
            anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers = failMixedBatch
            with self.assertRaisesRegex(RuntimeError, "forced mixed batch failure"):
                anisotropic.RCWASimulation(**common, precision="mixed").spectrum(
                    wavelengths,
                    theta=np.deg2rad(5.0),
                    phi=np.deg2rad(7.0),
                    polarizations=("TE",),
                    bidirectional=False,
                    workers=1,
                )
        finally:
            anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers = original

    @unittest.skipUnless(torchCudaAvailable(), "requires torch CUDA")
    def testAnisotropicBatchedSpectrumRejectsNonFinitePowers(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=2.2,
            size=(0.30, 0.25),
            shape=(18, 18),
            factorization="standard",
        )
        common = dict(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            workers=1,
            cacheModes=False,
        )
        wavelengths = np.array([0.96, 1.04])

        original = anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers

        def nonFiniteBatch(self, wavelengths, *args, **kwargs):
            values = np.asarray(wavelengths, dtype=float)
            return {label: (np.full(values.shape, np.nan), np.zeros(values.shape)) for label in kwargs["excitations"]}

        try:
            anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers = nonFiniteBatch
            with self.assertRaisesRegex(FloatingPointError, "non-finite powers"):
                anisotropic.RCWASimulation(**common, precision="mixed").spectrum(
                    wavelengths,
                    theta=np.deg2rad(5.0),
                    phi=np.deg2rad(7.0),
                    polarizations=("TE",),
                    bidirectional=False,
                    workers=1,
                )
        finally:
            anisotropic_simulation.RCWASimulation.prepareSpectrumBatchPowers = original

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
        self.assertEqual(anisotropic_simulation.spectrumParallelPlan(4, "auto", "cuda"), ("serial", 1))
        self.assertEqual(anisotropic_simulation.spectrumParallelPlan(4, "thread", "cuda"), ("thread", 4))
        with self.assertRaisesRegex(ValueError, "CUDA-only"):
            anisotropic_simulation.spectrumParallelPlan(4, "process", "cuda")

    def testAnisotropicPreparedCacheReusesPreparedStack(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[anisotropic.Layer(thickness=0.05, epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04))],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        first = simulation.preparedStack(1.0, theta=np.deg2rad(5), phi=0.0)
        second = simulation.preparedStack(1.0, theta=np.deg2rad(5), phi=0.0)
        third = simulation.preparedStack(1.0, theta=np.deg2rad(6), phi=0.0)

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(simulation.preparedCache), 2)

    def testAnisotropicSimulationSolveAndBatchUsePreparedCache(self) -> None:
        layer = anisotropic.rectangularPostLayer(
            period=(1.0, 1.0),
            thickness=0.05,
            background=1.0,
            post=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04),
            size=(0.28, 0.20),
            shape=(12, 12),
            factorization="standard",
        )
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layer],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        te = simulation.solve(1.0, theta=np.deg2rad(4.0), polarization="TE")
        tm = simulation.solve(1.0, theta=np.deg2rad(4.0), polarization="TM")
        self.assertEqual(len(simulation.preparedCache), 1)

        batch = simulation.solveExcitations(
            1.0,
            {"TE": (1.0, 0.0), "TM": (0.0, 1.0)},
            theta=np.deg2rad(4.0),
        )
        self.assertEqual(len(simulation.preparedCache), 1)
        self.assertAlmostEqual(batch["TE"].reflection, te.reflection, places=12)
        self.assertAlmostEqual(batch["TM"].transmission, tm.transmission, places=12)

    def testAnisotropicSimulationEmptyBatchSkipsPreparedStack(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[anisotropic.Layer(thickness=0.05, epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04))],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        self.assertEqual(simulation.solveExcitations(1.0, {}), {})
        self.assertEqual(simulation.solveBatchPowers(1.0, theta=0.0, phi=0.0, excitations={}), {})
        self.assertEqual(len(simulation.preparedCache), 0)

    def testAnisotropicPrepareReusesRepeatedLayerModesWithinStack(self) -> None:
        layer = anisotropic.Layer(
            thickness=0.05,
            epsilon=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04),
            name="repeated tensor",
        )
        prepared = anisotropic_solver.prepareStackSMatrix(
            layers=[layer, layer],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
            fullTotal=False,
        )

        self.assertIs(prepared.layerModes[0][0], prepared.layerModes[1][0])
        self.assertIs(prepared.layerModes[0][1], prepared.layerModes[1][1])
        self.assertTrue(np.isfinite(anisotropic_solver.evaluatePreparedStack(prepared).reflection))

    def testAnisotropicPrepareReusesCompiledLayerModesWithSharedTensorData(self) -> None:
        grid = np.ones((12, 12), dtype=complex)
        grid[3:9, 4:8] = 2.25
        compiled = compileAnisotropicLayersForTest(
            [anisotropic.Layer(thickness=0.05, epsilon=grid, factorization="standard")],
            orders=(1, 1),
            truncation="circular",
        )[0]
        repeated = anisotropic_solver.CompiledLayer(
            thickness=0.08,
            tensorData=compiled.tensorData,
            orders=compiled.orders,
            truncation=compiled.truncation,
            factorization=compiled.factorization,
            name="same pattern different thickness",
        )

        prepared = anisotropic_solver.prepareStackSMatrix(
            layers=[compiled, repeated],
            wavelength=1.0,
            period=(1.0, 1.0),
            orders=(1, 1),
            truncation="circular",
            backend="cuda",
            fullTotal=False,
        )

        self.assertIs(prepared.layerModes[0][0], prepared.layerModes[1][0])
        self.assertIs(prepared.layerModes[0][1], prepared.layerModes[1][1])

    def testAnisotropicDynamicHomogeneousLayersHaveStablePreparedCacheKeys(self) -> None:
        calls: list[float] = []

        def layerAt(wavelength: float) -> anisotropic.Layer:
            calls.append(float(wavelength))
            return anisotropic.Layer(
                thickness=0.05,
                epsilon=anisotropic.xzTensor(2.2 + 0.1 * wavelength, 2.1, 2.3, 0.04, 0.04),
            )

        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[layerAt],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        first = simulation.preparedStack(1.0, theta=0.0, phi=0.0)
        second = simulation.preparedStack(1.0, theta=0.0, phi=0.0)
        third = simulation.preparedStack(1.1, theta=0.0, phi=0.0)

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(calls, [1.0, 1.0, 1.1])
        self.assertEqual(len(simulation.preparedCache), 2)

    def testAnisotropicProfileSolveDoesNotPopulatePreparedCache(self) -> None:
        simulation = anisotropic.RCWASimulation(
            period=(1.0, 1.0),
            layers=[
                anisotropic.rectangularPostLayer(
                    period=(1.0, 1.0),
                    thickness=0.05,
                    background=1.0,
                    post=anisotropic.xzTensor(2.2, 2.1, 2.3, 0.04, 0.04),
                    size=(0.28, 0.20),
                    shape=(12, 12),
                    factorization="standard",
                )
            ],
            orders=(1, 1),
            truncation="circular",
            cacheModes=True,
        )

        profiled = simulation.solve(1.0, theta=np.deg2rad(4.0), polarization="TE", profile=True)
        self.assertEqual(len(profiled.layerEigTimings), 1)
        self.assertEqual(len(simulation.preparedCache), 0)

        simulation.solve(1.0, theta=np.deg2rad(4.0), polarization="TE")
        self.assertEqual(len(simulation.preparedCache), 1)

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
