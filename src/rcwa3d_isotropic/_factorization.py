from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fourier import Harmonics, epsilonConvolutionMatrixTorch
from .types import CompiledLayer, Layer


@dataclass(frozen=True)
class TorchLayerData:
    epsilonMatrix: Any
    epsilonInverse: Any | None
    displacementMatrices: tuple[Any, Any, Any, Any] | None
    homogeneousEpsilon: complex | None
    factorization: str = "standard"


def layerDataForTorch(layer: Layer | CompiledLayer, harmonics: Harmonics, torch: Any, device: Any) -> TorchLayerData:
    if hasattr(layer, "epsilonMatrix") and hasattr(layer, "epsilonInverse"):
        factorized = _compiledLayerDataForTorch(layer, harmonics, torch, device)
        return TorchLayerData(
            factorized.epsilonMatrix,
            factorized.epsilonInverse,
            factorized.displacementMatrices,
            factorized.homogeneousEpsilon,
            factorized.factorization,
        )

    epsilon = getattr(layer, "epsilon")
    factorizationMode = _normalizeFactorization(getattr(layer, "factorization", "auto"))
    normalField = getattr(layer, "normalField", None)
    needsFullFactorization = (
        _useAnalyticNormalVector(epsilon, factorizationMode)
        or factorizationMode in ("normal-vector", "jones")
        or (factorizationMode == "auto" and normalField is not None)
        or _shouldAutoGenerateNormalField(epsilon, factorizationMode)
    )
    if needsFullFactorization:
        factorized = _rawLayerDataForTorch(layer, harmonics, torch, device)
        return TorchLayerData(
            factorized.epsilonMatrix,
            factorized.epsilonInverse,
            factorized.displacementMatrices,
            factorized.homogeneousEpsilon,
            factorized.factorization,
        )

    epsilonMatrix = epsilonConvolutionMatrixTorch(epsilon, harmonics, torch, device)
    return TorchLayerData(
        epsilonMatrix=epsilonMatrix,
        epsilonInverse=None,
        displacementMatrices=None,
        homogeneousEpsilon=homogeneousEpsilon(epsilon),
        factorization=_analyticFactorizationName(epsilon),
    )


def pqMatricesTorch(
    epsilonMatrix: Any,
    harmonics: Harmonics,
    epsilonInverse: Any,
    displacementMatrices: tuple[Any, Any, Any, Any] | None,
    torch: Any,
    device: Any,
) -> tuple[Any, Any]:
    n = harmonics.count
    identity = torch.eye(n, dtype=torch.complex128, device=device)
    kx = toTorchComplex(harmonics.kx, torch, device)
    ky = toTorchComplex(harmonics.ky, torch, device)

    p11 = kx[:, None] * epsilonInverse * ky[None, :]
    p12 = identity - kx[:, None] * epsilonInverse * kx[None, :]
    p21 = ky[:, None] * epsilonInverse * ky[None, :] - identity
    p22 = -ky[:, None] * epsilonInverse * kx[None, :]

    if displacementMatrices is None:
        cxx = epsilonMatrix
        cxy = torch.zeros_like(epsilonMatrix)
        cyx = torch.zeros_like(epsilonMatrix)
        cyy = epsilonMatrix
    else:
        cxx, cxy, cyx, cyy = displacementMatrices

    q11 = -torch.diag(kx * ky) - cyx
    q12 = torch.diag(kx * kx) - cyy
    q21 = cxx - torch.diag(ky * ky)
    q22 = cxy + torch.diag(ky * kx)
    pMatrix = torch.cat([torch.cat([p11, p12], dim=1), torch.cat([p21, p22], dim=1)], dim=0)
    qMatrix = torch.cat([torch.cat([q11, q12], dim=1), torch.cat([q21, q22], dim=1)], dim=0)
    return pMatrix, qMatrix


def solveIdentityTorch(matrix: Any, torch: Any, device: Any) -> Any:
    return torch.linalg.solve(matrix, torch.eye(matrix.shape[0], dtype=torch.complex128, device=device))


def toTorchComplex(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex128)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex128, device=device)


def homogeneousEpsilon(epsilon: object) -> complex | None:
    if hasattr(epsilon, "convolutionMatrix"):
        return None
    if np.isscalar(epsilon):
        return complex(epsilon)
    array = np.asarray(epsilon, dtype=complex)
    if array.ndim == 0:
        return complex(array.item())
    if array.ndim == 2 and array.size > 0 and np.allclose(array, array.flat[0], rtol=0.0, atol=1e-14):
        return complex(array.flat[0])
    return None


def _compiledLayerDataForTorch(
    layer: Layer | CompiledLayer,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
) -> TorchLayerData:
    expectedOrders = harmonics.orders
    if getattr(layer, "orders") != expectedOrders:
        raise ValueError(f"compiled layer orders {getattr(layer, 'orders')} do not match requested orders {expectedOrders}")
    layerTruncation = getattr(layer, "truncation", "rectangular")
    if layerTruncation != harmonics.truncation:
        raise ValueError(
            f"compiled layer truncation {layerTruncation!r} does not match requested truncation "
            f"{harmonics.truncation!r}"
        )
    displacement = getattr(layer, "displacementMatrices", None)
    return TorchLayerData(
        toTorchComplex(getattr(layer, "epsilonMatrix"), torch, device),
        toTorchComplex(getattr(layer, "epsilonInverse"), torch, device),
        None if displacement is None else tuple(toTorchComplex(matrix, torch, device) for matrix in displacement),
        getattr(layer, "homogeneousEpsilon", None),
        getattr(layer, "factorization", "standard"),
    )


def _rawLayerDataForTorch(layer: Layer, harmonics: Harmonics, torch: Any, device: Any) -> TorchLayerData:
    epsilon = getattr(layer, "epsilon")
    epsilonMatrix = epsilonConvolutionMatrixTorch(epsilon, harmonics, torch, device)
    epsilonInverse = solveIdentityTorch(epsilonMatrix, torch, device)
    factorizationMode = _normalizeFactorization(getattr(layer, "factorization", "auto"))
    normalField = getattr(layer, "normalField", None)
    displacementMatrices: tuple[Any, Any, Any, Any] | None = None
    factorization = _analyticFactorizationName(epsilon)

    if normalField is None and _shouldAutoGenerateNormalField(epsilon, factorizationMode):
        normalField = _estimateNormalFieldTorch(epsilon, torch, device)

    if _useAnalyticNormalVector(epsilon, factorizationMode):
        displacementMatrices = _analyticNormalVectorDisplacementMatricesTorch(epsilon, harmonics, torch, device)
        factorization = "analytic-normal-vector-li"
    elif normalField is not None and factorizationMode in ("auto", "normal-vector", "jones"):
        displacementMatrices = _normalVectorDisplacementMatricesTorch(epsilon, normalField, harmonics, torch, device)
        factorization = "normal-vector-li"
    elif factorizationMode in ("normal-vector", "jones"):
        raise ValueError("normal-vector factorization requires a normalField or an analytic shape with normal vectors")

    return TorchLayerData(
        epsilonMatrix,
        epsilonInverse,
        displacementMatrices,
        homogeneousEpsilon(epsilon),
        factorization,
    )


def _normalizeFactorization(value: str) -> str:
    normalized = str(value).lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "standard": "standard",
        "none": "standard",
        "direct": "standard",
        "normal-vector": "normal-vector",
        "normal-vector-li": "normal-vector",
        "nv": "normal-vector",
        "jones": "jones",
        "jones-li": "jones",
    }
    if normalized not in aliases:
        raise ValueError("factorization must be 'auto', 'standard', 'normal-vector', or 'jones'")
    return aliases[normalized]


def _analyticFactorizationName(epsilon: object) -> str:
    if hasattr(epsilon, "convolutionMatrix"):
        return "analytic-li"
    return "standard"


def _useAnalyticNormalVector(epsilon: object, factorizationMode: str) -> bool:
    return bool(
        hasattr(epsilon, "normalVectorMatrices")
        and hasattr(epsilon, "reciprocalConvolutionMatrix")
        and (
            factorizationMode == "jones"
            or (factorizationMode == "auto" and getattr(epsilon, "factorization", "analytic") == "jones")
        )
    )


def _shouldAutoGenerateNormalField(epsilon: object, factorizationMode: str) -> bool:
    if factorizationMode not in ("auto", "normal-vector", "jones"):
        return False
    if hasattr(epsilon, "convolutionMatrix") or np.isscalar(epsilon):
        return False
    array = np.asarray(epsilon)
    if array.ndim != 2 or array.shape == (3, 3):
        return False
    if factorizationMode == "auto":
        return _looksPiecewiseConstant(array)
    return True


def _looksPiecewiseConstant(values: object) -> bool:
    grid = np.asarray(values)
    if grid.ndim != 2 or grid.size == 0:
        return False
    if not np.all(np.isfinite(grid)):
        return False

    scale = max(1.0, float(np.max(np.abs(grid))))
    tolerance = 1e-10 * scale
    rounded = np.round(grid.real / tolerance) + 1j * np.round(grid.imag / tolerance)
    uniqueCount = np.unique(rounded).size
    return 1 < uniqueCount <= max(16, grid.size // 4)


def _estimateNormalFieldTorch(values: object, torch: Any, device: Any) -> Any:
    grid = toTorchComplex(values, torch, device)
    if grid.ndim != 2:
        raise ValueError("normal-field estimation requires a 2D scalar grid")
    if grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("sampled epsilon grid must be non-empty")

    contrast = torch.abs(grid - torch.mean(grid))
    dx = 0.5 * (torch.roll(contrast, -1, dims=1) - torch.roll(contrast, 1, dims=1))
    dy = 0.5 * (torch.roll(contrast, -1, dims=0) - torch.roll(contrast, 1, dims=0))
    length = torch.sqrt(dx * dx + dy * dy)

    active = length > 1e-12 * max(1.0, float(torch.max(length).item()) if length.numel() else 1.0)
    normals = torch.zeros(grid.shape + (2,), dtype=torch.float64, device=device)
    safeLength = torch.where(active, length, torch.ones_like(length))
    normals[..., 0] = torch.where(active, dx / safeLength, torch.ones_like(length))
    normals[..., 1] = torch.where(active, dy / safeLength, torch.zeros_like(length))
    return normals


def _analyticNormalVectorDisplacementMatricesTorch(
    epsilon: object,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    direct = epsilonConvolutionMatrixTorch(epsilon, harmonics, torch, device)
    if hasattr(epsilon, "reciprocalConvolutionMatrixTorch"):
        reciprocal = epsilon.reciprocalConvolutionMatrixTorch(harmonics, torch, device)
    else:
        reciprocal = toTorchComplex(epsilon.reciprocalConvolutionMatrix(harmonics), torch, device)
    inverseRule = solveIdentityTorch(reciprocal, torch, device)
    if hasattr(epsilon, "normalVectorMatricesTorch"):
        nx, ny, tx, ty = epsilon.normalVectorMatricesTorch(harmonics, torch, device)
    else:
        nx, ny, tx, ty = tuple(toTorchComplex(matrix, torch, device) for matrix in epsilon.normalVectorMatrices(harmonics))
    return _normalVectorBlocks(direct, inverseRule, nx, ny, tx, ty)


def _normalVectorDisplacementMatricesTorch(
    epsilon: object,
    normalField: object,
    harmonics: Harmonics,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    grid = toTorchComplex(epsilon, torch, device)
    normals = _toTorchReal(normalField, torch, device)
    if grid.ndim != 2:
        raise ValueError("normal-vector factorization requires a 2D scalar epsilon grid")
    if tuple(normals.shape) != tuple(grid.shape) + (2,):
        raise ValueError("normalField must have shape (ny, nx, 2) matching epsilon")

    normalX = normals[..., 0]
    normalY = normals[..., 1]
    length = torch.sqrt(normalX * normalX + normalY * normalY)
    safe = length > 1e-12
    normalX = torch.where(safe, normalX / torch.where(safe, length, torch.ones_like(length)), torch.ones_like(length))
    normalY = torch.where(safe, normalY / torch.where(safe, length, torch.ones_like(length)), torch.zeros_like(length))
    tangentX = -normalY
    tangentY = normalX

    direct = epsilonConvolutionMatrixTorch(grid, harmonics, torch, device)
    inverseRule = solveIdentityTorch(epsilonConvolutionMatrixTorch(1.0 / grid, harmonics, torch, device), torch, device)
    nx = epsilonConvolutionMatrixTorch(normalX, harmonics, torch, device)
    ny = epsilonConvolutionMatrixTorch(normalY, harmonics, torch, device)
    tx = epsilonConvolutionMatrixTorch(tangentX, harmonics, torch, device)
    ty = epsilonConvolutionMatrixTorch(tangentY, harmonics, torch, device)
    return _normalVectorBlocks(direct, inverseRule, nx, ny, tx, ty)


def _normalVectorBlocks(
    direct: Any,
    inverseRule: Any,
    nx: Any,
    ny: Any,
    tx: Any,
    ty: Any,
) -> tuple[Any, Any, Any, Any]:
    cxx = nx @ inverseRule @ nx + tx @ direct @ tx
    cxy = nx @ inverseRule @ ny + tx @ direct @ ty
    cyx = ny @ inverseRule @ nx + ty @ direct @ tx
    cyy = ny @ inverseRule @ ny + ty @ direct @ ty
    return cxx, cxy, cyx, cyy


def _toTorchReal(value: object, torch: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(np.asarray(value), dtype=torch.float64, device=device)
