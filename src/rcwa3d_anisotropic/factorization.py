from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .fourier import Harmonics, epsilonConvolutionMatrix, epsilonConvolutionMatrices


ComplexArray = np.ndarray
TensorLike = object

AXIS = {"x": 0, "y": 1, "z": 2}
COMPONENT_NAMES = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yy": (1, 1),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
    "zz": (2, 2),
}


@dataclass(frozen=True)
class TensorConvolutionData:
    """Fourier convolution matrices for a relative-permittivity tensor.

    ``components[i][j]`` is the convolution matrix for epsilon_ij.  ``etaZz``
    is the Li inverse-rule matrix ``[epsilon_zz]^-1`` used to eliminate Ez via
    the continuous normal displacement Dz.
    """

    components: tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]
    etaZz: ComplexArray
    constantTensor: ComplexArray | None = None
    factorization: str = "z-li"


def layerTensorData(layer: object, harmonics: Harmonics) -> TensorConvolutionData:
    """Return tensor convolution data, validating precompiled layers."""

    if hasattr(layer, "tensorData"):
        expectedOrders = harmonics.orders
        if getattr(layer, "orders") != expectedOrders:
            raise ValueError(
                f"compiled layer orders {getattr(layer, 'orders')} do not match requested orders {expectedOrders}"
            )
        layerTruncation = getattr(layer, "truncation", "rectangular")
        if layerTruncation != harmonics.truncation:
            raise ValueError(
                f"compiled layer truncation {layerTruncation!r} does not match requested truncation "
                f"{harmonics.truncation!r}"
            )
        tensorData = getattr(layer, "tensorData")
        if not isinstance(tensorData, TensorConvolutionData):
            raise TypeError("compiled anisotropic layer contains invalid tensorData")
        return tensorData

    return tensorConvolutionData(
        getattr(layer, "epsilon"),
        harmonics,
        normalField=getattr(layer, "normalField", None),
        factorization=getattr(layer, "factorization", "auto"),
    )


def tensorConvolutionData(
    epsilon: TensorLike,
    harmonics: Harmonics,
    normalField: ArrayLike | None = None,
    factorization: str = "auto",
) -> TensorConvolutionData:
    """Build convolution matrices and the z-normal Li inverse factorization.

    Accepted epsilon layouts:

    - scalar or 2D ``(ny, nx)`` grid: isotropic material
    - constant ``(3, 3)`` tensor
    - sampled ``(ny, nx, 3, 3)`` tensor field
    - mapping with keys such as ``"xx"``, ``"xz"``, ``"zx"``. Missing
      off-diagonal terms are zero; diagonal terms must be supplied.
    """

    factorizationMode = normalizeFactorization(factorization)
    constantTensorValue = constantTensor(epsilon)
    if useAnalyticNormalVector(epsilon, factorizationMode):
        components = analyticNormalVectorScalarTensorMatrices(epsilon, harmonics)
        factorization = "analytic-normal-vector-li"
    elif isScalarGrid(epsilon):
        if normalField is None and shouldAutoGenerateNormalField(epsilon, factorizationMode):
            normalField = estimateNormalField(epsilon, harmonics.orders, harmonics.truncation)
        if normalField is not None:
            components = normalVectorScalarTensorMatrices(epsilon, normalField, harmonics)
            factorization = "normal-vector-li"
        elif factorizationMode == "normal-vector":
            raise ValueError("vector factorization requires a scalar epsilon grid with material contrast")
        else:
            components = tensorComponentMatrices(epsilon, harmonics)
            factorization = "z-li"
    elif normalField is not None:
        raise ValueError("normal-vector factorization requires a scalar 2D epsilon grid")
    elif factorizationMode == "normal-vector" and hasattr(epsilon, "convolutionMatrix"):
        raise ValueError("normal-vector factorization requires a normalField or an analytic shape with normal vectors")
    else:
        components = tensorComponentMatrices(epsilon, harmonics)
        factorization = "z-li"
    try:
        etaZz = solveIdentity(components[2][2])
    except np.linalg.LinAlgError as exc:
        raise ValueError("epsilon_zz convolution matrix is singular; Li inverse factorization failed") from exc
    return TensorConvolutionData(
        components=components,
        etaZz=etaZz,
        constantTensor=constantTensorValue if factorization == "z-li" else None,
        factorization=factorization,
    )


def tensorComponentMatrices(
    epsilon: TensorLike,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    if isinstance(epsilon, Mapping):
        return mappingTensorMatrices(epsilon, harmonics)

    if hasattr(epsilon, "convolutionMatrix"):
        scalar = convolution(epsilon, harmonics)
        zeroMatrix = zero(harmonics.count)
        return (
            (scalar, zeroMatrix, zeroMatrix),
            (zeroMatrix, scalar.copy(), zeroMatrix.copy()),
            (zeroMatrix.copy(), zeroMatrix.copy(), scalar.copy()),
        )

    if np.isscalar(epsilon):
        scalar = convolution(epsilon, harmonics)
        zeroMatrix = zero(harmonics.count)
        return (
            (scalar, zeroMatrix, zeroMatrix),
            (zeroMatrix, scalar.copy(), zeroMatrix.copy()),
            (zeroMatrix.copy(), zeroMatrix.copy(), scalar.copy()),
        )

    array = np.asarray(epsilon, dtype=complex)
    if array.ndim == 0:
        scalar = convolution(array.item(), harmonics)
        zeroMatrix = zero(harmonics.count)
        return (
            (scalar, zeroMatrix, zeroMatrix),
            (zeroMatrix, scalar.copy(), zeroMatrix.copy()),
            (zeroMatrix.copy(), zeroMatrix.copy(), scalar.copy()),
        )

    if array.ndim == 2 and array.shape == (3, 3):
        return constantTensorMatrices(array, harmonics)

    if array.ndim == 2:
        scalar = convolution(array, harmonics)
        zeroMatrix = zero(harmonics.count)
        return (
            (scalar, zeroMatrix, zeroMatrix),
            (zeroMatrix, scalar.copy(), zeroMatrix.copy()),
            (zeroMatrix.copy(), zeroMatrix.copy(), scalar.copy()),
        )

    if array.ndim == 4 and array.shape[-2:] == (3, 3):
        matrices = epsilonConvolutionMatrices(
            tuple(array[..., row, col] for row in range(3) for col in range(3)),
            harmonics,
        )
        return tuple(tuple(matrices[3 * row + col] for col in range(3)) for row in range(3))

    if array.ndim == 4 and array.shape[:2] == (3, 3):
        matrices = epsilonConvolutionMatrices(
            tuple(array[row, col, ...] for row in range(3) for col in range(3)),
            harmonics,
        )
        return tuple(tuple(matrices[3 * row + col] for col in range(3)) for row in range(3))

    raise ValueError(
        "anisotropic epsilon must be a scalar, a 2D scalar grid, a (3, 3) tensor, "
        "a (ny, nx, 3, 3) tensor grid, or a component mapping"
    )


def normalVectorScalarTensorMatrices(
    epsilon: TensorLike,
    normalField: ArrayLike,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    grid = np.asarray(epsilon, dtype=complex)
    normals = np.asarray(normalField, dtype=float)
    if grid.ndim != 2:
        raise ValueError("normal-vector factorization requires a 2D scalar epsilon grid")
    if normals.shape != grid.shape + (2,):
        raise ValueError("normalField must have shape (ny, nx, 2) matching the scalar epsilon grid")

    normalX = normals[..., 0]
    normalY = normals[..., 1]
    length = np.sqrt(normalX**2 + normalY**2)
    safe = length > 1e-12
    normalX = np.where(safe, normalX / np.where(safe, length, 1.0), 1.0)
    normalY = np.where(safe, normalY / np.where(safe, length, 1.0), 0.0)
    tangentX = -normalY
    tangentY = normalX

    nx, ny, tx, ty, direct, reciprocal = epsilonConvolutionMatrices(
        (normalX, normalY, tangentX, tangentY, grid, 1.0 / grid),
        harmonics,
    )
    inverseRule = solveIdentity(reciprocal)
    zeroMatrix = zero(harmonics.count)

    normalMatrices = np.stack((nx, ny), axis=0)
    tangentMatrices = np.stack((tx, ty), axis=0)
    normalBlocks = normalMatrices[:, None, :, :] @ inverseRule @ normalMatrices[None, :, :, :]
    tangentBlocks = tangentMatrices[:, None, :, :] @ direct @ tangentMatrices[None, :, :, :]
    displacementBlocks = normalBlocks + tangentBlocks
    cxx = displacementBlocks[0, 0]
    cxy = displacementBlocks[0, 1]
    cyx = displacementBlocks[1, 0]
    cyy = displacementBlocks[1, 1]
    return ((cxx, cxy, zeroMatrix.copy()), (cyx, cyy, zeroMatrix.copy()), (zeroMatrix.copy(), zeroMatrix.copy(), direct))


def analyticNormalVectorScalarTensorMatrices(
    epsilon: TensorLike,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    direct = convolution(epsilon, harmonics)
    reciprocal = np.asarray(epsilon.reciprocalConvolutionMatrix(harmonics), dtype=complex)
    inverseRule = solveIdentity(reciprocal)
    nx, ny, tx, ty = tuple(np.asarray(matrix, dtype=complex) for matrix in epsilon.normalVectorMatrices(harmonics))
    zeroMatrix = zero(harmonics.count)

    cxx = nx @ inverseRule @ nx + tx @ direct @ tx
    cxy = nx @ inverseRule @ ny + tx @ direct @ ty
    cyx = ny @ inverseRule @ nx + ty @ direct @ tx
    cyy = ny @ inverseRule @ ny + ty @ direct @ ty
    return ((cxx, cxy, zeroMatrix.copy()), (cyx, cyy, zeroMatrix.copy()), (zeroMatrix.copy(), zeroMatrix.copy(), direct))


def mappingTensorMatrices(
    epsilon: Mapping[object, ArrayLike | complex],
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    n = harmonics.count
    components: list[list[ComplexArray | None]] = [[None for ignored in range(3)] for ignored in range(3)]
    for key, value in epsilon.items():
        row, col = componentIndex(key)
        components[row][col] = convolution(value, harmonics)

    for index, name in enumerate(("xx", "yy", "zz")):
        if components[index][index] is None:
            raise ValueError(f"component mapping epsilon is missing required diagonal component '{name}'")

    zeroMatrix = zero(n)
    return tuple(
        tuple(components[row][col] if components[row][col] is not None else zeroMatrix.copy() for col in range(3))
        for row in range(3)
    )


def componentIndex(key: object) -> tuple[int, int]:
    if isinstance(key, str):
        normalized = key.lower().replace("epsilon", "").replace("eps", "").replace("_", "").replace("-", "")
        if normalized in COMPONENT_NAMES:
            return COMPONENT_NAMES[normalized]
        if len(normalized) == 2 and normalized[0] in AXIS and normalized[1] in AXIS:
            return AXIS[normalized[0]], AXIS[normalized[1]]
    if isinstance(key, tuple) and len(key) == 2:
        row, col = key
        if isinstance(row, str):
            row = AXIS[row.lower()]
        if isinstance(col, str):
            col = AXIS[col.lower()]
        row = int(row)
        col = int(col)
        if 0 <= row < 3 and 0 <= col < 3:
            return row, col
    raise ValueError(f"unknown tensor component key {key!r}")


def constantTensor(epsilon: TensorLike) -> ComplexArray | None:
    if hasattr(epsilon, "convolutionMatrix"):
        return None

    if isinstance(epsilon, Mapping):
        tensor = np.zeros((3, 3), dtype=complex)
        present = np.zeros((3, 3), dtype=bool)
        for key, value in epsilon.items():
            array = np.asarray(value, dtype=complex)
            if array.ndim != 0:
                return None
            row, col = componentIndex(key)
            tensor[row, col] = complex(array.item())
            present[row, col] = True
        if not (present[0, 0] and present[1, 1] and present[2, 2]):
            return None
        return tensor

    if np.isscalar(epsilon):
        return complex(epsilon) * np.eye(3, dtype=complex)

    array = np.asarray(epsilon, dtype=complex)
    if array.ndim == 0:
        return complex(array.item()) * np.eye(3, dtype=complex)
    if array.ndim == 2 and array.shape == (3, 3):
        return array.copy()
    return None


def isScalarGrid(epsilon: TensorLike) -> bool:
    if isinstance(epsilon, Mapping) or hasattr(epsilon, "convolutionMatrix") or np.isscalar(epsilon):
        return False
    array = np.asarray(epsilon)
    return bool(array.ndim == 2 and array.shape != (3, 3))


def normalizeFactorization(value: str) -> str:
    normalized = str(value).lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "standard": "standard",
        "none": "standard",
        "direct": "standard",
        "normal-vector": "normal-vector",
        "normal-vector-li": "normal-vector",
        "nv": "normal-vector",
    }
    if normalized not in aliases:
        raise ValueError("factorization must be 'auto', 'standard', or 'normal-vector'")
    return aliases[normalized]


def useAnalyticNormalVector(epsilon: TensorLike, factorizationMode: str) -> bool:
    return bool(
        hasattr(epsilon, "normalVectorMatrices")
        and hasattr(epsilon, "reciprocalConvolutionMatrix")
        and (
            factorizationMode == "normal-vector"
            or (factorizationMode == "auto" and getattr(epsilon, "factorization", "analytic") == "normal-vector")
        )
    )


def shouldAutoGenerateNormalField(epsilon: TensorLike, factorizationMode: str) -> bool:
    if factorizationMode not in ("auto", "normal-vector"):
        return False
    if not isScalarGrid(epsilon):
        return False
    if factorizationMode == "auto":
        return looksPiecewiseConstant(epsilon)
    return True


def looksPiecewiseConstant(values: ArrayLike) -> bool:
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


def estimateNormalField(
    values: ArrayLike,
    orders: int | tuple[int, int] | None = None,
    truncation: str | None = None,
) -> ComplexArray:
    del orders, truncation
    grid = np.asarray(values, dtype=complex)
    if grid.ndim != 2:
        raise ValueError("normal-field estimation requires a 2D scalar grid")
    if grid.shape[0] < 1 or grid.shape[1] < 1:
        raise ValueError("sampled epsilon grid must be non-empty")

    contrast = np.abs(grid - np.mean(grid))
    dx = 0.5 * (np.roll(contrast, -1, axis=1) - np.roll(contrast, 1, axis=1))
    dy = 0.5 * (np.roll(contrast, -1, axis=0) - np.roll(contrast, 1, axis=0))
    length = np.sqrt(dx**2 + dy**2)

    normals = np.zeros(grid.shape + (2,), dtype=float)
    active = length > 1e-12 * max(1.0, float(np.max(length)) if length.size else 1.0)
    normals[..., 0] = np.where(active, dx / np.where(active, length, 1.0), 1.0)
    normals[..., 1] = np.where(active, dy / np.where(active, length, 1.0), 0.0)
    return normals


def constantTensorMatrices(
    tensor: ComplexArray,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    identity = np.eye(harmonics.count, dtype=complex)
    return tuple(tuple(complex(tensor[row, col]) * identity for col in range(3)) for row in range(3))


def convolution(value: ArrayLike | complex, harmonics: Harmonics) -> ComplexArray:
    return epsilonConvolutionMatrix(value, harmonics)


def zero(size: int) -> ComplexArray:
    return np.zeros((size, size), dtype=complex)


def solveIdentity(matrix: ComplexArray) -> ComplexArray:
    return np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=complex))
