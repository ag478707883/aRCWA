from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .analytic import AnalyticDisk, analyticDiskConvolution, analyticDiskJonesMatrices
from .fourier import Harmonics, epsilonConvolutionMatrix


ComplexArray = np.ndarray
TensorLike = object

_AXIS = {"x": 0, "y": 1, "z": 2}
_COMPONENT_NAMES = {
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

    factorizationMode = _normalizeFactorization(factorization)
    constantTensor = _constantTensor(epsilon)
    if isinstance(epsilon, AnalyticDisk):
        components, factorization = _analyticDiskTensorMatrices(epsilon, harmonics)
    elif _isScalarGrid(epsilon):
        if normalField is None and _shouldAutoGenerateNormalField(epsilon, factorizationMode):
            normalField = _estimateNormalField(epsilon, harmonics.orders, harmonics.truncation)
        if normalField is not None:
            components = _normalVectorScalarTensorMatrices(epsilon, normalField, harmonics)
            factorization = "normal-vector-li"
        elif factorizationMode in ("normal-vector", "jones"):
            raise ValueError("vector factorization requires a scalar epsilon grid with material contrast")
        else:
            components = _tensorComponentMatrices(epsilon, harmonics)
            factorization = "z-li"
    elif normalField is not None:
        raise ValueError("normal-vector factorization requires a scalar 2D epsilon grid")
    else:
        components = _tensorComponentMatrices(epsilon, harmonics)
        factorization = "z-li"
    try:
        etaZz = _solveIdentity(components[2][2])
    except np.linalg.LinAlgError as exc:
        raise ValueError("epsilon_zz convolution matrix is singular; Li inverse factorization failed") from exc
    return TensorConvolutionData(
        components=components,
        etaZz=etaZz,
        constantTensor=constantTensor if factorization == "z-li" else None,
        factorization=factorization,
    )


def constantTensor(epsilon: TensorLike) -> ComplexArray | None:
    """Return a homogeneous 3x3 tensor for scalar/constant tensor inputs."""

    return _constantTensor(epsilon)


def liFactorizedSystemMatrix(data: TensorConvolutionData, harmonics: Harmonics) -> ComplexArray:
    """Build the full 4N x 4N first-order Maxwell matrix for tensor epsilon.

    The state vector is ``[Ex, Ey, Hx, Hy]`` and modes satisfy
    ``d f / d(k0 z) = i A f``.  Ez is eliminated through
    ``Dz = Ky Hx - Kx Hy`` using ``[epsilon_zz]^-1``.  This gives the Li
    inverse-rule Schur-complement blocks for all xz/zx tensor couplings.
    """

    n = harmonics.count
    identity = np.eye(n, dtype=complex)
    diagonal = np.arange(n)
    kx = harmonics.kx
    ky = harmonics.ky
    c = data.components
    eta = data.etaZz

    cxx, cxy, cxz = c[0]
    cyx, cyy, cyz = c[1]
    czx, czy, _czz = c[2]

    cxzEta = cxz @ eta
    cyzEta = cyz @ eta
    etaCzx = eta @ czx
    etaCzy = eta @ czy

    dxEx = cxx - cxzEta @ czx
    dxEy = cxy - cxzEta @ czy
    dxHx = cxzEta * ky[np.newaxis, :]
    dxHy = -cxzEta * kx[np.newaxis, :]

    dyEx = cyx - cyzEta @ czx
    dyEy = cyy - cyzEta @ czy
    dyHx = cyzEta * ky[np.newaxis, :]
    dyHy = -cyzEta * kx[np.newaxis, :]

    a11 = -kx[:, np.newaxis] * etaCzx
    a12 = -kx[:, np.newaxis] * etaCzy
    a13 = kx[:, np.newaxis] * eta * ky[np.newaxis, :]
    a14 = identity - kx[:, np.newaxis] * eta * kx[np.newaxis, :]

    a21 = -ky[:, np.newaxis] * etaCzx
    a22 = -ky[:, np.newaxis] * etaCzy
    a23 = ky[:, np.newaxis] * eta * ky[np.newaxis, :] - identity
    a24 = -ky[:, np.newaxis] * eta * kx[np.newaxis, :]

    a31 = -dyEx.copy()
    a31[diagonal, diagonal] -= kx * ky
    a32 = -dyEy.copy()
    a32[diagonal, diagonal] += kx * kx
    a33 = -dyHx
    a34 = -dyHy

    a41 = dxEx.copy()
    a41[diagonal, diagonal] -= ky * ky
    a42 = dxEy.copy()
    a42[diagonal, diagonal] += ky * kx
    a43 = dxHx
    a44 = dxHy

    system = np.empty((4 * n, 4 * n), dtype=complex)
    system[0:n, 0:n] = a11
    system[0:n, n : 2 * n] = a12
    system[0:n, 2 * n : 3 * n] = a13
    system[0:n, 3 * n : 4 * n] = a14
    system[n : 2 * n, 0:n] = a21
    system[n : 2 * n, n : 2 * n] = a22
    system[n : 2 * n, 2 * n : 3 * n] = a23
    system[n : 2 * n, 3 * n : 4 * n] = a24
    system[2 * n : 3 * n, 0:n] = a31
    system[2 * n : 3 * n, n : 2 * n] = a32
    system[2 * n : 3 * n, 2 * n : 3 * n] = a33
    system[2 * n : 3 * n, 3 * n : 4 * n] = a34
    system[3 * n : 4 * n, 0:n] = a41
    system[3 * n : 4 * n, n : 2 * n] = a42
    system[3 * n : 4 * n, 2 * n : 3 * n] = a43
    system[3 * n : 4 * n, 3 * n : 4 * n] = a44
    return system


def _tensorComponentMatrices(
    epsilon: TensorLike,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    if isinstance(epsilon, Mapping):
        return _mappingTensorMatrices(epsilon, harmonics)

    if np.isscalar(epsilon):
        scalar = _convolution(epsilon, harmonics)
        zero = _zero(harmonics.count)
        return ((scalar, zero, zero), (zero, scalar.copy(), zero.copy()), (zero.copy(), zero.copy(), scalar.copy()))

    array = np.asarray(epsilon, dtype=complex)
    if array.ndim == 0:
        scalar = _convolution(array.item(), harmonics)
        zero = _zero(harmonics.count)
        return ((scalar, zero, zero), (zero, scalar.copy(), zero.copy()), (zero.copy(), zero.copy(), scalar.copy()))

    if array.ndim == 2 and array.shape == (3, 3):
        return _constantTensorMatrices(array, harmonics)

    if array.ndim == 2:
        scalar = _convolution(array, harmonics)
        zero = _zero(harmonics.count)
        return ((scalar, zero, zero), (zero, scalar.copy(), zero.copy()), (zero.copy(), zero.copy(), scalar.copy()))

    if array.ndim == 4 and array.shape[-2:] == (3, 3):
        return tuple(
            tuple(_convolution(array[..., row, col], harmonics) for col in range(3)) for row in range(3)
        )

    if array.ndim == 4 and array.shape[:2] == (3, 3):
        return tuple(
            tuple(_convolution(array[row, col, ...], harmonics) for col in range(3)) for row in range(3)
        )

    raise ValueError(
        "anisotropic epsilon must be a scalar, a 2D scalar grid, a (3, 3) tensor, "
        "a (ny, nx, 3, 3) tensor grid, an AnalyticDisk, or a component mapping"
    )


def _analyticDiskTensorMatrices(
    disk: AnalyticDisk,
    harmonics: Harmonics,
) -> tuple[tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...], str]:
    zero = _zero(harmonics.count)
    if disk.factorization == "jones":
        cxx, cxy, cyx, cyy, czz = analyticDiskJonesMatrices(disk, harmonics)
        return ((cxx, cxy, zero.copy()), (cyx, cyy, zero.copy()), (zero.copy(), zero.copy(), czz)), "jones-li"

    direct = analyticDiskConvolution(disk, harmonics)
    return (
        (direct, zero.copy(), zero.copy()),
        (zero.copy(), direct.copy(), zero.copy()),
        (zero.copy(), zero.copy(), direct.copy()),
    ), "analytic-li"


def _normalVectorScalarTensorMatrices(
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

    nx = epsilonConvolutionMatrix(normalX, harmonics)
    ny = epsilonConvolutionMatrix(normalY, harmonics)
    tx = epsilonConvolutionMatrix(tangentX, harmonics)
    ty = epsilonConvolutionMatrix(tangentY, harmonics)

    direct = epsilonConvolutionMatrix(grid, harmonics)
    inverseRule = _solveIdentity(epsilonConvolutionMatrix(1.0 / grid, harmonics))
    zero = _zero(harmonics.count)

    cxx = nx @ inverseRule @ nx + tx @ direct @ tx
    cxy = nx @ inverseRule @ ny + tx @ direct @ ty
    cyx = ny @ inverseRule @ nx + ty @ direct @ tx
    cyy = ny @ inverseRule @ ny + ty @ direct @ ty
    return ((cxx, cxy, zero.copy()), (cyx, cyy, zero.copy()), (zero.copy(), zero.copy(), direct))


def _mappingTensorMatrices(
    epsilon: Mapping[object, ArrayLike | complex],
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    n = harmonics.count
    components: list[list[ComplexArray | None]] = [[None for _ in range(3)] for _ in range(3)]
    for key, value in epsilon.items():
        row, col = _componentIndex(key)
        components[row][col] = _convolution(value, harmonics)

    for index, name in enumerate(("xx", "yy", "zz")):
        if components[index][index] is None:
            raise ValueError(f"component mapping epsilon is missing required diagonal component '{name}'")

    zero = _zero(n)
    return tuple(
        tuple(components[row][col] if components[row][col] is not None else zero.copy() for col in range(3))
        for row in range(3)
    )


def _componentIndex(key: object) -> tuple[int, int]:
    if isinstance(key, str):
        normalized = key.lower().replace("epsilon", "").replace("eps", "").replace("_", "").replace("-", "")
        if normalized in _COMPONENT_NAMES:
            return _COMPONENT_NAMES[normalized]
        if len(normalized) == 2 and normalized[0] in _AXIS and normalized[1] in _AXIS:
            return _AXIS[normalized[0]], _AXIS[normalized[1]]
    if isinstance(key, tuple) and len(key) == 2:
        row, col = key
        if isinstance(row, str):
            row = _AXIS[row.lower()]
        if isinstance(col, str):
            col = _AXIS[col.lower()]
        row = int(row)
        col = int(col)
        if 0 <= row < 3 and 0 <= col < 3:
            return row, col
    raise ValueError(f"unknown tensor component key {key!r}")


def _constantTensor(epsilon: TensorLike) -> ComplexArray | None:
    if isinstance(epsilon, AnalyticDisk):
        return None
    if isinstance(epsilon, Mapping):
        tensor = np.zeros((3, 3), dtype=complex)
        present = np.zeros((3, 3), dtype=bool)
        for key, value in epsilon.items():
            array = np.asarray(value, dtype=complex)
            if array.ndim != 0:
                return None
            row, col = _componentIndex(key)
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


def _isScalarGrid(epsilon: TensorLike) -> bool:
    if isinstance(epsilon, AnalyticDisk) or isinstance(epsilon, Mapping) or np.isscalar(epsilon):
        return False
    array = np.asarray(epsilon)
    return bool(array.ndim == 2 and array.shape != (3, 3))


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


def _shouldAutoGenerateNormalField(epsilon: TensorLike, factorizationMode: str) -> bool:
    if factorizationMode not in ("auto", "normal-vector", "jones"):
        return False
    if not _isScalarGrid(epsilon):
        return False
    if factorizationMode == "auto":
        return _looksPiecewiseConstant(epsilon)
    return True


def _looksPiecewiseConstant(values: ArrayLike) -> bool:
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


def _estimateNormalField(
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


def _constantTensorMatrices(
    tensor: ComplexArray,
    harmonics: Harmonics,
) -> tuple[tuple[ComplexArray, ComplexArray, ComplexArray], ...]:
    identity = np.eye(harmonics.count, dtype=complex)
    return tuple(tuple(complex(tensor[row, col]) * identity for col in range(3)) for row in range(3))


def _convolution(value: ArrayLike | complex, harmonics: Harmonics) -> ComplexArray:
    if isinstance(value, AnalyticDisk):
        return analyticDiskConvolution(value, harmonics)
    return epsilonConvolutionMatrix(value, harmonics)


def _zero(size: int) -> ComplexArray:
    return np.zeros((size, size), dtype=complex)


def _solveIdentity(matrix: ComplexArray) -> ComplexArray:
    return np.linalg.solve(matrix, np.eye(matrix.shape[0], dtype=complex))
