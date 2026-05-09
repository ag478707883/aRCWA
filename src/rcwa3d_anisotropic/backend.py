from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    isCuda: bool = False
    isTorch: bool = False
    device: Any | None = None
    complexDtype: Any | None = None
    floatDtype: Any | None = None

    def asarray(self, value: Any) -> Any:
        return self.xp.asarray(value, dtype=self.complexDtype or complex)

    def toNumpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        eigenvalues, eigenvectors = self.xp.linalg.eig(matrix)
        if not finiteTorchEigResult(self.xp.torch, eigenvalues, eigenvectors):
            raise RuntimeError("CUDA eigensolve returned non-finite values for anisotropic layer matrix")
        if shouldValidateTorchEigResidual() and not acceptableTorchEigResult(
            self.xp.torch,
            matrix,
            eigenvalues,
            eigenvectors,
        ):
            raise RuntimeError("CUDA eigensolve residual check failed for anisotropic layer matrix")
        return eigenvalues, eigenvectors

    def solve(self, matrix: Any, rhs: Any) -> Any:
        return self.xp.linalg.solve(matrix, rhs)

    def factor(self, matrix: Any) -> Any:
        lu, pivots, info = self.xp.linalg.lu_factor_ex(matrix)
        if bool(self.xp.torch.any(info != 0)):
            raise RuntimeError("CUDA LU factorization failed for anisotropic linear solve")
        return lu, pivots

    def solveFactored(self, factorization: Any, rhs: Any) -> Any:
        lu, pivots = factorization
        squeeze = rhs.ndim == 1
        rhsMatrix = rhs[:, None] if squeeze else rhs
        solution = self.xp.linalg.lu_solve(lu, pivots, rhsMatrix)
        return solution[:, 0] if squeeze else solution

    def copy(self, value: Any) -> Any:
        return value.clone()

    def synchronize(self) -> None:
        if self.isCuda:
            self.xp.torch.cuda.synchronize(self.device)


def resolveBackend(
    name: str | ArrayBackend | None = "cuda",
    *,
    precision: str | None = None,
) -> ArrayBackend:
    """Resolve the public anisotropic array backend.

    The anisotropic solve path is CUDA-only and uses PyTorch, matching the
    isotropic package. ``auto`` intentionally does not fall back to CPU.
    """

    if isinstance(name, ArrayBackend):
        if not name.isTorch or not name.isCuda:
            raise ValueError("the anisotropic solver is CUDA-only; provide a CUDA PyTorch ArrayBackend")
        if precision is not None and normalizePrecision(precision) != backendPrecision(name):
            raise ValueError("precision cannot override an already-created ArrayBackend")
        return name

    value = "cuda" if name is None else str(name).lower()
    if value in ("cuda", "gpu", "torch", "torch-cuda", "auto"):
        torch = importTorch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the anisotropic solver requires a CUDA-enabled torch installation and visible CUDA device"
            )
        return cudaBackend(torch, precision=normalizePrecision(precision))
    if value in ("cpu", "numpy", "torch-cpu", "pytorch-cpu"):
        raise ValueError("the anisotropic solver is CUDA-only; use backend='cuda'")
    raise ValueError("backend must be 'cuda', 'gpu', 'torch', 'torch-cuda', 'auto', an ArrayBackend, or None")


def importTorch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        raise ImportError("PyTorch CUDA backend requested but torch is not installed") from exc
    return torch


def normalizePrecision(value: str | None) -> str:
    if value is None:
        value = os.environ.get("RCWA3D_ANISOTROPIC_PRECISION", "complex128")
    normalized = str(value).lower().replace("-", "").replace("_", "")
    aliases = {
        "complex128": "complex128",
        "128": "complex128",
        "double": "complex128",
        "float64": "complex128",
        "complex64": "complex64",
        "64": "complex64",
        "single": "complex64",
        "float32": "complex64",
        "mixed": "complex64",
    }
    if normalized not in aliases:
        raise ValueError("precision must be 'complex128', 'complex64', or 'mixed'")
    return aliases[normalized]


def backendPrecision(backend: ArrayBackend) -> str:
    dtype = backend.complexDtype
    torch = getattr(backend.xp, "torch", None)
    if torch is not None and dtype is torch.complex64:
        return "complex64"
    return "complex128"


def cudaBackend(torch: Any, *, precision: str = "complex128") -> ArrayBackend:
    device = torch.device("cuda")
    complexDtype = torch.complex64 if precision == "complex64" else torch.complex128
    floatDtype = torch.float32 if precision == "complex64" else torch.float64
    return ArrayBackend(
        name="cuda",
        xp=TorchNamespace(torch, device, complexDtype=complexDtype, floatDtype=floatDtype),
        isCuda=True,
        isTorch=True,
        device=device,
        complexDtype=complexDtype,
        floatDtype=floatDtype,
    )


class TorchNamespace:
    """Small NumPy-like facade for the solver's dense CUDA array operations."""

    def __init__(self, torch: Any, device: Any, *, complexDtype: Any, floatDtype: Any) -> None:
        self.torch = torch
        self.device = device
        self.linalg = torch.linalg
        self.complexDtype = complexDtype
        self.floatDtype = floatDtype

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        torch_dtype = self.dtype(dtype)
        if hasattr(value, "to") and hasattr(value, "device"):
            return value.to(device=self.device, dtype=torch_dtype or value.dtype)
        return self.torch.as_tensor(np.asarray(value), dtype=torch_dtype or self.complexDtype, device=self.device)

    def zeros(self, shape: Any, dtype: Any = complex) -> Any:
        return self.torch.zeros(shape, dtype=self.dtype(dtype), device=self.device)

    def zeros_like(self, value: Any) -> Any:
        return self.torch.zeros_like(value)

    def empty(self, shape: Any, dtype: Any = complex) -> Any:
        return self.torch.empty(shape, dtype=self.dtype(dtype), device=self.device)

    def eye(self, size: int, dtype: Any = complex) -> Any:
        return self.torch.eye(size, dtype=self.dtype(dtype), device=self.device)

    def diag(self, value: Any) -> Any:
        return self.torch.diag(value)

    def exp(self, value: Any) -> Any:
        return self.torch.exp(value)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return self.torch.cat(tuple(arrays), dim=axis)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self.torch.stack(tuple(arrays), dim=axis)

    def sum(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self.torch.sum(value)
        return self.torch.sum(value, dim=axis)

    def real(self, value: Any) -> Any:
        return self.torch.real(value)

    def conj(self, value: Any) -> Any:
        return self.torch.conj(value)

    def max(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self.torch.max(value)
        return self.torch.max(value, dim=axis).values

    def abs(self, value: Any) -> Any:
        return self.torch.abs(value)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        if not hasattr(x, "device"):
            x = self.torch.as_tensor(x, dtype=getattr(y, "dtype", self.complexDtype), device=self.device)
        if not hasattr(y, "device"):
            y = self.torch.as_tensor(y, dtype=getattr(x, "dtype", self.complexDtype), device=self.device)
        return self.torch.where(condition, x, y)

    def sqrt(self, value: Any) -> Any:
        return self.torch.sqrt(value)

    def argmax(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self.torch.argmax(value)
        return self.torch.argmax(value, dim=axis)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("device", self.device)
        return self.torch.arange(*args, **kwargs)

    def dtype(self, dtype: Any | None) -> Any:
        if dtype is None:
            return None
        if dtype in (complex, np.complex64, np.complex128):
            return self.complexDtype
        if dtype in (float, np.float32, np.float64):
            return self.floatDtype
        if dtype in (int, np.int32, np.int64):
            return self.torch.int64
        return dtype

def acceptableTorchEigResult(torch: Any, matrix: Any, eigenvalues: Any, eigenvectors: Any) -> bool:
    if not finiteTorchEigResult(torch, eigenvalues, eigenvectors):
        return False
    if matrix.numel() == 0:
        return True

    unit = torch.as_tensor(1.0, dtype=torch.real(matrix).dtype, device=matrix.device)
    scale = torch.maximum(torch.linalg.norm(matrix), unit)
    residual = matrix @ eigenvectors - eigenvectors * eigenvalues[..., None, :]
    error = torch.linalg.norm(residual)
    return bool(torch.isfinite(error) and error <= 1e-8 * scale)


def finiteTorchEigResult(torch: Any, eigenvalues: Any, eigenvectors: Any) -> bool:
    return bool(torch.all(torch.isfinite(eigenvalues)) and torch.all(torch.isfinite(eigenvectors)))


def shouldValidateTorchEigResidual() -> bool:
    return os.environ.get("RCWA3D_VALIDATE_CUDA_EIG", "").strip().lower() in ("1", "true", "yes", "on")
