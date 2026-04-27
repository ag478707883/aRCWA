from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from scipy.linalg import eig as scipyEig
from scipy.linalg import lu_factor as scipyLuFactor
from scipy.linalg import lu_solve as scipyLuSolve
from scipy.linalg import matrix_balance as scipyMatrixBalance
from scipy.linalg import solve as scipySolve


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    isGpu: bool = False
    isCuda: bool = False
    isTorch: bool = False
    device: Any | None = None

    def asarray(self, value: Any) -> Any:
        if self.isTorch:
            return self.xp.asarray(value, dtype=complex)
        return np.asarray(value)

    def toNumpy(self, value: Any) -> np.ndarray:
        if self.isTorch:
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        return np.asarray(value)

    def eig(self, matrix: Any) -> tuple[Any, Any]:
        if self.isTorch:
            eigenvalues, eigenvectors = self.xp.linalg.eig(matrix)
            if not _finiteTorchEigResult(self.xp._torch, eigenvalues, eigenvectors):
                raise RuntimeError("CUDA eigensolve returned non-finite values for anisotropic layer matrix")
            if _shouldValidateTorchEigResidual() and not _acceptableTorchEigResult(
                self.xp._torch,
                matrix,
                eigenvalues,
                eigenvectors,
            ):
                raise RuntimeError("CUDA eigensolve residual check failed for anisotropic layer matrix")
            return eigenvalues, eigenvectors

        array = np.asarray(matrix)
        try:
            eigenvalues, eigenvectors = np.linalg.eig(array)
            if _acceptableEigResult(array, eigenvalues, eigenvectors):
                return eigenvalues, eigenvectors
        except np.linalg.LinAlgError:
            pass

        eigenvalues, eigenvectors = scipyEig(array, overwrite_a=True, check_finite=False)
        if _acceptableEigResult(array, eigenvalues, eigenvectors):
            return eigenvalues, eigenvectors
        balanced, transform = scipyMatrixBalance(
            array,
            permute=False,
            scale=True,
            separate=False,
            overwrite_a=False,
        )
        eigenvalues, eigenvectors = scipyEig(balanced, overwrite_a=True, check_finite=False)
        return eigenvalues, transform @ eigenvectors

    def solve(self, matrix: Any, rhs: Any) -> Any:
        if self.isTorch:
            return self.xp.linalg.solve(matrix, rhs)
        return scipySolve(np.asarray(matrix), np.asarray(rhs), assume_a="gen", check_finite=False)

    def factor(self, matrix: Any) -> Any:
        if self.isTorch:
            lu, pivots, info = self.xp.linalg.lu_factor_ex(matrix)
            if bool(self.xp._torch.any(info != 0)):
                raise RuntimeError("CUDA LU factorization failed for anisotropic linear solve")
            return lu, pivots
        return scipyLuFactor(np.asarray(matrix), overwrite_a=False, check_finite=False)

    def solveFactored(self, factorization: Any, rhs: Any) -> Any:
        if self.isTorch:
            lu, pivots = factorization
            squeeze = rhs.ndim == 1
            rhsMatrix = rhs[:, None] if squeeze else rhs
            solution = self.xp.linalg.lu_solve(lu, pivots, rhsMatrix)
            return solution[:, 0] if squeeze else solution
        return scipyLuSolve(factorization, np.asarray(rhs), check_finite=False)

    def copy(self, value: Any) -> Any:
        if self.isTorch:
            return value.clone()
        return value.copy()

    def synchronize(self) -> None:
        if self.isCuda and self.isTorch:
            self.xp._torch.cuda.synchronize(self.device)


def resolveBackend(name: str | ArrayBackend | None = "cuda") -> ArrayBackend:
    """Resolve the public anisotropic array backend.

    The anisotropic solve path is CUDA-only and uses PyTorch, matching the
    isotropic package. ``auto`` intentionally does not fall back to CPU.
    """

    if isinstance(name, ArrayBackend):
        if not name.isTorch or not name.isCuda:
            raise ValueError("the anisotropic solver is CUDA-only; provide a CUDA PyTorch ArrayBackend")
        return name

    value = "cuda" if name is None else str(name).lower()
    if value in ("cuda", "gpu", "torch", "torch-cuda", "auto"):
        torch = _importTorch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the anisotropic solver requires a CUDA-enabled torch installation and visible CUDA device"
            )
        return _cudaBackend(torch)
    if value in ("cpu", "numpy", "torch-cpu", "pytorch-cpu"):
        raise ValueError("the anisotropic solver is CUDA-only; use backend='cuda'")
    raise ValueError("backend must be 'cuda', 'gpu', 'torch', 'torch-cuda', 'auto', an ArrayBackend, or None")


def cpuBackend() -> ArrayBackend:
    """Return the private NumPy backend used by CPU-only preprocessing helpers."""

    return ArrayBackend(name="cpu", xp=np, isGpu=False, isCuda=False, isTorch=False)


def _importTorch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        raise ImportError("PyTorch CUDA backend requested but torch is not installed") from exc
    return torch


def _cudaBackend(torch: Any) -> ArrayBackend:
    device = torch.device("cuda")
    return ArrayBackend(
        name="cuda",
        xp=_TorchNamespace(torch, device),
        isGpu=True,
        isCuda=True,
        isTorch=True,
        device=device,
    )


class _TorchNamespace:
    """Small NumPy-like facade for the solver's dense CUDA array operations."""

    def __init__(self, torch: Any, device: Any) -> None:
        self._torch = torch
        self.device = device
        self.linalg = torch.linalg

    def asarray(self, value: Any, dtype: Any | None = None) -> Any:
        torch_dtype = self._dtype(dtype)
        if hasattr(value, "to") and hasattr(value, "device"):
            return value.to(device=self.device, dtype=torch_dtype or value.dtype)
        return self._torch.as_tensor(np.asarray(value), dtype=torch_dtype or self._torch.complex128, device=self.device)

    def zeros(self, shape: Any, dtype: Any = complex) -> Any:
        return self._torch.zeros(shape, dtype=self._dtype(dtype), device=self.device)

    def zeros_like(self, value: Any) -> Any:
        return self._torch.zeros_like(value)

    def empty(self, shape: Any, dtype: Any = complex) -> Any:
        return self._torch.empty(shape, dtype=self._dtype(dtype), device=self.device)

    def eye(self, size: int, dtype: Any = complex) -> Any:
        return self._torch.eye(size, dtype=self._dtype(dtype), device=self.device)

    def diag(self, value: Any) -> Any:
        return self._torch.diag(value)

    def exp(self, value: Any) -> Any:
        return self._torch.exp(value)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return self._torch.cat(tuple(arrays), dim=axis)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self._torch.stack(tuple(arrays), dim=axis)

    def sum(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self._torch.sum(value)
        return self._torch.sum(value, dim=axis)

    def real(self, value: Any) -> Any:
        return self._torch.real(value)

    def conj(self, value: Any) -> Any:
        return self._torch.conj(value)

    def max(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self._torch.max(value)
        return self._torch.max(value, dim=axis).values

    def abs(self, value: Any) -> Any:
        return self._torch.abs(value)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        if not hasattr(x, "device"):
            x = self._torch.as_tensor(x, dtype=getattr(y, "dtype", self._torch.complex128), device=self.device)
        if not hasattr(y, "device"):
            y = self._torch.as_tensor(y, dtype=getattr(x, "dtype", self._torch.complex128), device=self.device)
        return self._torch.where(condition, x, y)

    def sqrt(self, value: Any) -> Any:
        return self._torch.sqrt(value)

    def argmax(self, value: Any, axis: int | None = None) -> Any:
        if axis is None:
            return self._torch.argmax(value)
        return self._torch.argmax(value, dim=axis)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("device", self.device)
        return self._torch.arange(*args, **kwargs)

    def _dtype(self, dtype: Any | None) -> Any:
        if dtype is None:
            return None
        if dtype in (complex, np.complex64, np.complex128):
            return self._torch.complex128
        if dtype in (float, np.float32, np.float64):
            return self._torch.float64
        if dtype in (int, np.int32, np.int64):
            return self._torch.int64
        return dtype


def _acceptableEigResult(matrix: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> bool:
    if not (np.all(np.isfinite(eigenvalues)) and np.all(np.isfinite(eigenvectors))):
        return False
    if matrix.size == 0:
        return True

    norm = float(np.linalg.norm(matrix))
    scale = max(1.0, norm)
    residual = matrix @ eigenvectors - eigenvectors * eigenvalues[np.newaxis, :]
    error = float(np.linalg.norm(residual))
    return np.isfinite(error) and error <= 1e-8 * scale


def _acceptableTorchEigResult(torch: Any, matrix: Any, eigenvalues: Any, eigenvectors: Any) -> bool:
    if not _finiteTorchEigResult(torch, eigenvalues, eigenvectors):
        return False
    if matrix.numel() == 0:
        return True

    unit = torch.as_tensor(1.0, dtype=torch.float64, device=matrix.device)
    scale = torch.maximum(torch.linalg.norm(matrix), unit)
    residual = matrix @ eigenvectors - eigenvectors * eigenvalues[None, :]
    error = torch.linalg.norm(residual)
    return bool(torch.isfinite(error) and error <= 1e-8 * scale)


def _finiteTorchEigResult(torch: Any, eigenvalues: Any, eigenvectors: Any) -> bool:
    return bool(torch.all(torch.isfinite(eigenvalues)) and torch.all(torch.isfinite(eigenvectors)))


def _shouldValidateTorchEigResidual() -> bool:
    return os.environ.get("RCWA3D_VALIDATE_CUDA_EIG", "").strip().lower() in ("1", "true", "yes", "on")
