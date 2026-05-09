from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    linalg: Any
    isCuda: bool = False
    isTorch: bool = False
    device: Any | None = None

    def asnumpy(self, value: Any) -> np.ndarray:
        if self.isTorch:
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        return np.asarray(value)

    def synchronize(self) -> None:
        if self.isCuda:
            self.xp.cuda.synchronize(self.device)


def resolveBackend(backend: str | ArrayBackend | None = None) -> ArrayBackend:
    """Resolve an array backend.

    The isotropic public solve path is CUDA-only.  Accepted aliases all map to
    a CUDA PyTorch backend; CPU and torch-cpu requests are rejected instead of
    falling back silently.
    """

    if isinstance(backend, ArrayBackend):
        if not backend.isTorch or not backend.isCuda:
            raise ValueError("the isotropic solver is CUDA-only; provide a CUDA PyTorch ArrayBackend")
        return backend

    value = "cuda" if backend is None else str(backend).lower()
    if value in ("cuda", "gpu", "torch", "torch-cuda", "auto"):
        torch = importTorch()
        if not torch.cuda.is_available():
            raise RuntimeError("the isotropic solver requires a CUDA-enabled torch installation and visible CUDA device")
        return cudaBackend(torch)
    if value in ("cpu", "numpy", "torch-cpu", "pytorch-cpu"):
        raise ValueError("the isotropic solver is CUDA-only; use backend='cuda'")
    raise ValueError("backend must be 'cuda', 'gpu', 'torch', 'torch-cuda', 'auto', an ArrayBackend, or None")


def importTorch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install
        raise ImportError("PyTorch backend requested but torch is not installed") from exc
    return torch


def cudaBackend(torch: Any) -> ArrayBackend:
    device = torch.device("cuda")
    return ArrayBackend("cuda", torch, torch.linalg, isTorch=True, isCuda=True, device=device)
