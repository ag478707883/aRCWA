from __future__ import annotations

from typing import Literal

import numpy as np


ComplexArray = np.ndarray
Axis = Literal["x", "y", "z"]
RotationMode = Literal["tensor", "coupling"]


def xzTensor(
    epsilonXx: complex,
    epsilonYy: complex | None = None,
    epsilonZz: complex | None = None,
    epsilonXz: complex = 0.0,
    epsilonZx: complex | None = None,
    *,
    twist: float = 0.0,
    twistAxis: Axis = "z",
    twistMode: RotationMode = "tensor",
) -> ComplexArray:
    """Return a 3x3 tensor whose local off-diagonal terms are only xz/zx.

    ``twistMode="tensor"`` applies a full tensor rotation ``R eps R.T``.
    ``twistMode="coupling"`` rotates only the xz/zx coupling direction around
    z while preserving the diagonal terms; this is useful for magneto-optic
    Voigt tensors whose bias direction is swept in the x-y plane.
    """

    epsilonYy = epsilonXx if epsilonYy is None else epsilonYy
    epsilonZz = epsilonXx if epsilonZz is None else epsilonZz
    epsilonZx = epsilonXz if epsilonZx is None else epsilonZx

    tensor = np.array(
        [
            [epsilonXx, 0.0, epsilonXz],
            [0.0, epsilonYy, 0.0],
            [epsilonZx, 0.0, epsilonZz],
        ],
        dtype=complex,
    )
    if abs(twist) <= 0.0:
        return tensor
    if twistMode == "tensor":
        return rotateTensor(tensor, angle=twist, axis=twistAxis)
    if twistMode == "coupling":
        return twistXzCoupling(tensor, angle=twist, axis=twistAxis)
    raise ValueError("twistMode must be 'tensor' or 'coupling'")


def reciprocalXzTensor(
    epsilonXx: complex,
    epsilonYy: complex | None = None,
    epsilonZz: complex | None = None,
    epsilonXz: complex = 0.0,
    *,
    twist: float = 0.0,
    twistAxis: Axis = "z",
    twistMode: RotationMode = "tensor",
) -> ComplexArray:
    """Convenience wrapper for symmetric xz/zx coupling."""

    return xzTensor(
        epsilonXx,
        epsilonYy,
        epsilonZz,
        epsilonXz,
        epsilonXz,
        twist=twist,
        twistAxis=twistAxis,
        twistMode=twistMode,
    )


def gyrotropicXzTensor(
    epsilonParallel: complex,
    epsilonY: complex,
    gyrotropy: complex,
    *,
    twist: float = 0.0,
    twistMode: RotationMode = "coupling",
) -> ComplexArray:
    """Return the common magneto-optic tensor with anti-symmetric xz/zx terms.

    The local tensor is ``[[eps_parallel, 0, g], [0, eps_y, 0],
    [-g, 0, eps_parallel]]``.  By default the gyrotropic coupling direction is
    twisted in the x-y plane without rotating the diagonal entries, matching the
    convention often used for in-plane magnetic bias.
    """

    return xzTensor(
        epsilonParallel,
        epsilonY,
        epsilonParallel,
        gyrotropy,
        -gyrotropy,
        twist=twist,
        twistAxis="z",
        twistMode=twistMode,
    )


def rotateTensor(tensor: ComplexArray, angle: float, axis: Axis = "z") -> ComplexArray:
    """Rotate a permittivity tensor into the lab frame."""

    array = np.asarray(tensor, dtype=complex)
    if array.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")
    rotation = rotationMatrix(angle, axis)
    return rotation @ array @ rotation.T


def rotationMatrix(angle: float, axis: Axis = "z") -> ComplexArray:
    """Right-handed 3D rotation matrix."""

    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=complex)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=complex)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=complex)
    raise ValueError("axis must be 'x', 'y', or 'z'")


def twistXzCoupling(tensor: ComplexArray, angle: float, axis: Axis = "z") -> ComplexArray:
    """Rotate only xz/zx coupling terms around z, preserving diagonal terms."""

    if axis != "z":
        raise ValueError("coupling-only twist is currently defined for axis='z'")
    array = np.asarray(tensor, dtype=complex)
    if array.shape != (3, 3):
        raise ValueError("tensor must have shape (3, 3)")

    c = np.cos(angle)
    s = np.sin(angle)
    twisted = array.copy()
    exz = array[0, 2]
    ezx = array[2, 0]
    twisted[0, 2] = exz * c
    twisted[1, 2] = exz * s
    twisted[2, 0] = ezx * c
    twisted[2, 1] = ezx * s
    return twisted
