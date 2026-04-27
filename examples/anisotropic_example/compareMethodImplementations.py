from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


# Runtime knobs. Edit these constants directly for local experiments.
ORDER = (1, 1)
TRUNCATION = "circular"
GRID = 20
POINTS = 61
SHOW = False

if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# One explicit anisotropic metasurface case for the stable CUDA S-matrix route.
PERIOD = (1.0, 1.0)
THICKNESS = 0.25
POST_SIZE = (0.52, 0.34)
THETA = math.radians(9.0)
PHI = math.radians(17.0)
WAVELENGTHS = np.linspace(0.95, 1.15, POINTS)
METHOD = "smatrix"


def build_compiled_layer() -> rcwa.CompiledLayer:
    """Create one static tensor metasurface layer for the CUDA S-matrix solve."""

    layer = rcwa.rectangularPostLayer(
        period=PERIOD,
        thickness=THICKNESS,
        background=1.0,
        post=rcwa.xzTensor(9.0, 9.4, 8.5, 0.3, 0.3),
        size=POST_SIZE,
        shape=(GRID, GRID),
        factorization="standard",
        name="anisotropic tensor post",
    )
    return rcwa.compileLayers([layer], orders=ORDER, truncation=TRUNCATION)[0]


def solve_smatrix_spectrum(layer: rcwa.CompiledLayer) -> dict[str, np.ndarray]:
    reflection = np.empty_like(WAVELENGTHS)
    transmission = np.empty_like(WAVELENGTHS)
    conservation = np.empty_like(WAVELENGTHS)

    for index, wavelength in enumerate(WAVELENGTHS):
        result = rcwa.solveStack(
            layers=[layer],
            wavelength=float(wavelength),
            period=PERIOD,
            orders=ORDER,
            epsIncident=1.0,
            epsTransmission=1.0,
            theta=THETA,
            phi=PHI,
            sAmplitude=0.0,
            pAmplitude=1.0,
            truncation=TRUNCATION,
            method=METHOD,
            backend="cuda",
        )
        reflection[index] = result.reflection
        transmission[index] = result.transmission
        conservation[index] = result.conservation

    return {
        "reflection": reflection,
        "transmission": transmission,
        "conservation": conservation,
    }


def plot_results(result: dict[str, np.ndarray]) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True, constrained_layout=True)

    axes[0].plot(WAVELENGTHS, result["reflection"], label=METHOD)
    axes[1].plot(WAVELENGTHS, result["transmission"], label=METHOD)
    axes[2].semilogy(WAVELENGTHS, np.abs(result["conservation"] - 1.0) + 1e-18, label=METHOD)

    axes[0].set_ylabel("Reflection")
    axes[1].set_ylabel("Transmission")
    axes[2].set_ylabel("|R + T - 1|")
    axes[2].set_xlabel("Wavelength")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    figure.suptitle(
        "Anisotropic metasurface stable CUDA S-matrix\n"
        f"order={ORDER}, grid={GRID}, theta={math.degrees(THETA):.1f} deg, phi={math.degrees(PHI):.1f} deg"
    )

    output = ROOT / "outputs" / "anisotropic_stable_cuda_smatrix.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if SHOW:
        plt.show()
    else:
        plt.close(figure)
    return output


def main() -> None:
    print(
        "Solving anisotropic stable CUDA S-matrix "
        f"on one tensor metasurface case, order={ORDER}, grid={GRID}, points={POINTS}"
    )
    start = time.perf_counter()
    layer = build_compiled_layer()
    result = solve_smatrix_spectrum(layer)
    output = plot_results(result)

    center = POINTS // 2
    print(f"figure: {output}")
    print(
        "center wavelength "
        f"{WAVELENGTHS[center]:.4f}: "
        f"R={result['reflection'][center]:.8f}, "
        f"T={result['transmission'][center]:.8f}, "
        f"R+T={result['conservation'][center]:.8f}"
    )
    print(f"max |R+T-1|={np.max(np.abs(result['conservation'] - 1.0)):.3e}")

    print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
