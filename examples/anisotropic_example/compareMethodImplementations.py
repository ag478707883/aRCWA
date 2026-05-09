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
BACKEND = "cuda"
PRECISION = "complex128"  # Use "complex64" or "mixed" for fast spectrum scans.
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


def build_layer() -> rcwa.Layer:
    """Create one static tensor metasurface layer for the CUDA S-matrix simulation."""

    return rcwa.rectangularPostLayer(
        period=PERIOD,
        thickness=THICKNESS,
        background=1.0,
        post=rcwa.xzTensor(9.0, 9.4, 8.5, 0.3, 0.3),
        size=POST_SIZE,
        shape=(GRID, GRID),
        factorization="standard",
        name="anisotropic tensor post",
    )


def make_simulation() -> rcwa.RCWASimulation:
    return rcwa.RCWASimulation(
        period=PERIOD,
        layers=[build_layer()],
        orders=ORDER,
        truncation=TRUNCATION,
        backend=BACKEND,
        precision=PRECISION,
        precompile=True,
        cacheModes=True,
        epsIncident=1.0,
        epsTransmission=1.0,
    )


def solveSMatrixSpectrum(simulation: rcwa.RCWASimulation) -> dict[str, np.ndarray]:
    spectrum = simulation.spectrum(
        WAVELENGTHS,
        theta=THETA,
        phi=PHI,
        polarizations=("TM",),
        bidirectional=False,
    )
    data = spectrum["TM"]
    return {
        "wavelengths": spectrum["wavelengths"],
        "absorptivity": data["absorptivity"],
    }


def plot_results(result: dict[str, np.ndarray]) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True, constrained_layout=True)
    wavelengths = result["wavelengths"]
    absorption = result["absorptivity"]

    axes[0].plot(wavelengths, absorption, label="A = 1 - R - T")
    axes[1].semilogy(wavelengths, np.abs(absorption) + 1e-18, label="|A|")

    axes[0].set_ylabel("Absorptivity")
    axes[1].set_ylabel("|A|")
    axes[1].set_xlabel("Wavelength")

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
        f"on one tensor metasurface case, order={ORDER}, grid={GRID}, points={POINTS}, "
        f"backend={BACKEND}, precision={PRECISION}"
    )
    start = time.perf_counter()
    simulation = make_simulation()
    result = solveSMatrixSpectrum(simulation)
    output = plot_results(result)

    center = POINTS // 2
    print(f"figure: {output}")
    print(
        "center wavelength "
        f"{WAVELENGTHS[center]:.4f}: "
        f"A={result['absorptivity'][center]:.8e}"
    )
    print(f"max |A|={np.max(np.abs(result['absorptivity'])):.3e}")

    print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
