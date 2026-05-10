from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


# Runtime knobs. Increase ORDER/GRID/POINTS for convergence studies.
ORDER = 7
TRUNCATION = "circular"
BACKEND = "cuda"
PRECISION = os.environ.get("RCWA3D_ANISOTROPIC_PRECISION", "complex128")
FACTORIZATION = "auto"
GRID = 512
WORKERS = 1
POINTS = 501
SHOW = True

# Fang et al. 2024, Near-infrared multi-band small-angle TE-polarization
# nonreciprocal thermal emitter. All geometric lengths are in um.
PERIOD = 1.2
RING_HEIGHT = 0.5
CEYIG_HEIGHT = 0.65
AG_HEIGHT = 0.2
INNER_WIDTH_RATIO = 0.3
OUTER_WIDTH_RATIO = 0.7
INNER_WIDTH = INNER_WIDTH_RATIO * PERIOD
OUTER_WIDTH = OUTER_WIDTH_RATIO * PERIOD

# Illumination and sweep for Fig. 2(c,d).
THETA = math.radians(4.0)
PHI = 0.0
WAVELENGTHS = np.linspace(1.65, 1.8, POINTS)
POLARIZATIONS = ("TE",)

# Reference peak positions reported in the paper, used only as plot guides.
PAPER_EMISSION_PEAKS_UM = np.array([1659.3, 1692.2, 1740.5, 1777.8]) / 1000.0
PAPER_ABSORPTION_PEAKS_UM = np.array([1668.8, 1694.3, 1728.0, 1776.2]) / 1000.0

# Material parameters.
SI_INDEX = 3.48
CEYIG_EPSILON = 4.0
CEYIG_B = 0.1

# Ag is intentionally copied from zou2024CylindricalGrating.py per request.
Ag_eps_inf = 3.4
Ag_wp = 1.39e16
Ag_gamma = 2.7e13


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def Ag_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = Ag_eps_inf - Ag_wp**2 / (omega**2 + 1j * Ag_gamma * omega)
    return scalar_tensor(epsilon)


def ceyig_tensor() -> np.ndarray:
    """Return the Ce:YIG Voigt tensor used by the paper.

    rcwa.gyrotropicXzTensor returns [[eps, 0, g], [0, eps_y, 0],
    [-g, 0, eps]].  With g=-1j*b this matches the paper convention
    [[eps, 0, -b*i], 
    [0,   eps, 0], 
    [b*i, 0, eps]].
    """

    return rcwa.gyrotropicXzTensor(
        epsilonParallel=CEYIG_EPSILON,
        epsilonY=CEYIG_EPSILON,
        gyrotropy=-1j * CEYIG_B,
        twist=PHI,
        twistMode="coupling",
    )


def square_boundary_normal(
    xx: np.ndarray,
    yy: np.ndarray,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal of the nearest edge of an axis-aligned square."""

    half_width = width / 2.0
    distance_x = np.abs(np.abs(xx) - half_width)
    distance_y = np.abs(np.abs(yy) - half_width)
    x_edge_is_closer = distance_x <= distance_y
    sign_x = np.where(xx >= 0.0, 1.0, -1.0)
    sign_y = np.where(yy >= 0.0, 1.0, -1.0)
    normal_x = np.where(x_edge_is_closer, sign_x, 0.0)
    normal_y = np.where(x_edge_is_closer, 0.0, sign_y)
    distance = np.minimum(distance_x, distance_y)
    return normal_x, normal_y, distance


def square_ring_normal_field(
    xx: np.ndarray,
    yy: np.ndarray,
    outer_width: float,
    inner_width: float,
) -> np.ndarray:
    """Composite normal field for the square ring's outer and inner walls."""

    outer_nx, outer_ny, outer_distance = square_boundary_normal(xx, yy, outer_width)
    inner_nx, inner_ny, inner_distance = square_boundary_normal(xx, yy, inner_width)
    use_inner_wall = inner_distance <= outer_distance
    normal_x = np.where(use_inner_wall, inner_nx, outer_nx)
    normal_y = np.where(use_inner_wall, inner_ny, outer_ny)
    return np.stack((normal_x, normal_y), axis=-1).astype(float)


def make_pattern_layer() -> rcwa.Layer:
    """Build the sampled silicon square-ring layer with an explicit boundary normal field."""

    layer = rcwa.PatternLayer(
        period=(PERIOD, PERIOD),
        thickness=RING_HEIGHT,
        background=1.0,
        shape=(GRID, GRID),
        factorization=FACTORIZATION,
        name="Si square-ring array",
    )
    layer.rectangle(size=(OUTER_WIDTH, OUTER_WIDTH), material=SI_INDEX**2, useNormal=False)
    layer.rectangle(size=(INNER_WIDTH, INNER_WIDTH), material=1.0, useNormal=False)

    xx, yy = layer.pattern.coordinates()
    layer.pattern.normalField = square_ring_normal_field(xx, yy, OUTER_WIDTH, INNER_WIDTH)
    return layer.toLayer()


def make_simulation() -> rcwa.RCWASimulation:
    return rcwa.buildSimulation(make_config(), make_layers())


def make_config() -> rcwa.SimulationConfig:
    return rcwa.SimulationConfig(
        period=(PERIOD, PERIOD),
        orders=ORDER,
        truncation=TRUNCATION,
        backend=BACKEND,
        precision=PRECISION,
        epsIncident=1.0,
        epsTransmission=SI_INDEX**2,
        precompile=True,
        cacheModes=True,
        workers=WORKERS,
    )


def make_layers() -> list[object]:
    return [
        make_pattern_layer(),
        rcwa.homogeneousLayer(CEYIG_HEIGHT, ceyig_tensor(), name="Ce:YIG"),
        rcwa.homogeneousLayer(AG_HEIGHT, Ag_tensor, name="Ag"),
    ]


def plot_spectrum(spectrum: dict[str, object]) -> Path:
    data = spectrum["TE"]
    wavelengths = spectrum["wavelengths"]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True, constrained_layout=True)
    axes[0].plot(wavelengths, data["absorptivity"], label="A(+theta), absorption")
    axes[0].plot(wavelengths, data["emissivity"], label="A(-theta), emission")
    axes[0].set_ylabel("TE absorptivity / emissivity")
    axes[0].set_ylim(-0.03, 1.03)

    axes[1].plot(wavelengths, data["nonreciprocity"], color="tab:red", label="|A(+theta)-A(-theta)|")
    axes[1].set_ylabel("TE nonreciprocity")
    axes[1].set_xlabel("Wavelength (um)")
    axes[1].set_ylim(-0.03, 1.03)

    for index, peak in enumerate(PAPER_ABSORPTION_PEAKS_UM):
        axes[0].axvline(
            peak,
            color="tab:blue",
            linestyle=":",
            alpha=0.35,
            label="paper absorption peaks" if index == 0 else None,
        )
    for index, peak in enumerate(PAPER_EMISSION_PEAKS_UM):
        axes[0].axvline(
            peak,
            color="tab:orange",
            linestyle=":",
            alpha=0.35,
            label="paper emission peaks" if index == 0 else None,
        )

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(f"Fang 2024 Si/Ce:YIG/Ag square-ring emitter, order={ORDER}, grid={GRID}")
    output = ROOT / "outputs" / "fang2024_square_ring_emitter_rcwa.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return output


def main() -> None:
    print(
        "RCWA "
        f"order={ORDER}, grid={GRID}, points={POINTS}, truncation={TRUNCATION}, "
        f"factorization={FACTORIZATION}, backend={BACKEND}, precision={PRECISION}, workers={WORKERS}"
    )
    print(
        "Fang 2024 geometry: "
        f"d={PERIOD} um, h1={RING_HEIGHT} um, h2={CEYIG_HEIGHT} um, h3={AG_HEIGHT} um, "
        f"w1={INNER_WIDTH:.3f} um, w2={OUTER_WIDTH:.3f} um, theta={math.degrees(THETA):.1f} deg"
    )
    start = time.perf_counter()
    spectrum = rcwa.solveSpectrum(
        make_config(),
        make_layers(),
        WAVELENGTHS,
        theta=THETA,
        phi=PHI,
        polarizations=POLARIZATIONS,
        bidirectional=True,
    )
    figure = plot_spectrum(spectrum)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
