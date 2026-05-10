from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
from scipy.constants import c, e, epsilon_0, m_e

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


# Runtime knobs. Increase ORDER/GRID/POINTS for convergence studies.
ORDER = 8
TRUNCATION = "circular"
BACKEND = "cuda"
PRECISION = "complex128"  # Use "complex64" or "mixed" only after checking convergence/stability.
FACTORIZATION = "auto"
GRID = 512
WORKERS = 1
POINTS = 241
SHOW = True

# Fang et al. 2023, International Communications in Heat and Mass Transfer 148,
# 107031, "Dual-polarization strong nonreciprocal thermal radiation under
# near-normal incidence", Fig. 2. All geometric lengths are in um.
PERIOD = 5.2
SI_HEIGHT = 4.3
INAS_HEIGHT = 3.9
AL_HEIGHT = 0.2
FILL_RATIO = 0.56
HOLE_DIAMETER = FILL_RATIO * PERIOD
HOLE_RADIUS = HOLE_DIAMETER / 2.0

# Illumination and sweep from the paper's optimized Fig. 2 case.
B_FIELD = 5.0
THETA = math.radians(1.5)
PHI = 0.0
WAVELENGTHS = np.linspace(15.00, 15.30, POINTS)
POLARIZATIONS = ("TE", "TM")

# Resonant wavelength guide marks quoted in the text around Figs. 6-7.
PAPER_TE_ABSORPTION_UM = 15.193
PAPER_TE_EMISSION_UM = 15.201
PAPER_TM_RESONANCE_UM = 15.127

# Material parameters from Eqs. (1)-(4) and (8).
SI_INDEX = 3.48
AL_EPS_INF = 1.0
AL_PLASMA = 2.24e16
AL_GAMMA = 1.24e14
INAS_EPS_INF = 12.37
INAS_GAMMA = 1.55e11
INAS_DENSITY = 7.8e17 * 1e6
INAS_MASS = 0.033 * m_e
INAS_HALL_SIGN = 1.0


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def al_tensor(wavelength_um: float) -> np.ndarray:
    """Return Al permittivity from the Drude model in Eq. (8)."""

    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = AL_EPS_INF - AL_PLASMA**2 / (omega * (omega + 1j * AL_GAMMA))
    return scalar_tensor(epsilon)


def inas_tensor(wavelength_um: float) -> np.ndarray:
    """Return magnetized InAs tensor from Eqs. (1)-(4), with B along y."""

    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    cyclotron = e * B_FIELD / INAS_MASS
    plasma = np.sqrt(INAS_DENSITY * e**2 / (epsilon_0 * INAS_MASS))
    denominator = omega * ((omega + 1j * INAS_GAMMA) ** 2 - cyclotron**2)

    eps_xx = INAS_EPS_INF - plasma**2 * (omega + 1j * INAS_GAMMA) / denominator
    eps_yy = INAS_EPS_INF - plasma**2 / (omega * (omega + 1j * INAS_GAMMA))
    eps_xz = INAS_HALL_SIGN * 1j * plasma**2 * cyclotron / denominator
    return rcwa.gyrotropicXzTensor(
        epsilonParallel=eps_xx,
        epsilonY=eps_yy,
        gyrotropy=eps_xz,
        twist=PHI,
        twistMode="coupling",
    )


def make_nanopore_layer() -> rcwa.Layer:
    """Build the sampled Si hole-array layer; RCWASimulation precompiles it once."""

    return rcwa.circularPostLayer(
        period=(PERIOD, PERIOD),
        thickness=SI_HEIGHT,
        background=SI_INDEX**2,
        post=1.0,
        radius=HOLE_RADIUS,
        shape=(GRID, GRID),
        factorization=FACTORIZATION,
        name="Si nanopore array",
    )


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
        epsTransmission=1.0,
        precompile=True,
        cacheModes=True,
        workers=WORKERS,
    )


def make_layers() -> list[object]:
    return [
        make_nanopore_layer(),
        rcwa.homogeneousLayer(INAS_HEIGHT, inas_tensor, name="InAs"),
        rcwa.homogeneousLayer(AL_HEIGHT, al_tensor, name="Al mirror"),
    ]


def top_view_map() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    n = 320
    x = np.linspace(-PERIOD / 2, PERIOD / 2, n)
    y = np.linspace(-PERIOD / 2, PERIOD / 2, n)
    xx, yy = np.meshgrid(x, y)
    material = np.ones_like(xx, dtype=int)
    material[xx * xx + yy * yy <= HOLE_RADIUS * HOLE_RADIUS] = 0
    return material, (-PERIOD / 2, PERIOD / 2, -PERIOD / 2, PERIOD / 2)


def cross_section_map() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    nx = 360
    nz = 280
    x = np.linspace(-PERIOD / 2, PERIOD / 2, nx)
    total_height = SI_HEIGHT + INAS_HEIGHT + AL_HEIGHT
    z = np.linspace(0.0, total_height, nz)
    xx, zz = np.meshgrid(x, z)

    material = np.zeros_like(xx, dtype=int)
    in_al = zz <= AL_HEIGHT
    in_inas = (zz > AL_HEIGHT) & (zz <= AL_HEIGHT + INAS_HEIGHT)
    in_si = zz > AL_HEIGHT + INAS_HEIGHT
    in_hole = in_si & (np.abs(xx) <= HOLE_RADIUS)
    material[in_al] = 3
    material[in_inas] = 2
    material[in_si] = 1
    material[in_hole] = 0
    return material, (-PERIOD / 2, PERIOD / 2, 0.0, total_height)


def draw_structure_guides(axis: plt.Axes) -> None:
    total_height = SI_HEIGHT + INAS_HEIGHT + AL_HEIGHT
    si_bottom = AL_HEIGHT + INAS_HEIGHT
    axis.add_patch(
        Rectangle(
            (-HOLE_RADIUS, si_bottom),
            HOLE_DIAMETER,
            SI_HEIGHT,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.0,
        )
    )
    for z_value in (AL_HEIGHT, si_bottom, total_height):
        axis.axhline(z_value, color="black", linewidth=0.7, alpha=0.55)


def plot_spectrum(spectrum: dict[str, object]) -> Path:
    wavelengths = spectrum["wavelengths"]
    top_map, top_extent = top_view_map()
    cross_map, cross_extent = cross_section_map()
    cmap = ListedColormap(["white", "#7da7d9", "#f6c65b", "#9c9c9c"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)

    axes[0, 0].imshow(top_map, origin="lower", extent=top_extent, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
    axes[0, 0].add_patch(Circle((0.0, 0.0), HOLE_RADIUS, fill=False, edgecolor="black", linestyle="--", linewidth=1.0))
    axes[0, 0].set_title("(a) top view")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")

    image = axes[1, 0].imshow(
        cross_map,
        origin="lower",
        extent=cross_extent,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=3,
    )
    draw_structure_guides(axes[1, 0])
    axes[1, 0].set_title("(b) x-z cross-section")
    axes[1, 0].set_xlabel("x (um)")
    axes[1, 0].set_ylabel("z (um)")
    colorbar = fig.colorbar(image, ax=axes[:, 0], ticks=[0.375, 1.125, 1.875, 2.625], shrink=0.78)
    colorbar.ax.set_yticklabels(["air", "Si", "InAs", "Al"])

    for axis, polarization in zip((axes[0, 1], axes[1, 1]), POLARIZATIONS):
        data = spectrum[polarization]
        axis.plot(wavelengths, data["absorptivity"], label="alpha(+theta)")
        axis.plot(wavelengths, data["emissivity"], label="epsilon(+theta)=A(-theta)")
        axis.plot(wavelengths, data["nonreciprocity"], "k:", linewidth=2.0, label="eta")
        axis.set_title(f"{polarization} spectrum")
        axis.set_ylabel("power")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    for axis in (axes[0, 1], axes[1, 1]):
        axis.axvline(PAPER_TM_RESONANCE_UM, color="tab:purple", linestyle="--", linewidth=0.9, alpha=0.5)
        axis.axvline(PAPER_TE_ABSORPTION_UM, color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.5)
        axis.axvline(PAPER_TE_EMISSION_UM, color="tab:orange", linestyle="--", linewidth=0.9, alpha=0.5)
        axis.set_xlabel("wavelength (um)")

    fig.suptitle(
        "Fang 2023 dual-polarization near-normal nonreciprocal radiator: "
        f"theta={math.degrees(THETA):.1f} deg, B={B_FIELD:g} T, order={ORDER}, grid={GRID}"
    )
    output = ROOT / "outputs" / "fang2023_dual_polarization_near_normal_rcwa.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return output


def print_peak_summary(spectrum: dict[str, object]) -> None:
    wavelengths = spectrum["wavelengths"]
    for polarization in POLARIZATIONS:
        data = spectrum[polarization]
        for key in ("absorptivity", "emissivity", "nonreciprocity"):
            values = np.asarray(data[key])
            index = int(np.nanargmax(values))
            print(f"{polarization} max {key}: lambda={wavelengths[index]:.5f} um, value={values[index]:.6f}")


def main() -> None:
    print(
        "RCWA "
        f"order={ORDER}, grid={GRID}, points={POINTS}, truncation={TRUNCATION}, "
        f"factorization={FACTORIZATION}, backend={BACKEND}, precision={PRECISION}, workers={WORKERS}"
    )
    print(
        "Fang 2023 geometry: "
        f"d={PERIOD} um, h1={SI_HEIGHT} um, h2={INAS_HEIGHT} um, h3={AL_HEIGHT} um, "
        f"w={HOLE_DIAMETER:.3f} um, f={FILL_RATIO}, theta={math.degrees(THETA):.1f} deg, B={B_FIELD:g} T"
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
    print_peak_summary(spectrum)
    figure = plot_spectrum(spectrum)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
