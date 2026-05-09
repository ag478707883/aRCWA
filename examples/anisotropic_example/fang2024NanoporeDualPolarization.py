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
ORDER = 5
TRUNCATION = "circular"
BACKEND = "cuda"
PRECISION = "complex128"  # Use "complex64" or "mixed" for fast spectrum scans.
FACTORIZATION = "auto"
GRID = 128
WORKERS = 1
POINTS = 101
SHOW = True

# Fang et al. 2024, Fig. 2 dual-polarization Si nanopore/InAs/Al emitter.
# All geometric lengths are in um.
PERIOD = 5.2
SI_HEIGHT = 5.0
INAS_HEIGHT = 2.0
AL_HEIGHT = 0.2
FILL_RATIO = 0.7
HOLE_DIAMETER = FILL_RATIO * PERIOD
HOLE_RADIUS = HOLE_DIAMETER / 2.0

# Illumination and sweep for Fig. 2.
B_FIELD = 4.0
THETA = math.radians(49.0)
PHI = 0.0
WAVELENGTHS = np.linspace(13.2, 13.4, POINTS)
POLARIZATIONS = ("TE", "TM")

# Literature guide wavelengths from the text near Fig. 2.
PAPER_TE_RESONANCES_UM = (13.314, 13.360)
PAPER_TM_RESONANCES_UM = (13.2, 13.4, 13.462)
PAPER_TE_NONRECIPROCITY = 0.75
PAPER_TM_NONRECIPROCITY = 0.90

# Material parameters.
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
    """Return Al Drude permittivity from Eq. (8) of the paper."""

    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = AL_EPS_INF - AL_PLASMA**2 / (omega * (omega + 1j * AL_GAMMA))
    return scalar_tensor(epsilon)


def inas_tensor(wavelength_um: float) -> np.ndarray:
    """Return the magnetized InAs tensor from Eqs. (1)-(4)."""

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
    """Build the sampled Si nanopore layer; RCWASimulation precompiles it once."""

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
        epsTransmission=SI_INDEX**2,
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
    nx = 320
    nz = 260
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


def draw_cross_section_guides(axis: plt.Axes) -> None:
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
        axis.axhline(z_value, color="black", linestyle="--", linewidth=0.9, alpha=0.75)


def plot_spectrum_panel(
    axis: plt.Axes,
    wavelengths: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    polarization: str,
    guide_wavelengths: tuple[float, ...],
    guide_eta: float,
) -> None:
    axis.plot(wavelengths, data["absorptivity"], label="A(+theta)")
    axis.plot(wavelengths, data["emissivity"], label="e(+theta)=A(-theta)")
    axis.plot(wavelengths, data["nonreciprocity"], "k:", linewidth=2.0, label="eta")
    for wavelength in guide_wavelengths:
        axis.axvline(wavelength, color="0.3", linestyle="--", linewidth=0.8, alpha=0.55)
    axis.axhline(guide_eta, color="0.35", linestyle=":", linewidth=0.8, alpha=0.5)
    axis.set_title(f"{polarization} spectrum")
    axis.set_xlabel("Wavelength (um)")
    axis.set_ylabel("Absorption / emission / eta")
    axis.set_ylim(-0.03, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)


def plot_result(spectrum: dict[str, object]) -> Path:
    wavelengths = spectrum["wavelengths"]
    fig = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    axes = fig.subplots(2, 2)

    cmap = ListedColormap(["#f3f6fa", "#3b73b9", "#7a3f9d", "#9a9a9a"])
    top_map, top_extent = top_view_map()
    top_image = axes[0, 0].imshow(
        top_map,
        extent=top_extent,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
    )
    axes[0, 0].add_patch(Circle((0.0, 0.0), HOLE_RADIUS, fill=False, edgecolor="black", linestyle="--", linewidth=1.1))
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("(a) Top view: Si nanopore unit cell")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    colorbar = fig.colorbar(top_image, ax=axes[0, 0], ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["air", "Si", "InAs", "Al"])

    section_map, section_extent = cross_section_map()
    axes[0, 1].imshow(
        section_map,
        extent=section_extent,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=3,
    )
    draw_cross_section_guides(axes[0, 1])
    axes[0, 1].set_title("(b) Cross section")
    axes[0, 1].set_xlabel("x (um)")
    axes[0, 1].set_ylabel("height from bottom (um)")

    plot_spectrum_panel(
        axes[1, 0],
        wavelengths,
        spectrum["TE"],
        polarization="(c) TE",
        guide_wavelengths=PAPER_TE_RESONANCES_UM,
        guide_eta=PAPER_TE_NONRECIPROCITY,
    )
    plot_spectrum_panel(
        axes[1, 1],
        wavelengths,
        spectrum["TM"],
        polarization="(d) TM",
        guide_wavelengths=PAPER_TM_RESONANCES_UM,
        guide_eta=PAPER_TM_NONRECIPROCITY,
    )

    fig.suptitle(
        "Fang 2024 Si nanopore/InAs/Al nonreciprocal radiator: "
        f"theta={math.degrees(THETA):.0f} deg, B={B_FIELD:g} T, order={ORDER}, grid={GRID}"
    )
    output = ROOT / "outputs" / "fang2024_nanopore_dual_polarization_rcwa.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
    return output


def summarize_polarization(label: str, spectrum: dict[str, np.ndarray]) -> str:
    eta_peak = int(np.argmax(spectrum["nonreciprocity"]))
    absorption_peak = int(np.argmax(spectrum["absorptivity"]))
    emission_peak = int(np.argmax(spectrum["emissivity"]))
    return (
        f"{label}: "
        f"max eta={spectrum['nonreciprocity'][eta_peak]:.4f} at {WAVELENGTHS[eta_peak]:.4f} um; "
        f"max A={spectrum['absorptivity'][absorption_peak]:.4f} at {WAVELENGTHS[absorption_peak]:.4f} um; "
        f"max e={spectrum['emissivity'][emission_peak]:.4f} at {WAVELENGTHS[emission_peak]:.4f} um"
    )


def main() -> None:
    print(
        "RCWA "
        f"order={ORDER}, grid={GRID}, points={POINTS}, truncation={TRUNCATION}, "
        f"factorization={FACTORIZATION}, backend={BACKEND}, precision={PRECISION}, workers={WORKERS}"
    )
    print(
        "Fang 2024 nanopore geometry: "
        f"d={PERIOD} um, h1={SI_HEIGHT} um, h2={INAS_HEIGHT} um, h3={AL_HEIGHT} um, "
        f"f={FILL_RATIO}, B={B_FIELD} T, theta={math.degrees(THETA):.1f} deg"
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
    print(summarize_polarization("TE", spectrum["TE"]))
    print(summarize_polarization("TM", spectrum["TM"]))
    figure = plot_result(spectrum)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
