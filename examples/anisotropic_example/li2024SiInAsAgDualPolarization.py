from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.constants import c, e, m_e

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


# Runtime knobs. Increase ORDER/GRID/POINTS for convergence studies.
ORDER = env_int("RCWA_LI2024_ORDER", 6)
TRUNCATION = os.environ.get("RCWA_LI2024_TRUNCATION", "circular")
BACKEND = os.environ.get("RCWA_LI2024_BACKEND", "cuda")
PRECISION = os.environ.get("RCWA_LI2024_PRECISION", "complex128")
FACTORIZATION = os.environ.get("RCWA_LI2024_FACTORIZATION", "auto")
GRID = env_int("RCWA_LI2024_GRID", 512)
ANALYTIC_GEOMETRY = env_bool("RCWA_LI2024_ANALYTIC_GEOMETRY", True)
NORMAL_VECTOR_RESOLUTION = env_int("RCWA_LI2024_NORMAL_VECTOR_RESOLUTION", 512)
WORKERS = env_int("RCWA_LI2024_WORKERS", 1)
POINTS = env_int("RCWA_LI2024_POINTS", 201)
SHOW = env_bool("RCWA_LI2024_SHOW", True)

# Li, Wang, and Wu, Applied Materials Today 39, 102345 (2024),
# "Si/InAs/Ag metamaterial for strong nonreciprocal thermal emitter with
# dual polarization under a 0.9 T magnetic field", optimized Fig. 2 case.
# All geometric lengths are in um.
PERIOD = 8.274
SI_WIDTH = 2.914
SI_HEIGHT = 3.149
INAS_HEIGHT = 2.177
AG_HEIGHT = 1.0

# Illumination and spectrum ranges from Figs. 2(a,b).
B_FIELD = 0.9
THETA = math.radians(26.0)
PHI = 0.0
TE_WAVELENGTHS = np.linspace(15.0, 15.4, POINTS)
TM_WAVELENGTHS = np.linspace(15.2, 15.6, POINTS)
POLARIZATIONS = ("TE", "TM")

# Paper guide values from the abstract/text and field-distribution figures.
PAPER_TE_PEAK_UM = 15.145
PAPER_TM_PEAK_UM = 15.422
PAPER_TE_NONRECIPROCITY = 0.902
PAPER_TM_NONRECIPROCITY = 0.892

# Material parameters used by related Si/InAs/Ag nonreciprocal thermal-emitter
# papers: Si is taken as nondispersive, Ag as Drude, and magnetized InAs as the
# Voigt tensor with B along y.
SI_INDEX = 3.48
AG_EPS_INF = 3.4
AG_PLASMA = 1.39e16
AG_GAMMA = 2.7e13
INAS_EPS_INF = 12.37
INAS_PLASMA = 2.7396e14
INAS_GAMMA = 1.55e11
INAS_MASS = 0.033 * m_e

# Change to +1 if matching a source that uses the opposite exp(+/-i omega t)
# convention or reverses the plotted magnetic-field direction.
INAS_HALL_SIGN = env_float("RCWA_LI2024_HALL_SIGN", -1.0)


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def ag_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = AG_EPS_INF - AG_PLASMA**2 / (omega**2 + 1j * AG_GAMMA * omega)
    return scalar_tensor(epsilon)


def inas_tensor(wavelength_um: float) -> np.ndarray:
    """Return the magnetized InAs tensor for B along y."""

    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    cyclotron = e * B_FIELD / INAS_MASS
    denominator = omega * ((omega + 1j * INAS_GAMMA) ** 2 - cyclotron**2)

    eps_xx = INAS_EPS_INF - INAS_PLASMA**2 * (omega + 1j * INAS_GAMMA) / denominator
    eps_yy = INAS_EPS_INF - INAS_PLASMA**2 / (omega**2 + 1j * INAS_GAMMA * omega)
    eps_xz = INAS_HALL_SIGN * 1j * INAS_PLASMA**2 * cyclotron / denominator
    return rcwa.gyrotropicXzTensor(
        epsilonParallel=eps_xx,
        epsilonY=eps_yy,
        gyrotropy=eps_xz,
        twist=PHI,
        twistMode="coupling",
    )


def make_cuboid_layer() -> rcwa.Layer:
    """Build the Si cuboid-array layer."""

    return rcwa.rectangularPostLayer(
        period=(PERIOD, PERIOD),
        thickness=SI_HEIGHT,
        background=1.0,
        post=SI_INDEX**2,
        size=(SI_WIDTH, SI_WIDTH),
        shape=(GRID, GRID),
        factorization=FACTORIZATION,
        analytic=ANALYTIC_GEOMETRY,
        normalVectorResolution=NORMAL_VECTOR_RESOLUTION,
        name="Si cuboid array",
    )


def make_config(wavelengths: np.ndarray | None = None) -> rcwa.SimulationConfig:
    del wavelengths
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
        make_cuboid_layer(),
        rcwa.homogeneousLayer(INAS_HEIGHT, inas_tensor, name="InAs"),
        rcwa.homogeneousLayer(AG_HEIGHT, ag_tensor, name="Ag reflector"),
    ]


def solve_dual_spectrum() -> dict[str, object]:
    layers = make_layers()
    te = rcwa.solveSpectrum(
        make_config(TE_WAVELENGTHS),
        layers,
        TE_WAVELENGTHS,
        theta=THETA,
        phi=PHI,
        polarizations=("TE",),
        bidirectional=True,
    )
    tm = rcwa.solveSpectrum(
        make_config(TM_WAVELENGTHS),
        layers,
        TM_WAVELENGTHS,
        theta=THETA,
        phi=PHI,
        polarizations=("TM",),
        bidirectional=True,
    )
    return {
        "TE": te["TE"],
        "TM": tm["TM"],
        "wavelengths": {"TE": TE_WAVELENGTHS.copy(), "TM": TM_WAVELENGTHS.copy()},
    }


def top_view_map() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    n = 320
    x = np.linspace(-PERIOD / 2, PERIOD / 2, n)
    y = np.linspace(-PERIOD / 2, PERIOD / 2, n)
    xx, yy = np.meshgrid(x, y)
    material = np.zeros_like(xx, dtype=int)
    inside = (np.abs(xx) <= SI_WIDTH / 2) & (np.abs(yy) <= SI_WIDTH / 2)
    material[inside] = 1
    return material, (-PERIOD / 2, PERIOD / 2, -PERIOD / 2, PERIOD / 2)


def cross_section_map() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    nx = 360
    nz = 300
    x = np.linspace(-PERIOD / 2, PERIOD / 2, nx)
    total_height = SI_HEIGHT + INAS_HEIGHT + AG_HEIGHT
    z = np.linspace(0.0, total_height, nz)
    xx, zz = np.meshgrid(x, z)

    material = np.zeros_like(xx, dtype=int)
    in_ag = zz <= AG_HEIGHT
    in_inas = (zz > AG_HEIGHT) & (zz <= AG_HEIGHT + INAS_HEIGHT)
    in_si = (zz > AG_HEIGHT + INAS_HEIGHT) & (np.abs(xx) <= SI_WIDTH / 2)
    material[in_ag] = 3
    material[in_inas] = 2
    material[in_si] = 1
    return material, (-PERIOD / 2, PERIOD / 2, 0.0, total_height)


def draw_cross_section_guides(axis: plt.Axes) -> None:
    inas_top = AG_HEIGHT + INAS_HEIGHT
    total_height = AG_HEIGHT + INAS_HEIGHT + SI_HEIGHT
    axis.add_patch(
        Rectangle(
            (-SI_WIDTH / 2, inas_top),
            SI_WIDTH,
            SI_HEIGHT,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.0,
        )
    )
    for z_value in (AG_HEIGHT, inas_top, total_height):
        axis.axhline(z_value, color="black", linewidth=0.8, linestyle="--", alpha=0.55)


def plot_spectrum_panel(
    axis: plt.Axes,
    wavelengths: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    title: str,
    guide_wavelength: float,
    guide_eta: float,
) -> None:
    axis.plot(wavelengths, data["absorptivity"], color="black", linewidth=2.0, label="alpha, A(+theta)")
    axis.plot(wavelengths, data["emissivity"], color="tab:red", linewidth=2.0, linestyle="--", label="e, A(-theta)")
    axis.plot(
        wavelengths,
        data["nonreciprocity"],
        color="tab:blue",
        linewidth=2.0,
        linestyle=":",
        label="eta",
    )
    axis.axvline(guide_wavelength, color="0.35", linestyle="--", linewidth=0.9, alpha=0.6)
    axis.axhline(guide_eta, color="0.35", linestyle=":", linewidth=0.8, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel("Wavelength (um)")
    axis.set_ylabel("alpha, e, eta")
    axis.set_ylim(-0.03, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)


def plot_result(spectrum: dict[str, object]) -> Path:
    fig = plt.figure(figsize=(11.0, 8.4), constrained_layout=True)
    axes = fig.subplots(2, 2)
    cmap = ListedColormap(["#f5f5f5", "#3e7b35", "#d97818", "#4f94c4"])

    top, top_extent = top_view_map()
    top_image = axes[0, 0].imshow(top, extent=top_extent, origin="lower", cmap=cmap, vmin=0, vmax=3)
    axes[0, 0].add_patch(
        Rectangle(
            (-SI_WIDTH / 2, -SI_WIDTH / 2),
            SI_WIDTH,
            SI_WIDTH,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.0,
        )
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("(a) top view")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    colorbar = fig.colorbar(top_image, ax=axes[0, 0], ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["air", "Si", "InAs", "Ag"])

    section, section_extent = cross_section_map()
    axes[0, 1].imshow(
        section,
        extent=section_extent,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=3,
    )
    draw_cross_section_guides(axes[0, 1])
    axes[0, 1].set_title("(b) x-z cross section")
    axes[0, 1].set_xlabel("x (um)")
    axes[0, 1].set_ylabel("height from bottom (um)")

    wavelengths = spectrum["wavelengths"]
    plot_spectrum_panel(
        axes[1, 0],
        wavelengths["TE"],
        spectrum["TE"],
        title="(c) TE spectrum",
        guide_wavelength=PAPER_TE_PEAK_UM,
        guide_eta=PAPER_TE_NONRECIPROCITY,
    )
    plot_spectrum_panel(
        axes[1, 1],
        wavelengths["TM"],
        spectrum["TM"],
        title="(d) TM spectrum",
        guide_wavelength=PAPER_TM_PEAK_UM,
        guide_eta=PAPER_TM_NONRECIPROCITY,
    )

    fig.suptitle(
        "Li 2024 Si/InAs/Ag dual-polarization nonreciprocal emitter: "
        f"theta={math.degrees(THETA):.0f} deg, B={B_FIELD:g} T, order={ORDER}, "
        f"geometry={'analytic' if ANALYTIC_GEOMETRY else f'grid {GRID}'}"
    )
    output = ROOT / "outputs" / "li2024_si_inas_ag_dual_polarization_rcwa.png"
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
        wave = wavelengths[polarization]
        for key in ("absorptivity", "emissivity", "nonreciprocity"):
            values = np.asarray(data[key])
            index = int(np.nanargmax(values))
            print(f"{polarization} max {key}: lambda={wave[index]:.5f} um, value={values[index]:.6f}")


def main() -> None:
    print(
        "RCWA "
        f"order={ORDER}, geometry={'analytic' if ANALYTIC_GEOMETRY else f'grid {GRID}'}, "
        f"normal_vector_resolution={NORMAL_VECTOR_RESOLUTION}, points={POINTS}, truncation={TRUNCATION}, "
        f"factorization={FACTORIZATION}, backend={BACKEND}, precision={PRECISION}, "
        f"workers={WORKERS}, hall_sign={INAS_HALL_SIGN:g}"
    )
    print(
        "Li 2024 geometry: "
        f"d={PERIOD} um, w={SI_WIDTH} um, h1={SI_HEIGHT} um, h2={INAS_HEIGHT} um, "
        f"h3={AG_HEIGHT} um, theta={math.degrees(THETA):.1f} deg, B={B_FIELD:g} T"
    )
    start = time.perf_counter()
    spectrum = solve_dual_spectrum()
    print_peak_summary(spectrum)
    figure = plot_result(spectrum)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
