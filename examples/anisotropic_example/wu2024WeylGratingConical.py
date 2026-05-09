from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.constants import c, e, epsilon_0, hbar, k

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa
from rcwa3d_anisotropic.phase import forwardKz, planeWaveFields


# Runtime knobs. Increase ORDER/GRID_X/POINTS for convergence studies.
ORDER = 15
TRUNCATION = "rectangular"
BACKEND = "cuda"
PRECISION = "complex128"  # Use "complex64" or "mixed" for fast spectrum scans.
FACTORIZATION = "auto"
GRID_X = 512
GRID_Y = 8
WORKERS = 1
POINTS = 201
SHOW = True

# Wu & Qing 2024, Fig. 2 TE-optimized conical-incidence structure.
# All geometric lengths are in um.
PERIOD = 4.8
SI_WIDTH = 3.0
SI_HEIGHT = 2.38
WEYL_HEIGHT = 0.63
AG_HEIGHT = 1.0

THETA = math.radians(56.0)
PHI = math.radians(33.0)
WAVELENGTHS = np.linspace(11, 13, POINTS)
POLARIZATIONS = ("TE",)
TARGET_WAVELENGTH = 12.0
FIELD_SHAPE = (220, 320)
FIELD_COLOR_MAX = 20.0

# Literature guide values from the text near Fig. 2.
PAPER_TARGET_ABSORPTION_LT = 0.049
PAPER_TARGET_EMISSION_GT = 0.998
PAPER_TARGET_NONRECIPROCITY = 0.95

# Material parameters.
SI_EPSILON = 11.9
SIO2_INDEX = 1.45
AG_EPS_INF = 3.4
AG_PLASMA = 1.39e16
AG_GAMMA = 2.7e13

# Weyl semimetal parameters copied from BerreMueller_example.py.
WEYL_EPS_B = 6.2
WEYL_B = 2e9
WEYL_VF = 0.83e5
WEYL_TAU = 1000e-15
WEYL_TEMPERATURE = 300.0
WEYL_EF_EV = 0.15
WEYL_CUTOFF = 3.0
WEYL_POINTS = 2.0
WEYL_INTEGRATION_POINTS = 1500
WEYL_HALL_SIGN = -1.0

EF = WEYL_EF_EV * e
RS = e**2 / (4 * np.pi * epsilon_0 * hbar * WEYL_VF)
XI = np.linspace(0.0, WEYL_CUTOFF, WEYL_INTEGRATION_POINTS)


def fermi_difference(energy: complex | np.ndarray) -> complex | np.ndarray:
    return 1 / (np.exp((-energy - EF) / (k * WEYL_TEMPERATURE)) + 1) - 1 / (
        np.exp((energy - EF) / (k * WEYL_TEMPERATURE)) + 1
    )


G_XI = fermi_difference(EF * XI)


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def ag_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = AG_EPS_INF - AG_PLASMA**2 / (omega**2 + 1j * AG_GAMMA * omega)
    return scalar_tensor(epsilon)


def weyl_diagonal_and_hall(wavelength_um: float) -> tuple[complex, complex]:
    """Return (epsilon_d, epsilon_a) from the BerreMueller Weyl model."""

    omega = complex(2 * np.pi * c / (float(wavelength_um) * 1e-6))
    epsilon_a = WEYL_B * e**2 / (2 * np.pi**2 * hbar * epsilon_0 * omega)
    omega_bar = hbar * (omega + 1j / WEYL_TAU) / EF

    g_omega = complex(fermi_difference(EF * omega_bar / 2.0))
    f1 = epsilon_0 * RS * WEYL_POINTS * EF / (6 * hbar) * omega_bar * g_omega
    f2 = 1j * epsilon_0 * RS * WEYL_POINTS * EF / (6 * np.pi * hbar)
    f3 = (1 + (np.pi**2 / 3) * (k * WEYL_TEMPERATURE / EF) ** 2) * 4.0 / omega_bar

    integrand = ((G_XI - g_omega) / (omega_bar**2 - 4.0 * XI**2)) * XI
    f4 = 8.0 * omega_bar * np.trapz(integrand, XI)
    sigma = f1 + f2 * (f3 + f4)
    epsilon_d = WEYL_EPS_B + 1j * sigma / (omega * epsilon_0)
    return epsilon_d, epsilon_a


def weyl_tensor(wavelength_um: float) -> np.ndarray:
    """Weyl tensor extracted from BerreMueller_example.py.

    BerreMueller_example.py uses an xz Hall coupling.  The sign below is the
    node-separation convention that makes A(+theta) low and e(-theta) high for
    the Fig. 2 geometry in this RCWA coordinate system.
    """

    epsilon_d, epsilon_a = weyl_diagonal_and_hall(wavelength_um)
    epsilon_a *= WEYL_HALL_SIGN
    return np.array(
        [
            [epsilon_d, 0.0, 1j * epsilon_a],
            [0.0, epsilon_d, 0.0],
            [-1j * epsilon_a, 0.0, epsilon_d],
        ],
        dtype=complex,
    )


def make_grating_layer() -> rcwa.Layer:
    """Build the 1D Si grating layer; RCWASimulation precompiles it once."""

    pattern = rcwa.SampledPattern(
        period=(PERIOD, PERIOD),
        shape=(GRID_Y, GRID_X),
        background=1.0,
        name="Si grating",
    )
    pattern.stripes(fillFraction=SI_WIDTH / PERIOD, material=SI_EPSILON, axis="x")
    return pattern.toLayer(SI_HEIGHT, factorization=FACTORIZATION)


def make_simulation() -> rcwa.RCWASimulation:
    return rcwa.buildSimulation(make_config(), make_layers())


def make_config() -> rcwa.SimulationConfig:
    return rcwa.SimulationConfig(
        period=(PERIOD, PERIOD),
        orders=(ORDER, 0),
        truncation=TRUNCATION,
        backend=BACKEND,
        precision=PRECISION,
        epsIncident=1.0,
        epsTransmission=SIO2_INDEX**2,
        precompile=True,
        cacheModes=True,
        workers=WORKERS,
    )


def make_layers() -> list[object]:
    return [
        make_grating_layer(),
        rcwa.homogeneousLayer(WEYL_HEIGHT, weyl_tensor, name="Weyl semimetal"),
        rcwa.homogeneousLayer(AG_HEIGHT, ag_tensor, name="Ag mirror"),
    ]


def structure_map() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    nx = 360
    nz = 260
    x = np.linspace(-PERIOD / 2, PERIOD / 2, nx)
    z_total = SI_HEIGHT + WEYL_HEIGHT + AG_HEIGHT
    z = np.linspace(0.0, z_total, nz)
    xx, zz = np.meshgrid(x, z)

    material = np.zeros_like(xx, dtype=int)
    in_ag = zz <= AG_HEIGHT
    in_weyl = (zz > AG_HEIGHT) & (zz <= AG_HEIGHT + WEYL_HEIGHT)
    in_si = (zz > AG_HEIGHT + WEYL_HEIGHT) & (np.abs(xx) <= SI_WIDTH / 2)
    material[in_ag] = 3
    material[in_weyl] = 2
    material[in_si] = 1
    return material, (-PERIOD / 2, PERIOD / 2, 0.0, z_total)


def incident_h_magnitude(theta: float) -> float:
    kx0 = np.sin(theta) * np.cos(PHI)
    ky0 = np.sin(theta) * np.sin(PHI)
    kz0 = forwardKz(1.0 - kx0**2 - ky0**2)[()]
    s_field, p_field = planeWaveFields(kx0, ky0, kz0, 1.0)
    hz0 = kx0 * s_field[1] - ky0 * s_field[0]
    return float(np.sqrt(np.real(np.abs(s_field[2]) ** 2 + np.abs(s_field[3]) ** 2 + np.abs(hz0) ** 2)))


def layer_tangential_fourier_fields(layer: rcwa.LayerFieldSolution, z: float) -> dict[str, np.ndarray]:
    if z < -1e-12 or z > layer.thickness + 1e-12:
        raise ValueError("z must be inside the selected layer")
    n_orders = layer.mx.size
    k0 = 2 * np.pi / layer.wavelength
    phase = np.exp(1j * layer.qValues * k0 * z)
    values = layer.modeMatrix @ (phase * layer.coefficients)
    ex = values[:n_orders]
    ey = values[n_orders : 2 * n_orders]
    hx = values[2 * n_orders : 3 * n_orders]
    hy = values[3 * n_orders :]
    hz = layer.kx * ey - layer.ky * ex
    return {"Hx": hx, "Hy": hy, "Hz": hz}


def magnetic_field_xz(
    result: rcwa.RCWAResult,
    theta: float,
    *,
    shape: tuple[int, int] = FIELD_SHAPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not result.layerSolutions:
        raise ValueError("result does not contain layer fields; call solveFields() for field maps")

    reference = result.layerSolutions[0]
    total_thickness = sum(layer.thickness for layer in result.layerSolutions)
    layer_starts = np.concatenate([[0.0], np.cumsum([layer.thickness for layer in result.layerSolutions[:-1]])])
    layer_ends = np.cumsum([layer.thickness for layer in result.layerSolutions])

    nz, nx = shape
    x_values = np.linspace(-PERIOD / 2, PERIOD / 2, nx)
    z_values = np.linspace(0.0, total_thickness, nz)
    xx, zz_top = np.meshgrid(x_values, z_values)
    h_map = np.zeros_like(xx, dtype=float)
    k0 = 2 * np.pi / reference.wavelength
    h0 = max(incident_h_magnitude(theta), 1e-300)
    lateral_phase = np.exp(1j * k0 * (reference.kx[:, None] * x_values[None, :]))

    for row, z_from_top in enumerate(z_values):
        layer_index = min(int(np.searchsorted(layer_ends, z_from_top, side="right")), len(result.layerSolutions) - 1)
        local_z = float(z_from_top - layer_starts[layer_index])
        fields = layer_tangential_fourier_fields(result.layerSolutions[layer_index], local_z)
        hx = np.sum(fields["Hx"][:, None] * lateral_phase, axis=0)
        hy = np.sum(fields["Hy"][:, None] * lateral_phase, axis=0)
        hz = np.sum(fields["Hz"][:, None] * lateral_phase, axis=0)
        h_map[row, :] = np.sqrt(np.maximum(np.real(np.abs(hx) ** 2 + np.abs(hy) ** 2 + np.abs(hz) ** 2), 0.0)) / h0

    height_from_bottom = total_thickness - zz_top
    # The solver's layer-local coordinate starts at the incident/top side, but
    # the paper-style plots use height measured upward from the Ag bottom.
    return xx[::-1, :], height_from_bottom[::-1, :], h_map[::-1, :]


def solve_magnetic_fields(
    simulation: rcwa.RCWASimulation,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    forward = simulation.solveFields(
        TARGET_WAVELENGTH,
        theta=THETA,
        phi=PHI,
        polarization="TE",
    )
    backward = simulation.solveFields(
        TARGET_WAVELENGTH,
        theta=-THETA,
        phi=PHI,
        polarization="TE",
    )
    return magnetic_field_xz(forward, THETA), magnetic_field_xz(backward, -THETA)


def draw_structure_outline(axis: plt.Axes, *, color: str = "white") -> None:
    total = SI_HEIGHT + WEYL_HEIGHT + AG_HEIGHT
    weyl_bottom = AG_HEIGHT
    grating_bottom = AG_HEIGHT + WEYL_HEIGHT
    axis.add_patch(
        Rectangle(
            (-SI_WIDTH / 2, grating_bottom),
            SI_WIDTH,
            SI_HEIGHT,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
            linestyle="--",
        )
    )
    for y_value in (weyl_bottom, grating_bottom, total):
        axis.axhline(y_value, color=color, linewidth=0.9, linestyle="--", alpha=0.9)


def plot_result(
    spectrum: dict[str, object],
    field_maps: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    wavelengths = spectrum["wavelengths"]
    data = spectrum["TE"]
    nearest = int(np.argmin(np.abs(wavelengths - TARGET_WAVELENGTH)))
    peak = int(np.argmax(data["nonreciprocity"]))

    fig = plt.figure(figsize=(11.2, 8.2), constrained_layout=True)
    axes = fig.subplots(2, 2)

    material, extent = structure_map()
    cmap = ListedColormap(["#f3f6fa", "#3b73b9", "#7a3f9d", "#b6b6b6"])
    image = axes[0, 0].imshow(material, extent=extent, interpolation="nearest", aspect="auto", cmap=cmap, vmin=0, vmax=3)
    colorbar = fig.colorbar(image, ax=axes[0, 0], ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["air", "Si", "WS", "Ag"])
    draw_structure_outline(axes[0, 0], color="black")
    axes[0, 0].set_title("(a) Structure")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("height from bottom (um)")

    axes[0, 1].plot(wavelengths, data["absorptivity"], label="A(+theta), absorption")
    axes[0, 1].plot(wavelengths, data["emissivity"], label="e(-theta), emission")
    axes[0, 1].plot(wavelengths, data["nonreciprocity"], label="eta")
    axes[0, 1].axvline(TARGET_WAVELENGTH, color="black", linestyle=":", linewidth=1.0, label="paper target 12 um")
    axes[0, 1].scatter([TARGET_WAVELENGTH], [PAPER_TARGET_ABSORPTION_LT], marker="v", color="tab:blue", label="paper A < 0.049")
    axes[0, 1].scatter([TARGET_WAVELENGTH], [PAPER_TARGET_EMISSION_GT], marker="^", color="tab:orange", label="paper e > 0.998")
    axes[0, 1].scatter(
        [TARGET_WAVELENGTH],
        [PAPER_TARGET_NONRECIPROCITY],
        marker="x",
        color="tab:green",
        label="paper eta approx 0.95",
    )
    axes[0, 1].set_title(
        f"(b) TE spectrum, theta={math.degrees(THETA):.0f} deg, phi={math.degrees(PHI):.0f} deg, order={ORDER}"
    )
    axes[0, 1].set_xlabel("Wavelength (um)")
    axes[0, 1].set_ylabel("Absorption / emission / eta")
    axes[0, 1].set_ylim(-0.03, 1.05)
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best", ncols=2, fontsize=7.5)

    field_titles = (
        f"(c) |H|/|H0|, lambda={TARGET_WAVELENGTH:g} um, theta=+{math.degrees(THETA):.0f} deg",
        f"(d) |H|/|H0|, lambda={TARGET_WAVELENGTH:g} um, theta=-{math.degrees(THETA):.0f} deg",
    )
    for axis, field, title in zip((axes[1, 0], axes[1, 1]), field_maps, field_titles):
        x_field, y_field, h_field = field
        field_image = axis.imshow(
            h_field,
            extent=(float(x_field.min()), float(x_field.max()), float(y_field.min()), float(y_field.max())),
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            cmap="hot",
            vmin=0.0,
            vmax=FIELD_COLOR_MAX,
        )
        draw_structure_outline(axis)
        axis.set_title(title)
        axis.set_xlabel("x (um)")
        axis.set_ylabel("height from bottom (um)")
        fig.colorbar(field_image, ax=axis, label="|H|/|H0|")

    fig.suptitle(
        "Peak sample: "
        f"lambda={wavelengths[peak]:.4f} um, "
        f"A={data['absorptivity'][peak]:.4f}, "
        f"e={data['emissivity'][peak]:.4f}, "
        f"eta={data['nonreciprocity'][peak]:.4f}; "
        "nearest 12 um: "
        f"A={data['absorptivity'][nearest]:.4f}, "
        f"e={data['emissivity'][nearest]:.4f}, "
        f"eta={data['nonreciprocity'][nearest]:.4f}"
    )
    output = ROOT / "outputs" / "wu2024_weyl_grating_conical_rcwa.png"
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
        f"order={ORDER}, grid=({GRID_Y}, {GRID_X}), points={POINTS}, truncation={TRUNCATION}, "
        f"factorization={FACTORIZATION}, backend={BACKEND}, precision={PRECISION}, workers={WORKERS}"
    )
    print(
        "Wu 2024 geometry: "
        f"d={PERIOD} um, w={SI_WIDTH} um, h={SI_HEIGHT} um, hWS={WEYL_HEIGHT} um, "
        f"hAg={AG_HEIGHT} um, theta={math.degrees(THETA):.1f} deg, phi={math.degrees(PHI):.1f} deg"
    )
    start = time.perf_counter()
    simulation = make_simulation()
    spectrum = simulation.spectrum(
        WAVELENGTHS,
        theta=THETA,
        phi=PHI,
        polarizations=POLARIZATIONS,
        bidirectional=True,
        workers=WORKERS,
    )
    field_maps = solve_magnetic_fields(simulation)
    data = spectrum["TE"]
    nearest = int(np.argmin(np.abs(spectrum["wavelengths"] - TARGET_WAVELENGTH)))
    peak = int(np.argmax(data["nonreciprocity"]))
    print(
        f"Nearest target point lambda={spectrum['wavelengths'][nearest]:.4f} um: "
        f"A={data['absorptivity'][nearest]:.6f}, "
        f"e={data['emissivity'][nearest]:.6f}, "
        f"eta={data['nonreciprocity'][nearest]:.6f}"
    )
    print(
        f"Peak eta point lambda={spectrum['wavelengths'][peak]:.4f} um: "
        f"A={data['absorptivity'][peak]:.6f}, "
        f"e={data['emissivity'][peak]:.6f}, "
        f"eta={data['nonreciprocity'][peak]:.6f}"
    )
    print(
        f"Field maxima at lambda={TARGET_WAVELENGTH:.4f} um: "
        f"|H|/|H0|(+theta)={float(np.max(field_maps[0][2])):.3f}, "
        f"|H|/|H0|(-theta)={float(np.max(field_maps[1][2])):.3f}"
    )
    figure = plot_result(spectrum, field_maps)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
