from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


# Runtime knobs. Increase ORDER/GRID/POINTS for convergence studies.
ORDER = 8
TRUNCATION = "rectangular"
BACKEND = "cuda"
PRECISION = "complex128"  # Use "complex64" or "mixed" for fast spectrum scans.
FACTORIZATION = "auto"
GRID = 512
WORKERS = 14
POINTS = 250
SHOW = True

# Device geometry (um).
PERIOD = 7.7
GE_HEIGHT = 8.13
INAS_HEIGHT = 5.44
AG_HEIGHT = 1.0
GE_WIDTH = 6.16
HOLE_DIAMETER = 3.08

# Illumination and sweep.
B_FIELD = 0.8
THETA = math.radians(4.7)
PHI = 0.0
WAVELENGTHS = np.linspace(17.30, 18.10, POINTS)
POLARIZATIONS = ("TE", "TM")

# Material parameters.
GE_N = 4.0
SIO2_N = 1.45
AG_EPS_INF = 3.4
AG_PLASMA = 1.39e16
AG_GAMMA = 2.7e13
INAS_EPS_INF = 12.37
INAS_GAMMA = 1.55e11
INAS_DENSITY = 7.8e17 * 1e6
INAS_MASS = 0.033 * m_e


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def ag_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = AG_EPS_INF - AG_PLASMA**2 / (omega**2 + 1j * AG_GAMMA * omega)
    return scalar_tensor(epsilon)


def inas_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    cyclotron = e * B_FIELD / INAS_MASS
    plasma = np.sqrt(INAS_DENSITY * e**2 / (epsilon_0 * INAS_MASS))
    denominator = omega * ((omega + 1j * INAS_GAMMA) ** 2 - cyclotron**2)

    eps_xx = INAS_EPS_INF - plasma**2 * (omega + 1j * INAS_GAMMA) / denominator
    eps_yy = INAS_EPS_INF - plasma**2 / (omega**2 + 1j * INAS_GAMMA * omega)
    gyrotropy = -1j * plasma**2 * cyclotron / denominator
    return rcwa.gyrotropicXzTensor(
        epsilonParallel=eps_xx,
        epsilonY=eps_yy,
        gyrotropy=gyrotropy,
        twist=PHI,
        twistMode="coupling",
    )


def make_pattern_layer() -> rcwa.Layer:
    """Build the patterned Ge layer; RCWASimulation precompiles it once."""

    layer = rcwa.rectangularHollowPostLayer(
        period=(PERIOD, PERIOD),
        thickness=GE_HEIGHT,
        background=1.0,
        post=GE_N**2,
        size=(GE_WIDTH, GE_WIDTH),
        holeRadius=HOLE_DIAMETER / 2,
        shape=(GRID, GRID),
        factorization=FACTORIZATION,
        name="Ge hollow array",
    )
    return layer


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
        epsTransmission=SIO2_N**2,
        precompile=True,
        cacheModes=True,
        workers=WORKERS,
    )


def make_layers() -> list[object]:
    return [
        make_pattern_layer(),
        rcwa.homogeneousLayer(INAS_HEIGHT, inas_tensor, name="InAs"),
        rcwa.homogeneousLayer(AG_HEIGHT, ag_tensor, name="Ag"),
    ]


def plot_spectrum(spectrum: dict[str, object]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True, constrained_layout=True)
    wavelengths = spectrum["wavelengths"]
    for ax, polarization in zip(axes, ("TE", "TM")):
        data = spectrum[polarization]
        ax.plot(wavelengths, data["absorptivity"], label="A(+theta)")
        ax.plot(wavelengths, data["emissivity"], label="A(-theta)")
        ax.plot(wavelengths, data["nonreciprocity"], "--", label="|delta A|")
        ax.set_ylabel(polarization)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Wavelength (um)")
    fig.suptitle(f"Shi 2025 Ge/InAs hollow array, order={ORDER}, grid={GRID}")
    output = ROOT / "outputs" / "shi2025_ge_inas_hollow_array_rcwa.png"
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
