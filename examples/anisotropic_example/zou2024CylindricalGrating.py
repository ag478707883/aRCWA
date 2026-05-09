from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_e

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_anisotropic as rcwa


# Runtime knobs. Edit these constants directly for local experiments.
ORDER = 5
TRUNCATION = "circular"
BACKEND = "cuda"
PRECISION = "mixed"  # Use "complex64" or "mixed" for fast spectrum scans.
FACTORIZATION = "auto"
GRID = 512
WORKERS = 6
POINTS = 501
SHOW = True

# Device geometry (um).
PERIOD = 5.33
CYLINDER_DIAMETER = 2.13
CYLINDER_HEIGHT = 5.94
INAS_HEIGHT = 1.962
AG_HEIGHT = 0.5

# Illumination and sweep.
theta = math.radians(33.0)
phi = 0.0
B = 3.0
WAVELENGTHS = np.linspace(11.70, 12.00, POINTS)

# Material parameters.
SI_INDEX = 3.48
Ag_eps_inf = 3.4
Ag_wp = 1.39e16
Ag_gamma = 2.7e13
INAS_EPS_INF = 12.37
INAS_PLASMA = 2.7396e14
INAS_GAMMA = 1.55e11
INAS_MASS = 0.033 * m_e


def scalar_tensor(epsilon: complex) -> np.ndarray:
    return complex(epsilon) * np.eye(3, dtype=complex)


def Ag_tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    epsilon = Ag_eps_inf - Ag_wp**2 / (omega**2 + 1j * Ag_gamma * omega)
    return scalar_tensor(epsilon)


def InAs_Tensor(wavelength_um: float) -> np.ndarray:
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    cyclotron = e * B / INAS_MASS
    denominator = omega * ((omega + 1j * INAS_GAMMA) ** 2 - cyclotron**2)

    eps_xx = INAS_EPS_INF - INAS_PLASMA**2 * (omega + 1j * INAS_GAMMA) / denominator
    eps_yy = INAS_EPS_INF - INAS_PLASMA**2 / (omega**2 + 1j * INAS_GAMMA * omega)
    eps_xz = -1j * INAS_PLASMA**2 * cyclotron / denominator
    return rcwa.gyrotropicXzTensor(
        epsilonParallel=eps_xx,
        epsilonY=eps_yy,
        gyrotropy=eps_xz,
        twist=phi,
        twistMode="coupling",
    )


def make_pattern_layer() -> rcwa.Layer:
    """Build the patterned Si cylinder layer; RCWASimulation precompiles it once."""

    layer = rcwa.PatternLayer(
        period=(PERIOD, PERIOD),
        thickness=CYLINDER_HEIGHT,
        background=1.0,
        shape=(GRID, GRID),
        factorization=FACTORIZATION,
        name="Si cylindrical grating",
    )
    layer.circle(radius=CYLINDER_DIAMETER / 2, material=SI_INDEX**2)
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
        epsTransmission=1.0,
        precompile=True,
        cacheModes=True,
        workers=WORKERS,
    )


def make_layers() -> list[object]:
    return [
        make_pattern_layer(),
        rcwa.homogeneousLayer(INAS_HEIGHT, InAs_Tensor, name="InAs"),
        rcwa.homogeneousLayer(AG_HEIGHT, Ag_tensor, name="Ag"),
    ]


def plot_spectrum(spectrum: dict[str, object]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    for ax, polarization in zip(axes, ("TE", "TM")):
        data = spectrum[polarization]
        ax.plot(WAVELENGTHS, data["absorptivity"], label="A(+theta)")
        ax.plot(WAVELENGTHS, data["emissivity"], label="A(-theta)")
        ax.plot(WAVELENGTHS, data["nonreciprocity"], "--", label="|A(+theta)-A(-theta)|")
        ax.set_ylabel(polarization)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Wavelength (um)")
    fig.suptitle(f"Zou 2024 Ag-InAs-Si cylindrical grating, order={ORDER}, grid={GRID}")
    output = ROOT / "outputs" / "zou2024_cylindrical_grating_rcwa.png"
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
        theta=theta,
        phi=phi,
        polarizations=("TE", "TM"),
        bidirectional=True,
    )
    figure = plot_spectrum(spectrum)
    print(f"Saved: {figure}")
    print(f"Elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
