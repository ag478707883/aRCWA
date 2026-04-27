from __future__ import annotations

import time
from pathlib import Path
import sys

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_isotropic as rcwa
from rcwa3d_isotropic.analytic import AnalyticRectangle


SHOW = True
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"

PERIOD_X = 0.5
PERIOD_Y = PERIOD_X
DUTY_CYCLE = 0.5
DEPTH = 0.6
ORDER = 10
ORDERS = (ORDER, 0)
NX = 1024
NY = 8

FREQUENCY_THZ = np.linspace(150.0, 300.0, 581)
THETA_CASES = [(0.0, 0.0), (30.0, np.deg2rad(30.0))]
POLARIZATION_CASES = [("TE", 1.0, 0.0), ("TM", 0.0, 1.0)]

EPS_COVER = 1.0
EPS_SUBSTRATE = 2.25
EPS_GRATING = 3.0**2
EPS_GROOVE = EPS_COVER

I3 = np.eye(3, dtype=complex)
EPS_COVER_TENSOR = EPS_COVER * I3
EPS_SUBSTRATE_TENSOR = EPS_SUBSTRATE * I3
EPS_GRATING_TENSOR = EPS_GRATING * I3
EPS_GROOVE_TENSOR = EPS_GROOVE * I3

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_yIndex, xIndex = np.mgrid[0:NY, 0:NX]
xCell = (xIndex + 0.5) / NX * PERIOD_X - PERIOD_X / 2
ridgeWidth = DUTY_CYCLE * PERIOD_X
ridgeMask = np.abs(xCell) <= ridgeWidth / 2

epsilonTensorCell = np.empty((NY, NX, 3, 3), dtype=complex)
epsilonTensorCell[:] = EPS_GROOVE_TENSOR
epsilonTensorCell[ridgeMask] = EPS_GRATING_TENSOR
epsilonCell = epsilonTensorCell[..., 0, 0]

gratingEpsilon = AnalyticRectangle(
    period=(PERIOD_X, PERIOD_Y),
    size=(ridgeWidth, PERIOD_Y),
    background=EPS_GROOVE_TENSOR[0, 0],
    inclusion=EPS_GRATING_TENSOR[0, 0],
)
layers = [
    rcwa.Layer(
        thickness=DEPTH,
        epsilon=gratingEpsilon,
        name="LAWP 2020 binary grating layer",
        factorization="standard",
    )
]
compiledLayers = rcwa.compileLayers(layers, orders=ORDERS, truncation=TRUNCATION)

reflection = {(pol, thetaDeg): np.zeros_like(FREQUENCY_THZ) for pol, *_ in POLARIZATION_CASES for thetaDeg, _ in THETA_CASES}
transmission = {(pol, thetaDeg): np.zeros_like(FREQUENCY_THZ) for pol, *_ in POLARIZATION_CASES for thetaDeg, _ in THETA_CASES}
conservation = {(pol, thetaDeg): np.zeros_like(FREQUENCY_THZ) for pol, *_ in POLARIZATION_CASES for thetaDeg, _ in THETA_CASES}

print("LAWP 2020 binary grating reproduction")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, order=({ORDER}, 0), points={FREQUENCY_THZ.size}")
print(f"epsilon cover tensor:\n{EPS_COVER_TENSOR}")
print(f"epsilon substrate tensor:\n{EPS_SUBSTRATE_TENSOR}")
print(f"epsilon grating tensor:\n{EPS_GRATING_TENSOR}")

startTime = time.perf_counter()
for polarizationName, sAmplitude, pAmplitude in POLARIZATION_CASES:
    for thetaDeg, theta in THETA_CASES:
        key = (polarizationName, thetaDeg)
        print(f"Solving {polarizationName}, theta={thetaDeg:.0f} deg")
        for index, frequency in enumerate(FREQUENCY_THZ):
            wavelength = 299.792458 / float(frequency)
            result = rcwa.solveStack(
                layers=compiledLayers,
                wavelength=wavelength,
                period=(PERIOD_X, PERIOD_Y),
                orders=ORDERS,
                epsIncident=EPS_COVER_TENSOR[0, 0],
                epsTransmission=EPS_SUBSTRATE_TENSOR[0, 0],
                theta=theta,
                phi=0.0,
                sAmplitude=sAmplitude,
                pAmplitude=pAmplitude,
                method=METHOD,
                truncation=TRUNCATION,
                backend=BACKEND,
            )
            reflection[key][index] = result.reflection
            transmission[key][index] = result.transmission
            conservation[key][index] = result.conservation
elapsed = time.perf_counter() - startTime

for key in [("TE", 0.0), ("TE", 30.0), ("TM", 0.0), ("TM", 30.0)]:
    peakIndex = int(np.argmax(reflection[key]))
    maxError = float(np.max(np.abs(conservation[key] - 1.0)))
    print(
        f"{key[0]}, theta={key[1]:.0f} deg: "
        f"max R={reflection[key][peakIndex]:.6f} at {FREQUENCY_THZ[peakIndex]:.3f} THz, "
        f"max |R+T-1|={maxError:.2e}"
    )
print(f"Elapsed: {elapsed:.2f} s")

if SAVE_PLOTS:
    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=180)
    axes[0].imshow(
        np.real(epsilonCell),
        origin="lower",
        extent=(-PERIOD_X / 2, PERIOD_X / 2, -PERIOD_Y / 2, PERIOD_Y / 2),
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set_title("Unit-cell permittivity")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    crossSection = np.vstack(
        [
            np.full((28, NX), EPS_COVER),
            np.tile(np.real(epsilonCell[0:1, :]), (72, 1)),
            np.full((34, NX), EPS_SUBSTRATE),
        ]
    )
    zMin = -0.22
    zMax = DEPTH + 0.18
    axes[1].imshow(
        crossSection,
        origin="lower",
        extent=(-PERIOD_X / 2, PERIOD_X / 2, zMin, zMax),
        cmap="viridis",
        aspect="auto",
    )
    axes[1].axhline(0.0, color="white", linewidth=1.0, alpha=0.85)
    axes[1].axhline(DEPTH, color="white", linewidth=1.0, alpha=0.85)
    axes[1].set_title("x-z cross section")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    fig.suptitle("Binary grating from LAWP 2020.3024640")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    cellOutput = outputDir / "lawp2020_binary_grating_cell.png"
    fig.savefig(cellOutput)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=180, sharex=True)
    panelInfo = [
        (axes[0, 0], ("TE", 0.0), "(a) TE, normal incidence"),
        (axes[0, 1], ("TE", 30.0), "(b) TE, theta=30 deg"),
        (axes[1, 0], ("TM", 0.0), "(c) TM, normal incidence"),
        (axes[1, 1], ("TM", 30.0), "(d) TM, theta=30 deg"),
    ]
    for axis, key, title in panelInfo:
        axis.plot(FREQUENCY_THZ, reflection[key], color="#0B65C2", linewidth=1.9, label="Reflection")
        axis.plot(FREQUENCY_THZ, conservation[key], color="#888888", linestyle="--", linewidth=0.9, alpha=0.9, label="R + T")
        axis.set_title(title)
        axis.set_xlim(float(FREQUENCY_THZ[0]), float(FREQUENCY_THZ[-1]))
        axis.set_ylim(-0.03, 1.05)
        axis.grid(True, alpha=0.28)
        axis.set_ylabel("Reflection coefficient")
    axes[1, 0].set_xlabel("Frequency (THz)")
    axes[1, 1].set_xlabel("Frequency (THz)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Reflection spectra of the LAWP 2020 binary grating, order={ORDER} in x")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    spectrumOutput = outputDir / "lawp2020_binary_grating_spectrum.png"
    fig.savefig(spectrumOutput)
    print(f"cell: {cellOutput}")
    print(f"spectrum: {spectrumOutput}")
    if SHOW:
        plt.show()
    else:
        plt.close(fig)
