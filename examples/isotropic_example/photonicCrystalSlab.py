from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_isotropic as rcwa
from rcwa3d_isotropic.visualization import plotEpsilon


SHOW = True
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"
PRECOMPILE = True
CACHE_MODES = True
WORKERS = 1

POLARIZATION = "TM"
POLARIZATIONS = ("TE", "TM") if POLARIZATION == "both" else (POLARIZATION,)
ORDER = 5
POINTS = 501
FREQUENCY_MIN = 0.50
FREQUENCY_MAX = 0.55
FACTORIZATION = "standard"

PERIOD = (1.0, 1.0)
RADIUS = 0.20
THICKNESS = 0.50
EPS_SLAB = rcwa.SI1550.epsilon()
EPS_HOLE = rcwa.AIR.epsilon()


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


normalizedFrequency = np.linspace(FREQUENCY_MIN, FREQUENCY_MAX, POINTS)
wavelengths = PERIOD[0] / normalizedFrequency

layer = rcwa.photonicCrystalSlab(
    period=PERIOD,
    thickness=THICKNESS,
    slab=EPS_SLAB,
    hole=EPS_HOLE,
    radius=RADIUS,
    analytic=True,
    factorization=FACTORIZATION,
    name="analytic photonic crystal slab",
)
simulation = rcwa.RCWASimulation(
    period=PERIOD,
    layers=[layer],
    orders=ORDER,
    truncation=TRUNCATION,
    epsIncident=EPS_HOLE,
    epsTransmission=EPS_HOLE,
    method=METHOD,
    backend=BACKEND,
    precompile=PRECOMPILE,
    cacheModes=CACHE_MODES,
    workers=WORKERS,
)

backend = rcwa.resolveBackend(BACKEND)
backend.synchronize()
startTime = time.perf_counter()
spectrum = simulation.spectrum(wavelengths, polarizations=POLARIZATIONS, workers=WORKERS)
backend.synchronize()
elapsed = time.perf_counter() - startTime
profileResult = simulation.solve(wavelengths[len(wavelengths) // 2], polarization=POLARIZATIONS[0], profile=True)

print("Photonic crystal slab")
print(f"method={METHOD}, truncation={TRUNCATION}, order={ORDER}, points={POINTS}, factorization={FACTORIZATION}")
print(f"frequency range a/lambda: {normalizedFrequency[0]:.4f} - {normalizedFrequency[-1]:.4f}")
print(f"epsilon slab={EPS_SLAB:.6g}, hole={EPS_HOLE:.6g}")
print(f"backend={BACKEND}, precompile={PRECOMPILE}, cacheModes={CACHE_MODES}, workers={WORKERS}, elapsed={elapsed:.3f} s")
for timing in profileResult.layerEigTimings:
    print(
        f"  layer {timing.layerIndex}: {timing.kind}, shape={timing.matrixShape}, "
        f"factor={timing.factorizationTimeSeconds:.4f} s, inv={timing.inverseTimeSeconds:.4f} s, "
        f"pq={timing.pqTimeSeconds:.4f} s, eig={timing.eigTimeSeconds:.4f} s, "
        f"min|q|={timing.minAbsQ:.3e}"
    )
if profileResult.stackTiming is not None:
    stackTiming = profileResult.stackTiming
    print(
        f"  stack: interfaces={stackTiming.interfaceTimeSeconds:.4f} s, "
        f"cascade={stackTiming.cascadeTimeSeconds:.4f} s, prepare={stackTiming.totalPrepareTimeSeconds:.4f} s, "
        f"max cond={stackTiming.maxInterfaceCondition:.3e}"
    )
    for warning in stackTiming.stabilityWarnings:
        print(f"  stability warning: {warning}")
for label in POLARIZATIONS:
    reflection = spectrum[label]["reflection"]
    peak = int(np.argmax(reflection))
    energyError = float(np.nanmax(spectrum[label]["energyError"]))
    print(
        f"  {label}: max R={reflection[peak]:.6f} at a/lambda={normalizedFrequency[peak]:.6f}; "
        f"max energy error={energyError:.2e}"
    )

if SAVE_PLOTS:
    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    displayGrid = 160
    pattern = rcwa.Pattern2D(period=PERIOD, shape=(displayGrid, displayGrid), background=EPS_SLAB, supersample=4)
    pattern.circle(radius=RADIUS, material=EPS_HOLE)
    epsilonCell = np.real(pattern.epsilon)

    unitCellOutput = outputDir / "photonic_crystal_slab_unit_cell.png"
    fig = plotEpsilon(epsilonCell, PERIOD, unitCellOutput, title="Photonic crystal slab cell: air hole in high-index slab")
    if not SHOW:
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=170, constrained_layout=True)
    image = axes[0].imshow(
        epsilonCell,
        origin="lower",
        extent=(-PERIOD[0] / 2, PERIOD[0] / 2, -PERIOD[1] / 2, PERIOD[1] / 2),
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set_title("Photonic crystal slab cell")
    axes[0].set_xlabel("x / a")
    axes[0].set_ylabel("y / a")
    fig.colorbar(image, ax=axes[0], label="Re(epsilon)")

    colors = {"TE": "#2B8CBE", "TM": "#88419D"}
    for label in POLARIZATIONS:
        data = spectrum[label]
        axes[1].plot(normalizedFrequency, data["reflection"], color=colors[label], linewidth=2.0, label=f"{label} R")
        axes[1].plot(
            normalizedFrequency,
            data["transmission"],
            color=colors[label],
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label=f"{label} T",
        )
        axes[1].plot(normalizedFrequency, data["conservation"], color="#777777", linewidth=0.9, linestyle=":", alpha=0.6)

    axes[1].set_title("Reflection / transmission spectrum")
    axes[1].set_xlabel("Normalized frequency a / lambda")
    axes[1].set_ylabel("Normalized power")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Photonic crystal slab RCWA (order={ORDER}, {TRUNCATION}, {METHOD})")

    outputPath = outputDir / "photonic_crystal_slab_spectrum.png"
    fig.savefig(outputPath)
    print(f"unit cell: {unitCellOutput}")
    print(f"spectrum: {outputPath}")

    if SHOW:
        plt.show()
    else:
        plt.close(fig)
