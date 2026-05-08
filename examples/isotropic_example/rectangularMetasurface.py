from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_isotropic as rcwa
from rcwa3d_isotropic.visualization import plotEpsilon, plotField, plotSpectrum


SHOW = True
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"
PRECOMPILE = True
CACHE_MODES = True

PERIOD = (0.72, 0.48)
THICKNESS = 0.28
ORDER = 4
POINTS = 301
WAVELENGTHS = np.linspace(0.25, 1.25, POINTS)
FIELD_WAVELENGTH = 0.75

EPS_AIR = rcwa.AIR.epsilon()
EPS_SI = rcwa.SI1550.epsilon()


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


pattern = rcwa.Pattern2D(period=PERIOD, shape=(72, 108), background=EPS_AIR, name="rectangular metasurface")
pattern.rectangle(size=(0.34, 0.18), angle=np.deg2rad(25), material=EPS_SI)
layer = pattern.toLayer(THICKNESS)
simulation = rcwa.RCWASimulation(
    period=PERIOD,
    layers=[layer],
    orders=ORDER,
    truncation=TRUNCATION,
    epsIncident=EPS_AIR,
    epsTransmission=EPS_AIR,
    method=METHOD,
    backend=BACKEND,
    precompile=PRECOMPILE,
    cacheModes=CACHE_MODES,
)

if FIELD_WAVELENGTH < WAVELENGTHS[0] or FIELD_WAVELENGTH > WAVELENGTHS[-1]:
    raise ValueError("FIELD_WAVELENGTH must lie inside the plotted wavelength range")

spectrum = simulation.spectrum(WAVELENGTHS, polarizations=("TE",))
reflection = spectrum["TE"]["reflection"]
transmission = spectrum["TE"]["transmission"]
conservation = spectrum["TE"]["conservation"]
fieldResult = simulation.solve(FIELD_WAVELENGTH, polarization="TE", returnFields=True)

print("Rectangular metasurface")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, precompile={PRECOMPILE}, cacheModes={CACHE_MODES}, order={ORDER}, points={POINTS}")
print(f"field wavelength={FIELD_WAVELENGTH:.4f}")
print(f"epsilon air={EPS_AIR:.6g}, silicon={EPS_SI:.6g}")
print(f"Max |R + T - 1|: {np.max(np.abs(conservation - 1)):.3e}")

if SAVE_PLOTS:
    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    fig = plotEpsilon(layer.epsilon, PERIOD, outputDir / "rectangular_metasurface_cell.png", title="Rectangular metasurface unit cell")
    plt.close(fig)
    spectrumOutput = outputDir / "rectangular_metasurface_spectrum.png"
    fig = plotSpectrum(
        WAVELENGTHS,
        reflection,
        transmission,
        None,
        xlabel="Wavelength",
        title=f"Rectangular-period metasurface, period={PERIOD}",
        conservation=conservation,
    )
    fig.axes[0].axvline(FIELD_WAVELENGTH, color="black", linewidth=1.0, linestyle="--", alpha=0.75)
    fig.savefig(spectrumOutput, bbox_inches="tight")
    plt.close(fig)

    x, y, ex = rcwa.fieldSliceXy(fieldResult, layerIndex=0, z=THICKNESS / 2, component="Ex", shape=(81, 101))
    fig = plotField(
        x,
        y,
        ex,
        outputDir / "rectangular_metasurface_field.png",
        title=f"Re(Ex), lambda={FIELD_WAVELENGTH:.3f} um, z=h/2",
    )
    plt.close(fig)

    print(f"figures: {outputDir}")
    if SHOW:
        plt.show()
