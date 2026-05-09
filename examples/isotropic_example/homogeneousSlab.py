from __future__ import annotations

from pathlib import Path
import sys

import matplotlib


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rcwa3d_isotropic as rcwa


SHOW = False
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"
PRECOMPILE = True
CACHE_MODES = True

WAVELENGTH = 1.0
PERIOD = (1.0, 1.0)
ORDER = 3
SLAB_THICKNESS = 0.30

EPS_INCIDENT = 1.0
EPS_TRANSMISSION = 1.0
EPS_SLAB = 2.25


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


layer = rcwa.Layer(thickness=SLAB_THICKNESS, epsilon=EPS_SLAB, name="glass slab")
simulation = rcwa.RCWASimulation(
    period=PERIOD,
    layers=[layer],
    orders=ORDER,
    truncation=TRUNCATION,
    epsIncident=EPS_INCIDENT,
    epsTransmission=EPS_TRANSMISSION,
    method=METHOD,
    backend=BACKEND,
    precompile=PRECOMPILE,
    cacheModes=CACHE_MODES,
)

result = simulation.solve(WAVELENGTH, polarization="TE")

print("Homogeneous slab")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, precompile={PRECOMPILE}, cacheModes={CACHE_MODES}, order={ORDER}")
print(f"epsilon slab={EPS_SLAB:.6g}")
print(f"R = {result.reflection:.8f}")
print(f"T = {result.transmission:.8f}")
print(f"R + T = {result.conservation:.8f}")
print(f"A = {result.absorption:.8f}")
print(f"energy error = {result.energyError:.3e}")

if SAVE_PLOTS:
    labels = ["Reflection", "Transmission", "Absorption", "R + T"]
    values = [result.reflection, result.transmission, result.absorption, result.conservation]
    colors = ["#31A354", "#2B8CBE", "#D73027", "#555555"]

    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=160, constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylim(0.0, max(1.05, max(values) + 0.05))
    ax.set_ylabel("Normalized power")
    ax.set_title("Homogeneous slab response")
    ax.text(
        0.03,
        0.95,
        f"lambda={WAVELENGTH:.2f}\nthickness={SLAB_THICKNESS:.2f}\nepsilon={EPS_SLAB:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    output = outputDir / "homogeneous_slab.png"
    fig.savefig(output)
    print(f"figure: {output}")
    if SHOW:
        plt.show()
    plt.close(fig)
