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


SHOW = False
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"

WAVELENGTH = 1.0
PERIOD = (0.8, 0.8)
ORDER = 2
THICKNESS = 0.25
FILL = 0.45
GRID = 96

EPS_BACKGROUND = 1.0
EPS_ROD = 3.4**2

I3 = np.eye(3, dtype=complex)
EPS_BACKGROUND_TENSOR = EPS_BACKGROUND * I3
EPS_ROD_TENSOR = EPS_ROD * I3


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


layer = rcwa.circularPostLayer(
    period=PERIOD,
    thickness=THICKNESS,
    background=EPS_BACKGROUND_TENSOR[0, 0],
    post=EPS_ROD_TENSOR[0, 0],
    radius=FILL * PERIOD[0] / 2,
    analytic=True,
    name="analytic square-lattice rods",
)
compiledLayers = rcwa.compileLayers([layer], orders=ORDER, truncation=TRUNCATION)

result = rcwa.solveStack(
    layers=compiledLayers,
    wavelength=WAVELENGTH,
    period=PERIOD,
    orders=ORDER,
    epsIncident=EPS_BACKGROUND_TENSOR[0, 0],
    epsTransmission=EPS_BACKGROUND_TENSOR[0, 0],
    sAmplitude=1.0,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
)

print("Binary grating / rod lattice")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, order={ORDER}")
print(f"epsilon background tensor:\n{EPS_BACKGROUND_TENSOR}")
print(f"epsilon rod tensor:\n{EPS_ROD_TENSOR}")
print(f"R = {result.reflection:.8f}")
print(f"T = {result.transmission:.8f}")
print(f"R + T = {result.conservation:.8f}")
print("Propagating diffraction orders:")

propagatingOrders = []
for diffractionOrder in result.orders:
    if diffractionOrder.reflectedPropagating or diffractionOrder.transmittedPropagating:
        propagatingOrders.append(diffractionOrder)
        print(
            f"  ({diffractionOrder.mx:+d}, {diffractionOrder.my:+d}) "
            f"R={diffractionOrder.reflectedPower:.6f} T={diffractionOrder.transmittedPower:.6f}"
        )

if SAVE_PLOTS:
    pattern = rcwa.Pattern2D(period=PERIOD, shape=(GRID, GRID), background=EPS_BACKGROUND_TENSOR[0, 0])
    pattern.circle(radius=FILL * PERIOD[0] / 2, material=EPS_ROD_TENSOR[0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160, constrained_layout=True)
    image = axes[0].imshow(
        pattern.epsilon.real,
        origin="lower",
        extent=(-PERIOD[0] / 2, PERIOD[0] / 2, -PERIOD[1] / 2, PERIOD[1] / 2),
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set_title("Square-lattice rod unit cell")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(image, ax=axes[0], label="epsilon")

    orderLabels = [f"({item.mx:+d}, {item.my:+d})" for item in propagatingOrders]
    reflectedPowers = [item.reflectedPower for item in propagatingOrders]
    transmittedPowers = [item.transmittedPower for item in propagatingOrders]
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_title("Propagating diffraction orders")
    axes[1].set_xlabel("(mx, my)")
    axes[1].set_ylabel("Power")
    axes[1].grid(True, axis="y", alpha=0.25)
    if propagatingOrders:
        xPositions = np.arange(len(orderLabels))
        barWidth = 0.34
        axes[1].bar(xPositions - barWidth / 2, reflectedPowers, width=barWidth, color="#31A354", label="Reflected")
        axes[1].bar(xPositions + barWidth / 2, transmittedPowers, width=barWidth, color="#2B8CBE", label="Transmitted")
        axes[1].set_xticks(xPositions, orderLabels)
        axes[1].set_ylim(0.0, max(1.05, float(np.max(reflectedPowers + transmittedPowers)) + 0.05))
        axes[1].legend()
    else:
        axes[1].set_xticks([])
        axes[1].set_ylim(0.0, 1.05)
        axes[1].text(0.5, 0.5, "No propagating diffraction orders", transform=axes[1].transAxes, ha="center", va="center")
    axes[1].text(
        0.03,
        0.95,
        f"Total R={result.reflection:.3f}\nTotal T={result.transmission:.3f}\nR+T={result.conservation:.3f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )

    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    output = outputDir / "binary_grating.png"
    fig.savefig(output)
    print(f"figure: {output}")
    if SHOW:
        plt.show()
    plt.close(fig)
