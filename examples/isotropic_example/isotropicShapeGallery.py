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
PRECOMPILE = True
CACHE_MODES = True

PERIOD = (1.0, 1.0)
WAVELENGTH = 1.15
ORDERS = (2, 2)
HEIGHT = 0.18
GRID = (96, 96)

EPS_BACKGROUND = 1.0
EPS_POST = 2.8**2

POLARIZATIONS = ("TE", "TM")


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


stripePattern = rcwa.Pattern2D(period=PERIOD, shape=GRID, background=EPS_BACKGROUND, name="sampled stripes")
stripePattern.stripes(fillFraction=0.38, material=EPS_POST, axis="x")

shapeCases = [
    (
        "analytic circle",
        rcwa.circularPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            radius=0.20,
            analytic=True,
            name="analytic circular post",
        ),
    ),
    (
        "analytic ellipse",
        rcwa.ellipticalPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            radii=(0.25, 0.12),
            angle=np.deg2rad(25),
            analytic=True,
        ),
    ),
    (
        "analytic rectangle",
        rcwa.rectangularPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            size=(0.34, 0.18),
            angle=np.deg2rad(20),
            analytic=True,
        ),
    ),
    (
        "analytic annulus",
        rcwa.annularPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            innerRadius=0.10,
            outerRadius=0.24,
            analytic=True,
        ),
    ),
    (
        "analytic hollow rectangle",
        rcwa.rectangularHollowPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            size=(0.46, 0.38),
            holeRadius=0.11,
            analytic=True,
        ),
    ),
    (
        "analytic cross",
        rcwa.crossPostLayer(
            PERIOD,
            HEIGHT,
            EPS_BACKGROUND,
            EPS_POST,
            armLengths=(0.52, 0.44),
            armWidths=(0.14, 0.12),
            angle=np.deg2rad(12),
            analytic=True,
        ),
    ),
    ("sampled stripes", stripePattern.toLayer(HEIGHT, factorization="standard")),
    (
        "sampled polygon",
        rcwa.polygonPostLayer(
            period=PERIOD,
            thickness=HEIGHT,
            background=EPS_BACKGROUND,
            post=EPS_POST,
            vertices=((-0.22, -0.14), (0.19, -0.21), (0.25, 0.12), (-0.08, 0.24), (-0.25, 0.02)),
            shape=GRID,
            factorization="standard",
            name="sampled polygon post",
        ),
    ),
]

rows = []
for label, layer in shapeCases:
    simulation = rcwa.RCWASimulation(
        period=PERIOD,
        layers=[layer],
        orders=ORDERS,
        truncation=TRUNCATION,
        epsIncident=EPS_BACKGROUND,
        epsTransmission=EPS_BACKGROUND,
        method=METHOD,
        backend=BACKEND,
        precompile=PRECOMPILE,
        cacheModes=CACHE_MODES,
    )
    spectrum = simulation.spectrum([WAVELENGTH], polarizations=POLARIZATIONS)
    row = {"shape": label}
    for polarizationName in POLARIZATIONS:
        row[f"{polarizationName} R"] = float(spectrum[polarizationName]["reflection"][0])
        row[f"{polarizationName} T"] = float(spectrum[polarizationName]["transmission"][0])
        row[f"{polarizationName} C"] = float(spectrum[polarizationName]["conservation"][0])
    rows.append(row)

print("Isotropic RCWA shape gallery")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, precompile={PRECOMPILE}, cacheModes={CACHE_MODES}, wavelength={WAVELENGTH:.3f}, orders={ORDERS}")
print(f"epsilon background={EPS_BACKGROUND:.6g}, post={EPS_POST:.6g}")
for row in rows:
    print(
        f"{row['shape']:>26s}: "
        f"TE R={row['TE R']:.5f}, TE T={row['TE T']:.5f}, TE C={row['TE C']:.5f}; "
        f"TM R={row['TM R']:.5f}, TM T={row['TM T']:.5f}, TM C={row['TM C']:.5f}"
    )

if SAVE_PLOTS:
    labels = [row["shape"] for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=160, constrained_layout=True)
    ax.bar(x - 1.5 * width, [row["TE R"] for row in rows], width, label="TE R", color="#2B8CBE")
    ax.bar(x - 0.5 * width, [row["TE T"] for row in rows], width, label="TE T", color="#7BCCC4")
    ax.bar(x + 0.5 * width, [row["TM R"] for row in rows], width, label="TM R", color="#88419D")
    ax.bar(x + 1.5 * width, [row["TM T"] for row in rows], width, label="TM T", color="#F16913")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Power")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center")
    ax.set_title("Isotropic RCWA shape gallery")

    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    output = outputDir / "isotropic_shape_gallery.png"
    fig.savefig(output)
    print(f"figure: {output}")
    if SHOW:
        plt.show()
    plt.close(fig)
