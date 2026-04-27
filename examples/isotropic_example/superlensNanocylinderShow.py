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


SHOW = True
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"
FACTORIZATION = "standard"
QUANTITY = "realEx"
NORMALIZE = False

PERIOD = 0.35
HEIGHT = 1.30
WAVELENGTH = 0.66
ORDER = 8
DISPLAY_GRID = 196
X_POINTS = 181
Z_POINTS = 241
XY_POINTS = 161
XY_DISTANCE_ABOVE_TOP = 0.30

EPS_CYLINDER = 4.1616
EPS_SUBSTRATE = 2.12074
EPS_INCIDENT = EPS_SUBSTRATE
EPS_EMITTING = 1.0

I3 = np.eye(3, dtype=complex)
EPS_CYLINDER_TENSOR = EPS_CYLINDER * I3
EPS_SUBSTRATE_TENSOR = EPS_SUBSTRATE * I3
EPS_INCIDENT_TENSOR = EPS_INCIDENT * I3
EPS_EMITTING_TENSOR = EPS_EMITTING * I3


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


pattern50 = rcwa.Pattern2D(period=(PERIOD, PERIOD), shape=(DISPLAY_GRID, DISPLAY_GRID), background=EPS_EMITTING_TENSOR[0, 0])
pattern50.circle(radius=0.05, material=EPS_CYLINDER_TENSOR[0, 0])
layer50 = rcwa.circularPostLayer(
    period=(PERIOD, PERIOD),
    thickness=HEIGHT,
    background=EPS_EMITTING_TENSOR[0, 0],
    post=EPS_CYLINDER_TENSOR[0, 0],
    radius=0.05,
    analytic=True,
    factorization=FACTORIZATION,
    name="analytic nanocylinder r=0.05",
)
compiledLayers50 = rcwa.compileLayers([layer50], orders=ORDER, truncation=TRUNCATION)
result50X = rcwa.solveStack(
    layers=compiledLayers50,
    wavelength=WAVELENGTH,
    period=(PERIOD, PERIOD),
    orders=ORDER,
    epsIncident=EPS_INCIDENT_TENSOR[0, 0],
    epsTransmission=EPS_EMITTING_TENSOR[0, 0],
    sAmplitude=0.0,
    pAmplitude=1.0,
    returnFields=True,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
)
layer100 = rcwa.circularPostLayer(
    period=(PERIOD, PERIOD),
    thickness=HEIGHT,
    background=EPS_EMITTING_TENSOR[0, 0],
    post=EPS_CYLINDER_TENSOR[0, 0],
    radius=0.10,
    analytic=True,
    factorization=FACTORIZATION,
    name="analytic nanocylinder r=0.10",
)
compiledLayers100 = rcwa.compileLayers([layer100], orders=ORDER, truncation=TRUNCATION)
result100X = rcwa.solveStack(
    layers=compiledLayers100,
    wavelength=WAVELENGTH,
    period=(PERIOD, PERIOD),
    orders=ORDER,
    epsIncident=EPS_INCIDENT_TENSOR[0, 0],
    epsTransmission=EPS_EMITTING_TENSOR[0, 0],
    sAmplitude=0.0,
    pAmplitude=1.0,
    returnFields=True,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
)

xPlot = np.linspace(0.0, PERIOD, X_POINTS)
zPlot = np.linspace(-0.20, HEIGHT + 0.30, Z_POINTS)
electricIntensityXz = {}
realExXz = {}
for label, result in (("r50", result50X), ("r100", result100X)):
    _xMap, _zMap, fieldMaps = rcwa.stackFieldComponentsXz(
        result,
        y=0.0,
        xSpan=(-PERIOD / 2, PERIOD / 2),
        zSpan=(float(zPlot.min()), float(zPlot.max())),
        shape=(zPlot.size, xPlot.size),
    )
    electricIntensityXz[label] = fieldMaps["EIntensity"]
    realExXz[label] = np.real(fieldMaps["Ex"])

xyPlaneZ = HEIGHT + XY_DISTANCE_ABOVE_TOP
_xXy, _yXy, fieldMapsXy = rcwa.stackFieldComponentsXy(
    result50X,
    z=xyPlaneZ,
    shape=(XY_POINTS, XY_POINTS),
)
electricIntensityXy = fieldMapsXy["EIntensity"]

print("Superlens nanocylinder RCWA check")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, order={ORDER}, factorization={FACTORIZATION}")
print("normal-incidence fields: p=x-polarized, s=y-polarized")
print(f"epsilon cylinder tensor:\n{EPS_CYLINDER_TENSOR}")
print(f"epsilon substrate tensor:\n{EPS_SUBSTRATE_TENSOR}")
print(f"epsilon emitting tensor:\n{EPS_EMITTING_TENSOR}")
print(f"r=50 nm x-pol: R={result50X.reflection:.6f}, T={result50X.transmission:.6f}, R+T={result50X.conservation:.6f}")
print(f"r=100 nm x-pol: R={result100X.reflection:.6f}, T={result100X.transmission:.6f}, R+T={result100X.conservation:.6f}")
print(f"Max |E|^2 in x-z slice, r=50 nm x-pol: {np.max(electricIntensityXz['r50']):.6f}")
print(f"Max |E|^2 in x-z slice, r=100 nm x-pol: {np.max(electricIntensityXz['r100']):.6f}")
print(
    f"Max |E|^2 in xy plane at z=h+{XY_DISTANCE_ABOVE_TOP:.3f}, "
    f"r=50 nm: {np.max(electricIntensityXy):.6f}"
)

if SAVE_PLOTS:
    if QUANTITY == "realEx":
        plotXz50 = realExXz["r50"]
        plotXz100 = realExXz["r100"]
        plotTitle = "real(Ex), x-pol"
        colorMap = "RdBu_r"
        colorLabel = "normalized real(Ex)" if NORMALIZE else "real(Ex)"
        if NORMALIZE:
            plotXz50 = plotXz50 / max(float(np.max(np.abs(plotXz50))), 1e-30)
            plotXz100 = plotXz100 / max(float(np.max(np.abs(plotXz100))), 1e-30)
        xzLimit = max(float(np.max(np.abs(plotXz50))), float(np.max(np.abs(plotXz100))), 1e-30)
    else:
        plotXz50 = electricIntensityXz["r50"]
        plotXz100 = electricIntensityXz["r100"]
        plotTitle = "|E|^2, x-pol"
        colorMap = "turbo"
        colorLabel = "normalized |E|^2" if NORMALIZE else "|E|^2"
        if NORMALIZE:
            plotXz50 = plotXz50 / max(float(np.max(plotXz50)), 1e-30)
            plotXz100 = plotXz100 / max(float(np.max(plotXz100)), 1e-30)
        xzLimit = None

    plotXyIntensity = electricIntensityXy
    if NORMALIZE:
        plotXyIntensity = plotXyIntensity / max(float(np.max(plotXyIntensity)), 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    epsilonImage = ax.imshow(
        np.real(pattern50.epsilon),
        origin="lower",
        extent=(-PERIOD / 2, PERIOD / 2, -PERIOD / 2, PERIOD / 2),
        cmap="viridis",
        aspect="equal",
    )
    ax.set_title("Unit cell epsilon, radius=50 nm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(epsilonImage, ax=ax, label="Re(epsilon)")

    for ax, data, title in (
        (axes[0, 1], plotXz50, f"{plotTitle}, radius=50 nm"),
        (axes[1, 0], plotXz100, f"{plotTitle}, radius=100 nm"),
    ):
        image = ax.imshow(
            data,
            origin="lower",
            extent=(-PERIOD / 2, PERIOD / 2, float(zPlot.min()), float(zPlot.max())),
            cmap=colorMap,
            aspect="auto",
            vmin=-xzLimit if xzLimit is not None else None,
            vmax=xzLimit if xzLimit is not None else None,
        )
        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
        ax.axhline(HEIGHT, color="k", linewidth=0.6, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        fig.colorbar(image, ax=ax, label=colorLabel)

    ax = axes[1, 1]
    image = ax.imshow(
        plotXyIntensity,
        origin="lower",
        extent=(-PERIOD / 2, PERIOD / 2, -PERIOD / 2, PERIOD / 2),
        cmap="turbo",
        aspect="equal",
    )
    ax.set_title(f"|E|^2, xy at z=h+{XY_DISTANCE_ABOVE_TOP:.2f}, radius=50 nm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, label="normalized |E|^2" if NORMALIZE else "|E|^2")

    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    output = outputDir / "superlens_nanocylinder_show.png"
    fig.savefig(output, dpi=180)
    print(f"figure: {output}")
    if SHOW:
        plt.show()
    plt.close(fig)
