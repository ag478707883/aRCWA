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
from rcwa3d_isotropic.analytic import AnalyticComposite, AnalyticRectangle, AnalyticTerm


SHOW = True
SAVE_PLOTS = True
METHOD = "smatrix"
TRUNCATION = "circular"
BACKEND = "cuda"
PRECOMPILE = False
CACHE_MODES = False
WORKERS = 1

SPECTRUM_ORDER = 8
FIELD_ORDER = 8
WAVELENGTH = 4.132
SPECTRUM_MIN = 3.0
SPECTRUM_MAX = 5
SPECTRUM_POINTS = 501
FIELD_X = 421
FIELD_Z = 301

PERIOD_1 = 3.5
PERIOD_2 = 0.7
FILL_2 = 0.37
GOLD_THICKNESS = 0.1
SILICON_HEIGHT = 0.7

EPS_AIR = 1.0
EPS_QUARTZ = 1.45**2


if not SHOW:
    matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


siCandidates = list(Path("D:/").glob("Lumerical RCWA*/Au-Si*/Si-palik.txt"))
auCandidates = list(Path("D:/").glob("Lumerical RCWA*/Au-Si*/Au-palik.txt"))
SI_DATA = siCandidates[0] if siCandidates else REPO_ROOT / "Si-palik.txt"
AU_DATA = auCandidates[0] if auCandidates else REPO_ROOT / "Au-palik.txt"

siData = np.loadtxt(SI_DATA, dtype=float)
auData = np.loadtxt(AU_DATA, dtype=float)

siSortIndex = np.lexsort((siData[:, 0],))
auSortIndex = np.lexsort((auData[:, 0],))
siData = siData[siSortIndex]
auData = auData[auSortIndex]
siWavelengths = siData[:, 0]
siN = siData[:, 1]
siK = siData[:, 2]
auWavelengths = auData[:, 0]
auN = auData[:, 1]
auK = auData[:, 2]

if WAVELENGTH < siWavelengths[0] or WAVELENGTH > siWavelengths[-1]:
    raise ValueError(f"WAVELENGTH={WAVELENGTH} um is outside {SI_DATA} data range")
if WAVELENGTH < auWavelengths[0] or WAVELENGTH > auWavelengths[-1]:
    raise ValueError(f"WAVELENGTH={WAVELENGTH} um is outside {AU_DATA} data range")
if SPECTRUM_MIN >= SPECTRUM_MAX:
    raise ValueError("SPECTRUM_MIN must be smaller than SPECTRUM_MAX")
if SPECTRUM_POINTS < 2:
    raise ValueError("SPECTRUM_POINTS must be at least 2")

period = (PERIOD_1, 1.0)
stripWidth = FILL_2 * PERIOD_2
stripCenters = (-PERIOD_2 / 2, PERIOD_2 / 2)


def interpolatedEpsilon(wavelength, wavelengthsTable, nTable, kTable, source):
    if wavelength < wavelengthsTable[0] or wavelength > wavelengthsTable[-1]:
        raise ValueError(f"wavelength {wavelength} um is outside {source} data range")
    nValue = float(np.interp(wavelength, wavelengthsTable, nTable))
    kValue = float(np.interp(wavelength, wavelengthsTable, kTable))
    return complex(nValue, kValue) ** 2


def siliconGoldLayers(wavelength):
    epsSiSample = interpolatedEpsilon(wavelength, siWavelengths, siN, siK, SI_DATA)
    epsAuSample = interpolatedEpsilon(wavelength, auWavelengths, auN, auK, AU_DATA)
    terms = tuple(
        AnalyticTerm(
            AnalyticRectangle(
                period=period,
                size=(stripWidth, period[1]),
                background=0.0,
                inclusion=1.0,
                center=(center, 0.0),
            ),
            epsSiSample - EPS_AIR,
        )
        for center in stripCenters
    )
    return [
        rcwa.Layer(
            thickness=SILICON_HEIGHT,
            epsilon=AnalyticComposite(period=period, background=EPS_AIR, terms=terms),
            name="analytic silicon strip pair in air",
            factorization="standard",
        ),
        rcwa.Layer(thickness=GOLD_THICKNESS, epsilon=epsAuSample, name="gold film", factorization="standard"),
    ]


epsSi = interpolatedEpsilon(WAVELENGTH, siWavelengths, siN, siK, SI_DATA)
epsAu = interpolatedEpsilon(WAVELENGTH, auWavelengths, auN, auK, AU_DATA)

wavelengths = np.linspace(SPECTRUM_MIN, SPECTRUM_MAX, SPECTRUM_POINTS)
reflection = np.empty_like(wavelengths)
transmission = np.empty_like(wavelengths)
absorptionSpectrum = np.empty_like(wavelengths)

spectrumSimulation = rcwa.RCWASimulation(
    period=period,
    orders=(SPECTRUM_ORDER, 0),
    layers=[siliconGoldLayers],
    epsIncident=EPS_AIR,
    epsTransmission=EPS_QUARTZ,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
    precompile=PRECOMPILE,
    cacheModes=CACHE_MODES,
    workers=WORKERS,
)
fieldSimulation = rcwa.RCWASimulation(
    period=period,
    orders=(FIELD_ORDER, 0),
    layers=[siliconGoldLayers],
    epsIncident=EPS_AIR,
    epsTransmission=EPS_QUARTZ,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
    precompile=PRECOMPILE,
    cacheModes=CACHE_MODES,
)
backend = rcwa.resolveBackend(BACKEND)
backend.synchronize()
startTime = time.perf_counter()
spectrum = spectrumSimulation.spectrum(wavelengths, polarizations=("TM",), workers=WORKERS)
reflection = spectrum["TM"]["reflection"]
transmission = spectrum["TM"]["transmission"]
absorptionSpectrum = spectrum["TM"]["absorption"]
result = fieldSimulation.solve(WAVELENGTH, polarization="TM", returnFields=True)
backend.synchronize()
elapsed = time.perf_counter() - startTime

absorption = result.absorption
bestAbsorptionIndex = int(np.argmax(absorptionSpectrum))

print("Appl. Sci. 2019 absorber: spectrum + Fig. 3(c) self-normalized magnetic field |H|")
print(
    f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}, precompile={PRECOMPILE}, "
    f"cacheModes={CACHE_MODES}, workers={WORKERS}"
)
print(f"spectrum orders=({SPECTRUM_ORDER}, 0), field orders=({FIELD_ORDER}, 0)")
print("geometry: analytic silicon strip pair in air + homogeneous gold film")
print(f"lambda={WAVELENGTH:.4f} um, period1={PERIOD_1:.3f} um, period2={PERIOD_2:.3f} um")
print(f"epsilon air={EPS_AIR:.6g}, quartz={EPS_QUARTZ:.6g}")
print(f"epsilon silicon={epsSi:.6g}, gold={epsAu:.6g}")
print(f"elapsed={elapsed:.3f} s")
print(f"R={result.reflection:.6f}, T={result.transmission:.6f}, A={absorption:.6f}, R+T={result.conservation:.6f}")
print(f"energy error diagnostic={result.energyError}")
for diagnostic in result.diagnostics:
    print(f"diagnostic: {diagnostic}")
print(f"spectrum peak A={absorptionSpectrum[bestAbsorptionIndex]:.6f} at lambda={wavelengths[bestAbsorptionIndex]:.4f} um")

if SAVE_PLOTS:
    zMinArticle = -0.5
    zMaxArticle = 1.5
    zSpanSolver = (SILICON_HEIGHT - zMaxArticle, SILICON_HEIGHT - zMinArticle)
    x, zSolver, fieldMaps = rcwa.stackFieldComponentsXz(
        result,
        y=0.0,
        xSpan=(-PERIOD_1 / 2, PERIOD_1 / 2),
        zSpan=zSpanSolver,
        shape=(FIELD_Z, FIELD_X),
    )
    zArticle = SILICON_HEIGHT - zSolver
    magneticField = fieldMaps["HSelfNormalizedMagnitude"]
    magneticEnhancement = fieldMaps["HNormalizedIntensity"]
    xPlot = x[::-1, :]
    zPlot = zArticle[::-1, :]
    magneticPlot = magneticField[::-1, :]

    fig, (spectrumAxis, fieldAxis) = plt.subplots(
        1,
        2,
        figsize=(8.4, 3.25),
        dpi=220,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    spectrumAxis.plot(wavelengths, absorptionSpectrum, color="#d73027", linewidth=2.0, label="A")
    spectrumAxis.plot(wavelengths, reflection, color="#2166ac", linewidth=1.6, label="R")
    spectrumAxis.plot(wavelengths, transmission, color="#1a9850", linewidth=1.6, label="T")
    spectrumAxis.axvline(WAVELENGTH, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    spectrumAxis.scatter([WAVELENGTH], [absorption], color="#d73027", edgecolor="white", linewidth=0.7, zorder=5)
    spectrumAxis.set_xlim(SPECTRUM_MIN, SPECTRUM_MAX)
    spectrumAxis.set_ylim(0.0, 1.05)
    spectrumAxis.set_xlabel("Wavelength (um)")
    spectrumAxis.set_ylabel("Power coefficient")
    spectrumAxis.set_title("Spectrum")
    spectrumAxis.grid(True, alpha=0.25)
    spectrumAxis.legend(fontsize=8, loc="best", frameon=False)

    image = fieldAxis.imshow(
        magneticPlot,
        origin="lower",
        extent=(float(np.min(xPlot)), float(np.max(xPlot)), float(np.min(zPlot)), float(np.max(zPlot))),
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="auto",
    )
    for center in stripCenters:
        fieldAxis.add_patch(
            patches.Rectangle(
                (center - stripWidth / 2, 0.0),
                stripWidth,
                SILICON_HEIGHT,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
            )
        )
    fieldAxis.axhline(0.0, color="black", linewidth=0.8)
    fieldAxis.axhline(-GOLD_THICKNESS, color="black", linewidth=0.45, alpha=0.65)
    fieldAxis.set_xlim(-PERIOD_1 / 2, PERIOD_1 / 2)
    fieldAxis.set_ylim(zMinArticle, zMaxArticle)
    fieldAxis.set_xlabel("x (um)")
    fieldAxis.set_ylabel("z (um)")
    fieldAxis.set_title("Self-normalized |H|")
    fieldAxis.text(0.83, 0.90, "|H|", transform=fieldAxis.transAxes, color="white", fontsize=10, fontweight="bold")
    colorbar = fig.colorbar(image, ax=fieldAxis, fraction=0.046, pad=0.03)
    colorbar.set_ticks([0.2, 0.4, 0.6, 0.8])
    print(f"self-normalized |H| range: {float(np.min(magneticPlot)):.6g} to {float(np.max(magneticPlot)):.6g}")
    print(f"max |H/H0|^2 in plotted window: {float(np.max(magneticEnhancement)):.6g}")

    outputDir = REPO_ROOT / "examples" / "outputs"
    outputDir.mkdir(parents=True, exist_ok=True)
    output = outputDir / "applsci2019_spectrum_and_figure3c_magnetic_field.png"
    fig.savefig(output)
    print(f"figure: {output}")
    if SHOW:
        plt.show()
    plt.close(fig)
