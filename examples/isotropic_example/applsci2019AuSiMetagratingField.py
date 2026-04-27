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

SPECTRUM_ORDER = 8
FIELD_ORDER = 8
WAVELENGTH = 4.132
SPECTRUM_MIN = 3.0
SPECTRUM_MAX = 5
SPECTRUM_POINTS = 501
GEOMETRY_GRID = (8, 1024)
FIELD_X = 421
FIELD_Z = 301

PERIOD_1 = 3.5
PERIOD_2 = 0.7
FILL_2 = 0.37
GOLD_THICKNESS = 0.1
SILICON_HEIGHT = 0.7

EPS_AIR = 1.0
EPS_QUARTZ = 1.45**2

I3 = np.eye(3, dtype=complex)
EPS_AIR_TENSOR = EPS_AIR * I3
EPS_QUARTZ_TENSOR = EPS_QUARTZ * I3


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

epsSi = complex(float(np.interp(WAVELENGTH, siWavelengths, siN)), float(np.interp(WAVELENGTH, siWavelengths, siK))) ** 2
epsAu = complex(float(np.interp(WAVELENGTH, auWavelengths, auN)), float(np.interp(WAVELENGTH, auWavelengths, auK))) ** 2
EPS_SI_TENSOR = epsSi * I3
EPS_AU_TENSOR = epsAu * I3

period = (PERIOD_1, 1.0)
stripWidth = FILL_2 * PERIOD_2
stripCenters = (-PERIOD_2 / 2, PERIOD_2 / 2)

wavelengths = np.linspace(SPECTRUM_MIN, SPECTRUM_MAX, SPECTRUM_POINTS)
reflection = np.empty_like(wavelengths)
transmission = np.empty_like(wavelengths)
absorptionSpectrum = np.empty_like(wavelengths)

for index, sampleWavelength in enumerate(wavelengths):
    if sampleWavelength < siWavelengths[0] or sampleWavelength > siWavelengths[-1]:
        raise ValueError(f"wavelength {sampleWavelength} um is outside {SI_DATA} data range")
    if sampleWavelength < auWavelengths[0] or sampleWavelength > auWavelengths[-1]:
        raise ValueError(f"wavelength {sampleWavelength} um is outside {AU_DATA} data range")

    epsSiSample = complex(float(np.interp(sampleWavelength, siWavelengths, siN)), float(np.interp(sampleWavelength, siWavelengths, siK))) ** 2
    epsAuSample = complex(float(np.interp(sampleWavelength, auWavelengths, auN)), float(np.interp(sampleWavelength, auWavelengths, auK))) ** 2
    epsSiTensorSample = epsSiSample * I3
    epsAuTensorSample = epsAuSample * I3

    spectrumGeometry = rcwa.LayerStack(period=period, shape=GEOMETRY_GRID)
    spectrumGeometry.addLayer(SILICON_HEIGHT, EPS_AIR_TENSOR, name="air layer patterned with silicon strips", factorization="standard")
    spectrumGeometry.addLayer(GOLD_THICKNESS, epsAuTensorSample, name="gold film", factorization="standard")
    for center in stripCenters:
        spectrumGeometry.setMaterial(
            epsSiTensorSample,
            x=(center - stripWidth / 2, center + stripWidth / 2),
            y=(-period[1] / 2, period[1] / 2),
            z=(0.0, SILICON_HEIGHT),
        )

    spectrumLayers = spectrumGeometry.toLayers()
    compiledSpectrumLayers = rcwa.compileLayers(spectrumLayers, orders=(SPECTRUM_ORDER, 0), truncation=TRUNCATION)
    spectrumResult = rcwa.solveStack(
        layers=compiledSpectrumLayers,
        wavelength=float(sampleWavelength),
        period=period,
        orders=(SPECTRUM_ORDER, 0),
        epsIncident=EPS_AIR_TENSOR[0, 0],
        epsTransmission=EPS_QUARTZ_TENSOR[0, 0],
        theta=0.0,
        phi=0.0,
        sAmplitude=0.0,
        pAmplitude=1.0,
        returnFields=False,
        method=METHOD,
        truncation=TRUNCATION,
        backend=BACKEND,
    )
    reflection[index] = spectrumResult.reflection
    transmission[index] = spectrumResult.transmission
    absorptionSpectrum[index] = 1.0 - spectrumResult.reflection - spectrumResult.transmission

geometry = rcwa.LayerStack(period=period, shape=GEOMETRY_GRID)
geometry.addLayer(SILICON_HEIGHT, EPS_AIR_TENSOR, name="air layer patterned with silicon strips", factorization="standard")
geometry.addLayer(GOLD_THICKNESS, EPS_AU_TENSOR, name="gold film", factorization="standard")
for center in stripCenters:
    geometry.setMaterial(
        EPS_SI_TENSOR,
        x=(center - stripWidth / 2, center + stripWidth / 2),
        y=(-period[1] / 2, period[1] / 2),
        z=(0.0, SILICON_HEIGHT),
    )

layers = geometry.toLayers()
compiledLayers = rcwa.compileLayers(layers, orders=(FIELD_ORDER, 0), truncation=TRUNCATION)
result = rcwa.solveStack(
    layers=compiledLayers,
    wavelength=WAVELENGTH,
    period=period,
    orders=(FIELD_ORDER, 0),
    epsIncident=EPS_AIR_TENSOR[0, 0],
    epsTransmission=EPS_QUARTZ_TENSOR[0, 0],
    theta=0.0,
    phi=0.0,
    sAmplitude=0.0,
    pAmplitude=1.0,
    returnFields=True,
    method=METHOD,
    truncation=TRUNCATION,
    backend=BACKEND,
)

absorption = 1.0 - result.reflection - result.transmission
bestAbsorptionIndex = int(np.argmax(absorptionSpectrum))

print("Appl. Sci. 2019 absorber: spectrum + Fig. 3(c) self-normalized magnetic field |H|")
print(f"method={METHOD}, truncation={TRUNCATION}, backend={BACKEND}")
print(f"spectrum orders=({SPECTRUM_ORDER}, 0), field orders=({FIELD_ORDER}, 0)")
print("geometry: full air-pattern layer + full gold layer, then Si set by x/y/z region selection")
print(f"lambda={WAVELENGTH:.4f} um, period1={PERIOD_1:.3f} um, period2={PERIOD_2:.3f} um")
print(f"epsilon air tensor:\n{EPS_AIR_TENSOR}")
print(f"epsilon quartz tensor:\n{EPS_QUARTZ_TENSOR}")
print(f"epsilon silicon tensor:\n{EPS_SI_TENSOR}")
print(f"epsilon gold tensor:\n{EPS_AU_TENSOR}")
print(f"R={result.reflection:.6f}, T={result.transmission:.6f}, A={absorption:.6f}, R+T={result.conservation:.6f}")
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
