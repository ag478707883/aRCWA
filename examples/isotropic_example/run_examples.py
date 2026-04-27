from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CASE = "all"

EXAMPLES = {
    "homogeneous": "homogeneousSlab.py",
    "binary": "binaryGrating.py",
    "metasurface": "rectangularMetasurface.py",
    "photonic": "photonicCrystalSlab.py",
    "applsci2019": "applsci2019AuSiMetagratingField.py",
    "shapes": "isotropicShapeGallery.py",
    "lawp2020": "lawp2020BinaryGratingSpectrum.py",
    "superlens": "superlensNanocylinderShow.py",
}

root = Path(__file__).resolve().parent
selected = tuple(EXAMPLES) if CASE == "all" else (CASE,)

for name in selected:
    script = root / EXAMPLES[name]
    print(f"\n=== {name}: {script.name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], check=True)
