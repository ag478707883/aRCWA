from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np


EpsilonLike = Union[complex, float, Callable[[float], complex]]


@dataclass(frozen=True)
class IsotropicMaterial:
    """Isotropic material described by relative permittivity epsilon."""

    name: str
    epsilonModel: EpsilonLike

    def epsilon(self, wavelength: float | None = None) -> complex:
        if callable(self.epsilonModel):
            if wavelength is None:
                raise ValueError(f"material {self.name!r} needs a wavelength")
            return complex(self.epsilonModel(float(wavelength)))
        return complex(self.epsilonModel)

    def index(self, wavelength: float | None = None) -> complex:
        return complex(np.sqrt(self.epsilon(wavelength) + 0j))


def constantEpsilon(epsilon: complex | float, name: str | None = None) -> IsotropicMaterial:
    label = name if name is not None else f"eps={epsilon:g}"
    return IsotropicMaterial(label, complex(epsilon))


def constantIndex(index: complex | float, name: str | None = None) -> IsotropicMaterial:
    label = name if name is not None else f"n={index:g}"
    return IsotropicMaterial(label, complex(index) ** 2)


AIR = constantIndex(1.0, "air")
VACUUM = constantIndex(1.0, "vacuum")
SIO2 = constantIndex(1.45, "SiO2")
SI1550 = constantIndex(3.464, "Si")
