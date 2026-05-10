from __future__ import annotations

from dataclasses import dataclass


TensorLike = object


@dataclass(frozen=True)
class ConstitutiveTensors:
    """Relative material tensors for anisotropic RCWA layers.

    The current solver implements the electric-magnetic subset
    ``D = epsilon E`` and ``B = mu H`` for homogeneous layers.  ``chi`` and
    ``xi`` are kept in the public data model so bi-anisotropic inputs can be
    rejected explicitly until the coupled Fourier factorization is implemented.
    """

    epsilon: TensorLike
    mu: TensorLike | None = None
    chi: TensorLike | None = None
    xi: TensorLike | None = None


def constitutiveTensors(
    epsilon: TensorLike,
    *,
    mu: TensorLike | None = None,
    chi: TensorLike | None = None,
    xi: TensorLike | None = None,
) -> ConstitutiveTensors:
    return ConstitutiveTensors(epsilon=epsilon, mu=mu, chi=chi, xi=xi)


def splitConstitutiveInput(
    epsilon: TensorLike,
    mu: TensorLike | None = None,
    chi: TensorLike | None = None,
    xi: TensorLike | None = None,
) -> tuple[TensorLike, TensorLike | None, TensorLike | None, TensorLike | None]:
    if isinstance(epsilon, ConstitutiveTensors):
        if mu is not None or chi is not None or xi is not None:
            raise ValueError("do not pass layer mu/chi/xi when epsilon is a ConstitutiveTensors object")
        return epsilon.epsilon, epsilon.mu, epsilon.chi, epsilon.xi
    return epsilon, mu, chi, xi


def magneticLayer(
    thickness: float,
    epsilon: TensorLike,
    mu: TensorLike,
    *,
    name: str = "",
    factorization: str = "auto",
) -> object:
    from .types import Layer

    return Layer(
        thickness=thickness,
        epsilon=epsilon,
        factorization=factorization,
        name=name,
        mu=mu,
    )
