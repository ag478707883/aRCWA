from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def save_if_requested(fig: Figure, path: str | Path | None) -> None:
    if path is None:
        return
    outputPath = Path(path)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath, bbox_inches="tight")


def plotEpsilon(
    epsilon: np.ndarray,
    period: tuple[float, float],
    path: str | Path | None = None,
    title: str = "Unit-cell permittivity",
) -> Figure:
    periodX, periodY = period
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    image = ax.imshow(
        np.real(epsilon),
        origin="lower",
        extent=(-periodX / 2, periodX / 2, -periodY / 2, periodY / 2),
        cmap="viridis",
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label="Re(epsilon)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    save_if_requested(fig, path)
    return fig


def plotSpectrum(
    x: np.ndarray,
    reflection: np.ndarray,
    transmission: np.ndarray,
    path: str | Path | None = None,
    xlabel: str = "Wavelength",
    title: str = "RCWA spectrum",
    conservation: np.ndarray | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.plot(x, transmission, color="#2B8CBE", linewidth=2.1, label="Transmission")
    ax.plot(x, reflection, color="#31A354", linewidth=2.1, label="Reflection")
    if conservation is not None:
        ax.plot(x, conservation, color="#555555", linewidth=1.2, linestyle="--", label="R + T")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized power")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    save_if_requested(fig, path)
    return fig


def plotField(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    path: str | Path | None = None,
    title: str = "Field",
    quantity: str = "real",
) -> Figure:
    if quantity == "real":
        values = np.real(field)
        label = "real(field)"
        cmap = "RdBu_r"
    elif quantity == "abs":
        values = np.abs(field)
        label = "|field|"
        cmap = "magma"
    else:
        raise ValueError("quantity must be 'real' or 'abs'")

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    image = ax.imshow(
        values,
        origin="lower",
        extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())),
        cmap=cmap,
        aspect="equal",
    )
    fig.colorbar(image, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    save_if_requested(fig, path)
    return fig
