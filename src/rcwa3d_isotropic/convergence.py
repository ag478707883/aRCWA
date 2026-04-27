from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .fields import fieldSliceXy, fieldSliceXz, stackFieldSliceXz
from .fourier import OrderSpec, normalizeOrders
from .solver import solveStack
from .types import CompiledLayer, Layer, RCWAResult
from .varrcwa import AdaptiveLayerSpec


@dataclass(frozen=True)
class OrderSweepPoint:
    orders: tuple[int, int]
    reflection: float
    transmission: float
    conservation: float
    deltaReflection: float | None
    deltaTransmission: float | None
    deltaConservation: float | None
    fieldDelta: float | None
    fieldRelativeDelta: float | None
    solvedBy: str
    result: RCWAResult


@dataclass(frozen=True)
class OrderSweepReport:
    points: tuple[OrderSweepPoint, ...]
    rtTolerance: float
    fieldTolerance: float | None = None

    @property
    def converged(self) -> bool:
        if len(self.points) < 2:
            return False
        last = self.points[-1]
        rtDelta = max(
            last.deltaReflection or 0.0,
            last.deltaTransmission or 0.0,
            last.deltaConservation or 0.0,
        )
        if rtDelta > self.rtTolerance:
            return False
        if self.fieldTolerance is not None:
            if last.fieldRelativeDelta is None:
                return False
            return last.fieldRelativeDelta <= self.fieldTolerance
        return True

    def rows(self) -> list[dict[str, float | str | tuple[int, int] | None]]:
        return [
            {
                "orders": point.orders,
                "R": point.reflection,
                "T": point.transmission,
                "R+T": point.conservation,
                "dR": point.deltaReflection,
                "dT": point.deltaTransmission,
                "d(R+T)": point.deltaConservation,
                "dField": point.fieldDelta,
                "relField": point.fieldRelativeDelta,
                "solvedBy": point.solvedBy,
            }
            for point in self.points
        ]

    def table(self) -> str:
        lines = [
            "orders       R             T             R+T           dR            dT            dField(rel)  solvedBy"
        ]
        for point in self.points:
            field = "-" if point.fieldRelativeDelta is None else f"{point.fieldRelativeDelta:.3e}"
            dR = "-" if point.deltaReflection is None else f"{point.deltaReflection:.3e}"
            dT = "-" if point.deltaTransmission is None else f"{point.deltaTransmission:.3e}"
            lines.append(
                f"{point.orders!s:<12} "
                f"{point.reflection:<13.6g} "
                f"{point.transmission:<13.6g} "
                f"{point.conservation:<13.6g} "
                f"{dR:<13} "
                f"{dT:<13} "
                f"{field:<12} "
                f"{point.solvedBy}"
            )
        return "\n".join(lines)


def sweepOrders(
    *,
    layers: Sequence[Layer | CompiledLayer | AdaptiveLayerSpec],
    wavelength: float,
    period: tuple[float, float],
    orderSequence: Iterable[OrderSpec] | None = None,
    maxOrder: int | None = None,
    epsIncident: complex = 1.0,
    epsTransmission: complex = 1.0,
    theta: float = 0.0,
    phi: float = 0.0,
    sAmplitude: complex = 1.0,
    pAmplitude: complex = 0.0,
    method: str = "smatrix",
    truncation: str = "circular",
    backend: str | object | None = "cuda",
    rtTolerance: float = 1e-3,
    fieldTolerance: float | None = None,
    fieldComponent: str | None = None,
    fieldPlane: str = "xz",
    fieldLayer: int = 0,
    fieldZ: float | None = None,
    fieldY: float = 0.0,
    fieldShape: tuple[int, int] = (121, 121),
    fieldXSpan: tuple[float, float] | None = None,
    fieldZSpan: tuple[float, float] | None = None,
    fieldZPadding: float | tuple[float, float] | None = None,
) -> OrderSweepReport:
    """Sweep Fourier orders and report adjacent R/T and optional field changes."""

    sequence = _orderSequence(orderSequence, maxOrder)
    previousResult: RCWAResult | None = None
    previousField: np.ndarray | None = None
    points: list[OrderSweepPoint] = []

    for orders in sequence:
        normalized = normalizeOrders(orders)
        result = solveStack(
            layers=layers,
            wavelength=wavelength,
            period=period,
            orders=normalized,
            epsIncident=epsIncident,
            epsTransmission=epsTransmission,
            theta=theta,
            phi=phi,
            sAmplitude=sAmplitude,
            pAmplitude=pAmplitude,
            returnFields=fieldComponent is not None,
            method=method,
            truncation=truncation,
            backend=backend,
        )
        fieldMap = (
            _sampleField(
                result,
                component=fieldComponent,
                plane=fieldPlane,
                layerIndex=fieldLayer,
                z=fieldZ,
                y=fieldY,
                shape=fieldShape,
                xSpan=fieldXSpan,
                zSpan=fieldZSpan,
                zPadding=fieldZPadding,
            )
            if fieldComponent is not None
            else None
        )
        fieldDelta, fieldRelativeDelta = _fieldDifference(fieldMap, previousField)
        points.append(
            OrderSweepPoint(
                orders=normalized,
                reflection=result.reflection,
                transmission=result.transmission,
                conservation=result.conservation,
                deltaReflection=None
                if previousResult is None
                else abs(result.reflection - previousResult.reflection),
                deltaTransmission=None
                if previousResult is None
                else abs(result.transmission - previousResult.transmission),
                deltaConservation=None
                if previousResult is None
                else abs(result.conservation - previousResult.conservation),
                fieldDelta=fieldDelta,
                fieldRelativeDelta=fieldRelativeDelta,
                solvedBy=result.solvedBy,
                result=result,
            )
        )
        previousResult = result
        previousField = fieldMap

    return OrderSweepReport(
        points=tuple(points),
        rtTolerance=float(rtTolerance),
        fieldTolerance=None
        if fieldComponent is None
        else float(rtTolerance if fieldTolerance is None else fieldTolerance),
    )


def _orderSequence(orderSequence: Iterable[OrderSpec] | None, maxOrder: int | None) -> tuple[OrderSpec, ...]:
    if orderSequence is not None:
        sequence = tuple(orderSequence)
    elif maxOrder is not None:
        if maxOrder < 0:
            raise ValueError("maxOrder must be non-negative")
        sequence = tuple(range(maxOrder + 1))
    else:
        raise ValueError("provide orderSequence or maxOrder")
    if not sequence:
        raise ValueError("order sweep requires at least one order")
    return sequence


def _sampleField(
    result: RCWAResult,
    *,
    component: str | None,
    plane: str,
    layerIndex: int,
    z: float | None,
    y: float,
    shape: tuple[int, int],
    xSpan: tuple[float, float] | None,
    zSpan: tuple[float, float] | None,
    zPadding: float | tuple[float, float] | None,
) -> np.ndarray:
    if component is None:
        raise ValueError("component is required for field sampling")
    normalized = plane.lower().replace("_", "-")
    if normalized == "xy":
        if z is None:
            z = 0.5 * result.layerSolutions[layerIndex].thickness
        _x, _y, values = fieldSliceXy(result, layerIndex=layerIndex, z=z, component=component, shape=shape)
    elif normalized == "xz":
        _x, _z, values = fieldSliceXz(result, layerIndex=layerIndex, y=y, component=component, shape=shape)
    elif normalized in ("stack-xz", "stack"):
        _x, _z, values = stackFieldSliceXz(
            result,
            y=y,
            xSpan=xSpan,
            zSpan=zSpan,
            zPadding=zPadding,
            component=component,
            shape=shape,
        )
    else:
        raise ValueError("fieldPlane must be 'xy', 'xz', or 'stack-xz'")
    return np.asarray(values)


def _fieldDifference(
    current: np.ndarray | None,
    previous: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if current is None or previous is None:
        return None, None
    difference = float(np.max(np.abs(current - previous)))
    reference = float(np.max(np.abs(previous)))
    return difference, difference / max(reference, 1e-300)
