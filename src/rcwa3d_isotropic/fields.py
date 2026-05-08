from __future__ import annotations

import numpy as np

from .fourier import forwardKz, planeWaveFields
from .types import LayerFieldSolution, RCWAResult


FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def reconstructFourierXy(
    coefficients: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    wavelength: float,
    period: tuple[float, float],
    shape: tuple[int, int] = (121, 121),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct one Fourier-space field component on an x/y plane."""

    del mx, my
    periodX, periodY = period
    ny, nx = shape
    x = np.linspace(-periodX / 2, periodX / 2, nx)
    y = np.linspace(-periodY / 2, periodY / 2, ny)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, reconstructFourierGrid(coefficients, kx, ky, wavelength, xx, yy)


def reconstructFourierGrid(
    coefficients: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    wavelength: float,
    x: np.ndarray,
    y: np.ndarray | float,
) -> np.ndarray:
    """Reconstruct one Fourier component on an arbitrary x/y grid."""

    k0 = 2 * np.pi / wavelength
    xArray = np.asarray(x)
    yArray = np.asarray(y)
    field = np.zeros(np.broadcast_shapes(xArray.shape, yArray.shape), dtype=complex)
    for coefficient, kxValue, kyValue in zip(coefficients, kx, ky):
        field += coefficient * np.exp(1j * k0 * (kxValue * xArray + kyValue * yArray))
    return field


def layerFourierFields(layerSolution: LayerFieldSolution, z: float) -> dict[str, np.ndarray]:
    """Return full electric and magnetic Fourier coefficients inside one layer.

    The stored modal state contains tangential coefficients.  Longitudinal
    terms are recovered from Maxwell constraints:

    ``Hz = kx * Ey - ky * Ex`` and
    ``Ez = epsilon^{-1} * (ky * Hx - kx * Hy)``.
    """

    if z < -1e-12 or z > layerSolution.thickness + 1e-12:
        raise ValueError("z must be inside the selected layer")
    n = layerSolution.mx.size
    modalCount = 2 * n
    k0 = 2 * np.pi / layerSolution.wavelength
    backwardRight = getattr(layerSolution, "backwardCoefficientsRight", None)
    if backwardRight is None:
        phase = np.exp(1j * layerSolution.qValues * k0 * z)
        values = layerSolution.modeMatrix @ (phase * layerSolution.coefficients)
    else:
        qForward = layerSolution.qValues[:modalCount]
        forward = layerSolution.coefficients[:modalCount]
        forwardValues = layerSolution.modeMatrix[:, :modalCount] @ (np.exp(1j * qForward * k0 * z) * forward)
        backwardValues = layerSolution.modeMatrix[:, modalCount:] @ (
            np.exp(1j * qForward * k0 * (layerSolution.thickness - z)) * backwardRight
        )
        values = forwardValues + backwardValues
    ex = values[:n]
    ey = values[n : 2 * n]
    hx = values[2 * n : 3 * n]
    hy = values[3 * n :]
    return _completeFourierFields(
        ex=ex,
        ey=ey,
        hx=hx,
        hy=hy,
        kx=layerSolution.kx,
        ky=layerSolution.ky,
        epsilonInverse=layerSolution.epsilonInverse,
    )


def homogeneousFourierFields(
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    sAmplitudes: np.ndarray,
    pAmplitudes: np.ndarray,
    epsilon: complex,
    z: float,
    wavelength: float,
) -> dict[str, np.ndarray]:
    """Return full Fourier fields in a homogeneous half-space.

    ``z`` is measured from the interface where the amplitudes are defined.
    The caller supplies the signed ``kz`` branch for each diffraction order.
    """

    k0 = 2 * np.pi / wavelength
    ex = np.zeros_like(kx, dtype=complex)
    ey = np.zeros_like(kx, dtype=complex)
    hx = np.zeros_like(kx, dtype=complex)
    hy = np.zeros_like(kx, dtype=complex)
    for index, (kxValue, kyValue, kzValue, sAmplitude, pAmplitude) in enumerate(
        zip(kx, ky, kz, sAmplitudes, pAmplitudes)
    ):
        sField, pField = planeWaveFields(kxValue, kyValue, kzValue, epsilon)
        phase = np.exp(1j * k0 * kzValue * z)
        field = phase * (sAmplitude * sField + pAmplitude * pField)
        ex[index] = field[0]
        ey[index] = field[1]
        hx[index] = field[2]
        hy[index] = field[3]
    return _completeFourierFields(ex=ex, ey=ey, hx=hx, hy=hy, kx=kx, ky=ky, epsilon=epsilon)


def fieldSliceXy(
    result: RCWAResult,
    layerIndex: int,
    z: float,
    component: str = "Ex",
    shape: tuple[int, int] = (121, 121),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct one x/y field map from an RCWA result.

    ``component`` may be one of ``Ex/Ey/Ez/Hx/Hy/Hz`` or a scalar-field alias
    such as ``EMagnitude`` / ``HIntensity``.
    """

    layer = _resultLayer(result, layerIndex)
    fourierFields = layerFourierFields(layer, z)
    maps = _fieldMapsXy(layer, fourierFields, shape)
    _addIncidentNormalizedIntensities(maps, result)
    return maps["x"], maps["y"], _selectComponent(maps, component)


def fieldComponentsXy(
    result: RCWAResult,
    layerIndex: int,
    z: float,
    shape: tuple[int, int] = (121, 121),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return all six field components and intensities on an x/y plane."""

    layer = _resultLayer(result, layerIndex)
    maps = _fieldMapsXy(layer, layerFourierFields(layer, z), shape)
    _addIncidentNormalizedIntensities(maps, result)
    x = maps.pop("x")
    y = maps.pop("y")
    return x, y, maps


def stackFieldSliceXy(
    result: RCWAResult,
    *,
    z: float,
    xSpan: tuple[float, float] | None = None,
    ySpan: tuple[float, float] | None = None,
    component: str = "EIntensity",
    shape: tuple[int, int] = (121, 121),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct one x/y map at a global stack ``z`` position.

    ``z`` is measured from the first finite-layer interface.  Negative values
    sample the incident half-space, values inside the finite stack sample the
    corresponding layer, and values above the total thickness sample the
    transmitted half-space.
    """

    x, y, maps = stackFieldComponentsXy(
        result,
        z=z,
        xSpan=xSpan,
        ySpan=ySpan,
        shape=shape,
    )
    return x, y, _selectComponent(maps, component)


def stackFieldComponentsXy(
    result: RCWAResult,
    *,
    z: float,
    xSpan: tuple[float, float] | None = None,
    ySpan: tuple[float, float] | None = None,
    shape: tuple[int, int] = (121, 121),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return full fields on an x/y plane through the whole stack."""

    if not result.layerSolutions:
        raise ValueError("result does not contain layer fields; solve with returnFields=True")
    reference = result.layerSolutions[0]
    periodX, periodY = reference.period
    if xSpan is None:
        xSpan = (-periodX / 2, periodX / 2)
    if ySpan is None:
        ySpan = (-periodY / 2, periodY / 2)

    ny, nx = shape
    xValues = np.linspace(float(xSpan[0]), float(xSpan[1]), nx)
    yValues = np.linspace(float(ySpan[0]), float(ySpan[1]), ny)
    xx, yy = np.meshgrid(xValues, yValues)
    totalThickness, layerStarts, layerEnds = _stackLayerBounds(result)
    fourierFields = _stackFourierFieldsAtZ(result, reference, layerStarts, layerEnds, totalThickness, float(z))

    maps = {
        component: reconstructFourierGrid(
            fourierFields[component],
            reference.kx,
            reference.ky,
            reference.wavelength,
            xx,
            yy,
        )
        for component in FIELD_COMPONENTS
    }
    maps["x"] = xx
    maps["y"] = yy
    _addIntensities(maps)
    _addIncidentNormalizedIntensities(maps, result)
    x = maps.pop("x")
    y = maps.pop("y")
    return x, y, maps


def fieldSliceXz(
    result: RCWAResult,
    layerIndex: int,
    y: float = 0.0,
    component: str = "Ex",
    shape: tuple[int, int] = (161, 121),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct one x/z field map inside a finite layer."""

    layer = _resultLayer(result, layerIndex)
    maps = _fieldMapsXz(layer, y, shape)
    _addIncidentNormalizedIntensities(maps, result)
    return maps["x"], maps["z"], _selectComponent(maps, component)


def fieldComponentsXz(
    result: RCWAResult,
    layerIndex: int,
    y: float = 0.0,
    shape: tuple[int, int] = (161, 121),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return all six field components and intensities inside a finite layer."""

    layer = _resultLayer(result, layerIndex)
    maps = _fieldMapsXz(layer, y, shape)
    _addIncidentNormalizedIntensities(maps, result)
    x = maps.pop("x")
    z = maps.pop("z")
    return x, z, maps


def stackFieldSliceXz(
    result: RCWAResult,
    *,
    y: float = 0.0,
    xSpan: tuple[float, float] | None = None,
    zSpan: tuple[float, float] | None = None,
    zPadding: float | tuple[float, float] | None = None,
    component: str = "EIntensity",
    shape: tuple[int, int] = (241, 181),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct one x/z map across incident, layer, and transmission regions.

    The default plane is the center plane ``y=0``.  ``xSpan`` defaults to one
    unit-cell period, and ``zSpan`` defaults to the full stack with a small
    free-space margin above and below.  ``component`` supports
    ``Ex/Ey/Ez/Hx/Hy/Hz``, ``realEx`` style real-part aliases, magnitude
    aliases such as ``HMagnitude``, and intensity aliases such as
    ``EIntensity`` / ``E2``.
    """

    x, z, maps = stackFieldComponentsXz(
        result,
        y=y,
        xSpan=xSpan,
        zSpan=zSpan,
        zPadding=zPadding,
        shape=shape,
    )
    return x, z, _selectComponent(maps, component)


def stackFieldComponentsXz(
    result: RCWAResult,
    *,
    y: float = 0.0,
    xSpan: tuple[float, float] | None = None,
    zSpan: tuple[float, float] | None = None,
    zPadding: float | tuple[float, float] | None = None,
    shape: tuple[int, int] = (241, 181),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return full fields on a center x/z plane through the whole stack."""

    if not result.layerSolutions:
        raise ValueError("result does not contain layer fields; solve with returnFields=True")
    reference = result.layerSolutions[0]
    periodX, _periodY = reference.period
    totalThickness = float(sum(layer.thickness for layer in result.layerSolutions))
    if xSpan is None:
        xSpan = (-periodX / 2, periodX / 2)
    if zSpan is None:
        before, after = _normalizePadding(zPadding, reference.wavelength)
        zSpan = (-before, totalThickness + after)

    nz, nx = shape
    xValues = np.linspace(float(xSpan[0]), float(xSpan[1]), nx)
    zValues = np.linspace(float(zSpan[0]), float(zSpan[1]), nz)
    xx, zz = np.meshgrid(xValues, zValues)
    maps = {component: np.zeros_like(xx, dtype=complex) for component in FIELD_COMPONENTS}

    totalThickness, layerStarts, layerEnds = _stackLayerBounds(result)

    for row, zValue in enumerate(zValues):
        fourierFields = _stackFourierFieldsAtZ(
            result,
            reference,
            layerStarts,
            layerEnds,
            totalThickness,
            float(zValue),
        )

        for component in FIELD_COMPONENTS:
            maps[component][row, :] = reconstructFourierGrid(
                fourierFields[component],
                reference.kx,
                reference.ky,
                reference.wavelength,
                xValues,
                y,
            )

    maps["x"] = xx
    maps["z"] = zz
    _addIntensities(maps)
    _addIncidentNormalizedIntensities(maps, result)
    x = maps.pop("x")
    z = maps.pop("z")
    return x, z, maps


def incidentFieldIntensities(result: RCWAResult) -> dict[str, float]:
    """Return incident zero-order electric and magnetic intensities.

    Fields in this package use the standard RCWA normalization where magnetic
    fields are scaled by the free-space admittance.  These values are therefore
    the correct denominators for ``|E/E0|^2`` and ``|H/H0|^2`` style maps for
    the incident medium, angle, and requested s/p amplitudes.
    """

    zeroOrder = None
    for order in result.orders:
        if order.mx == 0 and order.my == 0:
            zeroOrder = order
            break
    if zeroOrder is None:
        raise RuntimeError("zero diffraction order was not found")

    kx = np.asarray([zeroOrder.kx], dtype=complex)
    ky = np.asarray([zeroOrder.ky], dtype=complex)
    kz = forwardKz(result.epsIncident - kx**2 - ky**2)
    sField, pField = planeWaveFields(kx[0], ky[0], kz[0], result.epsIncident)
    tangential = result.sAmplitude * sField + result.pAmplitude * pField
    fields = _completeFourierFields(
        ex=np.asarray([tangential[0]], dtype=complex),
        ey=np.asarray([tangential[1]], dtype=complex),
        hx=np.asarray([tangential[2]], dtype=complex),
        hy=np.asarray([tangential[3]], dtype=complex),
        kx=kx,
        ky=ky,
        epsilon=result.epsIncident,
    )
    electric = float(np.real(np.abs(fields["Ex"][0]) ** 2 + np.abs(fields["Ey"][0]) ** 2 + np.abs(fields["Ez"][0]) ** 2))
    magnetic = float(np.real(np.abs(fields["Hx"][0]) ** 2 + np.abs(fields["Hy"][0]) ** 2 + np.abs(fields["Hz"][0]) ** 2))
    return {"EIntensity": electric, "HIntensity": magnetic}


def _completeFourierFields(
    *,
    ex: np.ndarray,
    ey: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    epsilon: complex | None = None,
    epsilonInverse: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    hz = kx * ey - ky * ex
    dz = ky * hx - kx * hy
    if epsilonInverse is not None:
        ez = epsilonInverse @ dz
    elif epsilon is not None:
        ez = dz / epsilon
    else:
        raise ValueError("epsilon or epsilonInverse is required to recover Ez")
    return {"Ex": ex, "Ey": ey, "Ez": ez, "Hx": hx, "Hy": hy, "Hz": hz}


def _fieldMapsXy(
    layer: LayerFieldSolution,
    fourierFields: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for component in FIELD_COMPONENTS:
        maps["x"], maps["y"], maps[component] = reconstructFourierXy(
            fourierFields[component],
            mx=layer.mx,
            my=layer.my,
            kx=layer.kx,
            ky=layer.ky,
            wavelength=layer.wavelength,
            period=layer.period,
            shape=shape,
        )
    _addIntensities(maps)
    return maps


def _fieldMapsXz(layer: LayerFieldSolution, y: float, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    nz, nx = shape
    periodX, _periodY = layer.period
    x = np.linspace(-periodX / 2, periodX / 2, nx)
    zValues = np.linspace(0.0, layer.thickness, nz)
    xx, zz = np.meshgrid(x, zValues)
    maps = {component: np.zeros_like(xx, dtype=complex) for component in FIELD_COMPONENTS}
    k0 = 2 * np.pi / layer.wavelength
    lateralPhase = np.exp(1j * k0 * (layer.kx[:, None] * x[None, :] + layer.ky[:, None] * y))
    for row, zValue in enumerate(zValues):
        fourierFields = layerFourierFields(layer, float(zValue))
        for component in FIELD_COMPONENTS:
            maps[component][row, :] = np.sum(fourierFields[component][:, None] * lateralPhase, axis=0)
    maps["x"] = xx
    maps["z"] = zz
    _addIntensities(maps)
    return maps


def _addIntensities(maps: dict[str, np.ndarray]) -> None:
    maps["EIntensity"] = np.maximum(
        np.real(
            np.abs(maps["Ex"]) ** 2 + np.abs(maps["Ey"]) ** 2 + np.abs(maps["Ez"]) ** 2
        ),
        0.0,
    )
    maps["HIntensity"] = np.maximum(
        np.real(
            np.abs(maps["Hx"]) ** 2 + np.abs(maps["Hy"]) ** 2 + np.abs(maps["Hz"]) ** 2
        ),
        0.0,
    )
    maps["EMagnitude"] = np.sqrt(maps["EIntensity"])
    maps["HMagnitude"] = np.sqrt(maps["HIntensity"])
    maps["ESelfNormalizedMagnitude"] = _normalizeByOwnMaximum(maps["EMagnitude"])
    maps["HSelfNormalizedMagnitude"] = _normalizeByOwnMaximum(maps["HMagnitude"])


def _normalizeByOwnMaximum(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.asarray(values, dtype=float))) if values.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values, dtype=float)
    return np.asarray(values, dtype=float) / scale


def _addIncidentNormalizedIntensities(maps: dict[str, np.ndarray], result: RCWAResult) -> None:
    incident = incidentFieldIntensities(result)
    electric = max(float(incident["EIntensity"]), 1e-300)
    magnetic = max(float(incident["HIntensity"]), 1e-300)
    maps["ENormalizedIntensity"] = maps["EIntensity"] / electric
    maps["HNormalizedIntensity"] = maps["HIntensity"] / magnetic
    maps["ENormalizedMagnitude"] = maps["EMagnitude"] / np.sqrt(electric)
    maps["HNormalizedMagnitude"] = maps["HMagnitude"] / np.sqrt(magnetic)


def _selectComponent(maps: dict[str, np.ndarray], component: str) -> np.ndarray:
    normalized = "".join(character for character in component.lower() if character.isalnum())
    aliases = {
        "ex": "Ex",
        "ey": "Ey",
        "ez": "Ez",
        "hx": "Hx",
        "hy": "Hy",
        "hz": "Hz",
        "e": "EMagnitude",
        "h": "HMagnitude",
        "abse": "EMagnitude",
        "absh": "HMagnitude",
        "emagnitude": "EMagnitude",
        "hmagnitude": "HMagnitude",
        "efieldmagnitude": "EMagnitude",
        "hfieldmagnitude": "HMagnitude",
        "e2": "EIntensity",
        "eintensity": "EIntensity",
        "electricintensity": "EIntensity",
        "abse2": "EIntensity",
        "h2": "HIntensity",
        "hintensity": "HIntensity",
        "magneticintensity": "HIntensity",
        "absh2": "HIntensity",
        "enormalizedintensity": "ENormalizedIntensity",
        "enormalisedintensity": "ENormalizedIntensity",
        "ee02": "ENormalizedIntensity",
        "eovere02": "ENormalizedIntensity",
        "eenhancement2": "ENormalizedIntensity",
        "hnormalizedintensity": "HNormalizedIntensity",
        "hnormalisedintensity": "HNormalizedIntensity",
        "hh02": "HNormalizedIntensity",
        "hoverh02": "HNormalizedIntensity",
        "henhancement2": "HNormalizedIntensity",
        "enormalizedmagnitude": "ENormalizedMagnitude",
        "enormalisedmagnitude": "ENormalizedMagnitude",
        "ee0": "ENormalizedMagnitude",
        "eovere0": "ENormalizedMagnitude",
        "eenhancement": "ENormalizedMagnitude",
        "hnormalizedmagnitude": "HNormalizedMagnitude",
        "hnormalisedmagnitude": "HNormalizedMagnitude",
        "hh0": "HNormalizedMagnitude",
        "hoverh0": "HNormalizedMagnitude",
        "henhancement": "HNormalizedMagnitude",
        "eselfnormalizedmagnitude": "ESelfNormalizedMagnitude",
        "eselfnormalisedmagnitude": "ESelfNormalizedMagnitude",
        "eselfnormalized": "ESelfNormalizedMagnitude",
        "eselfnormalised": "ESelfNormalizedMagnitude",
        "eoverepeak": "ESelfNormalizedMagnitude",
        "eoveremax": "ESelfNormalizedMagnitude",
        "hselfnormalizedmagnitude": "HSelfNormalizedMagnitude",
        "hselfnormalisedmagnitude": "HSelfNormalizedMagnitude",
        "hselfnormalized": "HSelfNormalizedMagnitude",
        "hselfnormalised": "HSelfNormalizedMagnitude",
        "hoverhpeak": "HSelfNormalizedMagnitude",
        "hoverhmax": "HSelfNormalizedMagnitude",
    }
    if normalized.startswith("real"):
        key = aliases.get(normalized[4:])
        if key is None:
            raise ValueError("real component must be realEx/realEy/realEz/realHx/realHy/realHz")
        return np.real(maps[key])
    if normalized.startswith("imag"):
        key = aliases.get(normalized[4:])
        if key is None:
            raise ValueError("imaginary component must be imagEx/imagEy/imagEz/imagHx/imagHy/imagHz")
        return np.imag(maps[key])
    key = aliases.get(normalized)
    if key is None:
        raise ValueError(
            "component must be Ex/Ey/Ez/Hx/Hy/Hz, a magnitude, or an intensity/normalized-intensity alias"
        )
    return maps[key]


def _resultLayer(result: RCWAResult, layerIndex: int) -> LayerFieldSolution:
    if not result.layerSolutions:
        raise ValueError("result does not contain layer fields; call RCWASimulation.solve(..., returnFields=True)")
    return result.layerSolutions[layerIndex]


def _incidentAmplitudeArrays(result: RCWAResult, layer: LayerFieldSolution) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(layer.mx.size, dtype=complex)
    p = np.zeros(layer.mx.size, dtype=complex)
    zero = np.where((layer.mx == 0) & (layer.my == 0))[0]
    if zero.size != 1:
        raise RuntimeError("zero diffraction order was not found")
    s[int(zero[0])] = result.sAmplitude
    p[int(zero[0])] = result.pAmplitude
    return s, p


def _stackLayerBounds(result: RCWAResult) -> tuple[float, np.ndarray, np.ndarray]:
    layerStarts = np.concatenate([[0.0], np.cumsum([layer.thickness for layer in result.layerSolutions[:-1]])])
    layerEnds = np.cumsum([layer.thickness for layer in result.layerSolutions])
    return float(layerEnds[-1]), layerStarts, layerEnds


def _stackFourierFieldsAtZ(
    result: RCWAResult,
    reference: LayerFieldSolution,
    layerStarts: np.ndarray,
    layerEnds: np.ndarray,
    totalThickness: float,
    z: float,
) -> dict[str, np.ndarray]:
    kx = reference.kx
    ky = reference.ky
    if z < 0.0:
        kzIncident = forwardKz(result.epsIncident - kx**2 - ky**2)
        incidentS, incidentP = _incidentAmplitudeArrays(result, reference)
        incident = homogeneousFourierFields(
            kx,
            ky,
            kzIncident,
            incidentS,
            incidentP,
            result.epsIncident,
            z,
            reference.wavelength,
        )
        reflected = homogeneousFourierFields(
            kx,
            ky,
            np.array([order.kzReflected for order in result.orders], dtype=complex),
            result.rAmplitudes[0::2],
            result.rAmplitudes[1::2],
            result.epsIncident,
            z,
            reference.wavelength,
        )
        return {component: incident[component] + reflected[component] for component in FIELD_COMPONENTS}

    if z > totalThickness:
        return homogeneousFourierFields(
            kx,
            ky,
            np.array([order.kzTransmitted for order in result.orders], dtype=complex),
            result.tAmplitudes[0::2],
            result.tAmplitudes[1::2],
            result.epsTransmission,
            z - totalThickness,
            reference.wavelength,
        )

    layerIndex = _layerIndexForZ(z, layerEnds)
    localZ = float(z - layerStarts[layerIndex])
    return layerFourierFields(result.layerSolutions[layerIndex], localZ)


def _normalizePadding(
    zPadding: float | tuple[float, float] | None,
    wavelength: float,
) -> tuple[float, float]:
    if zPadding is None:
        value = 0.25 * float(wavelength)
        return value, value
    if np.isscalar(zPadding):
        value = float(zPadding)
        return value, value
    return float(zPadding[0]), float(zPadding[1])


def _layerIndexForZ(z: float, layerEnds: np.ndarray) -> int:
    index = int(np.searchsorted(layerEnds, z, side="right"))
    return min(index, layerEnds.size - 1)
