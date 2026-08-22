"""Unit conversion between NNCM canonical units and whatever the template asks for.

The template declares each column's unit in header row 6. Writing a barg value
into a column configured for ``psi`` is a silent, study-wide error that is very
hard to spot downstream, so the writer converts explicitly and refuses when it
cannot.
"""

from __future__ import annotations

from typing import Callable

CANONICAL = {
    "temperature": "degC",
    "pressure": "bar",   # gauge, matching the "Pressure (gauge)" column
    "length": "m",
    "diameter": "mm",
    "mass": "kg",
}

_TEMPERATURE: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
    "degc": (lambda v: v, lambda v: v),
    "degf": (lambda v: v * 9.0 / 5.0 + 32.0, lambda v: (v - 32.0) * 5.0 / 9.0),
    "degk": (lambda v: v + 273.15, lambda v: v - 273.15),
    "k": (lambda v: v + 273.15, lambda v: v - 273.15),
    "degr": (lambda v: (v + 273.15) * 9.0 / 5.0, lambda v: v * 5.0 / 9.0 - 273.15),
}

# Multiplicative factors: canonical value * factor = value in the target unit.
_PRESSURE_FROM_BAR = {
    "bar": 1.0,
    "barg": 1.0,
    "at": 1.019716,
    "atm": 0.986923,
    "psi": 14.503774,
    "kpa": 100.0,
    "mpa": 0.1,
    "pa": 1.0e5,
    "n/m2": 1.0e5,
    "dyne/cm2": 1.0e6,
    "mmhg": 750.0617,
    "torr": 750.0617,
}

_LENGTH_FROM_M = {
    "m": 1.0,
    "cm": 100.0,
    "mm": 1000.0,
    "micron": 1.0e6,
    "um": 1.0e6,
    "km": 1.0e-3,
    "in": 39.37008,
    "ft": 3.280840,
    "yd": 1.093613,
    "mile": 6.213712e-4,
    "nm": 1.0e9,
}

_MASS_FROM_KG = {
    "kg": 1.0,
    "g": 1000.0,
    "lb": 2.204623,
    "tonne": 1.0e-3,
    "ton": 1.0e-3,
    "uston": 1.102311e-3,
    "st": 0.157473,
    "oz": 35.27396,
}


class UnitError(ValueError):
    """Raised when a value cannot be expressed in the unit a column declares."""


def convert(value: float, quantity: str, target_unit: str) -> float:
    """Convert a canonical-unit value into ``target_unit`` for ``quantity``."""
    if value is None:
        return value
    unit = (target_unit or "").strip().casefold()
    if not unit:
        return float(value)

    if quantity == "temperature":
        if unit not in _TEMPERATURE:
            raise UnitError(f"unsupported temperature unit '{target_unit}'")
        return float(_TEMPERATURE[unit][0](float(value)))

    table = {
        "pressure": _PRESSURE_FROM_BAR,
        "length": _LENGTH_FROM_M,
        "diameter": _LENGTH_FROM_M,
        "mass": _MASS_FROM_KG,
    }.get(quantity)
    if table is None:
        raise UnitError(f"unknown quantity '{quantity}'")
    if unit not in table:
        raise UnitError(f"unsupported {quantity} unit '{target_unit}'")

    if quantity == "diameter":
        # canonical diameter is mm, the length table is metre-based
        return float(value) / 1000.0 * table[unit]
    return float(value) * table[unit]


def to_canonical(value: float, quantity: str, source_unit: str) -> float:
    """Inverse of :func:`convert` — read a sheet value back into canonical units."""
    unit = (source_unit or "").strip().casefold()
    if not unit:
        return float(value)
    if quantity == "temperature":
        if unit not in _TEMPERATURE:
            raise UnitError(f"unsupported temperature unit '{source_unit}'")
        return float(_TEMPERATURE[unit][1](float(value)))
    factor = convert(1.0, quantity, source_unit)
    if factor == 0:
        raise UnitError(f"degenerate conversion for {quantity} '{source_unit}'")
    return float(value) / factor
