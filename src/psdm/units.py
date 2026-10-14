"""Unit conversion to the internal unit system.

Internally the model works in:

========================  ==============
quantity                  unit
========================  ==============
length, diameter, radius  cm
mass                      g
flow rate                 mL/min
density                   g/mL
time                      min
concentration             ug/L
========================  ==============
"""

from __future__ import annotations

LITERS_PER_GALLON = 3.785411784
CM_PER_FT = 30.48
G_PER_LB = 453.59237
MIN_PER_DAY = 1440.0
LB_FT3_PER_G_ML = 62.427973725314

_LENGTH = {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "ft": CM_PER_FT}
_MASS = {
    "g": 1.0,
    "gm": 1.0,
    "kg": 1e3,
    "lb": G_PER_LB,
    "ton": 2000 * G_PER_LB,
    "mton": 1e6,
    "metricton": 1e6,
}
_FLOW = {
    "ml/min": 1.0,
    "cm3/min": 1.0,
    "l/min": 1e3,
    "lpm": 1e3,
    "gal/min": LITERS_PER_GALLON * 1e3,
    "gpm": LITERS_PER_GALLON * 1e3,
    "mgd": LITERS_PER_GALLON * 1e9 / MIN_PER_DAY,
    "m3/min": 1e6,
    "ft3/min": CM_PER_FT**3,
}
_DENSITY = {"g/ml": 1.0, "g/cm3": 1.0, "kg/m3": 1e-3, "lb/ft3": 1 / LB_FT3_PER_G_ML}
_TIME = {
    "min": 1.0,
    "mins": 1.0,
    "minutes": 1.0,
    "h": 60.0,
    "hr": 60.0,
    "hrs": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    "d": MIN_PER_DAY,
    "day": MIN_PER_DAY,
    "days": MIN_PER_DAY,
}
_CONCENTRATION = {
    "ng/l": 1e-3,
    "ng": 1e-3,
    "ppt": 1e-3,
    "ug/l": 1.0,
    "ug": 1.0,
    "ppb": 1.0,
    "mg/l": 1e3,
    "mg": 1e3,
    "ppm": 1e3,
}

_TABLES = {
    "length": _LENGTH,
    "mass": _MASS,
    "flow": _FLOW,
    "density": _DENSITY,
    "time": _TIME,
    "concentration": _CONCENTRATION,
}


def factor(quantity: str, unit: str) -> float:
    """Return the multiplier that converts ``unit`` to the internal unit of ``quantity``.

    >>> factor("length", "m")
    100.0
    >>> factor("concentration", "ng/L")
    0.001
    """
    try:
        table = _TABLES[quantity]
    except KeyError:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {sorted(_TABLES)}"
        ) from None
    try:
        return table[unit.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown {quantity} unit {unit!r}; expected one of {sorted(table)}"
        ) from None


def convert(value: float, unit: str, quantity: str) -> float:
    """Convert ``value`` expressed in ``unit`` to the internal unit of ``quantity``."""
    return float(value) * factor(quantity, unit)
