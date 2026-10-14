"""Physical properties of water and dissolved compounds."""

from __future__ import annotations

import numpy as np


def water_viscosity(temperature: float) -> float:
    """Dynamic viscosity of water in g/(cm·s) at ``temperature`` (°C)."""
    t = temperature + 273.15
    return float(np.exp(-24.71 + 4209.0 / t + 0.04527 * t - 3.376e-5 * t**2) / 100.0)


def water_density(temperature: float) -> float:
    """Density of water in g/cm³ at ``temperature`` (°C)."""
    t = (temperature + 273.15) / 324.65
    return float(
        0.98396 * (-1.41768 + 8.97665 * t - 12.2755 * t**2 + 7.45844 * t**3 - 1.73849 * t**4)
    )


def liquid_diffusivity(molar_volume, viscosity: float):
    """Free-liquid diffusivity in cm²/s from the Hayduk–Laudie correlation.

    Parameters
    ----------
    molar_volume
        Molar volume at the normal boiling point, cm³/mol.
    viscosity
        Dynamic viscosity of water, g/(cm·s).
    """
    return 13.26e-5 / ((viscosity * 100.0) ** 1.14 * np.asarray(molar_volume) ** 0.589)
