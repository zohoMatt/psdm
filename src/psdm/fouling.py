"""Natural organic matter (NOM) fouling of the adsorbent.

Fouling reduces the Freundlich ``K`` over time by the multiplier

.. math::

    \\frac{K(t)}{K_0} = r_1 + r_2 t + r_3 e^{r_4 t}, \\qquad
    r_1 = b_1 a_1 + b_2,\\; r_2 = b_1 a_2,\\; r_3 = b_1 a_3,\\; r_4 = b_1 a_4

where ``a1..a4`` describe the water (:data:`WATER`) and ``b1, b2`` the compound class
(:data:`CHEMICAL`). ``t`` is in minutes. The tables are those of the AdDesignS database files
``water_co.txt`` and ``corr_com.txt``; the keys keep the names used by the US EPA PSDM code
("Portage" is AdDesignS's "Surface Water from the Keweenaw Waterway").
"""

from __future__ import annotations

import numpy as np

ORGANIC_FREE = "Organic Free"

WATER: dict[str, tuple[float, float, float, float]] = {
    ORGANIC_FREE: (1.0, 0.0, 0.0, 0.0),
    "Rhine": (0.35, -6.15e-8, 0.65, -8.93e-5),
    "Portage": (0.51, -9.21e-7, 0.49, -2.8e-5),
    "Karlsruhe": (0.65, -6.71e-7, 0.35, -1.0e-4),
    "Wausau": (0.83, -9.12e-7, 0.17, -2.65e-4),
    "Houghton": (0.66, -1.55e-7, 0.34, -1.04e-4),
}
"""Water-type coefficients ``(a1, a2, a3, a4)``."""

CHEMICAL: dict[str, tuple[float, float]] = {
    "halogenated alkanes": (1.2, -0.2),
    "halogenated alkenes": (1.0, 0.0),
    "trihalo-methanes": (1.0, 0.0),
    "aromatics": (0.9, 0.1),
    "nitro compounds": (0.75, 0.25),
    "chlorinated hydrocarbon": (0.59, 0.41),
    "acids": (0.65, 0.35),
    "phenols": (0.65, 0.35),
    "PNAs": (0.32, 0.68),
    "pesticides": (0.0, 0.05),
}
"""Compound-class coefficients ``(b1, b2)``."""

MIN_MULTIPLIER = 1e-3

TORTUOSITY_ONSET = 1.008e5
"""Minutes (70 days) after which the tortuosity correlation departs from 1."""


def chemical_coefficients(fouling_class: str | tuple[float, float]) -> tuple[float, float]:
    """Resolve ``(b1, b2)`` for a compound: a key of :data:`CHEMICAL` or an explicit pair."""
    if not isinstance(fouling_class, str):
        b1, b2 = fouling_class
        return float(b1), float(b2)
    try:
        return CHEMICAL[fouling_class]
    except KeyError:
        raise ValueError(
            f"unknown fouling class {fouling_class!r}; expected a (b1, b2) pair "
            f"or one of {sorted(CHEMICAL)}"
        ) from None


def water_coefficients(water: str) -> tuple[float, float, float, float]:
    """Return ``(a1, a2, a3, a4)`` for a water type in :data:`WATER`."""
    try:
        return WATER[water]
    except KeyError:
        raise ValueError(f"unknown water type {water!r}; expected one of {sorted(WATER)}") from None


def k_multiplier(t_min, water: str, b: tuple[float, float]):
    """Fraction of the virgin Freundlich ``K`` remaining after ``t_min`` minutes of fouling.

    The result is floored at ``1e-3``.
    """
    a1, a2, a3, a4 = water_coefficients(water)
    b1, b2 = b
    t = np.asarray(t_min, dtype=float)
    mult = (b1 * a1 + b2) + b1 * a2 * t + b1 * a3 * np.exp(b1 * a4 * t)
    return np.maximum(mult, MIN_MULTIPLIER)


def tortuosity(t_min):
    """Time-dependent pore tortuosity (``t_min`` in minutes).

    1 up to :data:`TORTUOSITY_ONSET`, then ``0.334 + 6.61e-6 t`` (AdDesignS ``DIFFUN``).
    """
    t = np.asarray(t_min, dtype=float)
    return np.where(t > TORTUOSITY_ONSET, 0.334 + 6.61e-6 * t, 1.0)
