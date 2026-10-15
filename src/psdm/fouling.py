"""Natural organic matter (NOM) fouling of the adsorbent.

Fouling reduces the Freundlich ``K`` over time by the multiplier

.. math::

    \\frac{K(t)}{K_0} = r_1 + r_2 t + r_3 e^{r_4 t}, \\qquad
    r_1 = b_1 a_1 + b_2,\\; r_2 = b_1 a_2,\\; r_3 = b_1 a_3,\\; r_4 = b_1 a_4

where ``a1..a4`` describe the water (:data:`WATER`) and ``b1, b2`` the compound class
(:data:`CHEMICAL`). ``t`` is in minutes. All tables are taken from the US EPA PSDM code, whose
comments state the water/compound-class values were edited to match AdDesignS and attribute
the PFAS values to a Burkhardt et al. (2020) manuscript submitted to AWWA Water Science. They
have not been checked against a primary publication.
"""

from __future__ import annotations

import numpy as np

ORGANIC_FREE = "Organic Free"

WATER: dict[str, tuple[float, float, float, float]] = {
    ORGANIC_FREE: (1.0, 0.0, 0.0, 0.0),
    "Rhine": (0.35, -6.15e-8, 0.65, -8.93e-5),
    "Portage": (0.510, -9.21e-7, 0.490, -2.80e-5),
    "Karlsruhe": (0.65, -6.71e-7, 0.35, -1.00e-4),
    "Wausau": (0.83, -9.12e-7, 0.17, -2.65e-4),
    "Houghton": (0.66, -1.55e-7, 0.34, -7.29e-5),
}
"""Water-type coefficients ``(a1, a2, a3, a4)``."""

CHEMICAL: dict[str, tuple[float, float]] = {
    "halogenated alkanes": (1.2, -0.2),
    "halogenated alkanes QSPR": (1.22, -0.12),
    "halogenated alkenes": (1.0, 0.0),
    "trihalo-methanes": (1.0, 0.0),
    "aromatics": (0.9, 0.1),
    "nitro compounds": (0.75, 0.25),
    "chlorinated hydrocarbon": (0.59, 0.41),
    "phenols": (0.65, 0.35),
    "PNAs": (0.32, 0.68),
    "pesticides": (0.0, 0.05),
}
"""Compound-class coefficients ``(b1, b2)``."""

PFAS: dict[str, tuple[float, float]] = {
    "PFBA": (0.82, 0.12),
    "PFPeA": (0.67, 0.19),
    "PFHxA": (0.55, 0.28),
    "PFHpA": (0.44, 0.36),
    "PFOA": (0.34, 0.44),
    "PFNA": (0.24, 0.53),
    "PFDA": (0.17, 0.61),
    "PFBS": (0.68, 0.24),
    "PFHxS": (0.44, 0.48),
    "PFHpS": (0.34, 0.57),
    "PFOS": (0.25, 0.66),
    "PFMOAA": (0.88, 0.055),
    "PFO2HxA": (0.63, 0.17),
    "PFO3OA": (0.43, 0.27),
    "GenX": (0.50, 0.34),
    "PFO4DA": (0.25, 0.37),
    "NafionBP2": (0.29, 0.63),
    "62FTS": (0.38, 0.62),
    "82FTS": (0.20, 0.79),
    # extrapolated / estimated from the values above
    "PFPrS": (0.63, 0.14),
    "PMPA": (0.63, 0.17),
    "PFPeS": (0.49, 0.35),
    "PFPrA": (0.88, 0.03),
}
"""Compound-specific ``(b1, b2)`` for PFAS (``fouling_class="PFAS"``)."""

PFAS_AVERAGE = (0.45, 0.41)

MIN_MULTIPLIER = 1e-3


def chemical_coefficients(
    fouling_class: str | tuple[float, float], name: str = ""
) -> tuple[float, float]:
    """Resolve ``(b1, b2)`` for a compound.

    ``fouling_class`` is a key of :data:`CHEMICAL`, an explicit ``(b1, b2)`` pair, or
    ``"PFAS"``, in which case ``name`` is looked up in :data:`PFAS` (falling back to the
    PFAS average).
    """
    if not isinstance(fouling_class, str):
        b1, b2 = fouling_class
        return float(b1), float(b2)
    if fouling_class == "PFAS":
        return PFAS.get(name, PFAS_AVERAGE)
    try:
        return CHEMICAL[fouling_class]
    except KeyError:
        raise ValueError(
            f"unknown fouling class {fouling_class!r}; expected 'PFAS', a (b1, b2) pair "
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
    """Time-dependent pore tortuosity in fouled water (``t_min`` in minutes), at least 1."""
    return np.maximum(0.334 + 6.61e-6 * np.asarray(t_min, dtype=float), 1.0)
