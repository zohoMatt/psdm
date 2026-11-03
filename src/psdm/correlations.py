"""Film mass-transfer correlation.

The liquid-phase correlation of AdDesignS (``Correlations.vb``, function ``kf``), labelled
"Gnielinski" there and "Chern and Chien" in the US EPA PSDM code:

.. math::

    k_f = SF \\, (1 + 1.5 (1 - \\varepsilon)) \\frac{D_l}{2R}
          \\left(2 + 0.644\\, Re^{1/2} Sc^{1/3}\\right)
"""

from __future__ import annotations


def sherwood(re, sc, ebed):
    """Sherwood number ``Sh = kf * 2R / D_l`` (without the shape factor)."""
    return (1.0 + 1.5 * (1.0 - ebed)) * (2.0 + 0.644 * re**0.5 * sc ** (1 / 3))


def film_transfer(
    diffusivity, radius: float, re: float, sc, ebed: float, shape_factor: float = 1.0
):
    """Film transfer coefficient ``kf`` (cm/s).

    Parameters
    ----------
    diffusivity
        Liquid diffusivity, cm²/s.
    radius
        Adsorbent particle radius, cm.
    re, sc
        Particle Reynolds and Schmidt numbers.
    ebed
        Bed void fraction.
    shape_factor
        Particle shape factor.
    """
    return shape_factor * diffusivity / (2.0 * radius) * sherwood(re, sc, ebed)
