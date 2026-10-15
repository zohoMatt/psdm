"""Film mass-transfer correlations.

Each correlation returns the Sherwood number ``Sh = kf * d_p / D_l``; :func:`film_transfer`
turns it into the film transfer coefficient ``kf`` (cm/s). The set and the names follow the US
EPA PSDM code, which points to the review by Xu, Cai & Pan (2013), "Mathematically modeling
fixed-bed adsorption in aqueous systems", *J. Zhejiang Univ. Sci. A* 14(3):155–176,
doi:10.1631/jzus.A1300029. Attributions to the original papers are as labelled there and
have not been individually verified.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _ohashi(re, sc, ebed):
    re = np.asarray(re, dtype=float)
    regimes = [re < 5.8, re < 500]
    coef = np.select(regimes, [1.58, 1.21], 0.59)
    exponent = np.select(regimes, [0.4, 0.5], 0.6)
    return 2.0 + coef * re**exponent * sc ** (1 / 3)


def _gnielinski(re, sc, ebed):
    sh_lam = 0.644 * re**0.5 * sc ** (1 / 3)
    sh_turb = 0.037 * re**0.8 * sc / (1.0 + 2.443 * re**-0.1 * (sc ** (2 / 3) - 1.0))
    return (2.0 + np.sqrt(sh_lam**2 + sh_turb**2)) * (1.0 + 1.5 * (1.0 - ebed))


SHERWOOD: dict[str, Callable] = {
    # modified Gnielinski form, labelled "Chern and Chien" in the EPA code; the PSDM default
    "Chern and Chien": lambda re, sc, eb: (
        (1 + 1.5 * (1 - eb)) * (2 + 0.644 * re**0.5 * sc ** (1 / 3))
    ),
    "Gnielinski": _gnielinski,
    "Tan et al": lambda re, sc, eb: 1.1 / eb * (re * sc) ** (1 / 3),
    "Wilson and Geankoplis": lambda re, sc, eb: 1.09 / eb * (re * sc) ** (1 / 3),
    "Ohashi et al": _ohashi,
    "Williamson et al": lambda re, sc, eb: 2.4 * eb * re**0.3 * sc**0.42,
    "Ko et al": lambda re, sc, eb: 0.325 / (eb * re**0.36 * sc ** (1 / 3)),
    "Wakao and Funazkri": lambda re, sc, eb: 2 + 1.1 * re**0.6 * sc ** (1 / 3),
    "Kataoka et al": lambda re, sc, eb: 1.85 * ((1 - eb) / eb) ** (1 / 3) * (re * sc) ** (1 / 3),
}
"""Available Sherwood-number correlations, keyed by name."""


def film_transfer(
    diffusivity, radius: float, re: float, sc, ebed: float, correlation: str = "Chern and Chien"
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
    correlation
        Key of :data:`SHERWOOD`.
    """
    try:
        sherwood = SHERWOOD[correlation]
    except KeyError:
        raise ValueError(
            f"unknown correlation {correlation!r}; expected one of {sorted(SHERWOOD)}"
        ) from None
    return diffusivity / (2.0 * radius) * sherwood(re, sc, ebed)
