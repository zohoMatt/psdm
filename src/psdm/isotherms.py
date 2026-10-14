"""Equilibrium isotherms and batch isotherm fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def freundlich(c, k, n_inv):
    """Freundlich isotherm ``q = K c^(1/n)``."""
    return k * np.asarray(c, dtype=float) ** n_inv


def langmuir(c, kl, qm):
    """Langmuir isotherm ``q = qm KL c / (1 + KL c)``."""
    c = np.asarray(c, dtype=float)
    return qm * kl * c / (1.0 + kl * c)


def redlich_peterson(c, a, b, m):
    """Redlich–Peterson isotherm ``q = A c / (B + c^M)``."""
    c = np.asarray(c, dtype=float)
    return a * c / (b + c**m)


MODELS = {
    "freundlich": (freundlich, ("K", "1/n")),
    "langmuir": (langmuir, ("KL", "qm")),
    "redlich-peterson": (redlich_peterson, ("A", "B", "M")),
}


@dataclass(frozen=True)
class IsothermFit:
    """Result of :func:`fit_isotherm`.

    Attributes
    ----------
    model
        Isotherm name.
    params, stderr
        Best-fit parameters and their standard errors.
    r2
        Coefficient of determination.
    ce, q
        Data used for the fit.
    """

    model: str
    params: dict[str, float]
    stderr: dict[str, float]
    r2: float
    ce: np.ndarray
    q: np.ndarray

    def predict(self, ce):
        """Solid-phase concentration at liquid concentration ``ce``."""
        func, _ = MODELS[self.model]
        return func(ce, *self.params.values())

    def bands(self, ce, level: float = 0.95):
        """Approximate confidence and prediction half-widths at ``ce``.

        Uses the linear-regression approximation of the original PSDM tools.
        Returns ``(confidence, prediction)``.
        """
        ce = np.asarray(ce, dtype=float)
        n, p = len(self.ce), len(self.params)
        dof = n - p
        if dof <= 0:
            raise ValueError("need more data points than parameters for interval estimates")
        s_err = np.sqrt(np.sum((self.q - self.predict(self.ce)) ** 2) / dof)
        t = stats.t.ppf(0.5 + level / 2, dof)
        lever = 1 / n + (ce - self.ce.mean()) ** 2 / np.sum((self.ce - self.ce.mean()) ** 2)
        return t * s_err * np.sqrt(lever), t * s_err * np.sqrt(1 + lever)


def batch_loading(c0, ce, mass, volume=1.0):
    """Solid-phase loading from a bottle-point test: ``q = (C0 - Ce) V / m``."""
    return (np.asarray(c0, float) - np.asarray(ce, float)) * volume / np.asarray(mass, float)


def fit_isotherm(ce, q, model: str = "freundlich", p0=None) -> IsothermFit:
    """Fit an isotherm to equilibrium data by nonlinear least squares.

    Parameters
    ----------
    ce
        Equilibrium liquid concentrations.
    q
        Solid-phase concentrations (see :func:`batch_loading`).
    model
        ``"freundlich"``, ``"langmuir"`` or ``"redlich-peterson"``.
    p0
        Optional initial guess.

    Units of the returned parameters follow the units of the data.
    """
    key = model.lower().replace("_", "-").replace(" ", "-")
    if key == "redlichpeterson":
        key = "redlich-peterson"
    try:
        func, names = MODELS[key]
    except KeyError:
        raise ValueError(f"unknown isotherm {model!r}; expected one of {sorted(MODELS)}") from None

    ce = np.asarray(ce, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = np.isfinite(ce) & np.isfinite(q)
    ce, q = ce[ok], q[ok]
    if len(ce) < len(names):
        raise ValueError(f"{model} needs at least {len(names)} data points")

    popt, pcov = curve_fit(func, ce, q, p0=p0, bounds=(0, np.inf), maxfev=10000)
    resid = q - func(ce, *popt)
    ss_tot = np.sum((q - q.mean()) ** 2)
    r2 = 1.0 - np.sum(resid**2) / ss_tot if ss_tot > 0 else float("nan")
    return IsothermFit(
        model=key,
        params=dict(zip(names, map(float, popt), strict=True)),
        stderr=dict(zip(names, map(float, np.sqrt(np.diag(pcov))), strict=True)),
        r2=float(r2),
        ce=ce,
        q=q,
    )
