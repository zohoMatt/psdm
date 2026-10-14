"""Adsorbent capacity from column breakthrough data when ``K`` is unknown.

The loading at breakthrough follows from a mass balance over the column,

.. math::

    q = \\frac{Q}{M} \\int_0^{t_b} \\left(C_{in}(t) - C_{out}(t)\\right) dt,

and a Freundlich ``K`` for an assumed ``1/n`` from :math:`K = q / \\bar C^{1/n}` with
:math:`\\bar C` the mean influent concentration up to breakthrough :math:`t_b`. When the
effluent has not yet reached the influent concentration, :math:`t_b` is extrapolated with a
logistic curve or a straight line, as in ``PSDM.__calculate_capacity`` of the US EPA code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

from . import units
from .model import Column

MAX_BREAKTHROUGH_DAYS = 2000.0
"""Extrapolated breakthrough times are capped at this many days."""

MIN_LOADING = 1e-6
"""Smallest loading (ug/g) returned when a breakthrough was found."""

METHODS = (
    "breakthrough",
    "exceeds mean influent",
    "logistic",
    "linear, single point",
    "linear regression",
    "minimum",
)
"""Possible values of :attr:`CapacityEstimate.method`.

``breakthrough``
    The effluent reached the influent concentration in the data.
``exceeds mean influent``
    The effluent reached the mean influent concentration in the data.
``logistic``
    Breakthrough extrapolated with :func:`logistic` (99 % of the plateau).
``linear, single point`` and ``linear regression``
    Breakthrough extrapolated with a straight line through the last effluent points.
``minimum``
    Too few non-zero effluent values; the whole data period is used, so ``q`` and ``K``
    are lower bounds.
"""


@dataclass(frozen=True)
class CapacityEstimate:
    """Output of :func:`estimate_capacity`.

    Attributes
    ----------
    k
        Freundlich ``K`` for :attr:`n_inv`, (ug/g)(L/ug)^(1/n).
    n_inv
        The assumed ``1/n``.
    q
        Adsorbent loading at breakthrough, ug/g.
    mean_influent
        Mean influent concentration up to breakthrough, in the data's concentration unit.
    breakthrough_time
        Time of (possibly extrapolated) breakthrough, in the data's time unit.
    method
        How the breakthrough time was found, one of :data:`METHODS`.
    """

    k: float
    n_inv: float
    q: float
    mean_influent: float
    breakthrough_time: float
    method: str
    _mean_influent_ug: float

    def k_for(self, n_inv):
        """Freundlich ``K`` that gives the same loading :attr:`q` for another ``1/n``."""
        return self.q / self._mean_influent_ug ** np.asarray(n_inv, dtype=float)


def logistic(t, c0: float, a: float, b: float):
    """Logistic breakthrough curve ``c0 / (1 + exp(a - b t))``."""
    return c0 / (1.0 + np.exp(a - b * np.asarray(t, dtype=float)))


def _interp(t, xp, fp):
    """Linear interpolation with linear extrapolation beyond both ends."""
    t = np.asarray(t, dtype=float)
    y = np.interp(t, xp, fp)
    if len(xp) > 1:
        lo, hi = t < xp[0], t > xp[-1]
        y[lo] = fp[0] + (t[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[hi] = fp[-1] + (t[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def _with_point(t, c, t_new, c_new):
    """Insert (or replace) the point ``(t_new, c_new)`` keeping ``t`` sorted."""
    keep = t != t_new
    t, c = np.append(t[keep], t_new), np.append(c[keep], c_new)
    order = np.argsort(t, kind="stable")
    return t[order], c[order]


def _integral(t_end, t, c):
    """Integral of the piecewise-linear (extrapolated) curve ``(t, c)`` from 0 to ``t_end``."""
    knots = np.unique(np.concatenate([[0.0, t_end], t[(t > 0) & (t < t_end)]]))
    y = _interp(knots, t, c)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(knots)))


def _last_time(t, below=np.inf):
    """Last positive, finite data time smaller than ``below`` (``0`` if there is none)."""
    ok = t[np.isfinite(t) & (t > 0) & (t < below)]
    return float(ok[-1]) if len(ok) else 0.0


def _breakthrough(t, c_in, c_out, mean_in, grid, cap):
    """Find or extrapolate the breakthrough time (days).

    Returns ``(time, method, t_in, c_in, t_out, c_out)``; the curves may gain an extra
    extrapolated point at the breakthrough time.
    """
    duration = t[-1]
    t_in, t_out = t.copy(), t.copy()

    nonzero = np.count_nonzero(c_out)
    if nonzero == 0:
        return duration, "minimum", t_in, c_in, t_out, c_out
    if nonzero == 1 and not (c_out[-1] > 0 and c_out[-1] / mean_in >= 0.25):
        # a single non-zero value that is not a high last point cannot be extrapolated
        return duration, "minimum", t_in, c_in, t_out, c_out

    if c_out.max() >= c_in.min():
        f_in, f_out = _interp(grid, t, c_in), _interp(grid, t, c_out)
        checks = ((f_out >= f_in, "breakthrough"), (f_out >= mean_in, "exceeds mean influent"))
        for crossed, method in checks:
            if crossed.any():
                time = max(grid[np.argmax(crossed)] - (grid[1] - grid[0]), 0.0)  # last before
                return time, method, t_in, c_in, t_out, c_out

    try:
        lo, hi = min(0.99 * mean_in, c_out[0]), max(mean_in, c_out[0])
        params, _ = curve_fit(
            logistic,
            t,
            c_out,
            p0=(mean_in, 5.0, 0.1),
            bounds=((lo, 0.0, 0.0), (hi, 1e3, 1.0)),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        params = None
    if params is not None:
        time = float(np.round((params[1] + np.log(99.0)) / params[2]))
        if time >= duration:
            t_in, c_in = _with_point(t_in, c_in, time, mean_in)
            t_out, c_out = _with_point(t_out, c_out, time, logistic(time, *params))
            return (duration if time > cap else time), "logistic", t_in, c_in, t_out, c_out
        if time > 0.25 * duration:
            return time, "logistic", t_in, c_in, t_out, c_out

    if c_out[-1] > 0 and c_out[-1] / mean_in >= 0.25 and len(t) > 1:
        method = "linear, single point"
        slope = c_out[-1] / (t[-1] - t[-2])
        intercept = -c_out[-1] * t[-2] / (t[-1] - t[-2])
    else:
        method = "linear regression"
        n = np.count_nonzero(c_out)
        if n < len(c_out) and c_out[-(n + 1)] != 0:
            n += 1
        slope, intercept = linregress(t[-(n + 1) :], c_out[-(n + 1) :])[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        time = float(np.round((mean_in - intercept) / slope))
    if not np.isfinite(time):
        return _last_time(t), method, t_in, c_in, t_out, c_out
    if time <= 0:
        time = _last_time(t)
    t_in, c_in = _with_point(t_in, c_in, time, mean_in)
    t_out, c_out = _with_point(t_out, c_out, time, mean_in)
    return time, method, t_in, c_in, t_out, c_out


def estimate_capacity(
    column: Column,
    influent: pd.Series,
    effluent: pd.Series,
    *,
    n_inv: float = 0.45,
    time_unit: str = "days",
    conc_unit: str = "ug/L",
    resolution: float = 1.0,
) -> CapacityEstimate:
    """Estimate Freundlich ``K`` from a breakthrough curve for an assumed ``1/n``.

    The breakthrough time is the last time before the effluent reaches the influent (or
    mean influent) concentration. If the effluent never gets there, a logistic curve
    ``C0 / (1 + exp(a - b t))`` (``t`` in days) is fitted to the effluent and breakthrough is
    taken at 99 % of ``C0``; if that fails or predicts breakthrough in the first quarter of
    the data, a straight line through the last effluent points is extrapolated to the mean
    influent concentration. Extrapolated breakthrough beyond :data:`MAX_BREAKTHROUGH_DAYS`
    falls back to the end of the data. The loading ``q`` is the integral of influent minus
    effluent (linearly interpolated, with the extrapolated point appended) up to
    breakthrough, times flow rate over adsorbent mass.

    This is ``PSDM.__calculate_capacity`` of the US EPA code with ``break_type='calc'``.

    Parameters
    ----------
    column
        The adsorber; only ``flow_rate`` and ``mass`` are used.
    influent, effluent
        Concentrations of one compound indexed by time (sharing the same time points, e.g.
        :attr:`psdm.io.Dataset.influent` and ``effluent`` columns).
    n_inv
        Assumed Freundlich ``1/n``; use :meth:`CapacityEstimate.k_for` for other values.
    time_unit, conc_unit
        Units of the index and values of ``influent`` and ``effluent``.
    resolution
        Time step, in days, of the search for the breakthrough time.

    Returns
    -------
    CapacityEstimate
    """
    if not resolution > 0:
        raise ValueError("resolution must be positive")
    infl = influent.astype(float).sort_index()
    effl = effluent.astype(float).sort_index()
    if not infl.index.equals(effl.index):
        effl = effl.reindex(infl.index).interpolate(method="index", limit_direction="both")
    if len(infl) < 2:
        raise ValueError("influent and effluent need at least two time points")
    if infl.isna().any() or effl.isna().any():
        raise ValueError("influent and effluent must not contain missing values")
    c_in = infl.to_numpy().copy()
    c_out = effl.to_numpy().copy()
    if c_in.sum() <= 0:
        raise ValueError("the influent contains no mass")
    c_in[c_in == 0] = 1e-3  # as upstream; avoids a zero mean concentration

    t_mult = units.factor("time", time_unit)  # -> minutes
    c_mult = units.factor("concentration", conc_unit)  # -> ug/L
    t = infl.index.to_numpy(dtype=float) * t_mult / units.MIN_PER_DAY  # days
    mean_in = float(c_in.mean())
    grid = np.arange(0.0, t[-1] + resolution, resolution)

    time, method, t_in, c_in, t_out, c_out = _breakthrough(
        t, c_in, c_out, mean_in, grid, MAX_BREAKTHROUGH_DAYS
    )
    if time > MAX_BREAKTHROUGH_DAYS:
        time = _last_time(t, MAX_BREAKTHROUGH_DAYS)
    elif time == 0:
        time = resolution  # breakthrough cannot be immediate

    # loading in ug/g: (conc · day) · (ug/L per conc) · (L/day) / g
    flow_l_per_day = column.flow_rate / 1e3 * units.MIN_PER_DAY
    q = (_integral(time, t_in, c_in) - _integral(time, t_out, c_out)) * c_mult
    q = max(q * flow_l_per_day / column.mass, MIN_LOADING)
    mean = float(np.mean(_interp(grid[grid <= time], t_in, c_in)))
    mean_ug = mean * c_mult
    return CapacityEstimate(
        k=float(q / mean_ug**n_inv),
        n_inv=float(n_inv),
        q=float(q),
        mean_influent=mean,
        breakthrough_time=time * units.MIN_PER_DAY / t_mult,
        method=method,
        _mean_influent_ug=mean_ug,
    )
