"""Fitting model parameters to column breakthrough data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .fouling import ORGANIC_FREE
from .model import Column, Compound
from .solver import Result, simulate
from .units import factor

_FAILED = 1e30


def sum_of_squares(
    result: Result, effluent: pd.Series, compound: str | None = None, until: float | None = None
) -> float:
    """Sum of squared residuals between simulated and observed effluent.

    Parameters
    ----------
    result
        Simulation output.
    effluent
        Observed effluent concentrations indexed by time (same units as the simulation).
    compound
        Compound to compare; defaults to the first simulated compound.
    until
        Only use observations at or before this time.
    """
    compound = compound or result.compounds[0].name
    obs = effluent.dropna()
    if until is not None:
        obs = obs[obs.index <= until]
    model = result(obs.index.to_numpy(dtype=float), compound)
    return float(np.sum((model - obs.to_numpy(dtype=float)) ** 2))


@dataclass
class FitResult:
    """Output of :func:`fit_freundlich`.

    Attributes
    ----------
    compound
        The compound with fitted ``k`` and ``n_inv``.
    ssq
        Sum of squared residuals at the optimum.
    result
        Simulation at the optimum.
    evaluations
        Every evaluated ``(k, n_inv, ssq)``.
    """

    compound: Compound
    ssq: float
    result: Result
    evaluations: pd.DataFrame


def fit_freundlich(
    column: Column,
    compound: Compound,
    influent: pd.Series | pd.DataFrame,
    effluent: pd.Series,
    *,
    n_inv_bounds: tuple[float, float] = (0.15, 1.0),
    q_factor_bounds: tuple[float, float] = (0.1, 10.0),
    until: float | None = None,
    max_evaluations: int = 200,
    **simulate_kw,
) -> FitResult:
    """Fit Freundlich ``K`` and ``1/n`` of one compound to observed breakthrough.

    The search runs over ``(log q_ref, 1/n)``, where ``q_ref = K C_ref^(1/n)`` is the
    loading at the mean influent concentration ``C_ref``. Holding ``q_ref`` while ``1/n``
    changes keeps the predicted capacity roughly constant, which decorrelates the two
    parameters (the same idea as ``recalc_k`` in the original PSDM). The starting point is
    the ``k`` and ``n_inv`` of ``compound``.

    Parameters
    ----------
    column, compound, influent
        As for :func:`psdm.simulate`.
    effluent
        Observed effluent for ``compound``, indexed by time.
    n_inv_bounds
        Allowed range of ``1/n``.
    q_factor_bounds
        Allowed range of ``q_ref`` relative to its starting value.
    until
        Only fit observations at or before this time (e.g. up to breakthrough).
    max_evaluations
        Maximum number of model simulations.
    **simulate_kw
        Forwarded to :func:`psdm.simulate` (``water``, ``time_unit``, ``conc_unit``, ...).
    """
    infl = influent[compound.name] if isinstance(influent, pd.DataFrame) else influent
    c_ref = float(infl.mean()) * factor("concentration", simulate_kw.get("conc_unit", "ug/L"))
    q0 = compound.k * c_ref**compound.n_inv
    rows = []

    def trial(x) -> tuple[Compound, float, Result | None]:
        log_q, n_inv = x
        k = float(np.exp(log_q)) / c_ref**n_inv
        trial_compound = compound.with_(k=k, n_inv=float(n_inv))
        try:
            res = simulate(column, trial_compound, infl.to_frame(compound.name), **simulate_kw)
            ssq = sum_of_squares(res, effluent, compound.name, until)
        except (RuntimeError, ValueError, FloatingPointError):
            res, ssq = None, _FAILED
        rows.append((k, float(n_inv), ssq))
        return trial_compound, ssq, res

    lo, hi = q_factor_bounds
    opt = minimize(
        lambda x: trial(x)[1],
        x0=[np.log(q0), np.clip(compound.n_inv, *n_inv_bounds)],
        method="Nelder-Mead",
        bounds=[(np.log(q0 * lo), np.log(q0 * hi)), n_inv_bounds],
        options={"maxfev": max_evaluations, "xatol": 1e-3, "fatol": 1e-9},
    )
    best, ssq, res = trial(opt.x)
    if res is None:
        raise RuntimeError("simulation failed at the best-fit parameters")
    return FitResult(best, ssq, res, pd.DataFrame(rows, columns=["k", "n_inv", "ssq"]))


@dataclass
class DsFitResult:
    """Output of :func:`fit_surface_diffusivity`.

    Attributes
    ----------
    compound
        The compound with the fitted ``ds``.
    ds
        Fitted surface diffusivity, cm²/s.
    ds_ref
        Reference surface diffusivity the search factors multiply, cm²/s.
    ssq
        Sum of squared residuals at the optimum.
    result
        Simulation at the optimum.
    evaluations
        Sum of squares for every evaluated ``ds`` (index).
    """

    compound: Compound
    ds: float
    ds_ref: float
    ssq: float
    result: Result
    evaluations: pd.Series


def fit_surface_diffusivity(
    column: Column,
    compound: Compound,
    influent: pd.Series | pd.DataFrame,
    effluent: pd.Series,
    *,
    factors=None,
    ds_ref: float | None = None,
    refine: bool = True,
    until: float | None = None,
    **simulate_kw,
) -> DsFitResult:
    """Fit the surface diffusivity ``ds`` of one compound to observed breakthrough.

    ``ds = factor · ds_ref`` is evaluated for every factor in ``factors`` and, with
    ``refine``, the best grid point is refined by a bounded scalar search in ``log ds``
    between its neighbours. ``K``, ``1/n``, ``kf`` and ``dp`` are those of ``compound``.
    This is ``PSDM.run_psdm_dsfit`` of the US EPA code, which uses the same default grid.

    Surface diffusion is switched off in fouled water, so ``water`` must be
    ``"Organic Free"``.

    Parameters
    ----------
    column, compound, influent
        As for :func:`psdm.simulate`.
    effluent
        Observed effluent for ``compound``, indexed by time.
    factors
        Multipliers of ``ds_ref`` to try; default ``numpy.linspace(1e-10, 1, 30)``.
    ds_ref
        Reference ``ds`` (cm²/s); default is the value from the pore-to-surface diffusion
        flux ratio ``column.psdfr`` that :func:`psdm.simulate` would use.
    refine
        Refine the best grid point with :func:`scipy.optimize.minimize_scalar`.
    until
        Only fit observations at or before this time.
    **simulate_kw
        Forwarded to :func:`psdm.simulate`.
    """
    if simulate_kw.get("water", ORGANIC_FREE) != ORGANIC_FREE:
        raise ValueError("surface diffusion is switched off in fouled water; nothing to fit")
    infl = influent[compound.name] if isinstance(influent, pd.DataFrame) else influent
    infl = infl.to_frame(compound.name)
    if ds_ref is None:
        base = simulate(column, compound.with_(ds=None), infl, **simulate_kw)
        ds_ref = float(base.mass_transfer.loc["ds", compound.name])
    factors = np.linspace(1e-10, 1.0, 30) if factors is None else np.asarray(factors, float)
    if len(factors) == 0 or not (factors > 0).all():
        raise ValueError("factors must be positive")
    factors = np.unique(factors)
    evaluated: dict[float, tuple[float, Result | None]] = {}

    def trial(ds: float) -> float:
        if ds not in evaluated:
            try:
                res = simulate(column, compound.with_(ds=ds), infl, **simulate_kw)
                evaluated[ds] = (sum_of_squares(res, effluent, compound.name, until), res)
            except (RuntimeError, ValueError, FloatingPointError):
                evaluated[ds] = (_FAILED, None)
        return evaluated[ds][0]

    grid = factors * ds_ref
    ssqs = np.array([trial(float(ds)) for ds in grid])
    i = int(np.argmin(ssqs))
    if refine and len(grid) > 1:
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        minimize_scalar(
            lambda x: trial(float(np.exp(x))),
            bounds=(np.log(lo), np.log(hi)),
            method="bounded",
            options={"xatol": 1e-3},
        )
    best = min(evaluated, key=lambda ds: evaluated[ds][0])
    ssq, res = evaluated[best]
    if res is None:
        raise RuntimeError("simulation failed at the best-fit parameters")
    evaluations = pd.Series({ds: v[0] for ds, v in sorted(evaluated.items())}, name="ssq")
    evaluations.index.name = "ds"
    return DsFitResult(compound.with_(ds=best), best, ds_ref, ssq, res, evaluations)
