"""Fitting model parameters to column breakthrough data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from . import fouling, units
from .capacity import CapacityEstimate, _interp, estimate_capacity
from .fouling import ORGANIC_FREE
from .model import Column, Compound
from .solver import Result, simulate
from .units import factor

_FAILED = 1e30

Callback = Callable[[int, float], None]
"""Progress callback ``callback(evaluations, best_ssq)``.

Called after every model simulation of a fit with the number of simulations so far and the
lowest sum of squares seen so far (``1e30`` stands for a failed simulation). Raising any
exception from the callback stops the fit immediately: the exception propagates to the
caller unchanged and no further simulations run.
"""


class _Progress:
    """Counts simulations and reports them to an optional callback."""

    def __init__(self, callback: Callback | None):
        self.callback = callback
        self.evaluations = 0
        self.best = float("inf")

    def __call__(self, ssq: float) -> None:
        self.evaluations += 1
        self.best = min(self.best, ssq)
        if self.callback is not None:
            self.callback(self.evaluations, self.best)


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
        Every simulated ``(k, n_inv, ssq)``, in order.
    method
        ``"epa"`` or ``"nelder-mead"``.
    ssq_grid
        ``method="epa"`` only: the search grid of the EPA algorithm (index ``K`` factor,
        columns ``1/n``) with the sum of squares of every visited node, ``NaN`` elsewhere.
    capacity
        ``method="epa"`` with ``start="capacity"`` only: the mass-balance estimate the search
        started from.
    breakthrough_time
        ``method="epa"`` only: the end of the sum-of-squares window.
    converged
        ``method="epa"``: ``False`` if the search stopped at ``search_limit``.
        ``method="nelder-mead"``: ``False`` if it stopped at ``max_evaluations``.
    """

    compound: Compound
    ssq: float
    result: Result
    evaluations: pd.DataFrame
    method: str = "nelder-mead"
    ssq_grid: pd.DataFrame | None = None
    capacity: CapacityEstimate | None = None
    breakthrough_time: float | None = None
    converged: bool = True


FIT_METHODS = ("epa", "nelder-mead")
"""Search methods of :func:`fit_freundlich`."""


class _Replay(Exception):
    pass


_EPA_FAILED = 1e20  # upstream run_all_smart's ssq for a failed simulation
_EPA_START_N_INV = 0.45
_EPA_MAX_DAYS = 2000.0
_EPA_K_BY_XN = np.arange(0.15, 1.001, 0.01)


def fit_freundlich(
    column: Column,
    compound: Compound,
    influent: pd.Series | pd.DataFrame,
    effluent: pd.Series,
    *,
    method: str = "epa",
    start: str = "capacity",
    pm: float = 10.0,
    num: int = 11,
    des_xn: float = 0.025,
    search_limit: int = 50,
    n_inv_bounds: tuple[float, float] = (0.15, 1.0),
    q_factor_bounds: tuple[float, float] = (0.1, 10.0),
    until: float | None = None,
    max_evaluations: int = 200,
    callback: Callback | None = None,
    **simulate_kw,
) -> FitResult:
    """Fit Freundlich ``K`` and ``1/n`` of one compound to observed breakthrough.

    ``method="epa"`` (default) is ``PSDM.run_all_smart`` of the US EPA code, ported step by
    step and checked against it (same sequence of trial parameters and the same result):

    1. ``start="capacity"`` (EPA without ``k_data``): a mass balance on the data
       (:func:`psdm.estimate_capacity`) gives the loading ``q`` and breakthrough time
       ``t_b``. The base ``K`` for each ``1/n`` keeps ``q`` at the mean influent
       concentration, and the ``K`` factor starts at the inverse mean fouling reduction of
       ``K`` up to ``t_b`` (1 in organic-free water).
       ``start="compound"`` (EPA with ``k_data``): the base ``K`` is ``compound.k`` for every
       ``1/n``, the factor starts at 1, and ``until`` is ``t_b`` (required; EPA
       ``k_data["brk"]``).
    2. The sum of squares is taken at whole time units ``0, 1, …, t_b − 1`` between the
       simulated effluent and the observations, both linearly interpolated and extrapolated.
    3. The search grid is ``K`` factor from ``1 − pm %`` to ``1 + pm %`` of the start in
       ``num`` steps each way, by ``1/n`` from 0.2 to 1.0 in steps of ``des_xn``. From
       ``1/n = 0.45`` it tries ``1/n`` = 0.2, 0.4, 0.6, 0.8 and 1.0, then repeatedly evaluates
       the four neighbours of the current node and moves towards lower sums of squares, until
       the centre is the best node, it cannot move, or ``search_limit`` loops have run.

    Like the EPA code, the result is a grid node and may lie on the grid edge (``1/n`` of 0.2
    or 1.0, or the ``K`` factor at ``1 ± pm %``); check :attr:`FitResult.ssq_grid`.

    ``method="nelder-mead"`` is the bounded Nelder–Mead search of psdm 0.1/0.2 (not in the EPA
    code). It uses ``n_inv_bounds``, ``q_factor_bounds``, ``until`` and ``max_evaluations``
    and starts from ``compound``.

    Parameters
    ----------
    column, compound, influent
        As for :func:`psdm.simulate`.
    effluent
        Observed effluent for ``compound``, indexed by time.
    method
        ``"epa"`` or ``"nelder-mead"``.
    start
        ``method="epa"`` only: ``"capacity"`` or ``"compound"``, see above.
    pm, num, des_xn, search_limit
        ``method="epa"`` only: the EPA ``run_all_smart`` arguments (``pm`` in percent).
    n_inv_bounds, q_factor_bounds, max_evaluations
        ``method="nelder-mead"`` only.
    until
        ``method="epa"``: end of the sum-of-squares window. With ``start="capacity"`` it
        replaces the estimated breakthrough time, which is a psdm addition; leave it ``None``
        for the EPA behaviour. ``method="nelder-mead"``: only fit observations at or before
        this time.
    callback
        Progress callback ``callback(evaluations, best_ssq)``, see :data:`Callback`. Raise
        from it to cancel the fit.
    **simulate_kw
        Forwarded to :func:`psdm.simulate` (``water``, ``time_unit``, ``conc_unit``, ...).
    """
    if method == "nelder-mead":
        return _fit_nelder_mead(
            column,
            compound,
            influent,
            effluent,
            n_inv_bounds=n_inv_bounds,
            q_factor_bounds=q_factor_bounds,
            until=until,
            max_evaluations=max_evaluations,
            callback=callback,
            **simulate_kw,
        )
    if method != "epa":
        raise ValueError(f"unknown method {method!r}; expected one of {FIT_METHODS}")
    return _fit_epa_smart(
        column,
        compound,
        influent,
        effluent,
        start=start,
        pm=pm,
        num=num,
        des_xn=des_xn,
        search_limit=search_limit,
        until=until,
        callback=callback,
        **simulate_kw,
    )


def _epa_fouling_factor(duration: float, brk: float, compound: Compound, simulate_kw) -> float:
    """Upstream ``foul_mult_est``: 1 / mean ``K`` multiplier at whole days ``0 … brk - 1``.

    Evaluated, like upstream, on its 500-point interpolation table of the multiplier.
    """
    water = simulate_kw.get("water", ORGANIC_FREE)
    t_mult = units.factor("time", simulate_kw.get("time_unit", "days"))
    b = fouling.chemical_coefficients(compound.fouling_class, compound.name)
    table_t = np.linspace(0.0, max(_EPA_MAX_DAYS, duration) * t_mult, 500)
    table = fouling.k_multiplier(table_t, water, b)
    times = np.arange(brk) * t_mult
    if len(times) == 0:
        return 1.25  # upstream fallback for a NaN estimate
    factor_ = float(1.0 / np.mean(_interp(times, table_t, table)))
    return factor_ if np.isfinite(factor_) else 1.25


def _fit_epa_smart(
    column: Column,
    compound: Compound,
    influent: pd.Series | pd.DataFrame,
    effluent: pd.Series,
    *,
    start: str,
    pm: float,
    num: int,
    des_xn: float,
    search_limit: int,
    until: float | None,
    callback: Callback | None,
    _ssq_fn: Callable[[float, float], float] | None = None,
    **simulate_kw,
) -> FitResult:
    """Port of ``PSDM.run_all_smart`` (US EPA PSDM, commit 3f514c2) for one compound.

    ``_ssq_fn(k, n_inv)`` replaces the simulation (tests replay upstream sums of squares).
    """
    infl = influent[compound.name] if isinstance(influent, pd.DataFrame) else influent
    infl_frame = infl.to_frame(compound.name)
    obs = effluent.dropna()
    obs_t, obs_c = obs.index.to_numpy(dtype=float), obs.to_numpy(dtype=float)
    progress = _Progress(callback)
    rows: list[tuple[float, float, float]] = []

    capacity = None
    if start == "capacity":
        aligned = pd.Series(np.interp(obs_t, infl.index.to_numpy(dtype=float), infl), obs_t)
        capacity = estimate_capacity(
            column,
            aligned,
            pd.Series(obs_c, obs_t),
            n_inv=_EPA_START_N_INV,
            time_unit=simulate_kw.get("time_unit", "days"),
            conc_unit=simulate_kw.get("conc_unit", "ug/L"),
        )
        brk = capacity.breakthrough_time if until is None else float(until)
        k_table = np.asarray(capacity.k_for(_EPA_K_BY_XN), dtype=float)

        def get_base_k(xn):  # upstream k_by_xn_factor: interp1d over 1/n = 0.15 … 1.00
            return float(np.interp(xn, _EPA_K_BY_XN, k_table))

        k_factor0 = _epa_fouling_factor(
            float(infl.index.max()), capacity.breakthrough_time, compound, simulate_kw
        )
    elif start == "compound":
        if until is None:
            raise ValueError('start="compound" needs until (the EPA k_data["brk"])')
        brk = float(until)

        def get_base_k(xn):
            # upstream with k_data returns the current k_data K: get_ssq writes k * factor
            # there, and only a real simulation (run_psdm_kfit) resets it afterwards, so a
            # cached node leaves it changed
            return state["k"]

        k_factor0 = 1.0
    else:
        raise ValueError(f'start must be "capacity" or "compound", got {start!r}')

    ssq_x = np.arange(brk)
    obs_at = _interp(ssq_x, obs_t, obs_c)

    def simulate_ssq(k_total: float, xn: float) -> float:
        trial = compound.with_(k=float(k_total), n_inv=float(xn))
        try:
            if _ssq_fn is not None:
                raise _Replay
            res = simulate(column, trial, infl_frame, **simulate_kw)
            t, c = res.effluent.index.to_numpy(), res.effluent[compound.name].to_numpy()
            ssq = float(np.sum((_interp(ssq_x, t, c) - obs_at) ** 2))
        except _Replay:
            ssq = float(_ssq_fn(float(k_total), float(xn)))
        except (RuntimeError, ValueError, FloatingPointError):
            ssq = _EPA_FAILED
        rows.append((float(k_total), float(xn), ssq))
        progress(ssq)  # outside the try, see _fit_nelder_mead
        return ssq

    def node(k_factor, xn):
        return (np.round(k_factor, 6), np.round(xn, 3))

    state = {"k": compound.k}

    def get_ssq(k, k_factor, xn, storage):
        # only simulates nodes that have no sum of squares yet
        state["k"] = k * k_factor * 1
        if np.isnan(storage.loc[node(k_factor, xn)]):
            ssq = simulate_ssq(k * k_factor * 1, xn * 1)
            state["k"] = compound.k  # upstream run_psdm_kfit resets k_data
            return ssq
        return storage.loc[node(k_factor, xn)]

    def store_if_new(k_factor, xn, value, storage):
        if np.isnan(storage.loc[node(k_factor, xn)]):
            storage.loc[node(k_factor, xn)] = value * 1

    # ---- PSDM.run_all_smart, line by line --------------------------------------------------
    best_xn = _EPA_START_N_INV
    best_k_factor = k_factor0
    base_k = get_base_k(best_xn)
    best_k = base_k * best_k_factor
    scale_k = best_k_factor * pm / 100 / num
    max_k = best_k_factor * (1 + pm / 100)
    min_k = best_k_factor * (1 - pm / 100)
    storage = pd.DataFrame(
        index=np.round(np.arange(min_k, max_k + scale_k / 2, scale_k), 6),
        columns=np.round(np.arange(0.2, 1 + des_xn / 2, des_xn), 3),
        dtype=float,
    )
    best_ssq = get_ssq(base_k, best_k_factor, best_xn, storage)
    storage.loc[node(best_k_factor, best_xn)] = best_ssq * 1

    xn_idx = np.where(storage.columns == np.round(best_xn, 3))[0][0]
    k_idx = np.where(storage.index == np.round(best_k_factor, 6))[0][0]
    xn_idx_max = len(storage.columns) - 1
    k_idx_max = len(storage.index) - 1

    # coarse start: 1/n = 0.2, 0.4, 0.6, 0.8, 1.0 at the starting K factor
    for xn_test in np.round(np.linspace(0, xn_idx_max, num=5), 0).astype(int):
        xn = storage.columns[xn_test]
        k_factor = storage.index[k_idx]
        test_base_k = get_base_k(xn)
        test_ssq = get_ssq(test_base_k, k_factor, xn, storage)
        storage.loc[node(k_factor, xn)] = test_ssq * 1
        if test_ssq < best_ssq:
            best_xn, best_k, best_ssq = xn * 1, k_factor * test_base_k * 1, test_ssq * 1
            xn_idx = xn_test * 1

    shift_xn = shift_k = 0
    loops_tried = 0
    converged = True
    done = False
    while not done:
        loops_tried += 1
        xn_idx += shift_xn
        k_idx += shift_k
        xn = storage.columns[xn_idx]
        k_factor = storage.index[k_idx]
        center_base_k = get_base_k(xn)

        base_ssq = get_ssq(center_base_k, k_factor, xn, storage)
        store_if_new(k_factor, xn, base_ssq, storage)
        if base_ssq < best_ssq:
            best_xn, best_k, best_ssq = xn * 1, k_factor * center_base_k * 1, base_ssq * 1

        # neighbours in 1/n (at the current K factor)
        right_ssq = left_ssq = 1e20
        if xn_idx < xn_idx_max:
            right_xn = storage.columns[xn_idx + 1]
            right_base_k = get_base_k(right_xn)
            right_ssq = get_ssq(right_base_k, k_factor, right_xn, storage)
            store_if_new(k_factor, right_xn, right_ssq, storage)
            if right_ssq < best_ssq:
                best_xn, best_k, best_ssq = right_xn * 1, k_factor * right_base_k * 1, right_ssq
        if xn_idx >= 1:
            left_xn = storage.columns[xn_idx - 1]
            left_base_k = get_base_k(left_xn)
            left_ssq = get_ssq(left_base_k, k_factor, left_xn, storage)
            store_if_new(k_factor, left_xn, left_ssq, storage)
            if left_ssq < best_ssq:
                best_xn, best_k, best_ssq = left_xn * 1, k_factor * left_base_k * 1, left_ssq
        if right_ssq <= base_ssq and xn_idx + 1 <= xn_idx_max:
            shift_xn = 1
        elif left_ssq <= base_ssq and xn_idx - 1 >= 0:
            shift_xn = -1
        else:
            shift_xn = 0

        # neighbours in K factor (at the current 1/n, base K of the centre)
        upper_ssq = lower_ssq = 1e20
        if k_idx < k_idx_max:
            upper = storage.index[k_idx + 1]
            upper_ssq = get_ssq(center_base_k, upper, xn, storage)
            store_if_new(upper, xn, upper_ssq, storage)
            if upper_ssq < best_ssq:
                best_xn, best_k, best_ssq = xn * 1, upper * center_base_k * 1, upper_ssq
        if k_idx >= 1:
            lower = storage.index[k_idx - 1]
            lower_ssq = get_ssq(center_base_k, lower, xn, storage)
            store_if_new(lower, xn, lower_ssq, storage)
            if lower_ssq < best_ssq:
                best_xn, best_k, best_ssq = xn * 1, lower * center_base_k * 1, lower_ssq
        if upper_ssq <= base_ssq and k_idx + 1 <= k_idx_max:
            shift_k = 1
        elif lower_ssq <= base_ssq and k_idx - 1 >= 0:
            shift_k = -1
        else:
            shift_k = 0

        if np.abs(base_ssq - best_ssq) <= 1e-7 or loops_tried >= search_limit:
            converged = bool(np.abs(base_ssq - best_ssq) <= 1e-7)
            if best_k <= 0:
                best_k = 1e-6  # upstream keeps K positive
            done = True
        elif shift_k == 0 and shift_xn == 0:
            done = True
    # ----------------------------------------------------------------------------------------

    best = compound.with_(k=float(best_k), n_inv=float(best_xn))
    res = simulate(column, best, infl_frame, **simulate_kw)
    storage.index.name, storage.columns.name = "k_factor", "n_inv"
    return FitResult(
        best,
        float(best_ssq),
        res,
        pd.DataFrame(rows, columns=["k", "n_inv", "ssq"]),
        method="epa",
        ssq_grid=storage,
        capacity=capacity,
        breakthrough_time=float(brk),
        converged=converged,
    )


def _fit_nelder_mead(
    column: Column,
    compound: Compound,
    influent: pd.Series | pd.DataFrame,
    effluent: pd.Series,
    *,
    n_inv_bounds: tuple[float, float] = (0.15, 1.0),
    q_factor_bounds: tuple[float, float] = (0.1, 10.0),
    until: float | None = None,
    max_evaluations: int = 200,
    callback: Callback | None = None,
    **simulate_kw,
) -> FitResult:
    """``fit_freundlich(method="nelder-mead")``: the psdm 0.1/0.2 search, not in the EPA code.

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
        Maximum number of model simulations of the search (one more is run at the optimum).
    callback
        Progress callback ``callback(evaluations, best_ssq)``, see :data:`Callback`. Raise
        from it to cancel the fit.
    **simulate_kw
        Forwarded to :func:`psdm.simulate` (``water``, ``time_unit``, ``conc_unit``, ...).
    """
    infl = influent[compound.name] if isinstance(influent, pd.DataFrame) else influent
    c_ref = float(infl.mean()) * factor("concentration", simulate_kw.get("conc_unit", "ug/L"))
    q0 = compound.k * c_ref**compound.n_inv
    rows = []
    progress = _Progress(callback)

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
        progress(ssq)  # outside the try: a cancelling exception must not count as a failure
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
    return FitResult(
        best,
        ssq,
        res,
        pd.DataFrame(rows, columns=["k", "n_inv", "ssq"]),
        method="nelder-mead",
        converged=bool(opt.success),
    )


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
    refine_maxiter: int = 50,
    callback: Callback | None = None,
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
        Reference ``ds`` (cm²/s); default is the value from the surface-to-pore diffusion
        flux ratio ``column.psdfr`` that :func:`psdm.simulate` would use.
    refine
        Refine the best grid point with :func:`scipy.optimize.minimize_scalar`.
    until
        Only fit observations at or before this time.
    refine_maxiter
        Maximum iterations of the refinement step.
    callback
        Progress callback ``callback(evaluations, best_ssq)``, see :data:`Callback`. Only the
        trial simulations are reported; the reference run that computes ``ds_ref`` (when it
        is not given) is not. :func:`surface_diffusivity_max_evaluations` gives the upper
        bound on ``evaluations``. Raise from the callback to cancel the fit.
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
    progress = _Progress(callback)

    def trial(ds: float) -> float:
        if ds not in evaluated:
            try:
                res = simulate(column, compound.with_(ds=ds), infl, **simulate_kw)
                evaluated[ds] = (sum_of_squares(res, effluent, compound.name, until), res)
            except (RuntimeError, ValueError, FloatingPointError):
                evaluated[ds] = (_FAILED, None)
            progress(evaluated[ds][0])  # outside the try, see fit_freundlich
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
            options={"xatol": 1e-3, "maxiter": refine_maxiter},
        )
    best = min(evaluated, key=lambda ds: evaluated[ds][0])
    ssq, res = evaluated[best]
    if res is None:
        raise RuntimeError("simulation failed at the best-fit parameters")
    evaluations = pd.Series({ds: v[0] for ds, v in sorted(evaluated.items())}, name="ssq")
    evaluations.index.name = "ds"
    return DsFitResult(compound.with_(ds=best), best, ds_ref, ssq, res, evaluations)


def surface_diffusivity_max_evaluations(
    factors=None, refine: bool = True, refine_maxiter: int = 50
) -> int:
    """Upper bound on the number of simulations :func:`fit_surface_diffusivity` runs.

    The grid points plus at most ``refine_maxiter + 1`` evaluations of the refinement (fewer
    when it revisits a grid point). Use it as the denominator of a progress bar.
    """
    factors = np.linspace(1e-10, 1.0, 30) if factors is None else np.asarray(factors, float)
    n = len(np.unique(factors))
    return n + (refine_maxiter + 1 if refine and n > 1 else 0)
