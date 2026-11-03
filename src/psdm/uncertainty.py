"""Model uncertainty bands from perturbed inputs.

Each requested input is raised and lowered by a relative amount and the model is rerun; the
band is the pointwise minimum and maximum over all runs, including the base case. This is
``PSDM.model_uncertainty`` of the US EPA code.

:func:`model_uncertainty` does everything in one call. To run the simulations elsewhere (e.g.
in a process pool), split it into :func:`plan`, which lists the runs, and :func:`envelope`,
which combines their results::

    p = psdm.uncertainty.plan(column, compounds, influent, capacity=0.1, k=0.2)
    results = [run.simulate() for run in p.runs]  # or pool.map(...)
    band = psdm.uncertainty.envelope(p, results)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fouling, units
from .model import Column, Compound
from .solver import (
    SURFACE_DIFFUSION_FOULED_FACTOR,
    Result,
    _as_compounds,
    _as_influent,
    simulate,
)

PARAMETERS = ("capacity", "k", "q_n_inv", "c0", "mass", "flow_rate", "kf", "dp", "ds")
"""Inputs :func:`model_uncertainty` can perturb.

``capacity``
    ``K·(1+f)`` with ``1/n·(1-f)`` (label ``capacity+…``) and ``K·(1-f)`` with ``1/n·(1+f)``
    (``capacity-…``), as in the EPA code. The loading ``q = K C^(1/n)`` changes by
    ``(1+f) C^(-f/n)``, so above ``C = (1+f)^(n/f)`` ug/L (about 9 ug/L for ``1/n = 0.43``
    and ``f = 0.1``) ``capacity+`` *lowers* the capacity and ``capacity-`` raises it; the
    labels name the direction of ``K``, not of the capacity. The envelope contains both
    runs. To change the capacity alone use ``k``.
``k``
    Freundlich ``K`` only.
``q_n_inv``
    ``1/n`` changed so that the loading at the mean influent concentration changes by
    ``f`` while ``K`` stays fixed.
``c0``
    All influent concentrations.
``mass``
    Adsorbent mass. When the mass exceeds what the bed holds at the bulk density
    (``Column.bed_density``, or the base column's mass per bed volume) the bed is
    lengthened.
``flow_rate``
    Flow rate, with the mass-transfer coefficients held at their base values.
``kf``, ``dp``, ``ds``
    Film coefficient, pore and surface diffusivity. ``ds`` has no effect in fouled water.
"""


@dataclass
class Uncertainty:
    """Output of :func:`model_uncertainty`.

    Attributes
    ----------
    effluent
        Base-case effluent per compound on the output time grid.
    lower, upper
        Pointwise envelope over all runs.
    runs
        Effluent of every perturbed run, keyed by a label such as ``"k+10%"``.
    """

    effluent: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    runs: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class Run:
    """One simulation of an uncertainty analysis.

    Attributes
    ----------
    label
        Perturbation label, e.g. ``"k+10%"``. With ``single=True`` one label has one run per
        compound.
    column, compounds, influent, simulate_kw
        Arguments for :func:`psdm.simulate`.
    """

    label: str
    column: Column
    compounds: list[Compound]
    influent: pd.DataFrame
    simulate_kw: dict

    def simulate(self) -> Result:
        """Run :func:`psdm.simulate` with this run's arguments."""
        return simulate(self.column, self.compounds, self.influent, **self.simulate_kw)


@dataclass
class Plan:
    """Output of :func:`plan`.

    Attributes
    ----------
    base
        Base-case effluent per compound on ``times``.
    runs
        The perturbed simulations still to run, in order.
    times
        Output time grid of the envelope.
    """

    base: pd.DataFrame
    runs: list[Run]
    times: np.ndarray


def _perturbations(amounts: dict[str, float | None]) -> list[tuple[str, dict[str, float]]]:
    out = []
    for key, f in amounts.items():
        if f is None:
            continue
        if not 0 < f < 1:
            raise ValueError(f"{key} must be a fraction between 0 and 1 (0.1 for 10 %), got {f}")
        pct = f"{100 * f:g}%"
        if key == "capacity":
            out.append((f"capacity+{pct}", {"k": 1 + f, "n_inv": 1 - f}))
            out.append((f"capacity-{pct}", {"k": 1 - f, "n_inv": 1 + f}))
        else:
            out.append((f"{key}+{pct}", {key: 1 + f}))
            out.append((f"{key}-{pct}", {key: 1 - f}))
    return out


def _perturbed_column(column: Column, change: dict[str, float]) -> Column:
    if "flow_rate" in change:
        column = column.with_(flow_rate=column.flow_rate * change["flow_rate"])
    if "mass" in change:
        rho = column.bed_density or column.mass / column.bed_volume
        mass = column.mass * change["mass"]
        length = max(column.length, mass / (column.area * rho))
        column = column.with_(mass=mass, length=length)
    return column


def _perturbed_compound(c: Compound, change: dict[str, float], c_mean_ug: float) -> Compound:
    k, n_inv = c.k * change.get("k", 1.0), c.n_inv * change.get("n_inv", 1.0)
    if "q_n_inv" in change:
        n_inv = n_inv + np.log(change["q_n_inv"]) / np.log(c_mean_ug)
    kw = {attr: getattr(c, attr) * change.get(attr, 1.0) for attr in ("kf", "dp", "ds")}
    return c.with_(k=k, n_inv=float(n_inv), **kw)


def plan(
    column: Column,
    compounds: Compound | Sequence[Compound],
    influent: pd.DataFrame | pd.Series,
    *,
    capacity: float | None = 0.1,
    k: float | None = None,
    q_n_inv: float | None = None,
    c0: float | None = None,
    mass: float | None = None,
    flow_rate: float | None = None,
    kf: float | None = None,
    dp: float | None = None,
    ds: float | None = None,
    single: bool = True,
    times=None,
    **simulate_kw,
) -> Plan:
    """List the perturbed simulations of an uncertainty analysis.

    Arguments are those of :func:`model_uncertainty`. The base case is simulated here (once
    per compound with ``single=True``), because its ``kf``, ``dp`` and ``ds`` are fixed in all
    perturbed runs; no other simulation is run. Run each of ``Plan.runs`` (for example with
    :meth:`Run.simulate`) and pass the results to :func:`envelope`.
    """
    comps = _as_compounds(compounds)
    names = [c.name for c in comps]
    influent = _as_influent(influent, names)
    if times is None:
        times = np.linspace(0.0, float(influent.index[-1]), 501)
    times = np.asarray(times, dtype=float)
    c_mult = units.factor("concentration", simulate_kw.get("conc_unit", "ug/L"))
    c_mean_ug = influent.mean() * c_mult

    groups = [[c] for c in comps] if single else [comps]

    # base case; remember the mass-transfer parameters it used
    base_parts, fixed = [], {}
    fouled = simulate_kw.get("water", fouling.ORGANIC_FREE) != fouling.ORGANIC_FREE
    for group in groups:
        group_names = [c.name for c in group]
        res = simulate(column, group, influent[group_names], **simulate_kw)
        base_parts.append(res(times))
        for c in group:
            mt = res.mass_transfer[c.name]
            ds_base = mt["ds"] / SURFACE_DIFFUSION_FOULED_FACTOR if fouled else mt["ds"]
            fixed[c.name] = c.with_(kf=float(mt["kf"]), dp=float(mt["dp"]), ds=float(ds_base))
    base = pd.concat(base_parts, axis=1)[names]

    amounts = {
        "capacity": capacity,
        "k": k,
        "q_n_inv": q_n_inv,
        "c0": c0,
        "mass": mass,
        "flow_rate": flow_rate,
        "kf": kf,
        "dp": dp,
        "ds": ds,
    }
    runs = []
    for label, change in _perturbations(amounts):
        col = _perturbed_column(column, change)
        infl = influent * change.get("c0", 1.0)
        for group in groups:
            perturbed = [
                _perturbed_compound(fixed[c.name], change, c_mean_ug[c.name]) for c in group
            ]
            group_names = [c.name for c in group]
            runs.append(Run(label, col, perturbed, infl[group_names], dict(simulate_kw)))
    return Plan(base=base, runs=runs, times=times)


def envelope(plan: Plan, results: Sequence[Result]) -> Uncertainty:
    """Combine the results of ``plan.runs`` (same order) into an :class:`Uncertainty`."""
    if len(results) != len(plan.runs):
        raise ValueError(f"expected {len(plan.runs)} results, got {len(results)}")
    names = list(plan.base.columns)
    parts: dict[str, list[pd.DataFrame]] = {}
    for run, res in zip(plan.runs, results, strict=True):
        parts.setdefault(run.label, []).append(res(plan.times))
    runs = {label: pd.concat(p, axis=1)[names] for label, p in parts.items()}

    base = plan.base
    stacked = np.stack([base.to_numpy(), *(r.to_numpy() for r in runs.values())])
    lower = pd.DataFrame(stacked.min(axis=0), index=base.index, columns=names)
    upper = pd.DataFrame(stacked.max(axis=0), index=base.index, columns=names)
    return Uncertainty(effluent=base, lower=lower, upper=upper, runs=runs)


def model_uncertainty(
    column: Column,
    compounds: Compound | Sequence[Compound],
    influent: pd.DataFrame | pd.Series,
    *,
    capacity: float | None = 0.1,
    k: float | None = None,
    q_n_inv: float | None = None,
    c0: float | None = None,
    mass: float | None = None,
    flow_rate: float | None = None,
    kf: float | None = None,
    dp: float | None = None,
    ds: float | None = None,
    single: bool = True,
    times=None,
    **simulate_kw,
) -> Uncertainty:
    """Rerun :func:`psdm.simulate` with perturbed inputs and return the envelope.

    Each argument from ``capacity`` to ``ds`` is a relative perturbation (``0.1`` for ±10 %)
    or ``None`` to leave that input unchanged; see :data:`PARAMETERS`. Every perturbation is
    applied on its own, giving two extra runs each. ``kf``, ``dp`` and ``ds`` are fixed to the
    base-case values (from the correlations unless set on the compound) in all perturbed runs,
    so e.g. a flow-rate change does not change ``kf``.

    Parameters
    ----------
    column, compounds, influent
        As for :func:`psdm.simulate`.
    single
        Simulate each compound on its own (default, as upstream) or all together with
        competition.
    times
        Output times; defaults to 501 points from 0 to the end of the influent.
    **simulate_kw
        Forwarded to :func:`psdm.simulate`.
    """
    p = plan(
        column,
        compounds,
        influent,
        capacity=capacity,
        k=k,
        q_n_inv=q_n_inv,
        c0=c0,
        mass=mass,
        flow_rate=flow_rate,
        kf=kf,
        dp=dp,
        ds=ds,
        single=single,
        times=times,
        **simulate_kw,
    )
    return envelope(p, [run.simulate() for run in p.runs])
