"""Model uncertainty bands from perturbed inputs.

Each requested input is raised and lowered by a relative amount and the model is rerun; the
band is the pointwise minimum and maximum over all runs, including the base case. This is
``PSDM.model_uncertainty`` of the US EPA code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fouling, units
from .model import Column, Compound
from .solver import SURFACE_DIFFUSION_FOULED_FACTOR, _as_compounds, _as_influent, simulate

PARAMETERS = ("capacity", "k", "q_n_inv", "c0", "mass", "flow_rate", "kf", "dp", "ds")
"""Inputs :func:`model_uncertainty` can perturb.

``capacity``
    ``K`` and ``1/n`` together in opposite directions (``K·(1+f)`` with ``1/n·(1-f)`` and
    vice versa), which moves the capacity most.
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
    comps = _as_compounds(compounds)
    names = [c.name for c in comps]
    influent = _as_influent(influent, names)
    if times is None:
        times = np.linspace(0.0, float(influent.index[-1]), 501)
    times = np.asarray(times, dtype=float)
    c_mult = units.factor("concentration", simulate_kw.get("conc_unit", "ug/L"))
    c_mean_ug = influent.mean() * c_mult

    groups = [[c] for c in comps] if single else [comps]

    def run(col: Column, group: list[Compound], infl: pd.DataFrame) -> pd.DataFrame:
        group_names = [c.name for c in group]
        res = simulate(col, group, infl[group_names], **simulate_kw)
        return res(times)

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
    runs = {}
    for label, change in _perturbations(amounts):
        col = _perturbed_column(column, change)
        infl = influent * change.get("c0", 1.0)
        parts = []
        for group in groups:
            perturbed = [
                _perturbed_compound(fixed[c.name], change, c_mean_ug[c.name]) for c in group
            ]
            parts.append(run(col, perturbed, infl))
        runs[label] = pd.concat(parts, axis=1)[names]

    stacked = np.stack([base.to_numpy(), *(r.to_numpy() for r in runs.values())])
    lower = pd.DataFrame(stacked.min(axis=0), index=base.index, columns=names)
    upper = pd.DataFrame(stacked.max(axis=0), index=base.index, columns=names)
    return Uncertainty(effluent=base, lower=lower, upper=upper, runs=runs)
