"""Full-scale design helpers: scale-up and scale-down of columns and operating predictions.

Ported from ``PSDM_extra.py`` and ``PSDM_tools.py`` of the US EPA code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import units
from .model import Column, Compound
from .solver import Result, simulate

SIEVE_OPENING_MM: dict[str, float] = {
    "4": 4.76, "5": 4.0, "6": 3.36, "7": 2.83, "8": 2.38, "10": 2.0, "12": 1.68,
    "14": 1.41, "16": 1.19, "18": 1.0, "20": 0.841, "25": 0.707, "30": 0.595, "35": 0.5,
    "40": 0.420, "45": 0.354, "50": 0.297, "60": 0.25, "70": 0.21, "80": 0.177,
    "100": 0.149, "120": 0.125, "140": 0.105, "170": 0.088, "200": 0.074, "230": 0.063,
    "270": 0.053, "325": 0.044, "400": 0.037,
}  # fmt: skip
"""US standard sieve openings, mm, by mesh number."""

EPA_PFAS_MCL: dict[str, float] = {
    "PFOA": 4.0,
    "PFOS": 4.0,
    "PFHxS": 10.0,
    "PFNA": 10.0,
    "HFPO-DA": 10.0,
    "GenX": 10.0,
}
"""US EPA PFAS maximum contaminant levels of April 2024, ng/L (``GenX`` = ``HFPO-DA``).

Check the current regulations before relying on these values.
"""

EPA_PFAS_HAZARD_INDEX: dict[str, float] = {
    "PFHxS": 10.0,
    "PFNA": 10.0,
    "HFPO-DA": 10.0,
    "GenX": 10.0,
    "PFBS": 2000.0,
}
"""Health-based water concentrations (ng/L) of the April 2024 US EPA hazard index.

The hazard index is the sum of each concentration divided by its value here; the MCL is 1.
"""

_LOADING_RATE = {  # -> m/h
    "m/h": 1.0,
    "m/hr": 1.0,
    "cm/s": 36.0,
    "gpm/ft2": units.LITERS_PER_GALLON * 1e-3 * 60.0 / (units.CM_PER_FT / 100.0) ** 2,
}


def mesh_diameter(mesh: str = "12x40") -> float:
    """Mean particle diameter, mm, of a ``"large x small"`` mesh size such as ``"12x40"``.

    >>> mesh_diameter("12x40")
    1.05
    """
    try:
        large, small = (s.strip() for s in mesh.lower().split("x"))
        return (SIEVE_OPENING_MM[large] + SIEVE_OPENING_MM[small]) / 2.0
    except (KeyError, ValueError):
        raise ValueError(
            f"invalid mesh {mesh!r}; expected 'NxM' with N, M in {list(SIEVE_OPENING_MM)}"
        ) from None


def _value(value, quantity: str) -> float:
    if isinstance(value, tuple):
        return units.convert(*value, quantity)
    return float(value)


def _particle_radius(particle_diameter, mesh: str | None) -> float | None:
    """Particle radius in cm from a diameter (cm or ``(value, unit)``) or a mesh size."""
    if particle_diameter is not None and mesh is not None:
        raise ValueError("give particle_diameter or mesh, not both")
    if particle_diameter is not None:
        return _value(particle_diameter, "length") / 2.0
    if mesh is not None:
        return mesh_diameter(mesh) / 10.0 / 2.0
    return None


def _bulk_density(column: Column) -> float:
    return column.bed_density or column.mass / column.bed_volume


def build_large_scale_column(
    carbon: Column,
    *,
    length,
    diameter,
    flow_rate=None,
    loading_rate=None,
    mass=None,
    bed_density=None,
    particle_diameter=None,
    mesh: str | None = None,
    name: str | None = None,
) -> Column:
    """Build a full-scale column packed with the adsorbent of ``carbon``.

    Particle density, porosity, tortuosity and ``psdfr`` are taken from ``carbon`` (e.g. a
    fitted laboratory column). Lengths, masses, flows and densities are ``(value, unit)``
    pairs or numbers in internal units (cm, g, mL/min, g/mL).

    Parameters
    ----------
    carbon
        Column that supplies the adsorbent properties.
    length, diameter
        Bed dimensions.
    flow_rate, loading_rate
        One of them. ``loading_rate`` is a ``(value, unit)`` pair with unit ``m/h``,
        ``cm/s`` or ``gpm/ft2``, or a number in m/h.
    mass, bed_density
        Adsorbent mass, or the bulk density to compute it from; default is the bulk density
        of ``carbon``.
    particle_diameter, mesh
        Particle size as a diameter or a mesh size such as ``"12x40"``; default is the
        particle size of ``carbon``.
    name
        Label of the new column.
    """
    length = _value(length, "length")
    diameter = _value(diameter, "length")
    area = math.pi * diameter**2 / 4.0
    if (flow_rate is None) == (loading_rate is None):
        raise ValueError("give exactly one of flow_rate and loading_rate")
    if flow_rate is not None:
        flow = _value(flow_rate, "flow")
    else:
        if isinstance(loading_rate, tuple):
            value, unit = loading_rate
            try:
                m_per_h = value * _LOADING_RATE[unit.strip().lower()]
            except KeyError:
                raise ValueError(
                    f"unknown loading rate unit {unit!r}; expected one of {list(_LOADING_RATE)}"
                ) from None
        else:
            m_per_h = float(loading_rate)
        flow = m_per_h * 100.0 / 60.0 * area  # cm/min · cm²
    density = _bulk_density(carbon) if bed_density is None else _value(bed_density, "density")
    mass = density * area * length if mass is None else _value(mass, "mass")
    radius = _particle_radius(particle_diameter, mesh) or carbon.particle_radius
    return carbon.with_(
        length=length,
        diameter=diameter,
        mass=mass,
        flow_rate=flow,
        particle_radius=radius,
        bed_density=density,
        name=name or carbon.name,
    )


@dataclass(frozen=True)
class RSSCTDesign:
    """Output of :func:`rssct_design`.

    Attributes
    ----------
    column
        The small-scale column (same adsorbent, smaller particles).
    scale_factor
        ``(d_small / d_large)^(2 - power)``; EBCT and run time scale with it.
    loading_rate
        Superficial velocity of the small column, m/h.
    duration
        Small-column run time matching ``operation_time`` of the large column, or ``None``.
    water_volume
        Water needed for ``duration``, L, or ``None``.
    """

    column: Column
    scale_factor: float
    loading_rate: float
    duration: float | None
    water_volume: float | None

    @property
    def ebct(self) -> float:
        """Empty-bed contact time of the small column, min."""
        return self.column.ebct


def rssct_design(
    large: Column,
    *,
    column_diameter,
    particle_diameter=None,
    mesh: str | None = None,
    power: float = 0.0,
    operation_time: float | None = None,
    time_unit: str = "days",
) -> RSSCTDesign:
    """Design a rapid small-scale column test (RSSCT) that mimics ``large``.

    With ``x = power``, the EBCT and run time scale with ``(d_small / d_large)^(2 - x)`` and
    the superficial velocity with ``d_large / d_small`` (Crittenden et al.); ``x = 0`` is the
    constant-diffusivity and ``x = 1`` the proportional-diffusivity design. The bed has the
    bulk density of ``large`` (``bed_density``, or its mass per bed volume).

    Parameters
    ----------
    large
        The full-scale column.
    column_diameter
        Inner diameter of the small column, cm or ``(value, unit)``.
    particle_diameter, mesh
        Small-column particle size as a diameter (cm or ``(value, unit)``) or mesh size.
    power
        Diffusivity exponent ``x``.
    operation_time
        Run time of the large column, in ``time_unit``; gives the RSSCT ``duration`` (same
        unit) and ``water_volume``.
    """
    radius = _particle_radius(particle_diameter, mesh)
    if radius is None:
        raise ValueError("give particle_diameter or mesh for the small column")
    diameter = _value(column_diameter, "length")
    scale = (radius / large.particle_radius) ** (2.0 - power)
    ebct = large.ebct * scale  # min
    velocity = large.superficial_velocity * large.particle_radius / radius  # cm/s
    length = ebct * 60.0 * velocity
    area = math.pi * diameter**2 / 4.0
    density = _bulk_density(large)
    column = large.with_(
        length=length,
        diameter=diameter,
        mass=density * area * length,
        flow_rate=area * length / ebct,
        particle_radius=radius,
        bed_density=density,
        name=f"{large.name} RSSCT",
    )
    duration = water = None
    if operation_time is not None:
        duration = operation_time * scale
        water = column.flow_rate * duration * units.factor("time", time_unit) / 1e3
    return RSSCTDesign(column, scale, velocity * 36.0, duration, water)


def rescaled_column(column: Column, ebct: float) -> Column:
    """Column with empty-bed contact time ``ebct`` (min).

    Shorter EBCTs than the column's are obtained by raising the flow rate; longer ones by
    enlarging the bed (length and diameter by the same factor, mass with the volume) at
    constant flow. Used by :func:`specific_throughput`.
    """
    s = ebct / column.ebct
    if s <= 1:
        return column.with_(flow_rate=column.flow_rate / s)
    r = s ** (1.0 / 3.0)
    return column.with_(
        length=column.length * r, diameter=column.diameter * r, mass=column.mass * s
    )


_rescaled = rescaled_column  # pre-0.2 private name


def _last_time_below(t, c, limit) -> float:
    """Last time ``t`` with ``c < limit``, without interpolation.

    Upstream ``specific_throughput``: ``results[results < ct].dropna().index[-1]``.
    """
    below = np.flatnonzero(c < limit)
    if len(below) == 0:
        raise ValueError("the effluent is never below the threshold")
    return float(t[below[-1]])


def throughput_from_result(
    result: Result, compound: str, threshold: float, *, time_unit: str = "days"
) -> float:
    """Specific throughput (L/g) of ``result`` up to the breakthrough of ``threshold``.

    As in the US EPA code, the breakthrough time is the last solver output time at which
    the effluent of ``compound`` is below ``threshold`` (not interpolated). If the effluent
    rises above the threshold and falls back below it, that is the later time; if it never
    reaches the threshold, it is the end of the simulation. ``ValueError`` if the effluent is
    never below it. ``threshold`` is in the concentration unit of the simulation and
    ``time_unit`` is the time unit it used. Together with
    :func:`rescaled_column` this is one point of :func:`specific_throughput`::

        col = psdm.fullscale.rescaled_column(column, ebct)
        res = psdm.simulate(col, compound, influent, ...)
        psdm.fullscale.throughput_from_result(res, compound.name, threshold)
    """
    col = result.column
    t = result.effluent.index.to_numpy()
    t_break = _last_time_below(t, result.effluent[compound].to_numpy(), threshold)
    return col.flow_rate / 1e3 * t_break * units.factor("time", time_unit) / col.mass


def specific_throughput(
    column: Column,
    compounds: Compound | Sequence[Compound],
    *,
    c0: float | Mapping[str, float],
    threshold: float | Mapping[str, float],
    ebcts,
    duration: float,
    time_unit: str = "days",
    conc_unit: str = "ug/L",
    **simulate_kw,
) -> pd.DataFrame:
    """Specific throughput (L of water per g of adsorbent) until ``threshold`` is reached.

    Each compound is simulated on its own with constant influent ``c0`` for every EBCT in
    ``ebcts``. Shorter EBCTs than the column's are obtained by raising the flow rate; longer
    ones by enlarging the bed (length and diameter by the same factor) at constant flow.

    Parameters
    ----------
    column, compounds
        As for :func:`psdm.simulate`.
    c0, threshold
        Influent and treatment-objective concentrations in ``conc_unit``, for all compounds
        or per compound name.
    ebcts
        Empty-bed contact times to evaluate, min.
    duration
        Simulated time, in ``time_unit``. Breakthrough is the last solver time with the
        effluent below ``threshold`` (see :func:`throughput_from_result`), so a compound
        that does not reach ``threshold`` within ``duration`` gets the throughput of the
        whole duration, as in the US EPA code.
    **simulate_kw
        Forwarded to :func:`psdm.simulate`.

    Returns
    -------
    pandas.DataFrame
        Index ``ebct`` (min), one column per compound.
    """
    comps = [compounds] if isinstance(compounds, Compound) else list(compounds)

    def pick(value, name):
        return value[name] if isinstance(value, Mapping) else value

    out = pd.DataFrame(index=pd.Index(np.asarray(ebcts, float), name="ebct"), dtype=float)
    for c in comps:
        infl = pd.DataFrame({c.name: [pick(c0, c.name)] * 2}, index=[0.0, float(duration)])
        values = []
        for ebct in out.index:
            col = rescaled_column(column, ebct)
            res = simulate(col, c, infl, time_unit=time_unit, conc_unit=conc_unit, **simulate_kw)
            values.append(
                throughput_from_result(res, c.name, pick(threshold, c.name), time_unit=time_unit)
            )
        out[c.name] = values
    return out


def _intervals(effluent: pd.DataFrame, limits, hazard_index, scale, step) -> pd.Series:
    t_end = float(effluent.index[-1])
    grid = np.arange(0.0, t_end + step / 2, step)
    x = effluent.index.to_numpy(dtype=float)
    daily = {c: np.interp(grid, x, effluent[c].to_numpy(dtype=float)) for c in effluent}
    out = {}
    for name, limit in limits.items():
        if name in daily:
            out[name] = daily[name] > limit / scale
    if hazard_index:
        present = [n for n in hazard_index if n in daily]
        if present:
            hi = sum(daily[n] * scale / hazard_index[n] for n in present)
            out["hazard index"] = hi > 1.0
    result = {}
    for name, exceeded in out.items():
        # last whole step below the limit
        result[name] = max(grid[np.argmax(exceeded)] - step, 0.0) if exceeded.any() else np.nan
    return pd.Series(result, dtype=float)


def replacement_interval(
    effluent: pd.DataFrame,
    *,
    limits: Mapping[str, float] = EPA_PFAS_MCL,
    hazard_index: Mapping[str, float] | None = EPA_PFAS_HAZARD_INDEX,
    conc_unit: str = "ng/L",
    lower: pd.DataFrame | None = None,
    upper: pd.DataFrame | None = None,
    step: float = 1.0,
) -> pd.DataFrame:
    """Operating time until the effluent first exceeds a regulatory limit.

    The effluent is interpolated onto steps of ``step`` (e.g. days) and the result is the
    last step before the first exceedance. This is ``calculate_replacement_interval`` of the
    US EPA code, which uses the 2024 PFAS rule.

    Parameters
    ----------
    effluent
        Effluent concentrations indexed by time, one column per compound (e.g.
        :attr:`psdm.Result.effluent`).
    limits
        Limits in ng/L by compound name; compounds not in ``effluent`` are skipped.
    hazard_index
        Reference concentrations (ng/L) of a hazard index whose limit is 1, or ``None``.
    conc_unit
        Concentration unit of ``effluent``.
    lower, upper
        Optional lower and upper effluent envelopes (e.g. from
        :func:`psdm.model_uncertainty`); the upper envelope gives the shortest interval.
    step
        Time resolution, in the time unit of ``effluent``.

    Returns
    -------
    pandas.DataFrame
        Index of limited quantities (compounds and ``"hazard index"``), columns
        ``predicted``, ``lower`` and ``upper``. ``nan`` means the limit is not exceeded
        within the data.
    """
    scale = units.factor("concentration", conc_unit) * 1e3  # -> ng/L
    out = pd.DataFrame({"predicted": _intervals(effluent, limits, hazard_index, scale, step)})
    out["lower"] = np.nan if upper is None else _intervals(upper, limits, hazard_index, scale, step)
    out["upper"] = np.nan if lower is None else _intervals(lower, limits, hazard_index, scale, step)
    return out


def staggered_bed_interval(
    effluent: pd.Series,
    *,
    total_beds: int,
    beds_per_cycle: int,
    target: float,
    max_interval: float | None = None,
    step: float = 1.0,
) -> tuple[float, pd.Series]:
    """Longest replacement interval of beds operated in parallel with staggered replacement.

    Every ``interval``, the ``beds_per_cycle`` oldest beds are replaced. The blended effluent
    over one interval is the flow-weighted mean of beds of every age group; with a remainder
    group (``total_beds`` not a multiple of ``beds_per_cycle``) it is taken to be the newest,
    the worst case. Ported from ``predict_full_scale`` of the US EPA code.

    Parameters
    ----------
    effluent
        Effluent of a single bed indexed by time (e.g. summed over compounds). Beyond the
        last time the last value is held.
    total_beds, beds_per_cycle
        Number of beds in operation and replaced per cycle.
    target
        Maximum allowed blended concentration.
    max_interval
        Longest interval to consider; default the effluent's last time.
    step
        Time resolution.

    Returns
    -------
    interval : float
        Longest interval whose blended effluent stays at or below ``target`` (``0`` if even
        the shortest step fails).
    blended : pandas.Series
        Blended effluent over one interval of that length.
    """
    if not 0 < beds_per_cycle <= total_beds:
        raise ValueError("need 0 < beds_per_cycle <= total_beds")
    full, rest = divmod(total_beds, beds_per_cycle)
    weights = ([rest / total_beds] if rest else []) + full * [beds_per_cycle / total_beds]
    x = effluent.index.to_numpy(dtype=float)
    y = effluent.to_numpy(dtype=float)
    max_interval = float(x[-1]) if max_interval is None else max_interval
    best, blended = 0.0, pd.Series(dtype=float)
    for n in range(1, int(max_interval / step) + 1):
        tau = np.arange(n) * step
        interval = n * step
        mix = sum(w * np.interp(j * interval + tau, x, y) for j, w in enumerate(weights))
        if mix.max() > target:
            break
        best, blended = interval, pd.Series(mix, index=pd.Index(tau, name="time"))
    return best, blended
