"""Numerical solution of the pore and surface diffusion model.

The model is the AdDesignS PSDM (Hokanson et al., Michigan Technological University) as
ported to Python by Burkhardt et al. (US EPA). Each compound has ``nr`` radial nodes in the
particle plus one bulk-liquid node at each of ``nz`` axial positions; competition between
compounds follows ideal adsorbed solution theory for Freundlich isotherms. The resulting
ODE system is solved in dimensionless time with :func:`scipy.integrate.solve_ivp`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp

from . import collocation, fouling, properties, units
from .correlations import film_transfer
from .model import Column, Compound

SURFACE_DIFFUSION_FOULED_FACTOR = 1e-10
"""Surface diffusion is effectively switched off in fouled water (AdDesignS behaviour)."""


@dataclass
class Result:
    """Output of :func:`simulate`.

    Attributes
    ----------
    effluent
        Effluent concentration per compound (columns) on the solver time grid (index), in
        the concentration and time units of the influent.
    mass_transfer
        ``kf`` (cm/s), ``dp`` (cm²/s) and ``ds`` (cm²/s) used for each compound.
    column
        The simulated column.
    compounds
        The simulated compounds.
    solution
        Raw :func:`scipy.integrate.solve_ivp` output.
    """

    effluent: pd.DataFrame
    mass_transfer: pd.DataFrame
    column: Column
    compounds: list[Compound]
    solution: object

    def __call__(self, t, compound: str | None = None):
        """Effluent concentration interpolated at times ``t``.

        Returns an array for a single ``compound`` or a DataFrame for all compounds.
        """
        t = np.asarray(t, dtype=float)
        x = self.effluent.index.to_numpy()
        if compound is not None:
            return np.interp(t, x, self.effluent[compound].to_numpy())
        return pd.DataFrame(
            {c: np.interp(t, x, self.effluent[c].to_numpy()) for c in self.effluent},
            index=np.atleast_1d(t),
        )


@dataclass
class _System:
    """The discretised ODE system ``dy/dt = rhs(t, y)`` in dimensionless time."""

    rhs: Callable[[float, np.ndarray], np.ndarray]
    shape: tuple[int, int, int]  # (compound, radial nodes + bulk, axial nodes)
    t_end: float
    time_scale: float  # dimensionless time per unit of user time
    conc_scale: np.ndarray  # user concentration per unit of dimensionless concentration
    names: list[str]
    mass_transfer: pd.DataFrame

    def sparsity(self) -> sparse.csr_matrix:
        n_comp, nr1, nz = self.shape
        return collocation.jacobian_sparsity(n_comp, nr1 - 1, nz)


def _as_compounds(compounds: Compound | Sequence[Compound]) -> list[Compound]:
    out = [compounds] if isinstance(compounds, Compound) else list(compounds)
    if not out:
        raise ValueError("at least one compound is required")
    names = [c.name for c in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate compound names: {names}")
    return out


def _as_influent(influent: pd.DataFrame | pd.Series, names: list[str]) -> pd.DataFrame:
    if isinstance(influent, pd.Series):
        if len(names) != 1:
            raise ValueError("a Series influent is only allowed for a single compound")
        influent = influent.to_frame(names[0])
    missing = [n for n in names if n not in influent.columns]
    if missing:
        raise ValueError(f"influent has no data for {missing}")
    influent = influent[names].sort_index()
    if len(influent) < 2:
        raise ValueError("influent needs at least two time points")
    if influent.index[0] != 0:
        raise ValueError("influent must start at time 0")
    return influent


def _build_system(
    column: Column,
    comps: list[Compound],
    influent: pd.DataFrame,
    *,
    water: str,
    temperature: float,
    time_unit: str,
    conc_unit: str,
    nr: int,
    nz: int,
    ne: int,
    correlation: str,
) -> _System:
    names = [c.name for c in comps]
    t_mult = units.factor("time", time_unit)  # -> minutes
    mass_mul = units.factor("concentration", conc_unit)  # -> ug/L
    fouling.water_coefficients(water)  # validate early
    fouled = water != fouling.ORGANIC_FREE

    def vec(attr):
        return np.array([getattr(c, attr) for c in comps], dtype=float)

    mw, molar_volume, k, xn = vec("mw"), vec("molar_volume"), vec("k"), vec("n_inv")
    xni = 1.0 / xn

    # --- column hydraulics and water properties
    vw = properties.water_viscosity(temperature)
    dw = properties.water_density(temperature)
    ebed = column.bed_porosity
    tau = column.residence_time
    rad = column.particle_radius
    epor = column.particle_porosity
    rhop = column.particle_density
    re = 2.0 * rad * column.superficial_velocity * dw / (ebed * vw)

    # --- influent in relative molar units (umol/L divided by the t = 0 value)
    t_data_min = influent.index.to_numpy(dtype=float) * t_mult
    c_molar = influent.to_numpy(dtype=float).T * mass_mul / mw[:, None]
    cb0 = c_molar[:, 0].copy()
    cb0[cb0 == 0.0] = 1.0
    c_rel = c_molar / cb0[:, None]

    # --- mass transfer parameters
    difl = properties.liquid_diffusivity(molar_volume, vw)
    sc = vw / (dw * difl)
    kf = film_transfer(difl, rad, re, sc, ebed, correlation)
    tortu0 = 1.0 if fouled else column.tortuosity
    dp = difl / tortu0
    molar_k = k / mw / (1.0 / mw) ** xn  # (umol/g)(L/umol)^(1/n)
    ds = epor * difl * cb0 * column.psdfr / (tortu0 * 1e3 * rhop * molar_k * cb0**xn)
    for i, c in enumerate(comps):
        if c.kf is not None:
            kf[i] = c.kf
        if c.dp is not None:
            dp[i] = c.dp
        if c.ds is not None:
            ds[i] = c.ds
    if fouled:
        ds = ds * SURFACE_DIFFUSION_FOULED_FACTOR
    mass_transfer = pd.DataFrame([kf, dp, ds], index=["kf", "dp", "ds"], columns=names)

    # --- dimensionless groups
    d = ds / dp
    qe = molar_k * cb0**xn
    ym = qe / qe.sum()
    dgs = rhop * qe * (1.0 - ebed) * 1000.0 / (ebed * cb0)  # solute distribution, surface
    dgp = epor * (1.0 - ebed) / ebed  # solute distribution, pore
    dg = dgs + dgp
    dgt = dg.sum()
    dg1 = 1.0 + dgt
    edd = dgt / dg
    eds = np.maximum(ds * dgs * tau / rad**2, 1e-130)  # surface diffusion modulus
    edp = dp * tortu0 * dgp * tau / rad**2  # pore diffusion modulus
    beds = (eds + d * edp) * edd
    bedp = edp * (1.0 - d) * edd
    stdv = kf * (1.0 - ebed) * tau / (ebed * rad) * dgt  # modified Stanton number
    stdv_dg = stdv / dg
    tconv = 60.0 / (tau * dg1)  # dimensionless time per minute

    # --- time-dependent inputs (time in minutes)
    b = [fouling.chemical_coefficients(c.fouling_class, c.name) for c in comps]

    def time_inputs(t_min):
        c_in = np.array([np.interp(t_min, t_data_min, row) for row in c_rel])
        if fouled:
            mult = np.array([fouling.k_multiplier(t_min, water, bi) for bi in b])
            foul_fac = (1.0 / mult) ** xni
            tort = fouling.tortuosity(t_min)
        else:
            foul_fac = np.ones(len(comps))
            tort = column.tortuosity
        fac = (1.0 / tort - d) / (1.0 - d)
        return c_in, foul_fac, fac

    # --- discretisation
    col = collocation.build(nr, nz, ne)
    shape = (len(comps), nr + 1, col.nz)
    br = col.br[:-1]  # interior radial rows
    wr_int, wr_s = col.wr[: nr - 1], col.wr[nr - 1]
    az_bulk = col.az[1:]
    ym3 = ym[:, None, None]
    xni3 = xni[:, None, None]
    xn_ym3 = (xn / ym)[:, None, None]

    def rhs(t, y):
        y = y.reshape(shape)
        q = y[:, :nr, :]  # dimensionless solid loading at radial nodes
        cb = y[:, nr, :]  # dimensionless bulk liquid concentration
        c_in, foul_fac, fac = time_inputs(t / tconv)

        # pore-liquid concentration in equilibrium with the solid (IAST, Freundlich)
        z = np.maximum(ym3 * q, 0.0)
        z_tot = z.sum(axis=0)
        z_tot[z_tot == 0.0] = 1e-30
        spread = (xni3 * z).sum(axis=0)
        cpore = (z / z_tot) * (xn_ym3 * spread) ** xni3 * foul_fac[:, None, None]
        cpore[:, spread <= 0.0] = 0.0
        c_surf = cpore[:, nr - 1, :]

        flux = (cb - c_surf) * stdv[:, None]  # film flux at each axial node
        flux[:, 0] = 0.0

        dy = np.empty(shape)
        # particle interior: pore + surface diffusion
        bb = (fac * bedp)[:, None, None] * (br @ cpore) + beds[:, None, None] * (br @ q)
        dy[:, : nr - 1, :] = bb
        # particle surface: mass balance with the film
        ww = np.einsum("r,crz->cz", wr_int, bb)
        dy[:, nr - 1, 0] = ((c_in - c_surf[:, 0]) * stdv_dg - ww[:, 0]) / wr_s
        dy[:, nr - 1, 1:] = (flux[:, 1:] / dg[:, None] - ww[:, 1:]) / wr_s
        # bulk liquid: advection and uptake; the inlet node is the influent itself
        dy[:, nr, 0] = 0.0
        advection = az_bulk[:, 0] * c_in[:, None] + cb[:, 1:] @ az_bulk[:, 1:].T
        dy[:, nr, 1:] = -dg1 * advection - 3.0 * flux[:, 1:]
        return dy.ravel()

    return _System(
        rhs=rhs,
        shape=shape,
        t_end=t_data_min[-1] * tconv,
        time_scale=tconv * t_mult,
        conc_scale=cb0 * mw / mass_mul,
        names=names,
        mass_transfer=mass_transfer,
    )


def simulate(
    column: Column,
    compounds: Compound | Sequence[Compound],
    influent: pd.DataFrame | pd.Series,
    *,
    water: str = fouling.ORGANIC_FREE,
    temperature: float = 20.0,
    time_unit: str = "days",
    conc_unit: str = "ug/L",
    nr: int = 14,
    nz: int = 19,
    ne: int = 1,
    correlation: str = "Chern and Chien",
    method: str = "BDF",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> Result:
    """Simulate effluent breakthrough from a fixed-bed adsorber.

    Parameters
    ----------
    column
        Adsorber description.
    compounds
        One or more compounds; several compounds compete for adsorption sites.
    influent
        Influent concentrations indexed by time, one column per compound name (a Series is
        accepted for a single compound). The first index value must be ``0`` and the
        simulation runs to the last index value. Concentrations are linearly interpolated.
    water
        Background water type for NOM fouling (see :data:`psdm.fouling.WATER`).
    temperature
        Water temperature, °C.
    time_unit, conc_unit
        Units of the influent index and values; the result uses the same units.
    nr, nz, ne
        Radial points, axial points per element, and axial elements. For many competing
        compounds with fouling, ``nz=3`` with ``ne`` of 20–30 is often more robust.
    correlation
        Film-transfer correlation, see :data:`psdm.correlations.SHERWOOD`.
    method, rtol, atol
        Passed to :func:`scipy.integrate.solve_ivp`.
    """
    comps = _as_compounds(compounds)
    influent = _as_influent(influent, [c.name for c in comps])
    system = _build_system(
        column,
        comps,
        influent,
        water=water,
        temperature=temperature,
        time_unit=time_unit,
        conc_unit=conc_unit,
        nr=nr,
        nz=nz,
        ne=ne,
        correlation=correlation,
    )

    options = {}
    if method in ("BDF", "Radau"):
        options["jac_sparsity"] = system.sparsity()
    sol = solve_ivp(
        system.rhs,
        (0.0, system.t_end),
        np.zeros(np.prod(system.shape)),
        method=method,
        rtol=rtol,
        atol=atol,
        **options,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    n_comp, nr1, mz = system.shape
    c_out = sol.y.reshape((n_comp, nr1, mz, -1))[:, nr1 - 1, mz - 1, :]
    c_out = np.maximum(c_out, 0.0) * system.conc_scale[:, None]
    time_index = pd.Index(sol.t / system.time_scale, name=influent.index.name or "time")
    effluent = pd.DataFrame(c_out.T, index=time_index, columns=system.names)
    return Result(
        effluent=effluent,
        mass_transfer=system.mass_transfer,
        column=column,
        compounds=comps,
        solution=sol,
    )
