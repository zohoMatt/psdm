"""Reading the PSDM Excel input template.

The workbook layout is the one used by the original US EPA PSDM examples:

``columnSpecs``
    Rows ``name | value | units`` with ``carbonID, radius, porosity, psdfr,
    particleDensity, apparentDensity, length, weight, flowrate, diameter, tortuosity,
    influentID, effluentID, units, time``.
``data``
    Long table with columns ``type | time | concentration | compound``; ``type`` is the
    influent or effluent identifier.
``Properties``
    Rows ``MW`` and ``MolarVol`` (other rows are ignored), one column per compound.
``Kdata`` (optional)
    Rows ``K`` and ``1/n``, one column per compound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike

import numpy as np
import pandas as pd

from . import units
from .model import Column, Compound
from .solver import Result, simulate


@dataclass
class Dataset:
    """Column test data read by :func:`read_excel`.

    Attributes
    ----------
    column
        The adsorber.
    influent, effluent
        Concentrations indexed by time, one column per compound, on a common time grid
        (gaps filled by linear interpolation).
    properties
        Compound properties (rows include ``MW`` and ``MolarVol``).
    k_data
        Freundlich parameters (rows ``K`` and ``1/n``), if provided.
    time_unit, conc_unit
        Units of the index and values of ``influent``/``effluent``.
    """

    column: Column
    influent: pd.DataFrame
    effluent: pd.DataFrame
    properties: pd.DataFrame
    k_data: pd.DataFrame | None = None
    time_unit: str = "days"
    conc_unit: str = "ug/L"
    compound_names: list[str] = field(default_factory=list)

    def compounds(
        self, fouling_class: str | tuple[float, float] = "halogenated alkenes", **overrides: dict
    ) -> list[Compound]:
        """Build :class:`~psdm.Compound` objects from ``properties`` and ``k_data``.

        ``overrides`` maps a compound name to keyword changes, e.g.
        ``compounds(PFOA={"k": 50.0, "n_inv": 0.4})``.
        """
        out = []
        for name in self.compound_names:
            kw = {
                "name": name,
                "fouling_class": fouling_class,
                "mw": float(self.properties.loc["MW", name]),
                "molar_volume": float(self.properties.loc["MolarVol", name]),
            }
            if self.k_data is not None and name in self.k_data.columns:
                kw["k"] = float(self.k_data.loc["K", name])
                kw["n_inv"] = float(self.k_data.loc["1/n", name])
            kw.update(overrides.get(name, {}))
            if "k" not in kw or "n_inv" not in kw:
                raise ValueError(f"no K and 1/n for {name!r}; pass them as overrides")
            out.append(Compound(**kw))
        return out

    def simulate(self, compounds: list[Compound] | None = None, **kw) -> Result:
        """Run :func:`psdm.simulate` on this dataset's column and influent."""
        kw.setdefault("time_unit", self.time_unit)
        kw.setdefault("conc_unit", self.conc_unit)
        return simulate(self.column, compounds or self.compounds(), self.influent, **kw)


def _spec(specs: pd.DataFrame, key: str, quantity: str | None = None, default=None):
    if key not in specs.index:
        if default is None:
            raise ValueError(f"columnSpecs is missing {key!r}")
        return default
    value = specs.loc[key, "value"]
    if quantity is None:
        return value
    return units.convert(value, str(specs.loc[key, "units"]), quantity)


def _wide(long: pd.DataFrame, kind: str, names: list[str]) -> pd.DataFrame:
    sub = long[long["type"] == kind]
    if sub.empty:
        raise ValueError(f"no rows with type {kind!r} in the data sheet")
    return sub.pivot_table(
        index="time", columns="compound", values="concentration", aggfunc="mean"
    ).reindex(columns=names)


def _fill(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Reindex to ``index`` filling gaps by linear interpolation (and extrapolation)."""
    out = pd.DataFrame(index=index, columns=df.columns, dtype=float)
    x = index.to_numpy(dtype=float)
    for name in df.columns:
        known = df[name].dropna()
        if known.empty:
            out[name] = 0.0
            continue
        xp, fp = known.index.to_numpy(dtype=float), known.to_numpy(dtype=float)
        if len(xp) == 1:
            out[name] = fp[0]
            continue
        y = np.interp(x, xp, fp)
        lo, hi = x < xp[0], x > xp[-1]
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[name] = y
    return out


def read_excel(
    path: str | PathLike,
    *,
    column_sheet: str = "columnSpecs",
    data_sheet: str = "data",
    properties_sheet: str = "Properties",
    kdata_sheet: str | None = "Kdata",
    reporting_limit: float = 0.0,
) -> Dataset:
    """Read a PSDM input workbook (requires ``openpyxl``).

    Parameters
    ----------
    path
        Workbook path.
    column_sheet, data_sheet, properties_sheet, kdata_sheet
        Sheet names; set ``kdata_sheet=None`` if there is no Freundlich data, or if the
        sheet is absent it is skipped.
    reporting_limit
        Concentrations below this value are set to zero.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    specs = sheets[column_sheet].set_index("name")
    specs.index = specs.index.astype(str).str.strip()

    column = Column(
        name=str(_spec(specs, "carbonID", default="column")),
        length=_spec(specs, "length", "length"),
        diameter=_spec(specs, "diameter", "length"),
        mass=_spec(specs, "weight", "mass"),
        flow_rate=_spec(specs, "flowrate", "flow"),
        particle_radius=_spec(specs, "radius", "length"),
        particle_density=_spec(specs, "particleDensity", "density"),
        bed_density=_spec(specs, "apparentDensity", "density"),
        particle_porosity=float(_spec(specs, "porosity")),
        tortuosity=float(_spec(specs, "tortuosity", default=1.0)),
        psdfr=float(_spec(specs, "psdfr", default=5.0)),
    )
    influent_id = str(_spec(specs, "influentID", default="influent"))
    effluent_id = str(_spec(specs, "effluentID", default="effluent"))
    conc_unit = str(_spec(specs, "units", default="ug")).strip()
    units.factor("concentration", conc_unit)
    time_unit = str(_spec(specs, "time", default="days")).strip()
    units.factor("time", time_unit)

    long = sheets[data_sheet].copy()
    long.columns = [str(c).strip().lower() for c in long.columns]
    if "time" not in long.columns and "days" in long.columns:
        long = long.rename(columns={"days": "time"})
    long["concentration"] = long["concentration"].where(
        long["concentration"] >= reporting_limit, 0.0
    )
    names = list(dict.fromkeys(long["compound"]))

    infl = _wide(long, influent_id, names)
    effl = _wide(long, effluent_id, names)
    index = infl.index.union(effl.index).union(pd.Index([0.0])).astype(float).sort_values()
    index = index[index >= 0]
    index.name = "time"
    influent, effluent = _fill(infl, index), _fill(effl, index)
    if 0.0 not in infl.index:
        influent.loc[0.0] = influent.loc[index[1]]

    properties = sheets[properties_sheet].set_index(sheets[properties_sheet].columns[0])
    missing = [n for n in names if n not in properties.columns]
    if missing:
        raise ValueError(f"no properties for {missing} in sheet {properties_sheet!r}")
    k_data = None
    if kdata_sheet is not None and kdata_sheet in sheets:
        k_data = sheets[kdata_sheet].set_index(sheets[kdata_sheet].columns[0])

    return Dataset(
        column=column,
        influent=influent,
        effluent=effluent,
        properties=properties,
        k_data=k_data,
        time_unit=time_unit,
        conc_unit=conc_unit,
        compound_names=names,
    )
