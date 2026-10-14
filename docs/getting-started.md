# Getting started

## Installation

```bash
pip install psdm
```

Optional extras: `psdm[excel]` adds `openpyxl` for reading Excel input files and
`psdm[plot]` adds `matplotlib` for {func}`psdm.plotting.plot_breakthrough`.

## A first simulation

A simulation needs three things: the **column**, the **compounds**, and the **influent**.

```python
import pandas as pd
import psdm

column = psdm.Column.from_units(
    length=(2.765, "m"),
    diameter=(10, "ft"),
    mass=(20000, "lb"),
    flow_rate=(567, "gpm"),
    particle_radius=(0.513, "mm"),
    particle_density=(0.803, "g/mL"),
    particle_porosity=0.641,
)
print(f"EBCT = {column.ebct:.1f} min")

tce = psdm.Compound(
    "TCE",
    mw=131.39,            # g/mol
    molar_volume=102.0,   # cm³/mol at the normal boiling point
    k=5026.04,            # Freundlich K, (ug/g)(L/ug)^(1/n)
    n_inv=0.43,           # Freundlich 1/n
    fouling_class="chlorinated hydrocarbon",
)

# time in days (index), concentration in mg/L
influent = pd.DataFrame({"TCE": [50.0, 50.0]}, index=[0.0, 174.0])

result = psdm.simulate(column, tce, influent, conc_unit="mg/L", water="Rhine")
```

`result.effluent` is a DataFrame of effluent concentration (same units as the influent)
on the solver's time grid. Calling the result interpolates:

```python
result([30, 60, 90], "TCE")   # numpy array
result(range(0, 175, 7))      # DataFrame, all compounds
result.mass_transfer          # kf, dp, ds that were used
```

To plot:

```python
from psdm.plotting import plot_breakthrough

plot_breakthrough(result, influent=influent)
```

## Units

{class}`~psdm.Column` stores values in internal units (cm, g, mL/min, g/mL); use
{meth}`Column.from_units <psdm.Column.from_units>` to pass `(value, unit)` pairs. The influent
can be in any time unit (`time_unit="days" | "hours" | "min"`) and concentration unit
(`conc_unit="ng/L" | "ug/L" | "mg/L"`); results come back in the same units. Freundlich `K`
is always in (ug/g)(L/ug)^(1/n).

## Several compounds

Pass a list of compounds and one influent column per compound. Compounds compete for
adsorption sites:

```python
pce = psdm.Compound("PCE", mw=165.83, molar_volume=119.0, k=10393.4, n_inv=0.458)
influent = pd.DataFrame({"TCE": [0.2, 0.2], "PCE": [0.2, 0.2]}, index=[0.0, 1000.0])
result = psdm.simulate(column, [tce, pce], influent, conc_unit="mg/L", nr=4, nz=8, ne=2)
```

With many compounds and fouling, the ODE system can be stiff; `nz=3` with `ne` of 20–30
(finite elements along the bed) is often more robust than one high-order element.
