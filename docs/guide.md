# User guide

## Excel input files

{func}`psdm.io.read_excel` reads the workbook layout used by the US EPA PSDM examples
(`Example_TCE.xlsx`, `Example_Multi.xlsx` in the repository's `examples/` folder):

| sheet | content |
|---|---|
| `columnSpecs` | rows `name / value / units`: `carbonID`, `radius`, `porosity`, `psdfr`, `particleDensity`, `apparentDensity`, `length`, `weight`, `flowrate`, `diameter`, `tortuosity`, `influentID`, `effluentID`, `units`, `time` |
| `data` | long table `type / time / concentration / compound`; `type` is the influent or effluent ID |
| `Properties` | rows `MW`, `MolarVol`; one column per compound |
| `Kdata` | rows `K`, `1/n`; one column per compound (optional) |

```python
from psdm.io import read_excel

data = read_excel("Example_TCE.xlsx")
data.column, data.influent, data.effluent

compounds = data.compounds(fouling_class="halogenated alkenes")
result = data.simulate(compounds, water="Rhine")
```

Missing influent or effluent points are filled by linear interpolation (and extrapolation)
onto a common time grid. Values below `reporting_limit` are set to zero. As in the EPA
`process_input_file`: if the data have no time 0, a t = 0 row of zeros is added (a t = 0 row
that only the influent or only the effluent has stays empty for the other), and every zero
influent concentration becomes 0.001.

## Fouling

Natural organic matter gradually reduces adsorption capacity. Set the background `water`
(one of {data}`psdm.fouling.WATER`) and give each compound a `fouling_class` from
{data}`psdm.fouling.CHEMICAL` or an explicit `(b1, b2)` pair; both tables are those of the
AdDesignS database. In fouled water the AdDesignS tortuosity correlation is on by default:
surface diffusion is switched off and the pore tortuosity is 1 for 70 days and then grows
with time (see {doc}`model`). Pass `tortuosity_correlation=False` to keep the SPDFR surface
diffusivity and `Column.tortuosity` in fouled water, or `True` to use the correlation in
organic-free water; AdDesignS offers both choices.

## Mass-transfer parameters

By default the film coefficient `kf` comes from the AdDesignS liquid-phase correlation (a
modified Gnielinski form, multiplied by `Column.shape_factor`), the pore diffusivity `dp` from the Hayduk–Laudie liquid diffusivity divided by tortuosity, and the
surface diffusivity `ds` from the surface-to-pore diffusion flux ratio `psdfr`. Any of them can
be fixed per compound:

```python
tce = tce.with_(kf=3.8e-3, dp=9e-6)
```

As in AdDesignS, `dp` and `ds` cannot be given while the tortuosity correlation is on.

## Migrating from the EPA `PSDM` package

| EPA PSDM | psdm |
|---|---|
| `PSDM.process_input_file` + `process_input_data` | {func}`psdm.io.read_excel` |
| `PSDM.PSDM(column_info, chem_data, raw_data, ...)` | {class}`psdm.Column`, {class}`psdm.Compound` |
| `column.run_psdm()` | {func}`psdm.simulate` (returns a {class}`~psdm.Result`, not a dict of interpolators) |
| `chem_type=` (for all compounds) | `Compound(fouling_class=...)` per compound |
| `water_type=` | `water=` |
| `mass_transfer={comp: {...}}` | `Compound(kf=..., dp=..., ds=...)` |
| fitting, capacity, uncertainty, `PSDM_tools`, `PSDM_extra` | not in psdm (no AdDesignS counterpart) |
