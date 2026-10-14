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

Missing influent or effluent points are filled by linear interpolation onto a common time
grid. Values below `reporting_limit` are set to zero.

## Fouling

Natural organic matter gradually reduces adsorption capacity. Set the background `water`
(one of {data}`psdm.fouling.WATER`) and give each compound a `fouling_class` from
{data}`psdm.fouling.CHEMICAL`, an explicit `(b1, b2)` pair, or `"PFAS"` to look the compound
name up in {data}`psdm.fouling.PFAS`. With fouling, surface diffusion is effectively switched
off and pore tortuosity increases with time, as in AdDesignS.

## Mass-transfer parameters

By default the film coefficient `kf` comes from the Chern & Chien (modified Gnielinski)
correlation (choose another with `correlation=`, see {data}`psdm.correlations.SHERWOOD`), the
pore diffusivity `dp` from the Hayduk–Laudie liquid diffusivity divided by tortuosity, and the
surface diffusivity `ds` from the pore-to-surface diffusion flux ratio `psdfr`. Any of them can
be fixed per compound:

```python
tce = tce.with_(kf=3.8e-3, dp=9e-6)
```

## Fitting Freundlich parameters to column data

```python
fit = psdm.fit_freundlich(column, tce, influent["TCE"], effluent["TCE"], water="Rhine")
fit.compound.k, fit.compound.n_inv, fit.ssq
```

The search starts from the compound's `k` and `n_inv` and minimises the squared error at the
observed effluent times. It runs over `(log q_ref, 1/n)` where `q_ref` is the capacity at the
mean influent concentration, so changing `1/n` does not change the predicted capacity much.
With a constant influent concentration, `q_ref` is well determined but `1/n` and `K` are not
individually — use isotherm data or several concentrations to pin down `1/n`.

## Batch isotherms

```python
q = psdm.isotherms.batch_loading(c0, ce, mass, volume)
fit = psdm.fit_isotherm(ce, q, "freundlich")   # or "langmuir", "redlich-peterson"
fit.params, fit.stderr, fit.r2
```

## Migrating from the EPA `PSDM` package

| EPA PSDM | psdm |
|---|---|
| `PSDM.process_input_file` + `process_input_data` | {func}`psdm.io.read_excel` |
| `PSDM.PSDM(column_info, chem_data, raw_data, ...)` | {class}`psdm.Column`, {class}`psdm.Compound` |
| `column.run_psdm()` | {func}`psdm.simulate` (returns a {class}`~psdm.Result`, not a dict of interpolators) |
| `column.run_psdm_kfit(comp)` with a single `test_range`/`xn_range` | {func}`psdm.simulate` + {func}`psdm.sum_of_squares` |
| `run_all`, `run_all_smart` | {func}`psdm.fit_freundlich` |
| `PSDM_tools.isotherm_fit` | {func}`psdm.fit_isotherm` |
| `chem_type=` (for all compounds) | `Compound(fouling_class=...)` per compound |
| `water_type=` | `water=` |
| `mass_transfer={comp: {...}}` | `Compound(kf=..., dp=..., ds=...)` |
