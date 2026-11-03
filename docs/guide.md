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
{data}`psdm.fouling.CHEMICAL`, an explicit `(b1, b2)` pair, or `"PFAS"` to look the compound
name up in {data}`psdm.fouling.PFAS`. With fouling, surface diffusion is effectively switched
off and pore tortuosity increases with time, following the EPA code (see {doc}`model`).

## Mass-transfer parameters

By default the film coefficient `kf` comes from the modified Gnielinski correlation labelled
"Chern and Chien" (choose another with `correlation=`, see {data}`psdm.correlations.SHERWOOD`), the
pore diffusivity `dp` from the Hayduk–Laudie liquid diffusivity divided by tortuosity, and the
surface diffusivity `ds` from the surface-to-pore diffusion flux ratio `psdfr`. Any of them can
be fixed per compound:

```python
tce = tce.with_(kf=3.8e-3, dp=9e-6)
```

## Fitting Freundlich parameters to column data

```python
fit = psdm.fit_freundlich(column, tce, influent["TCE"], effluent["TCE"], water="Rhine")
fit.compound.k, fit.compound.n_inv, fit.ssq
```

This is the EPA PSDM fitter `run_all_smart`, ported step by step and checked against it
(`tests/test_fit_epa.py`):

- `start="capacity"` (default): a mass balance on the data gives the loading `q` and the
  breakthrough time `t_b` ({func}`psdm.estimate_capacity`). For every `1/n` the base `K`
  keeps that `q`, and the `K` factor starts at the inverse mean fouling reduction up to `t_b`.
  `until` is not accepted here.
- `start="compound"`: the base `K` is the compound's `k` for every `1/n` and `until` is the
  window `t_b` (the EPA code with a `Kdata` sheet).
- The sum of squares is taken at whole days `0 … t_b − 1`, with simulated and observed
  effluent linearly interpolated and extrapolated.
- The search moves on a grid of `K` factor (±10 % in 11 steps each way) by `1/n`
  (0.2–1.0 in steps of 0.025), from `1/n` = 0.2, 0.4, … 1.0 towards lower sums of squares.

The result is a grid node, possibly on the grid edge; `fit.ssq_grid` shows every visited node.
The search compares sums of squares node by node, so when two nodes are nearly tied a tiny
numerical difference (another platform, solver tolerance, or the EPA code itself) can lead it
to a neighbouring end point; this is a property of the EPA algorithm.
With a constant influent concentration only the capacity at that concentration is well
determined, not `1/n` and `K` individually — use isotherm data or several concentrations to
pin down `1/n`.

The Nelder–Mead search of psdm 0.1/0.2 (`method="nelder-mead"`), which is not in the EPA
code, was removed in 0.3.0.

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
| `run_all_smart` | {func}`psdm.fit_freundlich` |
| `run_all`, `run_all_MP` (staged/brute grid) | not ported |
| `PSDM_tools.isotherm_fit` | {func}`psdm.fit_isotherm` |
| `PSDM(...)` without `k_data` (`__calculate_capacity`) | {func}`psdm.estimate_capacity` |
| `column.model_uncertainty(...)` | {func}`psdm.model_uncertainty` |
| `column.run_psdm_dsfit(comp)`, `analyze_all_ds` | {func}`psdm.fit_surface_diffusivity` |
| `PSDM_extra.calc_mesh_diameter` | {func}`psdm.fullscale.mesh_diameter` |
| `PSDM_extra.build_large_scale_column` | {func}`psdm.fullscale.build_large_scale_column` |
| `PSDM_extra.calc_RSSCT_specs` | {func}`psdm.fullscale.rssct_design` |
| `PSDM_tools.specific_throughput` | {func}`psdm.fullscale.specific_throughput` |
| `PSDM_tools.calculate_replacement_interval` | {func}`psdm.fullscale.replacement_interval` |
| `PSDM_tools.predict_full_scale` | {func}`psdm.simulate` + {func}`psdm.fullscale.staggered_bed_interval` |
| `chem_type=` (for all compounds) | `Compound(fouling_class=...)` per compound |
| `water_type=` | `water=` |
| `mass_transfer={comp: {...}}` | `Compound(kf=..., dp=..., ds=...)` |
