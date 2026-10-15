# Capacity, uncertainty and full-scale design

These tools are ported from the US EPA PSDM (`PSDM.py`, `PSDM_extra.py`, `PSDM_tools.py`).
The examples below reuse `column`, `tce` and `influent` from {doc}`getting-started`.

## Capacity from breakthrough data

When no isotherm is available, {func}`psdm.estimate_capacity` estimates Freundlich `K` for an
assumed `1/n` from a column mass balance:

```python
from psdm.io import read_excel

data = read_excel("Example_TCE.xlsx")
name = data.compound_names[0]
est = psdm.estimate_capacity(
    data.column, data.influent[name], data.effluent[name],
    n_inv=0.45, time_unit=data.time_unit, conc_unit=data.conc_unit,
)
est.k, est.q, est.breakthrough_time, est.method
est.k_for(0.6)        # K giving the same loading for another 1/n
```

The breakthrough time is where the effluent reaches the influent concentration. If it has
not got there yet, it is extrapolated with a logistic curve or, failing that, a straight
line; {data}`psdm.capacity.METHODS` lists the outcomes. With no effluent breakthrough at all
(`method == "minimum"`) the estimate is a lower bound. The result is a good starting point
for {func}`psdm.fit_freundlich`:

```python
start = data.compounds(**{name: {"k": est.k, "n_inv": est.n_inv}})[0]
fit = psdm.fit_freundlich(data.column, start, data.influent[name], data.effluent[name])
```

One deliberate difference from the EPA code: an effluent with a single non-zero value that is
the last point and at least 25 % of the mean influent is extrapolated instead of being
treated as "minimum" (the upstream check for this case never fires).

## Uncertainty bands

{func}`psdm.model_uncertainty` reruns the model with each input raised and lowered by a
fraction and returns the pointwise envelope:

```python
u = psdm.model_uncertainty(
    column, tce, influent, water="Rhine", conc_unit="mg/L",
    capacity=0.1,       # K up and 1/n down by 10 %, and vice versa
    flow_rate=0.05, mass=0.05, c0=0.1, kf=0.2,
)
u.effluent, u.lower, u.upper   # base case and envelope
u.runs["kf+20%"]               # any individual run
```

See {data}`psdm.uncertainty.PARAMETERS` for what each option changes. Mass-transfer
parameters stay at their base-case values in all runs. Fractions are given as `0.1` for 10 %
(the EPA code used `10`).

## Surface-diffusivity fitting

In organic-free water the surface diffusivity can be fitted to breakthrough data once `K` and
`1/n` are known:

```python
fit = psdm.fit_surface_diffusivity(column, tce, influent["TCE"], effluent, conc_unit="mg/L")
fit.ds, fit.ds / fit.ds_ref, fit.evaluations
```

`ds = factor × ds_ref` is evaluated on a grid of factors (the EPA default
`numpy.linspace(1e-10, 1, 30)`), where `ds_ref` is the value from the surface-to-pore
diffusion flux ratio, and the best point is then refined. The EPA code computed its
reference without the Freundlich `K` term, so its `ds_base` differs from `ds_ref`.

## Scaling columns

{mod}`psdm.fullscale` builds full-scale columns from a characterised adsorbent and designs
rapid small-scale column tests (RSSCTs):

```python
from psdm import fullscale

full = fullscale.build_large_scale_column(
    lab_column, length=(2.5, "m"), diameter=(3.66, "m"),
    loading_rate=(5, "gpm/ft2"), mesh="12x40",
)
rssct = fullscale.rssct_design(
    full, column_diameter=(1.1, "cm"), mesh="100x200", power=0.0, operation_time=365,
)
rssct.column, rssct.ebct, rssct.duration, rssct.water_volume
```

`power` is the diffusivity exponent `x`: `0` for the constant-diffusivity and `1` for the
proportional-diffusivity design. The results match worked examples 15-14 and 15-15 of
Crittenden et al., *MWH's Water Treatment*, as checked in the EPA code.

## Operating predictions

```python
# specific throughput (L/g) until 5 ug/L for a range of contact times (min)
fullscale.specific_throughput(
    full, tce, c0=50.0, threshold=5.0, ebcts=[5, 10, 15, 20], duration=1000,
)

# days until a regulatory limit is first exceeded (2024 US EPA PFAS rule by default)
res = psdm.simulate(full, pfas, influent_ng, conc_unit="ng/L")
fullscale.replacement_interval(res.effluent, lower=u.lower, upper=u.upper)

# beds in parallel with staggered replacement
interval, blended = fullscale.staggered_bed_interval(
    res.effluent.sum(axis=1), total_beds=8, beds_per_cycle=2, target=4.0,
)
```

{data}`psdm.fullscale.EPA_PFAS_MCL` holds the April 2024 maximum contaminant levels and
{data}`psdm.fullscale.EPA_PFAS_HAZARD_INDEX` the hazard-index reference values; check the
current regulations before relying on them. Unlike the EPA code, the PFBS contribution is
added to the hazard index (upstream added it to the PFBS column by mistake).
