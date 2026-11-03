# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-11-03

psdm now covers the PSDM of AdDesignS only, and its core follows the AdDesignS source
(USEPA/Environmental-Technologies-Design-Option-Tool, `VS2019_AdDesignS`). This is the first
release after 0.2.0: 0.3.0 was never published, and most of what it added is removed again
here. Upgrading from 0.2.0, read both sections.

### Removed (breaking)

Everything without an AdDesignS counterpart:

- `fit_freundlich`, `fit_surface_diffusivity`, `sum_of_squares`, `FitResult`, `DsFitResult`
  (`psdm.fitting`); `estimate_capacity`, `CapacityEstimate` (`psdm.capacity`).
- `model_uncertainty`, `Uncertainty` and `psdm.uncertainty`.
- `psdm.fullscale` (RSSCT design, specific throughput, replacement intervals, PFAS MCLs and
  hazard index).
- `fit_isotherm`, `IsothermFit` and `psdm.isotherms`.
- The film-transfer correlations other than the AdDesignS one, `psdm.correlations.SHERWOOD`
  and the `correlation=` argument of `simulate`.
- The PFAS fouling table (`fouling_class="PFAS"`, `fouling.PFAS`) and the "halogenated alkanes
  QSPR" class.

### Changed

- Houghton groundwater `a4` is `-1.04e-4`, as in AdDesignS `water_co.txt` (the EPA code has
  `-7.29e-5`).
- The pore diffusion modulus is `Edp = Dp Dgp tau / R²` and the tortuosity factor on the pore
  term is 1 without the tortuosity correlation, as in AdDesignS. The EPA code multiplied `Edp`
  by the tortuosity again, which changed results for `Column.tortuosity != 1` in organic-free
  water.
- The tortuosity correlation is 1 up to 1.008e5 min and `0.334 + 6.61e-6 t` after, as in the
  AdDesignS `DIFFUN` (previously `max(1, 0.334 + 6.61e-6 t)`).
- `fouling.chemical_coefficients` takes only the class (no compound name).

### Added

- `simulate(tortuosity_correlation=None | True | False)`: the AdDesignS tortuosity
  correlation as its own option. `None` keeps the previous behaviour (on in fouled water).
  While it is on, `Compound.dp`/`ds` raise `InputError`, as AdDesignS does not accept them.
- `Column.shape_factor` (default 1) multiplies the correlated `kf`, as in AdDesignS.
- Compound class `"acids"` `(0.65, 0.35)` from AdDesignS `corr_com.txt`.

### Tests

- `tests/test_tce_epa.py` compares `Example_TCE.xlsx` breakthrough curves with time-varying
  influent against the EPA code: `data_variable` with the `Kdata` values, and `data_optimize`
  at the K and 1/n where the EPA `run_all_smart` ends (including `1/n = 0.975` and the fit to
  the first 55 days). Both codes integrate with `rtol=1e-6` there: at the default
  `rtol=1e-3` each code's own integration error reaches ~0.7 % of C0 and depends on the
  platform. The largest difference is 0.06 % of the maximum influent concentration
  (tolerance 0.1 %).
- The tortuosity now has its own tests: it changes only `Dp` and `Ds`, as in AdDesignS.

## [0.3.0] - 2026-11-02 (not published)

### Changed

- **`fit_freundlich` is now the EPA fitter.** The default `method="epa"` is a step-by-step
  port of `PSDM.run_all_smart` (start from the mass-balance capacity estimate, sum of squares
  at whole days up to breakthrough, neighbour search on a `K` factor × `1/n` grid). Fed the
  EPA code's own sums of squares it visits the same parameters in the same order and returns
  the same result; end to end it reproduces the EPA result on the reference cases (see
  `tests/test_fit_epa.py`).
- **`fit_surface_diffusivity` is now exactly the EPA `run_psdm_dsfit`** (breaking): the best
  point of the grid `ds_ref × numpy.linspace(1e-10, 1, 30)`, with the sum of squares at whole
  time units up to the breakthrough time (`until`, else the `estimate_capacity` estimate)
  between the linearly interpolated and extrapolated simulated and observed effluent. The
  psdm refinement step and its `refine` and `refine_maxiter` arguments are gone, and `until`
  now sets the EPA window instead of cutting observations. `DsFitResult` gained
  `breakthrough_time`; `surface_diffusivity_max_evaluations(factors)` is the grid size.
  `ds_ref` is unchanged (the EPA `ds_base` leaves out the molar `K` and tortuosity terms;
  still open).
- **`psdm.io.read_excel` handles t = 0 and zero influent like the EPA
  `process_input_file`** (breaking): a missing t = 0 row is added as zeros (psdm copied the
  next influent value and extrapolated the effluent), a t = 0 row that only one data type
  has stays `NaN` for the other, and zero influent concentrations become 0.001.
- **`psdm.fullscale.specific_throughput` and `throughput_from_result` take the EPA
  breakthrough time** (breaking): the last solver output time with the effluent below the
  threshold, not interpolated (psdm interpolated the first exceedance). An effluent that
  never reaches the threshold now gives the throughput of the whole simulated time instead
  of `nan`, and one that is never below it raises `ValueError`.

### Removed

- **`fit_freundlich(method="nelder-mead")`** (breaking) and its arguments `n_inv_bounds`,
  `q_factor_bounds` and `max_evaluations`: the psdm 0.1/0.2 Nelder–Mead search was not in
  the EPA code. `method` accepts only `"epa"`.
- **`fit_freundlich(until=...)` with `start="capacity"`** (breaking) now raises: overriding
  the estimated breakthrough time was a psdm addition. `until` is the EPA `k_data["brk"]` of
  `start="compound"`.

### Added

- `fit_freundlich(start=, pm=, num=, des_xn=, search_limit=)` and the `FitResult`
  fields `method`, `ssq_grid`, `capacity`, `breakthrough_time` and `converged`.
- A `UserWarning` when `Column.bed_density` differs from mass / bed volume by more than 5 %
  (`psdm.model.BED_DENSITY_TOLERANCE`). The EPA example files trigger it.

### Fixed

- The documentation of the `capacity` perturbation of `model_uncertainty`: as in the EPA
  code, `capacity+` raises `K` and lowers `1/n`, which *lowers* the capacity above about
  9 ug/L. Behaviour is unchanged.

## [0.2.0] - 2026-10-30

All additions are backward compatible: 0.1.0 signatures are unchanged and results are
bit-for-bit identical.

### Added

- `psdm.errors`: `SolverError` (a `RuntimeError`) with `SolverDivergedError` and
  `StepSizeTooSmallError`, and `InputError` (a `ValueError`) with a `field` attribute.
  `simulate` raises these instead of bare `RuntimeError`/`ValueError`.
- `Result.diagnostics`: Stanton number, surface and pore diffusion moduli and Biot numbers
  per compound.
- `callback(evaluations, best_ssq)` keyword for `fit_freundlich` and
  `fit_surface_diffusivity`, called after every simulation; raising from it cancels the fit.
- `fit_surface_diffusivity(refine_maxiter=50)` and
  `psdm.fitting.surface_diffusivity_max_evaluations` for a progress denominator.
- `psdm.uncertainty.plan`, `envelope`, `Run` and `Plan`: `model_uncertainty` split into
  listing the runs and combining their results, so the simulations can run in parallel.
- `psdm.fullscale.rescaled_column` (previously private) and
  `psdm.fullscale.throughput_from_result`: one point of `specific_throughput`.

### Fixed

- Linear-algebra failures inside the ODE solver (e.g. a singular Newton matrix) are raised
  as `SolverDivergedError` instead of escaping from SciPy.

## [0.1.0] - 2026-10-14

First release.

### Added

- `psdm.simulate`: multi-component PSDM with NOM fouling, rewritten from the US EPA code.
- `psdm.Column` and `psdm.Compound` input objects with unit conversion.
- `psdm.fit_freundlich` and `psdm.fit_isotherm` for parameter estimation.
- `psdm.io.read_excel` for the EPA PSDM Excel template.
- Validation tests against AdDesignS and the original EPA implementation.
- `scripts/benchmark.py` and `benchmarks/report.html`: case-by-case comparison with the
  original EPA PSDM code (11 cases).
- `psdm.estimate_capacity`: Freundlich `K` from breakthrough data for an assumed `1/n`
  (upstream `__calculate_capacity`), with breakthrough detection and logistic/linear
  extrapolation. Validated against the EPA code.
- `psdm.model_uncertainty`: uncertainty bands from ± perturbations of capacity, `K`, `1/n`,
  influent concentration, mass, flow rate, `kf`, `dp` and `ds` (upstream `model_uncertainty`).
- `psdm.fit_surface_diffusivity`: grid search plus refinement for `ds` (upstream
  `run_psdm_dsfit`).
- `psdm.fullscale`: `mesh_diameter`, `build_large_scale_column`, `rssct_design`,
  `specific_throughput`, `replacement_interval` (2024 US EPA PFAS MCLs and hazard index) and
  `staggered_bed_interval` (upstream `PSDM_extra.py` and `PSDM_tools.py`).

### Fixed relative to the EPA code

- The surface-diffusivity reference includes the Freundlich `K` term.
- The PFBS term is added to the hazard index in the replacement interval.
- Longer EBCTs in `specific_throughput` enlarge the bed instead of shrinking it.
- RSSCT particle diameters given with units are converted correctly.

### Still not ported

- Lead–lag configurations (`lead_lag_class.py`, needs the `ixpy` package).
- The equilibrium column model (`ecm.py`).
- `run_all` / `run_all_MP` (staged and brute-force grid fitting) and `break_type='force'`
  breakthrough input.

[Unreleased]: https://github.com/zohoMatt/psdm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/zohoMatt/psdm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/zohoMatt/psdm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zohoMatt/psdm/releases/tag/v0.1.0
