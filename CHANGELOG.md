# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

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
- `run_all_smart` fouling-factor estimates and `break_type='force'` breakthrough input.

[Unreleased]: https://github.com/zohoMatt/psdm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/zohoMatt/psdm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zohoMatt/psdm/releases/tag/v0.1.0
