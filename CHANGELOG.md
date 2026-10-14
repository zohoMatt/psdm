# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

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

## [0.1.0] - 2026-10-14

### Added

- `psdm.simulate`: multi-component PSDM with NOM fouling, rewritten from the US EPA code.
- `psdm.Column` and `psdm.Compound` input objects with unit conversion.
- `psdm.fit_freundlich` and `psdm.fit_isotherm` for parameter estimation.
- `psdm.io.read_excel` for the EPA PSDM Excel template.
- Validation tests against AdDesignS and the original EPA implementation.

### Not yet ported from the EPA code (0.1.0)

- Capacity estimation from breakthrough data without `K` (`__calculate_capacity`).
- Lead–lag configurations, the equilibrium column model (ECM), full-scale prediction tools,
  model uncertainty bands and surface-diffusivity fitting.

[Unreleased]: https://github.com/zohoMatt/psdm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zohoMatt/psdm/releases/tag/v0.1.0
