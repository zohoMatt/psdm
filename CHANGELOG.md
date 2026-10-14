# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-14

### Added

- `psdm.simulate`: multi-component PSDM with NOM fouling, rewritten from the US EPA code.
- `psdm.Column` and `psdm.Compound` input objects with unit conversion.
- `psdm.fit_freundlich` and `psdm.fit_isotherm` for parameter estimation.
- `psdm.io.read_excel` for the EPA PSDM Excel template.
- Validation tests against AdDesignS and the original EPA implementation.

### Not yet ported from the EPA code

- Capacity estimation from breakthrough data without `K` (`__calculate_capacity`).
- Lead–lag configurations, the equilibrium column model (ECM), full-scale prediction tools,
  model uncertainty bands and surface-diffusivity fitting.

[Unreleased]: https://github.com/zohoMatt/psdm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zohoMatt/psdm/releases/tag/v0.1.0
