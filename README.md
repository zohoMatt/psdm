# psdm

[![CI](https://github.com/zohoMatt/psdm/actions/workflows/ci.yml/badge.svg)](https://github.com/zohoMatt/psdm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/psdm.svg)](https://pypi.org/project/psdm/)
[![Docs](https://readthedocs.org/projects/psdm/badge/?version=latest)](https://psdm.readthedocs.io)

The **Pore and Surface Diffusion Model (PSDM)** for fixed-bed granular activated carbon (GAC)
adsorbers, as a small, tested and documented Python package.

- Multi-component competitive adsorption (ideal adsorbed solution theory, Freundlich isotherms)
- Film, pore and surface diffusion with built-in correlations or user-supplied coefficients
- Natural organic matter fouling for the AdDesignS water types and compound classes, including PFAS
- Time-varying influent, any time / concentration / flow units
- Fitting Freundlich `K` and `1/n` to column data; batch isotherm fitting
- Reader for the US EPA PSDM Excel input template

## Installation

```bash
pip install psdm            # core
pip install "psdm[excel]"   # + Excel input files
pip install "psdm[plot]"    # + plotting
```

## Quick start

```python
import pandas as pd
import psdm

column = psdm.Column.from_units(
    length=(2.765, "m"), diameter=(10, "ft"), mass=(20000, "lb"), flow_rate=(567, "gpm"),
    particle_radius=(0.513, "mm"), particle_density=(0.803, "g/mL"), particle_porosity=0.641,
)
tce = psdm.Compound("TCE", mw=131.39, molar_volume=102.0, k=5026.04, n_inv=0.43)
influent = pd.DataFrame({"TCE": [50.0, 50.0]}, index=[0.0, 174.0])  # days, mg/L

result = psdm.simulate(column, tce, influent, conc_unit="mg/L", water="Rhine")
result.effluent          # DataFrame: effluent concentration vs. time
result([30, 60, 90])     # interpolated at given times
```

Reading the EPA Excel template:

```python
from psdm.io import read_excel

data = read_excel("Example_Multi.xlsx")
result = data.simulate(nr=4, nz=8, ne=2)
```

See the [documentation](https://psdm.readthedocs.io) for the model description, the full API
and more examples.

## Validation

The test suite checks the solver against

- breakthrough curves computed by **AdDesignS** for the verification case shipped with the EPA
  code (with and without fouling, Rhine and Portage waters), and
- the original EPA Python implementation (single- and multi-component).

Agreement is within 0.5 % of the influent concentration in all cases.

## Benchmark

[`scripts/benchmark.py`](scripts/benchmark.py) compares psdm with the original
[US EPA PSDM](https://github.com/USEPA/Water_Treatment_Models/tree/3f514c2/PSDM) (commit
`3f514c2`) on 11 cases. Each case is run 7 times, each run in a fresh process, with both
implementations in the same Python environment. The full report, with breakthrough curves,
timing breakdown, solver statistics and methods, is
[`benchmarks/report.html`](benchmarks/report.html) (download it and open it in a browser). The raw
numbers are in [`benchmarks/results.json`](benchmarks/results.json).

| Cases | Speed-up, model setup + run | Speed-up, ODE solver only | Max \|psdm − EPA\| / C0 |
|---|---|---|---|
| Single compound, constant influent (7 AdDesignS cases) | 2.5–3.0× | 2.4–2.8× | 0.28 % |
| Multi-component (`Example_Multi.xlsx`, 2 cases) | 2.5–2.9× | 1.0–1.2× | 0.33 % |
| Variable influent (`Example_TCE.xlsx`, 2 cases) | 1.2–1.4× | 1.0× | 0.20 % |

Measured with median timings on an AMD Ryzen 7 7800X3D, Windows 11, Python 3.10, numpy 2.2.6,
scipy 1.15.3 and pandas 2.3.3. Both codes take nearly the same number of solver steps. The gain
in the single-compound cases comes from a cheaper right-hand side; elsewhere it comes from less
overhead outside the integrator. In the single-compound cases, psdm stays within 0.57 % of C0 of
AdDesignS, and the EPA code within 0.62 %. These maxima are measured on a 349-point resampled
curve, so they differ slightly from the test-suite tolerance above. Re-run with
`uv run python scripts/benchmark.py`.

## Credits

This package is a rewrite of the Python PSDM from the US EPA
[Water Treatment Models](https://github.com/USEPA/Water_Treatment_Models) repository
(J. Burkhardt, L. Haupert, B. Datsov et al.), which converted the Fortran PSDM in AdDesignS™
(D.R. Hokanson, D.W. Hand, J.C. Crittenden, T.N. Rogers, E.J. Oman, Michigan Technological
University) to Python. It is not affiliated with or endorsed by the US EPA or Michigan
Technological University.

## License

MIT — see [LICENSE](LICENSE).
