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

## Credits

This package is a rewrite of the Python PSDM from the US EPA
[Water Treatment Models](https://github.com/USEPA/Water_Treatment_Models) repository
(J. Burkhardt, L. Haupert, B. Datsov et al.), which converted the Fortran PSDM in AdDesignS™
(D.R. Hokanson, D.W. Hand, J.C. Crittenden, T.N. Rogers, E.J. Oman, Michigan Technological
University) to Python. It is not affiliated with or endorsed by the US EPA or Michigan
Technological University.

## License

MIT — see [LICENSE](LICENSE).
