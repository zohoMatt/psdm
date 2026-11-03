"""Code paths not covered by tests/test_solver.py, compared with the original EPA code.

Reference curves: tests/data/reference/generate_paths.py (cases from PATHS in
scripts/benchmark.py). AdDesignS column, constant 5 mg/L influent for 174 d.
"""

import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA

RHINE = {"water": "Rhine", "fouling_class": "chlorinated hydrocarbon"}
CASES = {
    "tortu2": {"tortuosity": 2.0},
    "mt_of": {"kf": 2e-3, "dp": 4e-6, "ds": 1e-9},
    "mt_rhine": {**RHINE, "kf": 5e-3, "dp": 1.2e-5, "ds": 1e-9},
    "pfas": {"water": "Rhine", "fouling_class": "PFAS", "name": "PFOA"},
    "pfas_avg": {"water": "Rhine", "fouling_class": "PFAS", "name": "PFXX"},
    "gnielinski": {"correlation": "Gnielinski"},
    "wakao": {**RHINE, "correlation": "Wakao and Funazkri"},
    "temp10": {"temperature": 10.0},
    "temp30_rhine": {**RHINE, "temperature": 30.0},
    "ne3": {"nr": 8, "nz": 5, "ne": 3},
    "hours": {"time_unit": "hours"},
}
C0 = 5.0


@pytest.mark.parametrize("case", CASES)
def test_matches_original_epa_code(case, addesigns_column, tce):
    opts = dict(CASES[case])
    column = addesigns_column.with_(tortuosity=opts.pop("tortuosity", 1.0))
    compound = tce.with_(
        name=opts.pop("name", "TCE"),
        fouling_class=opts.pop("fouling_class", "halogenated alkenes"),
        **{k: opts.pop(k) for k in ("kf", "dp", "ds") if k in opts},
    )
    scale = 24.0 if opts.get("time_unit") == "hours" else 1.0
    influent = pd.DataFrame({compound.name: [C0, C0]}, index=[0.0, 174.0 * scale])
    result = psdm.simulate(column, compound, influent, conc_unit="mg/L", **opts)

    ref = pd.read_csv(DATA / "reference" / "paths.csv", index_col=0)[case]
    got = result(ref.index.to_numpy() * scale, compound.name)
    assert np.abs(got - ref.to_numpy()).max() < 5e-3 * C0
