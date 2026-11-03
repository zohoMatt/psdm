"""Example_TCE breakthrough curves against the upstream EPA PSDM (generate_tce.py).

Real column data with time-varying influent. The ``optimize_*`` cases use the K and 1/n at
which upstream ``run_all_smart`` ends on the ``data_optimize`` sheet, including the extreme
end point ``1/n = 0.975`` and the fit to the first 55 days only, where the starting K predicts
no breakthrough: these are the curves the EPA fitter compares with the data.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import DATA, EXAMPLES
from psdm.io import read_excel

PARAMS = pd.read_csv(DATA / "reference" / "tce_params.csv", index_col="case")
CURVES = pd.read_csv(DATA / "reference" / "tce_curves.csv", header=[0, 1])
TOLERANCE = 5e-3  # of the largest influent concentration, as for the other EPA comparisons


@pytest.mark.parametrize("case", PARAMS.index)
def test_tce_curve_matches_original_epa_code(case):
    p = PARAMS.loc[case]
    data = read_excel(EXAMPLES / "Example_TCE.xlsx", data_sheet=p["sheet"])
    (name,) = data.compound_names
    compounds = data.compounds(**{name: {"k": float(p["k"]), "n_inv": float(p["n_inv"])}})
    result = data.simulate(compounds, water=p["water"])

    t = CURVES[(case, "time")].to_numpy()
    expected = CURVES[(case, "effluent")].to_numpy()
    c0 = data.influent[name].max()
    diff = np.abs(result(t, name) - expected)
    assert diff.max() < TOLERANCE * c0, f"max diff {diff.max():.3g} ug/L at t = {t[diff.argmax()]}"


def test_tce_cases_cover_breakthrough():
    """Every reference curve breaks through, so the comparison is not of two zero curves."""
    for case in PARAMS.index:
        assert CURVES[(case, "effluent")].max() > 0.05 * 1000, case
