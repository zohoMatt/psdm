"""Run the upstream USEPA ``PSDM.run_all_smart`` K & 1/n fitter on the Example_TCE data.

usage: python generate_fit.py <upstream PSDM dir> <output dir>

Writes ``fit_smart.csv`` (final K, 1/n and the search inputs per case) and
``fit_smart_calls.csv`` (every ``run_psdm_kfit`` call in order, with its ssq).
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
up = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(up))
import psdm.PSDM as P  # noqa: E402

fn = up / "Example_TCE.xlsx"
chem = P.process_input_data(fn, sheet_name="Properties")
k_sheet = pd.read_excel(fn, sheet_name="Kdata", index_col=0)
raw, cinfo, comps, carbons = P.process_input_file(
    fn, data_sheet="data_optimize", column_sheet="columnSpecs"
)
column = cinfo[carbons[0]]
comp = comps[0]

cases = {
    # no k_data: start from __calculate_capacity (k_by_xn_factor, fouling estimate)
    "capacity_organic_free": {"water_type": "Organic Free"},
    "capacity_rhine": {"water_type": "Rhine"},
    # k_data given: constant K for every 1/n, fouling factor 1, ssq up to k_data["brk"]
    "given_organic_free": {"water_type": "Organic Free", "k_data": k_sheet.copy()},
}

summary, calls = [], []
for name, kw in cases.items():
    with contextlib.redirect_stdout(io.StringIO()):
        col = P.PSDM(column, chem, raw, optimize=False, **kw)
    start = {
        "case": name,
        "brk": float(col.k_data[comp]["brk"]),
        "k_start": float(col.k_data[comp]["K"]),
        "q": float(col.k_data[comp]["q"]),
        "fouling_factor": float(col.foul_mult_estimates[comp]),
        "k_by_xn_045": float(col.k_by_xn_factor[comp](0.45))
        if comp in col.k_by_xn_factor
        else np.nan,
    }
    original = col.run_psdm_kfit

    def logged(compound, _orig=original, _name=name, _col=col):
        k = float(_col.k_data.loc["K", compound])
        xn = float(_col.k_data.loc["1/n", compound])
        result = _orig(compound)
        calls.append({"case": _name, "k": k, "n_inv": xn, "ssq": float(result[3].values[0][0])})
        return result

    col.run_psdm_kfit = logged
    with contextlib.redirect_stdout(io.StringIO()):
        col.run_all_smart()
    start.update({"k": float(col.k_data[comp]["K"]), "n_inv": float(col.k_data[comp]["1/n"])})
    summary.append(start)
    print(name, start, sum(c["case"] == name for c in calls), "calls")

pd.DataFrame(summary).set_index("case").to_csv(out / "fit_smart.csv")
pd.DataFrame(calls).to_csv(out / "fit_smart_calls.csv", index=False)
