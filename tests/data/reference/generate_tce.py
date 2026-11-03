"""Run the upstream USEPA PSDM on the Example_TCE workbook and save breakthrough curves.

usage: python generate_tce.py <upstream PSDM dir> <output dir>

Writes ``tce_curves.csv`` (effluent on a 301-point grid per case, ug/L) and
``tce_params.csv`` (K, 1/n and water per case). Every case is one ``run_psdm_kfit`` call with
a single K and 1/n, i.e. one simulation with time-varying influent:

- ``data_variable`` with the Kdata sheet values, in organic-free and Rhine water;
- ``data_optimize`` with the end points of upstream ``run_all_smart`` on that sheet: from the
  capacity estimate (organic-free and Rhine water), from the Kdata sheet, and from the
  capacity estimate on the first 55 days only (where the starting K predicts no breakthrough).
  These are the curves the fitter compares with the data; the fit itself is not part of psdm.

The fits run with upstream's default solver tolerances (scipy's ``rtol=1e-3``), as a user
would get them. The curves are then integrated with ``rtol=RTOL, atol=ATOL``: at the default
tolerances the integration error of either code on these cases is up to ~0.7 % of the influent
concentration and depends on the platform, which would hide or fake a model difference.
"""

import contextlib
import functools
import io
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
up = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
# upstream's psdm/ has no __init__.py and would be shadowed by an installed psdm package
pkg = types.ModuleType("epa_psdm")
pkg.__path__ = [str(up / "psdm")]
sys.modules["epa_psdm"] = pkg
import epa_psdm.PSDM as P  # noqa: E402

RTOL, ATOL = 1e-6, 1e-9  # keep in sync with tests/test_tce_epa.py
default_solve_ivp = P.solve_ivp
fn = up / "Example_TCE.xlsx"
chem = P.process_input_data(fn, sheet_name="Properties")
k_sheet = pd.read_excel(fn, sheet_name="Kdata", index_col=0)


def load(sheet, until=None):
    raw, cinfo, comps, carbons = P.process_input_file(
        fn, data_sheet=sheet, column_sheet="columnSpecs"
    )
    if until is not None:
        raw = raw[raw.index <= until]
    return raw, cinfo[carbons[0]], comps[0]


def fitted(sheet, water, k_data=None, until=None):
    raw, column, comp = load(sheet, until)
    kw = {} if k_data is None else {"k_data": k_data.copy()}
    with contextlib.redirect_stdout(io.StringIO()):
        col = P.PSDM(column, chem, raw, water_type=water, optimize=False, **kw)
        col.run_all_smart()
    return float(col.k_data[comp]["K"]), float(col.k_data[comp]["1/n"])


def curve(sheet, water, k, xn):
    raw, column, comp = load(sheet)
    kd = k_sheet.copy()
    kd.loc["K", comp], kd.loc["1/n", comp] = k, xn
    P.solve_ivp = functools.partial(default_solve_ivp, rtol=RTOL, atol=ATOL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            col = P.PSDM(column, chem, raw, water_type=water, k_data=kd, optimize=False)
            col.test_range = np.array([k])
            col.xn_range = np.array([xn])
            md = col.run_psdm_kfit(comp)[-1]
    finally:
        P.solve_ivp = default_solve_ivp
    grid = np.linspace(0, float(raw.index.max()), 301)
    return grid, np.interp(grid, md.index.values, md.values[:, 0])


comp_name = load("data")[2]
kd_k, kd_xn = float(k_sheet.loc["K", comp_name]), float(k_sheet.loc["1/n", comp_name])
cases = {
    "variable_organic_free": ("data_variable", "Organic Free", kd_k, kd_xn),
    "variable_rhine": ("data_variable", "Rhine", kd_k, kd_xn),
}
for name, water, k_data, until in [
    ("optimize_capacity_organic_free", "Organic Free", None, None),
    ("optimize_capacity_rhine", "Rhine", None, None),
    ("optimize_given_organic_free", "Organic Free", k_sheet, None),
    ("optimize_capacity_55d_organic_free", "Organic Free", None, 55.0),
]:
    k, xn = fitted("data_optimize", water, k_data, until)
    cases[name] = ("data_optimize", water, k, xn)
    print(name, k, xn, flush=True)

params, curves = [], {}
for name, (sheet, water, k, xn) in cases.items():
    grid, c = curve(sheet, water, k, xn)
    params.append({"case": name, "sheet": sheet, "water": water, "k": k, "n_inv": xn})
    curves[(name, "time")] = grid
    curves[(name, "effluent")] = c
pd.DataFrame(params).set_index("case").to_csv(out / "tce_params.csv")
pd.DataFrame(curves).to_csv(out / "tce_curves.csv", index=False)
print("done")
