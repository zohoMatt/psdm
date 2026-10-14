"""Run the upstream USEPA PSDM code and save reference breakthrough curves.

usage: python make_golden.py <upstream PSDM dir> <output dir>
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
up = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(up))
import psdm.PSDM as P  # noqa: E402

# ---------------------------------------------------------------- AdDesignS single-component cases
inp = pd.read_excel(up / "test.xlsx", sheet_name="data", index_col=[0])
column_prop = inp.T[["epor", "diam", "L", "wt", "rhop", "rhof", "flrt", "rad", "influentID"]].T
column_prop.loc["tortu"] = 1
column_prop.loc["psdfr"] = 5
chem = inp.T[["MW", "MolarVol"]].T
kd = inp.T[["K", "kf", "dp", "1/n"]].T
kd.loc["q"] = 1
kd.loc["brk"] = 100
kd.columns = pd.MultiIndex.from_tuples([("Test", "Test")])
print(column_prop, chem, kd, sep="\n")

idx = pd.MultiIndex.from_tuples([("concentration", "Test", "Test")])
t = np.arange(175)
rows = {}
for name, c0, water, chem_type in [
    ("rhine50", 50.0, "Rhine", "chlorinated hydrocarbon"),
    ("rhine5", 5.0, "Rhine", "chlorinated hydrocarbon"),
    ("nofoul5", 5.0, "Organic Free", "halogenated alkenes"),
    ("nofoul50", 50.0, "Organic Free", "halogenated alkenes"),
    ("portage50", 50.0, "Portage", "chlorinated hydrocarbon"),
]:
    infl = pd.DataFrame(c0 * 1e6, index=t, columns=idx)["concentration"]
    t0 = time.time()
    col = P.PSDM(
        column_prop["Test"],
        chem,
        infl,
        conc_type="ng",
        chem_type=chem_type,
        water_type=water,
        k_data=kd["Test"],
        test_range=kd.loc["K"].values,
        xn_range=kd.loc["1/n"].values,
        optimize=False,
    )
    _, K, xn, ssq, md = col.run_psdm_kfit("Test")
    print(name, f"{time.time() - t0:.1f}s", md.shape)
    grid = np.linspace(0, 174, 349)
    f = np.interp(grid, md.index.values, md.values[:, 0])
    rows[name] = f
pd.DataFrame(rows, index=pd.Index(np.linspace(0, 174, 349), name="days")).to_csv(
    out / "addesigns_single.csv"
)

# ---------------------------------------------------------------- multi-component example
fn = up / "Example_Multi.xlsx"
chem = P.process_input_data(fn, sheet_name="Properties")
kd = pd.read_excel(fn, sheet_name="Kdata", index_col=0)
raw, cinfo, comps, carbons = P.process_input_file(fn, data_sheet="data", column_sheet="columnSpecs")
print(cinfo, chem, kd, sep="\n")
for water, chem_type in [("Organic Free", "halogenated alkenes"), ("Rhine", "halogenated alkanes")]:
    t0 = time.time()
    col = P.PSDM(
        cinfo[carbons[0]],
        chem,
        raw,
        nz=8,
        nr=4,
        ne=2,
        chem_type=chem_type,
        water_type=water,
        k_data=kd,
        solver="BDF",
    )
    res = col.run_psdm()
    grid = np.linspace(0, float(raw.index.max()), 301)
    df = pd.DataFrame({c: res[c](grid) for c in res}, index=pd.Index(grid, name="days"))
    df.to_csv(out / f"multi_{water.replace(' ', '_').lower()}.csv")
    print(water, f"{time.time() - t0:.1f}s")
print("done")
