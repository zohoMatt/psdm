"""Run upstream USEPA ``model_uncertainty`` and ``run_psdm_dsfit`` on the AdDesignS column.

usage: python generate_uncertainty_dsfit.py <upstream PSDM dir> <output dir>

``dsfit_effluent.csv`` (synthetic TCE effluent made with ``psdm.simulate`` and
``ds = 0.3 ds_correlation``) must already be in the output dir. Writes ``uncertainty.csv``
and ``dsfit.csv``.
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

inp = pd.read_excel(up / "test.xlsx", sheet_name="data", index_col=[0])
column_prop = inp.T[["epor", "diam", "L", "wt", "rhop", "rhof", "flrt", "rad", "influentID"]].T
column_prop.loc["tortu"] = 1
column_prop.loc["psdfr"] = 5
column_prop = column_prop["Test"]
column_prop.name = "GAC"
chem = inp.T[["MW", "MolarVol"]].T
chem.columns = ["Test"]
kd = inp.T[["K", "1/n"]].T
kd.columns = ["Test"]
kd.loc["q"] = 1
kd.loc["brk"] = 174
kd.loc["AveC"] = 50e6
kd.loc["MW"] = chem.loc["MW", "Test"]
kd.loc["MolarVol"] = chem.loc["MolarVol", "Test"]
kd = kd.astype(float)
t = np.arange(175.0)


def make(c0, effluent=None):
    cols = pd.MultiIndex.from_tuples([(column_prop["influentID"], "Test"), ("GAC", "Test")])
    eff = np.zeros_like(t) if effluent is None else effluent
    data = pd.DataFrame(np.column_stack([np.full_like(t, c0), eff]), index=t, columns=cols)
    with contextlib.redirect_stdout(io.StringIO()):
        return P.PSDM(
            column_prop,
            chem,
            data,
            conc_type="ng",
            k_data=kd.copy(),
            chem_type="halogenated alkenes",
            water_type="Organic Free",
            nr=8,
            nz=10,
            optimize=False,
        )


# ---------------------------------------------------------------- uncertainty
col = make(50e6)
col.model_uncertainty(capacity=10, mass=10, flrt=10, kf=20, dp=20, k=10)
grid = np.linspace(0, 174, 175)
res = (
    pd.DataFrame(
        {
            "base": np.interp(grid, col.results.index, col.results["Test"].astype(float)),
            "lower": np.interp(
                grid, col.results.index, col.uncertainty_results["lower"]["Test"].astype(float)
            ),
            "upper": np.interp(
                grid, col.results.index, col.uncertainty_results["upper"]["Test"].astype(float)
            ),
        },
        index=pd.Index(grid, name="days"),
    )
    / 1e6
)  # mg/L
res.to_csv(out / "uncertainty.csv")
print(res.iloc[::20])

# ---------------------------------------------------------------- ds fit
eff = pd.read_csv(out / "dsfit_effluent.csv")
col = make(5e6, np.interp(t, eff["time"], eff["effluent"]) * 1e6)
_, best_ds, ssqs, _, ds_base = col.run_psdm_dsfit("Test")
pd.DataFrame({"ds_base": [ds_base], "best_ds": [best_ds]}).to_csv(out / "dsfit.csv", index=False)
print(ds_base, best_ds, best_ds / ds_base)
