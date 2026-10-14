"""Run the upstream USEPA ``PSDM.__calculate_capacity`` on synthetic breakthrough data.

usage: python generate_capacity.py <upstream PSDM dir> <output dir>

Writes ``capacity_cases.csv`` (inputs) and ``capacity.csv`` (upstream K, q, breakthrough).
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
raw, cinfo, comps, carbons = P.process_input_file(fn, data_sheet="data", column_sheet="columnSpecs")
column = cinfo[carbons[0]]
comp = comps[0]


def logistic(t, c0, a, b):
    return c0 / (1 + np.exp(a - b * t))


t = np.arange(0.0, 101.0, 5.0)
rng = np.random.default_rng(1)
cases = {
    # full breakthrough within the data
    "breakthrough": (np.full_like(t, 50.0), logistic(t, 52.0, 6.0, 0.1)),
    # variable influent, effluent passes the mean influent only
    "variable": (50.0 + 10.0 * np.sin(t / 7.0), logistic(t, 48.0, 5.0, 0.09)),
    # partial breakthrough, extrapolated with the logistic curve
    "logistic": (np.full_like(t, 50.0), logistic(t, 50.0, 8.0, 0.07)),
    "logistic_noisy": (
        np.full_like(t, 50.0),
        np.maximum(logistic(t, 50.0, 9.0, 0.06) + rng.normal(0, 0.5, t.size), 0.0),
    ),
    # early, flat leakage: the logistic fit is rejected and a line is extrapolated
    "leakage": (np.full_like(t, 50.0), np.where(t > 0, 10.0 + 0.05 * t, 0.0)),
    # no breakthrough at all
    "minimum": (np.full_like(t, 50.0), np.zeros_like(t)),
}

rows, inputs = {}, []
for name, (c_in, c_out) in cases.items():
    inputs.append(pd.DataFrame({"case": name, "time": t, "influent": c_in, "effluent": c_out}))
    cols = pd.MultiIndex.from_tuples([(column["influentID"], comp), (carbons[0], comp)])
    data = pd.DataFrame(np.column_stack([c_in, c_out]), index=t, columns=cols)
    for xn in (0.45, 0.7):
        with contextlib.redirect_stdout(io.StringIO()):
            col = P.PSDM(column, chem, data, xn=xn, nr=4, nz=4)
        kd = col.k_data[comp]
        rows[(name, xn)] = {"K": kd["K"], "q": kd["q"], "brk": kd["brk"], "AveC": kd["AveC"]}
        print(name, xn, rows[(name, xn)])

pd.concat(inputs).to_csv(out / "capacity_cases.csv", index=False)
res = pd.DataFrame(rows).T
res.index.names = ["case", "n_inv"]
res.to_csv(out / "capacity.csv")
