"""Run upstream USEPA ``process_input_file`` on the example workbooks and a crafted one.

usage: python generate_io.py <upstream PSDM dir> <output dir> <psdm examples dir>

Writes ``io_missing_t0.xlsx`` (the crafted workbook: no t = 0 rows and zero influent values,
column specs copied from ``Example_TCE.xlsx``) into ``<output dir>/..`` and
``io_process_input_file.csv``: the influent and effluent of every (file, sheet) case in long
form ``file, sheet, time, type, compound, concentration``.

Run it with ``pandas<3``: upstream sets zero influent to 0.001 by chained assignment
(``rawdata_df[influent][rawdata_df[influent]==0.] = 0.001``), which pandas 3 (copy on write)
silently ignores.
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.simplefilter("ignore")
up = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
examples = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(up))
import psdm.PSDM as P  # noqa: E402

crafted = out.parent / "io_missing_t0.xlsx"
specs = pd.read_excel(examples / "Example_TCE.xlsx", sheet_name="columnSpecs")
props = pd.read_excel(examples / "Example_TCE.xlsx", sheet_name="Properties")
props["Tetrachloroethylene"] = props["Trichloroethylene"]
rows = {
    # neither influent nor effluent at t = 0; zero influent values later on
    "data": [
        ("influent", 1.0, 10.0, "Trichloroethylene"),
        ("influent", 3.0, 0.0, "Trichloroethylene"),
        ("influent", 5.0, 12.0, "Trichloroethylene"),
        ("influent", 8.0, 0.0, "Trichloroethylene"),
        ("effluent", 2.0, 1.0, "Trichloroethylene"),
        ("effluent", 4.0, 0.0, "Trichloroethylene"),
        ("effluent", 6.0, 3.0, "Trichloroethylene"),
        ("effluent", 8.0, 4.0, "Trichloroethylene"),
        ("influent", 1.0, 20.0, "Tetrachloroethylene"),
        ("influent", 3.0, 22.0, "Tetrachloroethylene"),
        ("influent", 5.0, 0.0, "Tetrachloroethylene"),
        ("influent", 8.0, 30.0, "Tetrachloroethylene"),
        ("effluent", 2.0, 0.0, "Tetrachloroethylene"),
        ("effluent", 4.0, 2.0, "Tetrachloroethylene"),
        ("effluent", 6.0, 5.0, "Tetrachloroethylene"),
        ("effluent", 8.0, 6.0, "Tetrachloroethylene"),
    ],
    # influent rows for t = 0 present (zero), effluent starts later
    "data_infl_t0": [
        ("influent", 0.0, 0.0, "Trichloroethylene"),
        ("influent", 2.0, 50.0, "Trichloroethylene"),
        ("influent", 9.0, 40.0, "Trichloroethylene"),
        ("effluent", 3.0, 1.0, "Trichloroethylene"),
        ("effluent", 6.0, 2.0, "Trichloroethylene"),
        ("effluent", 9.0, 7.0, "Trichloroethylene"),
    ],
}
with pd.ExcelWriter(crafted) as w:
    specs.to_excel(w, sheet_name="columnSpecs", index=False)
    props.to_excel(w, sheet_name="Properties", index=False)
    for sheet, r in rows.items():
        frame = pd.DataFrame(r, columns=["type", "time", "concentration", "compound"])
        frame.to_excel(w, sheet_name=sheet, index=False)

cases = [
    (examples / "Example_TCE.xlsx", "data"),
    (examples / "Example_TCE.xlsx", "data_variable"),
    (examples / "Example_TCE.xlsx", "data_optimize"),
    (examples / "Example_Multi.xlsx", "data"),
    (crafted, "data"),
    (crafted, "data_infl_t0"),
]
long = []
for fn, sheet in cases:
    with contextlib.redirect_stdout(io.StringIO()):
        raw, cinfo, comps, carbons = P.process_input_file(
            fn, data_sheet=sheet, column_sheet="columnSpecs"
        )
    column = cinfo[carbons[0]]
    for kind, key in [("influent", column["influentID"]), ("effluent", carbons[0])]:
        frame = raw[key]
        for comp in frame.columns:
            for t, c in frame[comp].items():
                long.append((fn.name, sheet, float(t), kind, comp, float(c)))
    print(fn.name, sheet)
    print(raw)

pd.DataFrame(long, columns=["file", "sheet", "time", "type", "compound", "concentration"]).to_csv(
    out / "io_process_input_file.csv", index=False
)
