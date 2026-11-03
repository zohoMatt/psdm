"""Run the upstream USEPA PSDM code on the ``PATHS`` cases of scripts/benchmark.py and save
reference breakthrough curves for tests/test_paths.py.

usage: python generate_paths.py <upstream PSDM dir> <output dir>

e.g. from the repository root (the dev environment has matplotlib, which upstream imports)::

    uv run python tests/data/reference/generate_paths.py <PSDM dir> tests/data/reference
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark import PATHS, SINGLE_T_END, load_upstream, run_path_upstream  # noqa: E402

P = load_upstream(sys.argv[1])
out = Path(sys.argv[2]).resolve()
rows = {}
for name in PATHS:
    _, rows[name] = run_path_upstream(P, name)  # mg/L on 349 points over 0-174 d
    print(name)
days = pd.Index(np.linspace(0, SINGLE_T_END, 349), name="days")
pd.DataFrame(rows, index=days).to_csv(out / "paths.csv", float_format="%.9g")
