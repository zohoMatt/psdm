"""Benchmark psdm against the original US EPA PSDM code and write an HTML report.

usage (from the repository root)::

    uv run python scripts/benchmark.py [--upstream PATH] [--repeats 7]
    uv run python scripts/benchmark.py --from-json benchmarks/results.json   # re-render only

``--upstream`` is the ``PSDM`` folder of a checkout of
https://github.com/USEPA/Water_Treatment_Models at commit 3f514c2. Without it the
repository is cloned into a temporary directory (requires ``git``).

The upstream code is run twice for the single-compound cases: with inputs built as in upstream's
``test_PSDM.py`` (object-dtype frames, which make upstream's right-hand side run Python-object
matmuls) and, as a control group, with the same inputs converted to float64.

Every timed run happens in a fresh worker process (``python -I``) started with the same
interpreter, so both implementations use identical numpy/scipy/pandas builds. Inside each
worker the case is run twice: the first ("cold") call includes one-off costs such as lazy
imports inside scipy/pandas, the second ("warm") call is the steady state. Package import
time is measured but not included in either.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
EXAMPLES = ROOT / "examples"
UPSTREAM_REPO = "https://github.com/USEPA/Water_Treatment_Models.git"
UPSTREAM_COMMIT = "3f514c2"

# name: (kind, label, details)
CASES = {
    "rhine5": (
        "single",
        "Rhine, 5 mg/L",
        dict(c0=5.0, water="Rhine", cls="chlorinated hydrocarbon", sheet="5"),
    ),
    "rhine25": (
        "single",
        "Rhine, 25 mg/L",
        dict(c0=25.0, water="Rhine", cls="chlorinated hydrocarbon", sheet="25"),
    ),
    "rhine50": (
        "single",
        "Rhine, 50 mg/L",
        dict(c0=50.0, water="Rhine", cls="chlorinated hydrocarbon", sheet="50"),
    ),
    "nofoul5": (
        "single",
        "Organic Free, 5 mg/L",
        dict(c0=5.0, water="Organic Free", cls="halogenated alkenes", sheet="5no"),
    ),
    "nofoul25": (
        "single",
        "Organic Free, 25 mg/L",
        dict(c0=25.0, water="Organic Free", cls="halogenated alkenes", sheet="25no"),
    ),
    "nofoul50": (
        "single",
        "Organic Free, 50 mg/L",
        dict(c0=50.0, water="Organic Free", cls="halogenated alkenes", sheet="50no"),
    ),
    "portage50": (
        "single",
        "Portage, 50 mg/L",
        dict(c0=50.0, water="Portage", cls="chlorinated hydrocarbon", sheet="50K"),
    ),
    "multi_of": (
        "multi",
        "Multi-component, Organic Free",
        dict(water="Organic Free", cls="halogenated alkenes"),
    ),
    "multi_rhine": (
        "multi",
        "Multi-component, Rhine",
        dict(water="Rhine", cls="halogenated alkanes"),
    ),
    "tce_var_of": (
        "variable",
        "Variable influent, Organic Free",
        dict(water="Organic Free", cls="halogenated alkenes"),
    ),
    "tce_var_rhine": (
        "variable",
        "Variable influent, Rhine",
        dict(water="Rhine", cls="halogenated alkanes"),
    ),
}
SINGLE_T_END = 174.0

# Code paths not reached by CASES. Accuracy only (one run each, not timed). Every case is the
# AdDesignS column with a constant TCE-like influent for 174 d; the keys below override the
# defaults in path_spec(). tests/test_paths.py runs the same cases against reference curves
# written by tests/data/reference/generate_paths.py, which calls run_path_upstream().
PATHS = {
    "tortu2": ("Tortuosity 2, no fouling", dict(tortu=2.0)),
    "mt_of": ("Manual kf/dp/ds, no fouling", dict(kf=2e-3, dp=4e-6, ds=1e-9)),
    "mt_rhine": (
        "Manual kf/dp/ds, Rhine",
        dict(water="Rhine", cls="chlorinated hydrocarbon", kf=5e-3, dp=1.2e-5, ds=1e-9),
    ),
    "pfas": ("PFAS fouling, PFOA in Rhine", dict(water="Rhine", cls="PFAS", name="PFOA")),
    "pfas_avg": (
        "PFAS fouling, unlisted PFAS (average b1, b2)",
        dict(water="Rhine", cls="PFAS", name="PFXX"),
    ),
    "gnielinski": ("Gnielinski film-transfer correlation", dict(corr="Gnielinski")),
    "wakao": (
        "Wakao and Funazkri correlation, Rhine",
        dict(corr="Wakao and Funazkri", water="Rhine", cls="chlorinated hydrocarbon"),
    ),
    "temp10": ("Water at 10 °C", dict(temp=10.0)),
    "temp30_rhine": (
        "Water at 30 °C, Rhine",
        dict(temp=30.0, water="Rhine", cls="chlorinated hydrocarbon"),
    ),
    "ne3": ("Single compound, ne = 3 (nr 8, nz 5)", dict(nr=8, nz=5, ne=3)),
    "hours": ("Time in hours", dict(time_unit="hours")),
}
PATH_TOL = 5e-3  # max |psdm - upstream| / C0

# K and 1/n fits on examples/Example_TCE.xlsx, sheet data_optimize (ug/L, days).
FITS = {
    "fit_full": ("data_optimize, all data", dict(until=None)),
    "fit_55": ("data_optimize, t ≤ 55 d only", dict(until=55.0)),
}
FIT_KW = dict(nr=8, nz=12)  # as in upstream Example_TCE.py


def path_spec(name):
    d = dict(
        c0=5.0,
        water="Organic Free",
        cls="halogenated alkenes",
        name="TCE",
        tortu=1.0,
        temp=20.0,
        nr=14,
        nz=19,
        ne=1,
        corr="Chern and Chien",
        time_unit="days",
        kf=None,
        dp=None,
        ds=None,
    )
    d.update(PATHS[name][1])
    return d


# ============================================================================ workers
class SolveLog:
    """Wraps ``solve_ivp`` to time the integration and record solver statistics."""

    def __init__(self, func):
        self.func = func
        self.calls = []

    def __call__(self, *args, **kw):
        t0 = time.perf_counter()
        sol = self.func(*args, **kw)
        self.calls.append(
            dict(
                seconds=time.perf_counter() - t0,
                nsteps=len(sol.t) - 1,
                nfev=int(sol.nfev),
                njev=int(sol.njev),
                nlu=int(sol.nlu),
            )
        )
        return sol

    def take(self):
        calls, self.calls = self.calls, []
        out = {k: sum(c[k] for c in calls) for k in ("seconds", "nsteps", "nfev", "njev", "nlu")}
        out["integrate_s"] = out.pop("seconds")
        out["solver_calls"] = len(calls)
        return out


def _timed(func):
    t0 = time.perf_counter()
    out = func()
    return out, time.perf_counter() - t0


def load_upstream(upstream):
    """Import upstream ``psdm/PSDM.py``. Its ``psdm/`` folder has no ``__init__.py`` and would
    be shadowed by the installed psdm package, so it is loaded as the package ``epa_psdm``."""
    import types

    pkg = types.ModuleType("epa_psdm")
    pkg.__path__ = [str(Path(upstream).resolve() / "psdm")]
    sys.modules["epa_psdm"] = pkg
    import epa_psdm.PSDM as P

    return P


def worker_upstream(case, upstream, float_inputs=False):
    import numpy as np
    import pandas as pd

    t0 = time.perf_counter()
    P = load_upstream(upstream)
    import_s = time.perf_counter() - t0
    log = SolveLog(P.solve_ivp)
    P.solve_ivp = log
    kind, _, d = CASES[case]

    if kind == "single":

        def read():
            return pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name="data", index_col=[0])

        def build(inp):
            # identical to tests/data/reference/generate.py, which follows upstream test_PSDM.py:
            # the `data` sheet mixes numbers with the string influentID, so the transposed frames
            # are dtype object. float_inputs=True is the control group with float64 inputs.
            column_prop = inp.T[
                ["epor", "diam", "L", "wt", "rhop", "rhof", "flrt", "rad", "influentID"]
            ].T
            column_prop.loc["tortu"] = 1
            column_prop.loc["psdfr"] = 5
            chem = inp.T[["MW", "MolarVol"]].T
            kd = inp.T[["K", "kf", "dp", "1/n"]].T
            kd.loc["q"] = 1
            kd.loc["brk"] = 100
            kd.columns = pd.MultiIndex.from_tuples([("Test", "Test")])
            if float_inputs:
                chem, kd = chem.astype(float), kd.astype(float)
                column_prop = column_prop.map(lambda v: v if isinstance(v, str) else float(v))
            idx = pd.MultiIndex.from_tuples([("concentration", "Test", "Test")])
            infl = pd.DataFrame(d["c0"] * 1e6, index=np.arange(175), columns=idx)["concentration"]
            return P.PSDM(
                column_prop["Test"],
                chem,
                infl,
                conc_type="ng",
                chem_type=d["cls"],
                water_type=d["water"],
                k_data=kd["Test"],
                test_range=kd.loc["K"].values,
                xn_range=kd.loc["1/n"].values,
                optimize=False,
            )

        def run(col):
            md = col.run_psdm_kfit("Test")[-1]
            grid = np.linspace(0, SINGLE_T_END, 349)
            return grid, {"TCE": np.interp(grid, md.index.values, md.values[:, 0]) / 1e6}

        def c0(_):
            return {"TCE": d["c0"]}

    else:
        fn = EXAMPLES / ("Example_Multi.xlsx" if kind == "multi" else "Example_TCE.xlsx")
        sheet = "data" if kind == "multi" else "data_variable"

        def read():
            chem = P.process_input_data(fn, sheet_name="Properties")
            kd = pd.read_excel(fn, sheet_name="Kdata", index_col=0)
            raw, cinfo, comps, carbons = P.process_input_file(
                fn, data_sheet=sheet, column_sheet="columnSpecs"
            )
            return chem, kd, raw, cinfo, comps, carbons

        def build(inp):
            chem, kd, raw, cinfo, _comps, carbons = inp
            kw = dict(nz=8, nr=4, ne=2) if kind == "multi" else dict(nr=8, nz=12, optimize=False)
            col = P.PSDM(
                cinfo[carbons[0]],
                chem,
                raw,
                chem_type=d["cls"],
                water_type=d["water"],
                k_data=kd,
                **kw,
            )
            col.benchmark_inputs = inp
            return col

        def run(col):
            _chem, kd, raw, _cinfo, comps, _carbons = col.benchmark_inputs
            grid = np.linspace(0, float(raw.index.max()), 349)
            if kind == "multi":
                res = col.run_psdm()
                return grid, {c: np.asarray(res[c](grid)) for c in res}
            (comp,) = comps
            col.test_range = np.array([kd[comp]["K"]])
            col.xn_range = np.array([kd[comp]["1/n"]])
            md = col.run_psdm_kfit(comp)[-1]
            return grid, {comp: np.interp(grid, md.index.values, md.values[:, 0])}

        def c0(inp):
            raw, cinfo, comps, carbons = inp[2:]
            infl = raw[cinfo[carbons[0]]["influentID"]]
            return {c: float(infl[c].max()) for c in comps}

    return _run_iterations(import_s, read, build, run, c0, log)


def worker_psdm(case):
    import numpy as np
    import pandas as pd

    t0 = time.perf_counter()
    import psdm
    import psdm.solver
    from psdm.io import read_excel

    import_s = time.perf_counter() - t0
    log = SolveLog(psdm.solver.solve_ivp)
    psdm.solver.solve_ivp = log
    kind, _, d = CASES[case]

    if kind == "single":

        def read():
            return pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name="data", index_col=0)[
                "Test"
            ]

        def build(p):
            column = psdm.Column(
                length=float(p["L"]),
                diameter=float(p["diam"]),
                mass=float(p["wt"]),
                flow_rate=float(p["flrt"]),
                particle_radius=float(p["rad"]),
                particle_density=float(p["rhop"]),
                particle_porosity=float(p["epor"]),
                bed_density=float(p["rhof"]),
            )
            tce = psdm.Compound(
                "TCE",
                mw=float(p["MW"]),
                molar_volume=float(p["MolarVol"]),
                k=float(p["K"]),
                n_inv=float(p["1/n"]),
                fouling_class=d["cls"],
            )
            influent = pd.DataFrame({"TCE": [d["c0"], d["c0"]]}, index=[0.0, SINGLE_T_END])
            return column, tce, influent

        def run(objs):
            column, tce, influent = objs
            res = psdm.simulate(column, tce, influent, water=d["water"], conc_unit="mg/L")
            grid = np.linspace(0, SINGLE_T_END, 349)
            return grid, {"TCE": res(grid, "TCE")}

        def c0(_):
            return {"TCE": d["c0"]}

    else:
        fn = EXAMPLES / ("Example_Multi.xlsx" if kind == "multi" else "Example_TCE.xlsx")
        sheet = "data" if kind == "multi" else "data_variable"

        def read():
            return read_excel(fn, data_sheet=sheet)

        def build(ds):
            return ds, ds.compounds(fouling_class=d["cls"])

        def run(objs):
            ds, comps = objs
            kw = dict(nr=4, nz=8, ne=2) if kind == "multi" else dict(nr=8, nz=12)
            res = ds.simulate(comps, water=d["water"], **kw)
            grid = np.linspace(0, float(ds.influent.index.max()), 349)
            return grid, {c: res(grid, c) for c in res.effluent.columns}

        def c0(ds):
            return {c: float(ds.influent[c].max()) for c in ds.compound_names}

    return _run_iterations(import_s, read, build, run, c0, log)


def _path_grid(d):
    """349 output times over 0-174 d, in the case's time unit."""
    import numpy as np

    return np.linspace(0, SINGLE_T_END, 349) * (24.0 if d["time_unit"] == "hours" else 1.0)


def run_path_upstream(P, name):
    """Run one PATHS case with upstream ``run_psdm_kfit``; returns (grid, effluent mg/L)."""
    import numpy as np
    import pandas as pd

    d = path_spec(name)
    comp = d["name"]
    inp = pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name="data", index_col=[0])["Test"]
    num = pd.to_numeric(inp, errors="coerce")
    column_prop = num[["epor", "diam", "L", "wt", "rhop", "rhof", "flrt", "rad"]].astype(object)
    column_prop["influentID"] = "Test"
    column_prop["tortu"] = d["tortu"]
    column_prop["psdfr"] = 5.0
    column_prop.name = "Test"
    chem = pd.DataFrame({comp: num[["MW", "MolarVol"]]}).astype(float)
    kd = pd.DataFrame({comp: [num["K"], num["1/n"], 1.0, 100.0]}, index=["K", "1/n", "q", "brk"])
    scale = 24.0 if d["time_unit"] == "hours" else 1.0
    idx = pd.MultiIndex.from_tuples([("Test", comp)])
    infl = pd.DataFrame(d["c0"] * 1e6, index=np.arange(175) * scale, columns=idx)

    def build(**extra):
        return P.PSDM(
            column_prop,
            chem,
            infl,
            conc_type="ng",
            chem_type=d["cls"],
            water_type=d["water"],
            k_data=kd,
            test_range=kd.loc["K"].values,
            xn_range=kd.loc["1/n"].values,
            optimize=False,
            nr=d["nr"],
            nz=d["nz"],
            ne=d["ne"],
            temp=d["temp"],
            time_type=d["time_unit"],
            **extra,
        )

    mt = {k: d[k] for k in ("kf", "dp", "ds") if d[k] is not None}
    if d["corr"] != "Chern and Chien":
        # upstream run_psdm hard-codes the Chern and Chien correlation; the others are only
        # reachable through kf_calc, so evaluate it as run_psdm would and pass kf explicitly
        col = build()
        difl = 13.26e-5 / (((col.vw * 100.0) ** 1.14) * (chem[comp]["MolarVol"] ** 0.589))
        sc = col.vw / (col.dw * difl)
        kf = P.kf_calc(difl / (2 * col.rad), col.re, sc, col.ebed, corr=d["corr"])
        mt["kf"] = float(np.squeeze(kf))
    col = build(mass_transfer={comp: mt}) if mt else build()
    md = col.run_psdm_kfit(comp)[-1]
    grid = _path_grid(d)
    return grid, np.interp(grid, md.index.values, md.values[:, 0]) / 1e6


def run_path_psdm(name):
    import pandas as pd

    import psdm

    d = path_spec(name)
    p = pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name="data", index_col=0)["Test"]
    column = psdm.Column(
        length=float(p["L"]),
        diameter=float(p["diam"]),
        mass=float(p["wt"]),
        flow_rate=float(p["flrt"]),
        particle_radius=float(p["rad"]),
        particle_density=float(p["rhop"]),
        particle_porosity=float(p["epor"]),
        bed_density=float(p["rhof"]),
        tortuosity=d["tortu"],
    )
    compound = psdm.Compound(
        d["name"],
        mw=float(p["MW"]),
        molar_volume=float(p["MolarVol"]),
        k=float(p["K"]),
        n_inv=float(p["1/n"]),
        kf=d["kf"],
        dp=d["dp"],
        ds=d["ds"],
        fouling_class=d["cls"],
    )
    grid = _path_grid(d)
    influent = pd.DataFrame({d["name"]: [d["c0"], d["c0"]]}, index=[0.0, grid[-1]])
    res = psdm.simulate(
        column,
        compound,
        influent,
        water=d["water"],
        temperature=d["temp"],
        time_unit=d["time_unit"],
        conc_unit="mg/L",
        nr=d["nr"],
        nz=d["nz"],
        ne=d["ne"],
        correlation=d["corr"],
    )
    return grid, res(grid, d["name"]), res.mass_transfer[d["name"]].to_dict()


def worker_path(impl, name, upstream):
    t0 = time.perf_counter()
    if impl == "upstream":
        grid, y = run_path_upstream(load_upstream(upstream), name)
        extra = {}
    else:
        grid, y, mt = run_path_psdm(name)
        extra = {"mass_transfer": mt}
    d = path_spec(name)
    days = grid / (24.0 if d["time_unit"] == "hours" else 1.0)
    return dict(
        grid=list(map(float, days)),
        curve=list(map(float, y)),
        c0=d["c0"],
        seconds=time.perf_counter() - t0,
        **extra,
    )


def fit_data():
    """Observed influent and effluent of data_optimize as written in the sheet (ug/L, days)."""
    import pandas as pd

    raw = pd.read_excel(EXAMPLES / "Example_TCE.xlsx", sheet_name="data_optimize")
    out = {}
    for kind, g in raw.groupby("type"):
        out[kind] = g.set_index("time")["concentration"].astype(float).sort_index()
    return out["influent"], out["effluent"]


def worker_fit(impl, name, upstream):
    """Fit K and 1/n on data_optimize (optionally only t <= until) and report the result."""
    import numpy as np
    import pandas as pd

    until = FITS[name][1]["until"]
    grid = np.linspace(0, 175, 351)
    if impl == "upstream":
        P = load_upstream(upstream)
        fn = EXAMPLES / "Example_TCE.xlsx"
        chem = P.process_input_data(fn, sheet_name="Properties")
        raw, cinfo, (comp,), carbons = P.process_input_file(
            fn, data_sheet="data_optimize", column_sheet="columnSpecs"
        )
        window = raw if until is None else raw[raw.index <= until]
        calls = [0]
        run_psdm = P.PSDM.run_psdm

        def counted(self):
            calls[0] += 1
            return run_psdm(self)

        P.PSDM.run_psdm = counted
        t0 = time.perf_counter()
        # no k_data: the start point comes from upstream's __calculate_capacity, as in normal use
        col = P.PSDM(cinfo[carbons[0]], chem, window, optimize=False, **FIT_KW)
        start = col.k_data[comp][["K", "1/n", "q", "brk"]].astype(float).to_dict()
        col.run_all_smart(save_file=False)
        seconds = time.perf_counter() - t0
        evaluations = calls[0]
        k, n_inv = (float(col.k_data[comp][key]) for key in ("K", "1/n"))
        own_ssq = float(np.nanmin(col.ssq_storage.values.astype(float)))
        # simulate the fitted parameters over the full data set with upstream itself
        kd = pd.DataFrame({comp: [k, n_inv, 1.0, 1e3, 1.0]}, index=["K", "1/n", "q", "brk", "AveC"])
        full = P.PSDM(cinfo[carbons[0]], chem, raw, optimize=False, k_data=kd, **FIT_KW)
        curve = full.run_psdm()[comp](grid)
        extra = dict(start=start, own_ssq=own_ssq, own_ssq_note="min of ssq_storage")
    else:
        import psdm
        from psdm.io import read_excel

        ds = read_excel(EXAMPLES / "Example_TCE.xlsx", data_sheet="data_optimize")
        (start,) = ds.compounds()
        comp = start.name
        infl, eff = ds.influent[comp], ds.effluent[comp]
        t0 = time.perf_counter()
        fit = psdm.fit_freundlich(ds.column, start, infl, eff, until=until, **FIT_KW)
        seconds = time.perf_counter() - t0
        evaluations = len(fit.evaluations) + 1
        k, n_inv = fit.compound.k, fit.compound.n_inv
        curve = fit.result(grid, comp)
        extra = dict(
            start={"K": start.k, "1/n": start.n_inv},
            own_ssq=fit.ssq,
            own_ssq_note="fit.ssq (gap-filled effluent)",
            path=fit.evaluations.to_dict(orient="list"),
        )
    return dict(
        k=k,
        n_inv=n_inv,
        evaluations=evaluations,
        seconds=seconds,
        grid=list(map(float, grid)),
        curve=list(map(float, curve)),
        **extra,
    )


def _run_iterations(import_s, read, build, run, c0, log, n=2):
    iters = []
    for i in range(n):
        inp, read_s = _timed(read)
        obj, build_s = _timed(lambda: build(inp))  # noqa: B023
        (grid, curves), run_s = _timed(lambda: run(obj))  # noqa: B023
        iters.append(dict(read_s=read_s, build_s=build_s, run_s=run_s, **log.take()))
        if i == 0:
            out = dict(
                grid=list(map(float, grid)),
                curves={k: list(map(float, v)) for k, v in curves.items()},
                c0=c0(inp),
            )
    return dict(import_s=import_s, iterations=iters, **out)


# ============================================================================ orchestration
def clone_upstream(dest: Path) -> Path:
    subprocess.run(["git", "clone", "-q", UPSTREAM_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", UPSTREAM_COMMIT], check=True)
    return dest / "PSDM"


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def environment(upstream: Path):
    def ver(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "?"

    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    psdm_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version() + " (" + platform.python_implementation() + ")",
        "numpy": ver("numpy"),
        "scipy": ver("scipy"),
        "pandas": ver("pandas"),
        "openpyxl": ver("openpyxl"),
        "psdm": ver("psdm") + (f" (git {psdm_commit})" if psdm_commit else ""),
        "upstream": f"USEPA/Water_Treatment_Models PSDM @ {commit or UPSTREAM_COMMIT}",
    }


def run_worker(impl, case, upstream, out_file):
    cmd = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        impl,
        "--case",
        case,
        "--upstream",
        str(upstream),
        "--worker-out",
        str(out_file),
    ]
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode:
        sys.exit(f"worker {impl}/{case} failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(Path(out_file).read_text())


def impls(case):
    """Implementations run for a case: the float64-input control group only applies where
    the upstream inputs are built as object-dtype frames (single-compound cases)."""
    return ["upstream", "psdm"] + (["upstream_f64"] if CASES[case][0] == "single" else [])


def benchmark(upstream: Path, repeats: int, cases: list[str]):
    raw = {c: {i: [] for i in impls(c)} for c in cases}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "w.json"
        for r in range(repeats):
            for case in cases:
                order = impls(case)
                k = (r + cases.index(case)) % len(order)
                order = order[k:] + order[:k]
                for impl in order:
                    res = run_worker(impl, case, upstream, out)
                    raw[case][impl].append(res)
                    it = res["iterations"]
                    print(
                        f"[{r + 1}/{repeats}] {case:14s} {impl:12s} cold {it[0]['build_s'] + it[0]['run_s']:6.3f}s"
                        f"  warm {it[1]['build_s'] + it[1]['run_s']:6.3f}s",
                        flush=True,
                    )
    return raw


def accuracy_runs(upstream: Path, paths: list[str], fits: list[str]):
    """One run per implementation of each PATHS and FITS case, each in a fresh process."""
    out = {"paths": {}, "fits": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "w.json"
        for group, names in (("paths", paths), ("fits", fits)):
            for name in names:
                out[group][name] = {}
                for impl in ("upstream", "psdm"):
                    res = run_worker(f"{group[:-1]}_{impl}", name, upstream, tmp_out)
                    out[group][name][impl] = res
                    print(
                        f"{group[:-1]:5s} {name:14s} {impl:9s} {res['seconds']:7.2f}s", flush=True
                    )
    return out


# ============================================================================ analysis
def fit_reference(fits_raw):
    """Score every fitted (K, 1/n) with one simulator (psdm) and one data set (the effluent
    rows of the sheet), and scan a grid for the best SSQ reachable in each window."""
    import numpy as np

    import psdm
    from psdm.io import read_excel

    ds = read_excel(EXAMPLES / "Example_TCE.xlsx", data_sheet="data_optimize")
    (base,) = ds.compounds()
    infl = ds.influent[[base.name]]
    _, eff = fit_data()

    def ssq(k, n_inv):
        try:
            res = psdm.simulate(ds.column, base.with_(k=k, n_inv=n_inv), infl, **FIT_KW)
        except (RuntimeError, ValueError):
            return {"window": {}, "full": float("nan")}
        y = res(eff.index.to_numpy(), base.name) - eff.to_numpy()
        win = {
            name: float(np.sum(y[eff.index <= (FITS[name][1]["until"] or np.inf)] ** 2))
            for name in FITS
        }
        return {"window": win, "full": float(np.sum(y**2))}

    c_ref = float(ds.influent[base.name].mean())
    q0 = base.k * c_ref**base.n_inv
    scan = []
    for qf in np.geomspace(0.1, 10, 25):
        for n_inv in np.round(np.arange(0.2, 1.0001, 0.1), 3):
            k = q0 * qf / c_ref**n_inv
            scan.append(dict(k=float(k), n_inv=float(n_inv), **ssq(k, n_inv)))
    out = {}
    for name, runs in fits_raw.items():
        entry = {"label": FITS[name][0], "until": FITS[name][1]["until"]}
        for impl, r in runs.items():
            s = ssq(r["k"], r["n_inv"])
            yu = np.interp(eff.index.to_numpy(), r["grid"], r["curve"]) - eff.to_numpy()
            within = eff.index <= (entry["until"] or np.inf)
            entry[impl] = {
                **{k: v for k, v in r.items() if k not in ("grid", "curve", "path")},
                "ssq_window": s["window"][name],
                "ssq_full": s["full"],
                "ssq_window_own_sim": float(np.sum(yu[within] ** 2)),
                "ssq_full_own_sim": float(np.sum(yu**2)),
            }
            entry.setdefault("curves", {"grid": r["grid"]})[impl] = r["curve"]
        start_ssq = ssq(base.k, base.n_inv)
        entry["start_psdm"] = {
            "k": base.k,
            "n_inv": base.n_inv,
            "ssq_window": start_ssq["window"][name],
        }
        best = min(scan, key=lambda row: row["window"][name])
        entry["grid_best"] = {
            "k": best["k"],
            "n_inv": best["n_inv"],
            "ssq_window": best["window"][name],
            "ssq_full": best["full"],
        }
        entry["observed"] = {"t": eff.index.tolist(), "c": eff.tolist()}
        out[name] = entry
    return out


def summarise_paths(paths_raw):
    import numpy as np

    out = {}
    for name, runs in paths_raw.items():
        up, ps = runs["upstream"], runs["psdm"]
        grid = np.asarray(up["grid"])
        yu = np.asarray(up["curve"]) / up["c0"]
        yp = np.interp(grid, ps["grid"], ps["curve"]) / up["c0"]
        diff = yp - yu
        out[name] = {
            "label": PATHS[name][0],
            "spec": {k: v for k, v in path_spec(name).items()},
            "max_abs_rel": float(np.abs(diff).max()),
            "rms_rel": float(np.sqrt(np.mean(diff**2))),
            "pass": bool(np.abs(diff).max() < PATH_TOL),
            "seconds": {"upstream": up["seconds"], "psdm": ps["seconds"]},
            "mass_transfer": ps["mass_transfer"],
            "grid": up["grid"],
            "upstream": yu.tolist(),
            "psdm": yp.tolist(),
        }
    return out


def addesigns(sheet):
    import pandas as pd

    out = pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name=sheet, index_col=0)["C, mg/L"]
    out = out[
        out > 0
    ]  # sheet pads with negative sentinel rows; same filter as tests/test_solver.py
    return out.index.to_numpy() / 1440.0, out.to_numpy()


def summarise(raw):
    import numpy as np

    summary = {}
    for case, runs in raw.items():
        kind, label, d = CASES[case]
        s = {"kind": kind, "label": label, "timing": {}, "accuracy": {}, "solver": {}}
        for impl, rs in runs.items():
            t = {}
            for phase, which in (("cold", 0), ("warm", 1)):
                its = [r["iterations"][which] for r in rs]
                for key in ("read_s", "build_s", "run_s", "integrate_s"):
                    t[f"{phase}_{key}"] = [i[key] for i in its]
                t[f"{phase}_total_s"] = [i["build_s"] + i["run_s"] for i in its]
            t["import_s"] = [r["import_s"] for r in rs]
            s["timing"][impl] = t
            s["solver"][impl] = {
                k: rs[0]["iterations"][0][k]
                for k in ("nsteps", "nfev", "njev", "nlu", "solver_calls")
            }
            stable = all(
                r["iterations"][i]["nsteps"] == s["solver"][impl]["nsteps"]
                for r in rs
                for i in (0, 1)
            )
            s["solver"][impl]["identical_across_runs"] = stable

        up, ps = runs["upstream"][0], runs["psdm"][0]
        grid = np.asarray(up["grid"])
        curves = {"grid": up["grid"], "c0": up["c0"], "compounds": {}}
        worst = {"max": 0.0, "rms": 0.0}
        for comp, c0 in up["c0"].items():
            yu = np.asarray(up["curves"][comp])
            yp = np.interp(grid, ps["grid"], ps["curves"][comp])
            diff = (yp - yu) / c0
            m, rms = float(np.abs(diff).max()), float(np.sqrt(np.mean(diff**2)))
            worst["max"], worst["rms"] = max(worst["max"], m), max(worst["rms"], rms)
            entry = {
                "upstream": (yu / c0).tolist(),
                "psdm": (yp / c0).tolist(),
                "max_abs_rel": m,
                "rms_rel": rms,
                "c0": c0,
                "c0_psdm": ps["c0"].get(comp),
            }
            if kind == "single":
                ta, ca = addesigns(d["sheet"])
                for impl, y in (("upstream", yu), ("psdm", yp)):
                    da = (np.interp(ta, grid, y) - ca) / c0
                    entry[f"{impl}_vs_addesigns"] = {
                        "max": float(np.abs(da).max()),
                        "rms": float(np.sqrt(np.mean(da**2))),
                    }
                entry["addesigns"] = {"t": ta.tolist(), "c": (ca / c0).tolist()}
            curves["compounds"][comp] = entry
        s["accuracy"] = {"vs_upstream": worst}
        if kind == "single":
            e = curves["compounds"]["TCE"]
            s["accuracy"]["upstream_vs_addesigns"] = e["upstream_vs_addesigns"]
            s["accuracy"]["psdm_vs_addesigns"] = e["psdm_vs_addesigns"]
        if "upstream_f64" in runs:
            f64 = runs["upstream_f64"][0]
            for key, a, b in (("f64_vs_upstream", f64, up), ("psdm_vs_f64", ps, f64)):
                diff = np.interp(grid, a["grid"], a["curves"]["TCE"]) - np.asarray(
                    b["curves"]["TCE"]
                )
                s["accuracy"][key] = {"max": float(np.abs(diff).max()) / d["c0"]}
        s["curves"] = curves
        summary[case] = s
    return summary


# ============================================================================ HTML
CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;
--rule:#e4e3df;--grid:#efeeea;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#008300;--accent-bg:#f3f2ee}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#141413;
--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e86;--rule:#33332f;--grid:#262624;--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--good:#3fae3f;--accent-bg:#22221f}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#141413;--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;
--muted:#8f8e86;--rule:#33332f;--grid:#262624;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#3fae3f;--accent-bg:#22221f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}h2{font-size:20px;margin:40px 0 12px}h3{font-size:15px;margin:0 0 4px}
p,li{color:var(--ink2);max-width:76ch}.lede{font-size:16px;margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:26px;font-variant-numeric:tabular-nums;color:var(--ink)}
.kpi span{font-size:13px;color:var(--ink2)}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-weight:600;color:var(--ink2);background:var(--accent-bg)}
th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}
td small{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 14px 8px;position:relative}
.card p.meta{font-size:12.5px;margin:0 0 6px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin:8px 0 4px}
.legend i{display:inline-block;width:18px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.legend i.dash{border-top-style:dashed}.legend i.dot{width:8px;height:8px;border:0;border-radius:50%;background:var(--s3)}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:11px;font-family:inherit}
svg .gl{stroke:var(--grid);stroke-width:1}svg .ax{stroke:var(--rule);stroke-width:1}
svg .up{stroke:var(--s1);stroke-width:2;fill:none}svg .ps{stroke:var(--s2);stroke-width:2;fill:none;stroke-dasharray:6 4}
svg .ad{fill:var(--s3);stroke:var(--surface);stroke-width:1.5}svg .diff{stroke:var(--s2);stroke-width:1.5;fill:none}
svg .zero{stroke:var(--muted);stroke-width:1}svg .hair{stroke:var(--muted);stroke-width:1;visibility:hidden}
svg .ttl{fill:var(--ink2);font-size:11.5px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--rule);border-radius:6px;
padding:6px 8px;font-size:12px;color:var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap;
font-variant-numeric:tabular-nums;z-index:2}
.tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
code{font-size:13px;background:var(--accent-bg);padding:1px 4px;border-radius:4px}
.env{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:13.5px;color:var(--ink2)}
.env b{color:var(--ink);font-weight:600}
.verdict{border-left:3px solid var(--s1);padding:2px 0 2px 14px;margin:16px 0}.verdict p{margin:4px 0}.verdict ul{margin:4px 0;padding-left:20px}
"""

JS = """
document.querySelectorAll('[data-chart]').forEach(card=>{
  const d=JSON.parse(card.querySelector('script').textContent), svg=card.querySelector('svg.main'),
        tip=card.querySelector('.tip'), hair=svg.querySelector('.hair');
  const pt=svg.createSVGPoint();
  function move(ev){
    pt.x=ev.clientX;pt.y=ev.clientY;const p=pt.matrixTransform(svg.getScreenCTM().inverse());
    if(p.x<d.x0||p.x>d.x1){leave();return}
    const t=d.tmin+(p.x-d.x0)/(d.x1-d.x0)*(d.tmax-d.tmin);
    let i=0,best=1e9;d.t.forEach((v,j)=>{const e=Math.abs(v-t);if(e<best){best=e;i=j}});
    const x=d.x0+(d.t[i]-d.tmin)/(d.tmax-d.tmin)*(d.x1-d.x0);
    hair.setAttribute('x1',x);hair.setAttribute('x2',x);hair.style.visibility='visible';
    let rows=`<b>${d.t[i].toFixed(1)} d</b>`;
    d.series.forEach(s=>{rows+=`<br><i style="background:var(${s.color})"></i>${s.name}: ${(100*s.y[i]).toFixed(2)} % of C0`});
    if(d.ad){let k=-1,b=1e9;d.ad.t.forEach((v,j)=>{const e=Math.abs(v-d.t[i]);if(e<b){b=e;k=j}});
      if(k>=0&&b<1.5)rows+=`<br><i style="background:var(--s3)"></i>AdDesignS (${d.ad.t[k].toFixed(1)} d): ${(100*d.ad.c[k]).toFixed(2)} %`}
    tip.innerHTML=rows;tip.style.display='block';
    const r=card.getBoundingClientRect(),w=tip.offsetWidth;let left=ev.clientX-r.left+14;
    if(left+w>r.width-4)left=ev.clientX-r.left-w-14;
    tip.style.left=Math.max(4,left)+'px';tip.style.top=(ev.clientY-r.top+12)+'px';
  }
  function leave(){tip.style.display='none';hair.style.visibility='hidden'}
  svg.addEventListener('mousemove',move);svg.addEventListener('mouseleave',leave);
});
"""


def nice_ticks(lo, hi, n=5):
    import math

    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (m * step) <= n:
            step *= m
            break
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _path(xs, ys, sx, sy):
    return "M" + "L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys, strict=False))


def svg_chart(
    t,
    series,
    ad=None,
    *,
    w=520,
    h=230,
    ylo=0.0,
    yhi=1.1,
    ylabel="C / C0",
    pct=False,
    cls="main",
    hair=True,
):
    ml, mr, mt, mb = 44, 10, 10, 28
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    tmin, tmax = 0.0, max(t)

    def sx(v):
        return x0 + (v - tmin) / (tmax - tmin) * (x1 - x0)

    def sy(v):
        return y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)

    out = [
        f'<svg class="{cls}" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(ylabel)} vs days">'
    ]
    for v in nice_ticks(ylo, yhi, 4 if not pct else 5):
        lab = f"{v * 100:g}%" if pct else f"{v:g}"
        out.append(
            f'<line class="gl" x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
            f'<text x="{x0 - 6}" y="{sy(v) + 3.5:.1f}" text-anchor="end">{lab}</text>'
        )
    for v in nice_ticks(tmin, tmax, 6):
        out.append(f'<text x="{sx(v):.1f}" y="{y1 + 16}" text-anchor="middle">{v:g}</text>')
    out.append(f'<line class="ax" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    if pct:
        out.append(f'<line class="zero" x1="{x0}" x2="{x1}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    out.append(f'<text class="ttl" x="{x0}" y="{y0 - 1}" dy="-1">{html.escape(ylabel)}</text>')
    out.append(f'<text x="{x1}" y="{h - 1}" text-anchor="end">days</text>')
    for s in series:
        out.append(f'<path class="{s["cls"]}" d="{_path(t, s["y"], sx, sy)}"/>')
    if ad is not None:
        ta, ca = ad
        step = max(1, len(ta) // 36)
        for a, c in list(zip(ta, ca, strict=False))[::step]:
            if a <= tmax:
                out.append(f'<circle class="ad" cx="{sx(a):.1f}" cy="{sy(c):.1f}" r="3.5"/>')
    if hair:
        out.append(f'<line class="hair" x1="0" x2="0" y1="{y0}" y2="{y1}"/>')
    out.append("</svg>")
    return "".join(out), dict(x0=x0, x1=x1, tmin=tmin, tmax=tmax)


def med(xs):
    return statistics.median(xs)


def fmt_s(xs):
    return f"{med(xs):.3f} s<br><small>{min(xs):.3f}–{max(xs):.3f}</small>"


def pct(x, digits=2):
    return f"{100 * x:.{digits}f} %"


def chart_card(title, meta, t, series, ad=None, yhi=1.1, ylabel="C / C0", resid=None):
    """One hoverable chart card; ``series`` items are (name, css class, colour var, y)."""
    main, geo = svg_chart(
        t, [{"y": y, "cls": cls} for _, cls, _, y in series], ad, yhi=yhi, ylabel=ylabel
    )
    dsvg = ""
    if resid is not None:
        lim = max(1e-3, max(abs(v) for v in resid) * 1.15)
        dsvg, _ = svg_chart(
            t,
            [{"y": resid, "cls": "diff"}],
            h=110,
            ylo=-lim,
            yhi=lim,
            ylabel="psdm − upstream, % of C0",
            pct=True,
            cls="resid",
            hair=False,
        )
    data = dict(t=t, series=[dict(name=n, color=c, y=y) for n, _, c, y in series], **geo)
    if ad:
        data["ad"] = {"t": ad[0], "c": ad[1]}
    return (
        f'<div class="card" data-chart><h3>{html.escape(title)}</h3><p class="meta">{html.escape(meta)}</p>'
        f'{main}{dsvg}<div class="tip"></div>'
        f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script></div>'
    )


def _spec_text(d):
    default = {
        "c0": 5.0,
        "water": "Organic Free",
        "cls": "halogenated alkenes",
        "name": "TCE",
        "tortu": 1.0,
        "temp": 20.0,
        "nr": 14,
        "nz": 19,
        "ne": 1,
        "corr": "Chern and Chien",
        "time_unit": "days",
        "kf": None,
        "dp": None,
        "ds": None,
    }
    keys = {
        "water": "water",
        "cls": "fouling class",
        "name": "compound",
        "tortu": "tortuosity",
        "temp": "°C",
        "nr": "nr",
        "nz": "nz",
        "ne": "ne",
        "corr": "kf correlation",
        "time_unit": "time unit",
        "kf": "kf",
        "dp": "dp",
        "ds": "ds",
    }
    return ", ".join(
        f"{keys[k]} {d[k]:g}" if isinstance(d[k], float) else f"{keys[k]} {d[k]}"
        for k in keys
        if d[k] != default[k]
    )


def render_paths(paths):
    rows, charts = [], []
    for p in paths.values():
        ok = '<span style="color:var(--good)">pass</span>' if p["pass"] else "<b>FAIL</b>"
        mt = p["mass_transfer"]
        rows.append(
            f"<tr><td>{html.escape(p['label'])}<br><small>{html.escape(_spec_text(p['spec']))}</small></td>"
            f"<td>{pct(p['max_abs_rel'])}</td><td>{pct(p['rms_rel'], 3)}</td><td>{ok}</td>"
            f"<td>{mt['kf']:.3g}</td><td>{mt['dp']:.3g}</td><td>{mt['ds']:.3g}</td></tr>"
        )
        diff = [a - b for a, b in zip(p["psdm"], p["upstream"], strict=True)]
        charts.append(
            chart_card(
                p["label"],
                f"{_spec_text(p['spec'])} · max |Δ| {pct(p['max_abs_rel'])}",
                p["grid"],
                [("upstream", "up", "--s1", p["upstream"]), ("psdm", "ps", "--s2", p["psdm"])],
                yhi=max(1.1, max(p["upstream"]) * 1.05),
                resid=diff,
            )
        )
    worst = max(p["max_abs_rel"] for p in paths.values())
    npass = sum(p["pass"] for p in paths.values())
    return f"""<h2>Other code paths</h2>
<p>Options the timed cases above do not exercise, each run once by both codes and compared point by point. All
cases use the AdDesignS column with a constant 5 mg/L influent (TCE properties) for 174 d and the default
nr = 14, nz = 19 unless stated. Pass means max |psdm − upstream| / C0 &lt; {pct(PATH_TOL, 1)}.
<b>{npass} of {len(paths)} pass; worst {pct(worst)}.</b> The same cases run in the test suite
(<code>tests/test_paths.py</code>). kf, dp and ds are the values psdm used (<code>Result.mass_transfer</code>; in
fouled water ds is divided by 1e10, as upstream does, also when ds is given).</p>
<div class="tablewrap"><table><thead><tr><th>Case</th><th>max |Δ|/C0</th><th>RMS</th><th>0.5 %</th>
<th>kf, cm/s</th><th>dp, cm²/s</th><th>ds, cm²/s</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p>How each option reaches upstream: tortuosity through <code>column_data['tortu']</code> (used only in Organic
Free water by both codes); manual coefficients through the <code>mass_transfer=</code> keyword; PFAS through
<code>chem_type='PFAS'</code> with the compound name looked up in upstream's PFAS table (an unlisted name falls back
to the average); temperature through <code>temp=</code>; time unit through <code>time_type='hours'</code>.
Upstream's <code>run_psdm</code> hard-codes the Chern and Chien film-transfer correlation, so for the other
correlations kf was computed with upstream's own <code>kf_calc</code> as <code>run_psdm</code> would and passed
through <code>mass_transfer=</code>. Gnielinski gives nearly the same kf as Chern and Chien for this column; the
Wakao and Funazkri case is the one that moves kf.</p>
<p><b>Intentional differences in psdm</b> that show up as small residuals here, not bugs:</p>
<ul>
<li><b>Fouling is evaluated analytically.</b> psdm computes the K multiplier
<code>rk1 + rk2 t + rk3 exp(rk4 t)</code> (floored at 1e-3) at each solver time. Upstream samples it at 500
points over max(2000 d, run length), interpolates linearly, and then tabulates the interpolant once per simulated
minute. Over 174 d those 500 points are about 4 d apart, so upstream's multiplier is a chord of the curve.</li>
<li><b>Influent and tortuosity are evaluated on the solver's time scale.</b> Upstream builds per-minute
tables and looks values up in them; psdm interpolates the influent and evaluates the tortuosity function
directly.</li>
<li><b>The film-transfer correlation is an argument.</b> <code>simulate(correlation=...)</code> uses any entry of
<code>psdm.correlations.SHERWOOD</code>; upstream <code>run_psdm</code> always uses Chern and Chien, and the others
are reachable only by passing kf by hand.</li>
<li>Equal on purpose, although it looks odd: in fouled water both codes divide ds by 1e10 even when the
user gives ds, and with a user-given dp and tortuosity ≠ 1 both multiply dp by the tortuosity in the pore
diffusion modulus.</li>
</ul>
<div class="legend"><span><i style="border-color:var(--s1)"></i>upstream EPA PSDM</span>
<span><i class="dash" style="border-color:var(--s2)"></i>psdm</span></div>
<div class="grid">{"".join(charts)}</div>"""


def _fit_notes(r, lo, hi, max_sims=None):
    notes = []
    if min(abs(r["n_inv"] - lo), abs(r["n_inv"] - hi)) < 1e-3:
        notes.append(f"1/n at its bound ({lo:g}–{hi:g})")
    if max_sims and r["evaluations"] > max_sims:
        notes.append(f"stopped at max_evaluations = {max_sims}")
    return "; ".join(notes)


def render_fits(fits):
    rows, charts, verdict = [], [], []
    for f in fits.values():
        up, ps, gb = f["upstream"], f["psdm"], f["grid_best"]
        ratio = ps["ssq_window"] / up["ssq_window"]
        better = "better than" if ratio < 0.99 else "worse than" if ratio > 1.01 else "equal to"
        notes = [f"psdm: {n}" for n in [_fit_notes(ps, 0.15, 1.0, 200)] if n] + [
            f"upstream: {n}" for n in [_fit_notes(up, 0.2, 1.0)] if n
        ]
        verdict.append(
            f"<li><b>{html.escape(f['label'])}</b>: psdm window SSQ {ps['ssq_window']:.3g} is "
            f"<b>{better}</b> upstream's {up['ssq_window']:.3g} ({ratio:.2g}×), with "
            f"{ps['evaluations']} simulations against {up['evaluations']}. Best grid point in the "
            f"window: {gb['ssq_window']:.3g} (psdm reaches {ps['ssq_window'] / gb['ssq_window']:.2g}× "
            f"that, upstream {up['ssq_window'] / gb['ssq_window']:.2g}×). All data: psdm "
            f"{ps['ssq_full']:.3g}, upstream {up['ssq_full']:.3g}."
            + (f" <small>{html.escape('; '.join(notes))}.</small>" if notes else "")
            + "</li>"
        )
        rows.append(f'<tr><td colspan="7"><b>{html.escape(f["label"])}</b></td></tr>')
        for name, r in (("upstream run_all_smart", up), ("psdm fit_freundlich", ps)):
            rows.append(
                f"<tr><td>{name}</td><td>{r['k']:,.1f}</td><td>{r['n_inv']:.3f}</td>"
                f"<td>{r['ssq_window']:.4g}<br><small>own sim {r['ssq_window_own_sim']:.4g}</small></td>"
                f"<td>{r['ssq_full']:.4g}<br><small>own sim {r['ssq_full_own_sim']:.4g}</small></td>"
                f"<td>{r['evaluations']}</td><td>{r['seconds']:.1f}</td></tr>"
            )
        rows.append(
            f"<tr><td><small>grid scan, best in window</small></td><td>{gb['k']:,.1f}</td>"
            f"<td>{gb['n_inv']:.3f}</td><td>{gb['ssq_window']:.4g}</td><td>{gb['ssq_full']:.4g}</td>"
            f"<td>225</td><td>–</td></tr>"
            f"<tr><td><small>psdm start (Kdata sheet)</small></td><td>{f['start_psdm']['k']:,.1f}</td>"
            f"<td>{f['start_psdm']['n_inv']:.3f}</td><td>{f['start_psdm']['ssq_window']:.4g}</td>"
            f"<td>–</td><td>–</td><td>–</td></tr>"
        )
        scale = 1750.0
        cv = f["curves"]
        obs = f["observed"]
        meta = "effluent / 1750 µg/L (max influent)"
        if f["until"]:
            meta += f" · fitted on t ≤ {f['until']:g} d only, all observations shown"
        charts.append(
            chart_card(
                f["label"],
                meta,
                cv["grid"],
                [
                    ("upstream fit", "up", "--s1", [v / scale for v in cv["upstream"]]),
                    ("psdm fit", "ps", "--s2", [v / scale for v in cv["psdm"]]),
                ],
                ad=(obs["t"], [v / scale for v in obs["c"]]),
                yhi=1.1,
                ylabel="C / 1750 µg/L",
            )
        )
    return f"""<h2>Fitting K and 1/n</h2>
<p>psdm's <code>fit_freundlich</code> (Nelder–Mead over log q_ref and 1/n, started from the K and 1/n in the
<code>Kdata</code> sheet) against upstream's <code>run_all_smart</code> (a grid walk over a K multiplier and 1/n,
started from the capacity estimated by upstream's <code>__calculate_capacity</code>, with no <code>k_data</code>
given, as in normal use), both with nr = 8, nz = 12 on <code>examples/Example_TCE.xlsx</code>, sheet
<code>data_optimize</code>. For the window case upstream was given only the rows with t ≤ 55 d; psdm was given
all data with <code>until=55</code>. The two codes minimise different objectives (upstream: daily-interpolated
effluent up to its estimated breakthrough day; psdm: the observed points, after <code>read_excel</code> fills the
effluent at influent-only times), so every fitted (K, 1/n) is re-scored the same way: simulated with psdm and
compared with the effluent rows of the sheet (<em>own sim</em> repeats the score with each code's own
simulation). SSQ in (µg/L)². <em>Simulations</em> counts calls of upstream <code>run_psdm</code> and psdm
<code>simulate</code>; the grid scan is 25 q_ref factors (0.1–10×) × 9 values of 1/n (0.2–1.0).</p>
<div class="tablewrap"><table><thead><tr><th>Method</th><th>K</th><th>1/n</th><th>SSQ in window</th>
<th>SSQ all data</th><th>simulations</th><th>time, s</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<div class="verdict"><ul>{"".join(verdict)}</ul>
<p>Criterion: psdm's SSQ should be no worse than upstream's in the fitted window; identical parameters are not
expected. Both searches can stop at a poor point. In the t ≤ 55 d window the starting K predicts zero effluent at
every observation, the SSQ surface around it is flat, and psdm's Nelder–Mead does not leave the plateau
(<code>fit_freundlich</code> in psdm 0.2.0 has no coarse start-point scan). Upstream's
<code>run_all_smart</code> takes a few steps from its own capacity estimate and stops when no neighbouring grid
point improves, often at an edge of its 1/n grid. Upstream's other optimiser, <code>run_all</code>, raises
<code>ValueError</code> in its grid-refinement step under pandas 2.3 (upstream marks it
<em>TODO: check this still functions</em>), so it is not compared. A good fit to t ≤ 55 d alone does not
predict the later data; the grid point that fits the window best is far off over all data.</p></div>
<div class="legend"><span><i style="border-color:var(--s1)"></i>upstream fit</span>
<span><i class="dash" style="border-color:var(--s2)"></i>psdm fit</span>
<span><i class="dot"></i>observed effluent</span></div>
<div class="grid">{"".join(charts)}</div>"""


def render(summary, env, repeats, paths=None, fits=None):
    cases = list(summary)
    rows, detail, solver = [], [], []
    worst_up, worst_ad = 0.0, 0.0
    by_kind = {}
    f64_cut = []  # (upstream as-is, upstream float64, psdm) warm build + run, single cases
    rhs_ratio = []
    for c in cases:
        s = summary[c]
        tu, tp = s["timing"]["upstream"], s["timing"]["psdm"]
        tf = s["timing"].get("upstream_f64")
        base = tf or tu  # fair baseline: upstream with float64 inputs where that differs

        def outside(t):
            return med(t["warm_build_s"]) + med(t["warm_run_s"]) - med(t["warm_integrate_s"])

        sw = med(base["warm_total_s"]) / med(tp["warm_total_s"])
        sc = med(base["cold_total_s"]) / med(tp["cold_total_s"])
        si = med(base["warm_integrate_s"]) / med(tp["warm_integrate_s"])
        so = outside(base) / max(1e-9, outside(tp))
        by_kind.setdefault(s["kind"], []).append((sw, si))
        sb = s["solver"]["upstream_f64" if tf else "upstream"]
        sp = s["solver"]["psdm"]
        rhs_ratio.append(
            (med(base["warm_integrate_s"]) / sb["nfev"])
            / (med(tp["warm_integrate_s"]) / sp["nfev"])
        )
        if tf:
            f64_cut.append(
                (med(tu["warm_total_s"]), med(tf["warm_total_s"]), med(tp["warm_total_s"]))
            )
        acc = s["accuracy"]
        worst_up = max(worst_up, acc["vs_upstream"]["max"])
        if "psdm_vs_addesigns" in acc:
            worst_ad = max(worst_ad, acc["psdm_vs_addesigns"]["max"])
            ad_cell = (
                f"{pct(acc['psdm_vs_addesigns']['max'])}<br><small>upstream "
                f"{pct(acc['upstream_vs_addesigns']['max'])}</small>"
            )
        else:
            ad_cell = "<small>no AdDesignS output</small>"
        if tf:
            sa = med(tu["warm_total_s"]) / med(tp["warm_total_s"])
            f64_cell = fmt_s(tf["warm_total_s"])
            asis_cell = f"{sa:.2f}×"
        else:
            f64_cell = "<small>inputs already float64</small>"
            asis_cell = "<small>same</small>"
        rows.append(
            f"<tr><td>{html.escape(s['label'])}</td><td>{fmt_s(tu['warm_total_s'])}</td>"
            f"<td>{f64_cell}</td><td>{fmt_s(tp['warm_total_s'])}</td>"
            f"<td><b>{sw:.2f}×</b><br><small>cold {sc:.2f}×</small></td><td>{asis_cell}</td>"
            f"<td>{si:.2f}×</td><td>{so:.1f}×</td>"
            f"<td>{pct(acc['vs_upstream']['max'])}<br><small>RMS {pct(acc['vs_upstream']['rms'], 3)}</small></td>"
            f"<td>{ad_cell}</td></tr>"
        )
        impl_rows = [("upstream", tu)] + ([("upstream, float64", tf)] if tf else [])
        for impl, t in [*impl_rows, ("psdm", tp)]:
            detail.append(
                f"<tr><td>{html.escape(s['label'])}</td><td>{impl}</td>"
                f"<td>{med(t['cold_read_s']):.3f}</td><td>{med(t['warm_build_s']):.3f}</td>"
                f"<td>{med(t['warm_run_s']):.3f}</td><td>{med(t['warm_integrate_s']):.3f}</td>"
                f"<td>{med(t['warm_run_s']) - med(t['warm_integrate_s']):.3f}</td>"
                f"<td>{med(t['cold_build_s']) + med(t['cold_run_s']):.3f}</td>"
                f"<td>{med(t['import_s']):.2f}</td></tr>"
            )
        su = s["solver"]["upstream"]
        cnt = "".join(
            f"<td>{su[k]}</td><td>{sb[k] if tf else '–'}</td><td>{sp[k]}</td>"
            for k in ("nsteps", "nfev", "njev", "nlu")
        )
        ms = [
            med(tu["warm_integrate_s"]) / su["nfev"] * 1e3,
            med(tf["warm_integrate_s"]) / sb["nfev"] * 1e3 if tf else None,
            med(tp["warm_integrate_s"]) / sp["nfev"] * 1e3,
        ]
        solver.append(
            f"<tr><td>{html.escape(s['label'])}</td>{cnt}"
            + "".join(f"<td>{v:.3f}</td>" if v is not None else "<td>–</td>" for v in ms)
            + "</tr>"
        )

    # largest relative difference in step and RHS-evaluation counts between implementations
    max_count_dev = 0.0
    for s in summary.values():
        sp = s["solver"]["psdm"]
        for su in s["solver"].values():
            for k in ("nsteps", "nfev"):
                max_count_dev = max(max_count_dev, abs(su[k] - sp[k]) / sp[k])

    gm = statistics.geometric_mean
    charts = []
    for c in cases:
        s = summary[c]
        t = s["curves"]["grid"]
        for comp, e in s["curves"]["compounds"].items():
            ad = (e["addesigns"]["t"], e["addesigns"]["c"]) if "addesigns" in e else None
            yhi = max(1.1, max(max(e["upstream"]), max(e["psdm"])) * 1.05)
            main, geo = svg_chart(
                t, [{"y": e["upstream"], "cls": "up"}, {"y": e["psdm"], "cls": "ps"}], ad, yhi=yhi
            )
            diff = [p - u for p, u in zip(e["psdm"], e["upstream"], strict=False)]
            lim = max(1e-3, max(abs(v) for v in diff) * 1.15)
            dsvg, _ = svg_chart(
                t,
                [{"y": diff, "cls": "diff"}],
                h=110,
                ylo=-lim,
                yhi=lim,
                ylabel="psdm − upstream, % of C0",
                pct=True,
                cls="resid",
                hair=False,
            )
            data = dict(
                t=t,
                series=[
                    dict(name="upstream", color="--s1", y=e["upstream"]),
                    dict(name="psdm", color="--s2", y=e["psdm"]),
                ],
                **geo,
            )
            if ad:
                data["ad"] = {"t": ad[0], "c": ad[1]}
            title = s["label"] if len(s["curves"]["compounds"]) == 1 else f"{s['label']} — {comp}"
            meta = f"C0 = {e['c0']:g} {'mg/L' if s['kind'] == 'single' else 'µg/L'}"
            if s["kind"] == "variable":
                meta = f"max influent = {e['c0']:g} µg/L (curves scaled by it)"
            meta += f" · max |Δ| vs upstream {pct(e['max_abs_rel'])}"
            if "psdm_vs_addesigns" in e:
                meta += f" · vs AdDesignS {pct(e['psdm_vs_addesigns']['max'])}"
            charts.append(
                f'<div class="card" data-chart><h3>{html.escape(title)}</h3><p class="meta">{html.escape(meta)}</p>'
                f'{main}{dsvg}<div class="tip"></div>'
                f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script></div>'
            )

    env_rows = "".join(
        f"<b>{html.escape(k)}</b><span>{html.escape(str(v))}</span>" for k, v in env.items()
    )
    all_fair = [a for v in by_kind.values() for a, _ in v]
    all_int = [b for v in by_kind.values() for _, b in v]
    kinds = {
        "single": "single compound, constant influent",
        "multi": "multi-component",
        "variable": "variable influent",
    }

    def rng(v):
        lo, hi = f"{min(v):.2f}", f"{max(v):.2f}"
        return f"{lo}×" if lo == hi else f"{lo}–{hi}×"

    verdict = "".join(
        f"<li><b>{kinds[k]}</b> ({len(v)} case{'s' if len(v) > 1 else ''}): build + run {rng([a for a, _ in v])}, "
        f"inside <code>solve_ivp</code> {rng([b for _, b in v])}.</li>"
        for k, v in by_kind.items()
    )
    obj_txt = ""
    if f64_cut:
        f64_dev = max(
            summary[c]["accuracy"]["f64_vs_upstream"]["max"]
            for c in cases
            if "f64_vs_upstream" in summary[c]["accuracy"]
        )
        obj_txt = f"""<p><b>Most of the single-compound gap is an input-dtype artefact, not a psdm improvement.</b>
Upstream's single-compound inputs were built the way upstream's <code>test_PSDM.py</code> builds them, by
transposing the <code>data</code> sheet of <code>test.xlsx</code>. That sheet mixes numbers with the string
<code>influentID</code>, so the chemical-property and K-data frames have dtype <code>object</code>. The object
dtype carries through to the bed coefficients, and the two <code>np.matmul</code> calls in upstream's
right-hand side then run as Python-object loops. Built that way, upstream is
{rng([a / p for a, _, p in f64_cut])} slower than psdm. Converting the same inputs to float64 (the
<em>float64 inputs</em> column) makes upstream {rng([a / f for a, f, _ in f64_cut])} faster with no change to its
code, and leaves psdm only {rng([f / p for _, f, p in f64_cut])} ahead. The float64 run changes upstream's output
by at most {pct(f64_dev)} of C0, through a slightly different step sequence.</p>"""

    paths_html = render_paths(paths) if paths else ""
    fits_html = render_fits(fits) if fits else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSDM Benchmark</title><style>{CSS}</style></head><body><main>
<h1>psdm vs the original EPA PSDM</h1>
<p class="lede">Speed and agreement of the <code>psdm</code> package against the US EPA
<a href="https://github.com/USEPA/Water_Treatment_Models/tree/{UPSTREAM_COMMIT}/PSDM">PSDM</a> code it was rewritten
from, on {len(cases)} cases, {repeats} fresh-process runs per case and implementation. Generated by
<code>scripts/benchmark.py</code> on {html.escape(env["date"])}.</p>
<div class="kpis">
<div class="kpi"><b>{gm(all_fair):.2f}×</b><span>geometric-mean speed-up, build + run, against upstream with float64 inputs (range {rng(all_fair)})</span></div>
<div class="kpi"><b>{gm(all_int):.2f}×</b><span>geometric-mean speed-up inside <code>solve_ivp</code> (range {rng(all_int)})</span></div>
<div class="kpi"><b>{pct(worst_up)}</b><span>worst max |psdm − upstream| / C0 across all cases</span></div>
<div class="kpi"><b>{pct(worst_ad)}</b><span>worst max |psdm − AdDesignS| / C0 (single-compound cases)</span></div>
</div>
<div class="verdict">
<p><b>The ODE algorithm is the same.</b> Both codes integrate the same equations with scipy BDF, the same
tolerances and the same Jacobian sparsity pattern. Step and RHS-evaluation counts agree within
{pct(max_count_dev, 1)}.</p>
{obj_txt}
<p><b>What psdm actually gains</b>, against upstream with float64 inputs:</p>
<ul>{verdict}</ul>
<p>Most of that gain is outside the integrator. Upstream rebuilds per-minute tables for influent, fouling and
tortuosity on every run, one point per simulated minute (about 250,000 points for 174 days and 1.44 million for
the 1,000-day multi-component case), plus the Jacobian pattern; it also copies
DataFrames and computes a fit residual inside <code>run_psdm_kfit</code>. psdm builds these on the solver's own
time scale. Inside the integrator the two are close: per RHS evaluation (integrate time / RHS evaluations,
which includes Jacobian and LU work) upstream takes {rng(rhs_ratio)} as long as psdm.</p></div>

<h2>Summary</h2>
<p>Times are the median of {repeats} runs (min–max below) of object construction plus simulation, second call
inside each process. Two upstream variants are shown: <em>upstream</em> builds the inputs exactly as upstream's
<code>test_PSDM.py</code> does (object dtype in the single-compound cases), and <em>float64 inputs</em> converts the
same inputs to float64 first. <b>Speed-up</b> is against the float64 variant where it exists, otherwise against
upstream (whose inputs are already float64 there); the small line is the same ratio for the first, cold call.
<em>vs as-is</em> is the ratio against the object-dtype variant. <em>solve_ivp</em> and <em>outside</em> split the
speed-up into the time inside the integrator and everything else (build + run − integrate). Accuracy is the
largest absolute difference over the curve divided by the influent concentration C0, against the as-is upstream
run.</p>
<div class="tablewrap"><table>
<thead><tr><th>Case</th><th>upstream<br><small>test_PSDM.py inputs</small></th><th>upstream<br><small>float64 inputs</small></th>
<th>psdm</th><th>speed-up</th><th>vs as-is</th><th>solve_ivp</th><th>outside</th><th>max |Δ|/C0 vs upstream</th>
<th>max |Δ|/C0 vs AdDesignS</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>

<h2>Breakthrough curves</h2>
<div class="legend"><span><i style="border-color:var(--s1)"></i>upstream EPA PSDM</span>
<span><i class="dash" style="border-color:var(--s2)"></i>psdm</span>
<span><i class="dot"></i>AdDesignS (every few points shown)</span></div>
<p>Effluent concentration as a fraction of C0; the strip under each chart is psdm − upstream (as-is run). Hover a
chart for values.</p>
<div class="grid">{"".join(charts)}</div>

{paths_html}
{fits_html}

<h2>Where the time goes</h2>
<p>Medians in seconds. <em>read</em>: Excel input (cold call); <em>build</em>: constructing the model objects;
<em>run</em>: the simulation call; <em>integrate</em>: time inside <code>scipy.integrate.solve_ivp</code> only;
<em>run − integrate</em>: everything else in the run call (setup of matrices, per-minute influent/fouling tables,
output conversion and interpolation; upstream also computes a fit residual it does not need here). <em>cold</em>:
build + run on the first call in the process. <em>import</em>: <code>import psdm</code> (or the upstream module) in
a fresh process.</p>
<div class="tablewrap"><table><thead><tr><th>Case</th><th>impl</th><th>read</th><th>build</th><th>run</th>
<th>integrate</th><th>run − integrate</th><th>cold build+run</th><th>import</th></tr></thead>
<tbody>{"".join(detail)}</tbody></table></div>

<h2>Solver work</h2>
<p>All implementations call <code>solve_ivp</code> with BDF, default tolerances (rtol 1e-3, atol 1e-6) and the same
sparse Jacobian pattern; scipy estimates the Jacobian by finite differences. Counts are from the first run and were
identical in every repeat. The small differences come from floating-point round-off in different but equivalent
arithmetic. Cost per RHS evaluation = warm integrate time / nfev. It includes the finite-difference Jacobian and LU
work, so it is a per-step cost proxy, not a pure RHS timing.</p>
<div class="tablewrap"><table><thead><tr><th rowspan="2">Case</th><th colspan="3">steps</th><th colspan="3">RHS evals</th>
<th colspan="3">Jacobians</th><th colspan="3">LU decomps</th><th colspan="3">ms / RHS eval</th></tr>
<tr>{"<th>up</th><th>up f64</th><th>psdm</th>" * 5}</tr></thead><tbody>{"".join(solver)}</tbody></table></div>

<h2>Methods</h2>
<ul>
<li><b>Implementations.</b> Upstream: <code>psdm/PSDM.py</code> from USEPA/Water_Treatment_Models at commit
{UPSTREAM_COMMIT}, unmodified, imported from a fresh clone. New: the <code>psdm</code> package from this repository.
All run in the same virtual environment, so numpy, scipy and pandas builds are identical.</li>
<li><b>Upstream input variants.</b> In the single-compound cases the upstream inputs are built as in
upstream's <code>test_PSDM.py</code> and this repository's <code>tests/data/reference/generate.py</code>, by
transposing the <code>data</code> sheet, which produces object-dtype frames. The <em>float64 inputs</em> control group
applies <code>.astype(float)</code> to the chemical-property and K-data frames and converts the numeric entries of
the column Series; nothing else changes. The multi-component and variable-influent cases read their inputs with
upstream's <code>process_input_file</code> and <code>process_input_data</code>, which already return numeric
dtypes, so they have no separate control group.</li>
<li><b>Process isolation.</b> Each timed sample is a new <code>python -I</code> process that imports one
implementation and runs one case twice. The order of implementations rotates per case and repeat to spread out
drift from thermal or background load.</li>
<li><b>What is timed.</b> <em>build + run</em> = upstream <code>PSDM(...)</code> construction plus
<code>run_psdm_kfit</code> (single-compound and variable-influent cases, the call the upstream examples use) or
<code>run_psdm</code> (multi-component); for psdm, building <code>Column</code>/<code>Compound</code> objects
(or <code>Dataset.compounds()</code>) plus <code>simulate()</code>. Excel reading is timed separately and excluded
from the headline; so are interpreter start-up and package import.</li>
<li><b>Cases.</b> Single compound (TCE) from the AdDesignS verification set in <code>tests/data/addesigns_test.xlsx</code>
(the upstream <code>test.xlsx</code>), constant influent for 174 d, default nr = 14, nz = 19. This adds the 25 mg/L
Rhine and Organic Free cases to the five used in the test suite, because AdDesignS output exists for them.
Multi-component: <code>examples/Example_Multi.xlsx</code>, nr = 4, nz = 8, ne = 2, as in
<code>tests/data/reference/generate.py</code>. Variable influent: <code>examples/Example_TCE.xlsx</code> sheet
<code>data_variable</code> (piecewise-linear TCE influent 3–5 mg/L), nr = 8, nz = 12, as in upstream
<code>Example_TCE.py</code>, with and without Rhine fouling.</li>
<li><b>Accuracy.</b> Effluent curves are linearly interpolated onto 349 evenly spaced times and compared;
max and RMS of |psdm − upstream| / C0 are reported (for multi-component cases, the worst compound). AdDesignS
comparisons use the sheets 5, 25, 50, 5no, 25no, 50no and 50K at their own output times, keeping only rows with
C &gt; 0, as in the test suite. The sheets end in negative sentinel rows, and their first rows are ±1e-3 noise.
AdDesignS output stops at 105 d for the Rhine cases.</li>
</ul>
<h2>Limitations</h2>
<ul>
<li>One machine, one Python and library set (below). Absolute times will differ elsewhere; ratios are more portable
but still depend on the BLAS and scipy version.</li>
<li>The cases are small (tens to a few hundred ODEs). Large multi-component problems were not benchmarked.</li>
<li>Upstream <code>run_psdm_kfit</code> also computes a sum of squares against (empty) effluent data and copies
DataFrames; that overhead is part of its normal API and is counted. The <em>solve_ivp</em> column isolates the
integrator for a like-for-like comparison.</li>
<li>Fitting is compared once on one data set (above), not timed repeatedly; ds fitting, capacity and
uncertainty tools are neither timed nor compared here. Upstream's fitting
loops call <code>run_psdm</code> many times, so the per-run setup cost above would multiply there.</li>
<li>Agreement with upstream is not agreement with reality; the only external reference here is AdDesignS.</li>
</ul>
<h2>Environment</h2>
<div class="env">{env_rows}</div>
</main><script>{JS}</script></body></html>
"""


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--upstream", type=Path, help="PSDM folder of the upstream checkout")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "report.html")
    ap.add_argument("--json", type=Path, default=ROOT / "benchmarks" / "results.json")
    ap.add_argument("--from-json", type=Path, help="re-render the report from saved results")
    ap.add_argument("--paths", nargs="*", default=list(PATHS), choices=list(PATHS))
    ap.add_argument("--fits", nargs="*", default=list(FITS), choices=list(FITS))
    ap.add_argument(
        "--worker",
        choices=[
            "upstream",
            "upstream_f64",
            "psdm",
            "path_upstream",
            "path_psdm",
            "fit_upstream",
            "fit_psdm",
        ],
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--case", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        warnings.simplefilter("ignore")
        if a.worker == "psdm":
            res = worker_psdm(a.case)
        elif a.worker.startswith("path_"):
            res = worker_path(a.worker[5:], a.case, a.upstream)
        elif a.worker.startswith("fit_"):
            res = worker_fit(a.worker[4:], a.case, a.upstream)
        else:
            res = worker_upstream(a.case, a.upstream, float_inputs=a.worker == "upstream_f64")
        a.worker_out.write_text(json.dumps(res))
        return

    if a.from_json:
        saved = json.loads(a.from_json.read_text())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            upstream = a.upstream or clone_upstream(Path(tmp) / "wtm")
            env = environment(upstream)
            print("\n".join(f"{k}: {v}" for k, v in env.items()), flush=True)
            acc = accuracy_runs(upstream, a.paths, a.fits)
            raw = benchmark(upstream, a.repeats, a.cases)
        saved = {
            "environment": env,
            "repeats": a.repeats,
            "summary": summarise(raw),
            "paths": summarise_paths(acc["paths"]),
            "fits": fit_reference(acc["fits"]),
        }
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(saved, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        render(
            saved["summary"],
            saved["environment"],
            saved["repeats"],
            saved.get("paths", {}),
            saved.get("fits", {}),
        ),
        encoding="utf-8",
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
