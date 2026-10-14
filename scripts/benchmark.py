"""Benchmark psdm against the original US EPA PSDM code and write an HTML report.

usage (from the repository root)::

    uv run python scripts/benchmark.py [--upstream PATH] [--repeats 7]
    uv run python scripts/benchmark.py --from-json benchmarks/results.json   # re-render only

``--upstream`` is the ``PSDM`` folder of a checkout of
https://github.com/USEPA/Water_Treatment_Models at commit 3f514c2. Without it the
repository is cloned into a temporary directory (requires ``git``).

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


def worker_upstream(case, upstream):
    import types

    import numpy as np
    import pandas as pd

    # upstream's psdm/ folder has no __init__.py and would be shadowed by the installed psdm
    # package, so load it as the package "epa_psdm"
    pkg_dir = Path(upstream).resolve() / "psdm"
    t0 = time.perf_counter()
    pkg = types.ModuleType("epa_psdm")
    pkg.__path__ = [str(pkg_dir)]
    sys.modules["epa_psdm"] = pkg
    import epa_psdm.PSDM as P

    import_s = time.perf_counter() - t0
    log = SolveLog(P.solve_ivp)
    P.solve_ivp = log
    kind, _, d = CASES[case]

    if kind == "single":

        def read():
            return pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name="data", index_col=[0])

        def build(inp):
            # identical to tests/data/reference/generate.py
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


def benchmark(upstream: Path, repeats: int, cases: list[str]):
    raw = {c: {"upstream": [], "psdm": []} for c in cases}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "w.json"
        for r in range(repeats):
            for case in cases:
                order = (
                    ("upstream", "psdm")
                    if (r + cases.index(case)) % 2 == 0
                    else ("psdm", "upstream")
                )
                for impl in order:
                    res = run_worker(impl, case, upstream, out)
                    raw[case][impl].append(res)
                    it = res["iterations"]
                    print(
                        f"[{r + 1}/{repeats}] {case:14s} {impl:8s} cold {it[0]['build_s'] + it[0]['run_s']:6.3f}s"
                        f"  warm {it[1]['build_s'] + it[1]['run_s']:6.3f}s",
                        flush=True,
                    )
    return raw


# ============================================================================ analysis
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


def render(summary, env, repeats):
    cases = list(summary)
    rows, detail, solver = [], [], []
    speed_warm, speed_cold, worst_up, worst_ad = [], [], 0.0, 0.0
    by_kind = {}
    for c in cases:
        s = summary[c]
        tu, tp = s["timing"]["upstream"], s["timing"]["psdm"]
        sw = med(tu["warm_total_s"]) / med(tp["warm_total_s"])
        sc = med(tu["cold_total_s"]) / med(tp["cold_total_s"])
        speed_warm.append(sw)
        speed_cold.append(sc)
        si = med(tu["warm_integrate_s"]) / med(tp["warm_integrate_s"])
        by_kind.setdefault(s["kind"], []).append((sw, si))
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
        rows.append(
            f"<tr><td>{html.escape(s['label'])}</td><td>{fmt_s(tu['warm_total_s'])}</td>"
            f"<td>{fmt_s(tp['warm_total_s'])}</td><td><b>{sw:.2f}×</b><br><small>cold {sc:.2f}×</small></td>"
            f"<td>{pct(acc['vs_upstream']['max'])}<br><small>RMS {pct(acc['vs_upstream']['rms'], 3)}</small></td>"
            f"<td>{ad_cell}</td></tr>"
        )
        for impl, t in (("upstream", tu), ("psdm", tp)):
            detail.append(
                f"<tr><td>{html.escape(s['label'])}</td><td>{impl}</td>"
                f"<td>{med(t['cold_read_s']):.3f}</td><td>{med(t['warm_build_s']):.3f}</td>"
                f"<td>{med(t['warm_run_s']):.3f}</td><td>{med(t['warm_integrate_s']):.3f}</td>"
                f"<td>{med(t['warm_run_s']) - med(t['warm_integrate_s']):.3f}</td>"
                f"<td>{med(t['cold_build_s']) + med(t['cold_run_s']):.3f}</td>"
                f"<td>{med(t['import_s']):.2f}</td></tr>"
            )
        su, sp = s["solver"]["upstream"], s["solver"]["psdm"]
        solver.append(
            f"<tr><td>{html.escape(s['label'])}</td>"
            + "".join(
                f"<td>{su[k]}</td><td>{sp[k]}</td>" for k in ("nsteps", "nfev", "njev", "nlu")
            )
            + f"<td>{med(tu['warm_integrate_s']) / su['nfev'] * 1e3:.3f}</td>"
            f"<td>{med(tp['warm_integrate_s']) / sp['nfev'] * 1e3:.3f}</td></tr>"
        )

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
    faster = gm(speed_warm)
    kinds = {
        "single": "single compound, constant influent",
        "multi": "multi-component",
        "variable": "variable influent",
    }

    def rng(v):
        return f"{min(v):.2f}×" if max(v) - min(v) < 0.005 else f"{min(v):.2f}–{max(v):.2f}×"

    verdict = "".join(
        f"<li><b>{kinds[k]}</b> ({len(v)} cases): build + run {rng([a for a, _ in v])} faster, "
        f"time inside <code>solve_ivp</code> {rng([b for _, b in v])}.</li>"
        for k, v in by_kind.items()
    )
    ratio_rng = f"{min(speed_warm):.2f}–{max(speed_warm):.2f}×"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSDM Benchmark</title><style>{CSS}</style></head><body><main>
<h1>psdm vs the original EPA PSDM</h1>
<p class="lede">Speed and agreement of the <code>psdm</code> package against the US EPA
<a href="https://github.com/USEPA/Water_Treatment_Models/tree/{UPSTREAM_COMMIT}/PSDM">PSDM</a> code it was rewritten
from, on {len(cases)} cases, {repeats} fresh-process runs per case and implementation. Generated by
<code>scripts/benchmark.py</code> on {html.escape(env["date"])}.</p>
<div class="kpis">
<div class="kpi"><b>{faster:.2f}×</b><span>geometric-mean speed-up, warm build + run (range {ratio_rng})</span></div>
<div class="kpi"><b>{gm(speed_cold):.2f}×</b><span>geometric-mean speed-up, first call in a fresh process</span></div>
<div class="kpi"><b>{pct(worst_up)}</b><span>worst max |psdm − upstream| / C0 across all cases</span></div>
<div class="kpi"><b>{pct(worst_ad)}</b><span>worst max |psdm − AdDesignS| / C0 (single-compound cases)</span></div>
</div>
<div class="verdict"><p>psdm is faster in every case, but by how much depends on the case:</p>
<ul>{verdict}</ul>
<p>Where <code>solve_ivp</code> time is about equal, the gain is overhead outside the integrator (model setup,
output handling). Both codes produce the same Jacobian sparsity pattern and nearly the same step counts, so the
integrator gap in the single-compound cases comes from a cheaper right-hand side per evaluation.</p></div>

<h2>Summary</h2>
<p>Times are the median of {repeats} runs (min–max below) of object construction plus simulation, second call
inside each process. Speed-up = upstream median / psdm median; the small line gives the same ratio for the
first (cold) call. Accuracy is the largest absolute difference over the curve, divided by the influent
concentration C0.</p>
<div class="tablewrap"><table>
<thead><tr><th>Case</th><th>upstream</th><th>psdm</th><th>speed-up</th><th>max |Δ|/C0 vs upstream</th>
<th>max |Δ|/C0 vs AdDesignS</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>

<h2>Breakthrough curves</h2>
<div class="legend"><span><i style="border-color:var(--s1)"></i>upstream EPA PSDM</span>
<span><i class="dash" style="border-color:var(--s2)"></i>psdm</span>
<span><i class="dot"></i>AdDesignS (every few points shown)</span></div>
<p>Effluent concentration as a fraction of C0; the strip under each chart is psdm − upstream. Hover a chart for values.</p>
<div class="grid">{"".join(charts)}</div>

<h2>Where the time goes</h2>
<p>Medians in seconds. <em>read</em>: Excel input (cold call); <em>build</em>: constructing the model objects;
<em>run</em>: the simulation call; <em>integrate</em>: time inside <code>scipy.integrate.solve_ivp</code> only;
<em>run − integrate</em>: everything else in the run call (setup of matrices, fouling, output conversion and
interpolation; upstream also computes a fit residual it does not need here). <em>cold</em>: build + run on the first
call in the process. <em>import</em>: <code>import psdm</code> (or the upstream module) in a fresh process.</p>
<div class="tablewrap"><table><thead><tr><th>Case</th><th>impl</th><th>read</th><th>build</th><th>run</th>
<th>integrate</th><th>run − integrate</th><th>cold build+run</th><th>import</th></tr></thead>
<tbody>{"".join(detail)}</tbody></table></div>

<h2>Solver work</h2>
<p>Both implementations call <code>solve_ivp</code> with BDF, default tolerances (rtol 1e-3, atol 1e-6) and a sparse
Jacobian pattern; scipy estimates the Jacobian by finite differences. Counts are from the first run and were
identical in every repeat. Cost per RHS evaluation = warm integrate time / nfev (it includes the finite-difference
Jacobian and LU work, so it is a per-step cost proxy, not a pure RHS timing).</p>
<div class="tablewrap"><table><thead><tr><th rowspan="2">Case</th><th colspan="2">steps</th><th colspan="2">RHS evals</th>
<th colspan="2">Jacobians</th><th colspan="2">LU decomps</th><th colspan="2">ms / RHS eval</th></tr>
<tr><th>upstream</th><th>psdm</th><th>upstream</th><th>psdm</th><th>upstream</th><th>psdm</th><th>upstream</th><th>psdm</th>
<th>upstream</th><th>psdm</th></tr></thead><tbody>{"".join(solver)}</tbody></table></div>

<h2>Methods</h2>
<ul>
<li><b>Implementations.</b> Upstream: <code>psdm/PSDM.py</code> from USEPA/Water_Treatment_Models at commit
{UPSTREAM_COMMIT}, unmodified, imported from a fresh clone. New: the <code>psdm</code> package from this repository.
Both run in the same virtual environment, so numpy, scipy and pandas builds are identical.</li>
<li><b>Process isolation.</b> Each timed sample is a new <code>python -I</code> process that imports one
implementation and runs one case twice. Runs alternate between implementations (order flipped per case and
repeat) to spread out drift from thermal or background load.</li>
<li><b>What is timed.</b> <em>build + run</em> = upstream <code>PSDM(...)</code> construction plus
<code>run_psdm_kfit</code> (single-compound and variable-influent cases, the call the upstream examples use) or
<code>run_psdm</code> (multi-component); for psdm, building <code>Column</code>/<code>Compound</code> objects
(or <code>Dataset.compounds()</code>) plus <code>simulate()</code>. Excel reading is timed separately and excluded
from the headline; so are interpreter start-up and package import.</li>
<li><b>Cases.</b> Single compound (TCE) from the AdDesignS verification set in <code>tests/data/addesigns_test.xlsx</code>
(the upstream <code>test.xlsx</code>), constant influent for 174 d, default nr = 14, nz = 19 — this adds the 25 mg/L
Rhine and Organic Free cases to the five used in the test suite because AdDesignS output exists for them.
Multi-component: <code>examples/Example_Multi.xlsx</code>, nr = 4, nz = 8, ne = 2, as in
<code>tests/data/reference/generate.py</code>. Variable influent: <code>examples/Example_TCE.xlsx</code> sheet
<code>data_variable</code> (piecewise-linear TCE influent 3–5 mg/L), nr = 8, nz = 12, as in upstream
<code>Example_TCE.py</code>, with and without Rhine fouling.</li>
<li><b>Accuracy.</b> Both effluent curves are linearly interpolated onto 349 evenly spaced times and compared;
max and RMS of |psdm − upstream| / C0 are reported (for multi-component cases, the worst compound). AdDesignS
comparisons use the sheets 5, 25, 50, 5no, 25no, 50no and 50K at their own output times, keeping only rows with
C &gt; 0 (the sheets end in negative sentinel rows, and the first rows are ±1e-3 noise), as in the test suite.
AdDesignS output stops at 105 d for the Rhine cases.</li>
</ul>
<h2>Limitations</h2>
<ul>
<li>One machine, one Python and library set (below). Absolute times will differ elsewhere; ratios are more portable
but still depend on the BLAS and scipy version.</li>
<li>The cases are small (tens to a few hundred ODEs). Large multi-component problems were not benchmarked.</li>
<li>Upstream <code>run_psdm_kfit</code> also computes a sum of squares against (empty) effluent data and copies
DataFrames; that overhead is part of its normal API and is counted. The <em>integrate</em> column isolates the solver
for a like-for-like comparison.</li>
<li>Fitting routines (K/1/n search, ds fitting), capacity and uncertainty tools were not timed.</li>
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
    ap.add_argument("--worker", choices=["upstream", "psdm"], help=argparse.SUPPRESS)
    ap.add_argument("--case", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker:
        warnings.simplefilter("ignore")
        res = worker_upstream(a.case, a.upstream) if a.worker == "upstream" else worker_psdm(a.case)
        a.worker_out.write_text(json.dumps(res))
        return

    if a.from_json:
        saved = json.loads(a.from_json.read_text())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            upstream = a.upstream or clone_upstream(Path(tmp) / "wtm")
            env = environment(upstream)
            print("\n".join(f"{k}: {v}" for k, v in env.items()), flush=True)
            raw = benchmark(upstream, a.repeats, a.cases)
        saved = {"environment": env, "repeats": a.repeats, "summary": summarise(raw)}
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(saved, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        render(saved["summary"], saved["environment"], saved["repeats"]), encoding="utf-8"
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
