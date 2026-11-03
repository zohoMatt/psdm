"""Tests for the 0.2 additions: errors, diagnostics, fit callbacks, split uncertainty/throughput."""

import itertools

import numpy as np
import pandas as pd
import pytest

import psdm
from psdm import errors, fullscale, uncertainty
from psdm.fitting import surface_diffusivity_max_evaluations

KW = {"conc_unit": "mg/L", "nr": 6, "nz": 8}


@pytest.fixture
def influent():
    return pd.Series([50.0, 50.0], index=[0.0, 174.0])


@pytest.fixture
def observed(addesigns_column, tce, influent):
    res = psdm.simulate(addesigns_column, tce, influent, **KW)
    t = np.arange(0, 175, 7.0)
    return pd.Series(res(t, "TCE"), index=t)


# --- errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("compounds", "infl", "field"),
    [
        ("none", pd.DataFrame({"TCE": [1.0, 1.0]}), "compounds"),
        ("dup", pd.DataFrame({"TCE": [1.0, 1.0]}), "compounds"),
        ("one", pd.DataFrame({"PCE": [1.0, 1.0]}), "influent"),
        ("one", pd.Series([1.0, 1.0], index=[1.0, 2.0]), "influent"),
        ("one", pd.Series([1.0], index=[0.0]), "influent"),
        ("two", pd.Series([1.0, 1.0], index=[0.0, 1.0]), "influent"),
    ],
)
def test_input_errors_carry_field(compounds, infl, field, addesigns_column, tce):
    comps = {"none": [], "dup": [tce, tce], "one": [tce], "two": [tce, tce.with_(name="PCE")]}[
        compounds
    ]
    with pytest.raises(errors.InputError) as info:
        psdm.simulate(addesigns_column, comps, infl)
    assert info.value.field == field
    assert isinstance(info.value, ValueError)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solver_failure_is_specific_and_backward_compatible(addesigns_column, tce, influent):
    # an absurdly tight tolerance makes the Newton matrix of the BDF step singular
    tight = {"rtol": 1e-13, "atol": 1e-300}
    with pytest.raises(errors.SolverDivergedError):
        psdm.simulate(addesigns_column, tce, influent, **tight, **KW)
    with pytest.raises(RuntimeError):  # pre-0.2 handlers still catch it
        psdm.simulate(addesigns_column, tce, influent, **tight, **KW)


def test_fits_treat_solver_errors_as_failed_trials(
    addesigns_column, tce, influent, observed, monkeypatch
):
    # failed trials score 1e20, like upstream run_all_smart, and do not abort the search
    real = psdm.fitting.simulate

    def failing(column, compound, *args, **kw):
        if compound.n_inv >= 0.8:
            raise errors.SolverDivergedError("diverged")
        return real(column, compound, *args, **kw)

    monkeypatch.setattr(psdm.fitting, "simulate", failing)
    fit = psdm.fit_freundlich(
        addesigns_column, tce, influent, observed, start="compound", until=174, search_limit=2,
        **KW,
    )  # fmt: skip
    failed = fit.evaluations[fit.evaluations["n_inv"] >= 0.8]
    assert len(failed) and (failed["ssq"] == 1e20).all()
    assert np.isfinite(fit.ssq) and fit.compound.n_inv < 0.8


def test_until_is_only_for_given_k(addesigns_column, tce, influent, observed):
    with pytest.raises(ValueError, match="until is only for"):
        psdm.fit_freundlich(addesigns_column, tce, influent, observed, until=100.0, **KW)


def test_nelder_mead_is_gone(addesigns_column, tce, influent, observed):
    with pytest.raises(ValueError, match="unknown method"):
        psdm.fit_freundlich(addesigns_column, tce, influent, observed, method="nelder-mead")


# --- diagnostics --------------------------------------------------------------------------


def _expected_groups(column, compound, c0_ug, kf, dp, ds):
    vw, dw = psdm.properties.water_viscosity(20), psdm.properties.water_density(20)
    ebed, tau, rad = column.bed_porosity, column.residence_time, column.particle_radius
    cb0 = c0_ug / compound.mw  # umol/L
    molar_k = compound.k / compound.mw / (1 / compound.mw) ** compound.n_inv
    qe = molar_k * cb0**compound.n_inv
    dgs = column.particle_density * qe * (1 - ebed) * 1000 / (ebed * cb0)
    dgp = column.particle_porosity * (1 - ebed) / ebed
    assert vw > 0 and dw > 0
    st = kf * (1 - ebed) * tau / (ebed * rad)
    eds = max(ds * dgs * tau / rad**2, 1e-130)
    edp = dp * dgp * tau / rad**2
    return {"St": st, "Eds": eds, "Edp": edp, "Bi_s": st / eds, "Bi_p": st / edp}


@pytest.mark.parametrize("water", ["Organic Free", "Rhine"])
def test_diagnostics_match_independent_formulas(water, addesigns_column, tce, influent):
    res = psdm.simulate(addesigns_column, tce, influent, water=water, **KW)
    mt = res.mass_transfer["TCE"]
    expected = _expected_groups(addesigns_column, tce, 50_000, mt["kf"], mt["dp"], mt["ds"])
    diag = res.diagnostics["TCE"]
    assert list(res.diagnostics.index) == ["St", "Eds", "Edp", "Bi_s", "Bi_p"]
    for key, value in expected.items():
        assert diag[key] == pytest.approx(value, rel=1e-12), key
    assert np.isfinite(diag).all() and (diag > 0).all()
    if water != "Organic Free":
        assert diag["Eds"] < 1e-9  # surface diffusion switched off


def test_diagnostics_are_unit_independent(addesigns_column, tce):
    days = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    a = psdm.simulate(addesigns_column, tce, days, conc_unit="mg/L", nr=6, nz=8)
    b = psdm.simulate(
        addesigns_column,
        tce,
        pd.Series([50_000.0, 50_000.0], index=[0.0, 100 * 1440.0]),
        conc_unit="ug/L",
        time_unit="min",
        nr=6,
        nz=8,
    )
    pd.testing.assert_frame_equal(a.diagnostics, b.diagnostics, rtol=1e-12)


def test_diagnostics_multicomponent(addesigns_column, tce):
    pce = psdm.Compound("PCE", mw=165.83, molar_volume=119, k=10393.4, n_inv=0.458)
    infl = pd.DataFrame({"TCE": [5.0, 5.0], "PCE": [2.0, 2.0]}, index=[0.0, 50.0])
    res = psdm.simulate(addesigns_column, [tce, pce], infl, **KW)
    assert list(res.diagnostics.columns) == ["TCE", "PCE"]
    assert (res.diagnostics > 0).all().all()


# --- fit callbacks ------------------------------------------------------------------------


class _Recorder:
    def __init__(self, stop_after=None):
        self.calls = []
        self.stop_after = stop_after

    def __call__(self, evaluations, best_ssq):
        self.calls.append((evaluations, best_ssq))
        if self.stop_after is not None and evaluations >= self.stop_after:
            raise _Cancelled


class _Cancelled(Exception):
    pass


def _assert_monotone(calls):
    ev = [c[0] for c in calls]
    best = [c[1] for c in calls]
    assert ev == list(range(1, len(calls) + 1))
    assert all(b2 <= b1 for b1, b2 in itertools.pairwise(best))


def test_freundlich_callback(addesigns_column, tce, influent, observed):
    rec = _Recorder()
    start = tce.with_(k=3000.0, n_inv=0.5)
    fit = psdm.fit_freundlich(
        addesigns_column,
        start,
        influent,
        observed,
        start="compound",
        until=174,
        search_limit=2,
        callback=rec,
        **KW,
    )
    assert len(rec.calls) == len(fit.evaluations)
    _assert_monotone(rec.calls)
    assert rec.calls[-1][1] == pytest.approx(fit.evaluations["ssq"].min())


def test_ds_callback_and_bound(addesigns_column, tce, influent, observed):
    rec = _Recorder()
    fit = psdm.fit_surface_diffusivity(
        addesigns_column,
        tce,
        influent,
        observed,
        factors=np.linspace(0.1, 1, 6),
        callback=rec,
        **KW,
    )
    assert len(rec.calls) == len(fit.evaluations)
    _assert_monotone(rec.calls)
    assert len(rec.calls) == surface_diffusivity_max_evaluations(np.linspace(0.1, 1, 6)) == 6
    assert surface_diffusivity_max_evaluations() == 30


@pytest.mark.parametrize("which", ["freundlich", "ds"])
def test_callback_cancels(which, addesigns_column, tce, influent, observed, monkeypatch):
    sims = []
    real = psdm.fitting.simulate

    def counting(*args, **kw):
        sims.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(psdm.fitting, "simulate", counting)
    rec = _Recorder(stop_after=3)
    with pytest.raises(_Cancelled):
        if which == "freundlich":
            psdm.fit_freundlich(addesigns_column, tce, influent, observed, callback=rec, **KW)
        else:
            psdm.fit_surface_diffusivity(
                addesigns_column, tce, influent, observed, ds_ref=1e-9, callback=rec, **KW
            )
    assert len(rec.calls) == 3
    assert len(sims) == 3  # nothing ran after the cancelling callback


# --- uncertainty plan / envelope ----------------------------------------------------------


@pytest.mark.parametrize("single", [True, False])
def test_plan_envelope_equals_model_uncertainty(single, addesigns_column, tce, monkeypatch):
    pce = psdm.Compound("PCE", mw=165.83, molar_volume=119, k=10393.4, n_inv=0.458)
    infl = pd.DataFrame({"TCE": [50.0, 50.0], "PCE": [20.0, 20.0]}, index=[0.0, 174.0])
    opts = {"capacity": 0.1, "flow_rate": 0.1, "mass": 0.1, "single": single, "water": "Rhine"}
    direct = psdm.model_uncertainty(addesigns_column, [tce, pce], infl, **opts, **KW)

    calls = []
    real = uncertainty.simulate
    monkeypatch.setattr(uncertainty, "simulate", lambda *a, **k: calls.append(1) or real(*a, **k))
    plan = uncertainty.plan(addesigns_column, [tce, pce], infl, **opts, **KW)
    assert len(calls) == (2 if single else 1)  # only the base case
    assert len(plan.runs) == 6 * (2 if single else 1)

    split = uncertainty.envelope(plan, [run.simulate() for run in plan.runs])
    for attr in ("effluent", "lower", "upper"):
        pd.testing.assert_frame_equal(getattr(split, attr), getattr(direct, attr), check_exact=True)
    assert list(split.runs) == list(direct.runs)
    for label in direct.runs:
        pd.testing.assert_frame_equal(split.runs[label], direct.runs[label], check_exact=True)


def test_envelope_checks_length(addesigns_column, tce, influent):
    plan = uncertainty.plan(addesigns_column, tce, influent, **KW)
    with pytest.raises(ValueError, match="expected 2 results"):
        uncertainty.envelope(plan, [])


# --- specific throughput split ------------------------------------------------------------


def test_throughput_from_result_matches_specific_throughput(addesigns_column, tce):
    ebcts = [5.0, 10.0, 20.0]
    table = fullscale.specific_throughput(
        addesigns_column, tce, c0=50.0, threshold=5.0, ebcts=ebcts, duration=300.0, **KW
    )
    infl = pd.DataFrame({"TCE": [50.0, 50.0]}, index=[0.0, 300.0])
    for ebct in ebcts:
        col = fullscale.rescaled_column(addesigns_column, ebct)
        assert col.ebct == pytest.approx(ebct)
        res = psdm.simulate(col, tce, infl, **KW)
        value = fullscale.throughput_from_result(res, "TCE", 5.0)
        assert value == table.loc[ebct, "TCE"]


def test_step_size_failure_is_classified(addesigns_column, tce, influent, monkeypatch):
    # No physical case reliably triggers it, so replay scipy's own BDF failure message.
    from types import SimpleNamespace

    import psdm.solver

    msg = "Required step size is less than spacing between numbers."
    monkeypatch.setattr(
        psdm.solver, "solve_ivp", lambda *a, **k: SimpleNamespace(success=False, message=msg)
    )
    with pytest.raises(errors.StepSizeTooSmallError):
        psdm.simulate(addesigns_column, tce, influent, **KW)


def test_runs_are_picklable(addesigns_column, tce, influent):
    import pickle

    plan = uncertainty.plan(addesigns_column, tce, influent, **KW)
    run = pickle.loads(pickle.dumps(plan.runs[0]))
    expected = plan.runs[0].simulate().effluent
    pd.testing.assert_frame_equal(run.simulate().effluent, expected, check_exact=True)
    assert pickle.loads(pickle.dumps(uncertainty.Run.simulate)) is uncertainty.Run.simulate
