"""Tests for typed errors and diagnostics."""

import numpy as np
import pandas as pd
import pytest

import psdm
from psdm import errors

KW = {"conc_unit": "mg/L", "nr": 6, "nz": 8}


@pytest.fixture
def influent():
    return pd.Series([50.0, 50.0], index=[0.0, 174.0])


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
