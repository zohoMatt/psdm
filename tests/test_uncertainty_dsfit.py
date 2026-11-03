import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA

REF = DATA / "reference"


def test_uncertainty_matches_original_epa_code(addesigns_column, tce):
    ref = pd.read_csv(REF / "uncertainty.csv", index_col=0)
    infl = pd.DataFrame({"TCE": [50.0, 50.0]}, index=[0.0, 174.0])
    u = psdm.model_uncertainty(
        addesigns_column, tce, infl, conc_unit="mg/L", capacity=0.1, k=0.1, mass=0.1,
        flow_rate=0.1, kf=0.2, dp=0.2, nr=8, nz=10, times=ref.index,
    )  # fmt: skip
    for got, col in [(u.effluent, "base"), (u.lower, "lower"), (u.upper, "upper")]:
        assert np.abs(got["TCE"].to_numpy() - ref[col].to_numpy()).max() < 5e-3 * 50
    assert len(u.runs) == 12
    assert (u.lower <= u.effluent).all().all() and (u.effluent <= u.upper).all().all()


def test_uncertainty_options(addesigns_column, tce):
    infl = pd.DataFrame({"TCE": [50.0, 50.0]}, index=[0.0, 100.0])
    pce = tce.with_(name="PCE", k=2 * tce.k)
    two = pd.DataFrame({"TCE": [50.0, 50.0], "PCE": [20.0, 20.0]}, index=[0.0, 100.0])
    kw = {"conc_unit": "mg/L", "nr": 6, "nz": 8, "capacity": None}
    u = psdm.model_uncertainty(addesigns_column, tce, infl, c0=0.1, q_n_inv=0.1, **kw)
    assert set(u.runs) == {"c0+10%", "c0-10%", "q_n_inv+10%", "q_n_inv-10%"}
    assert u.runs["c0+10%"]["TCE"].iloc[-1] == pytest.approx(55.0, rel=1e-3)
    multi = psdm.model_uncertainty(addesigns_column, [tce, pce], two, single=False, ds=0.5, **kw)
    assert list(multi.upper.columns) == ["TCE", "PCE"]
    nothing = psdm.model_uncertainty(addesigns_column, tce, infl, **kw)
    assert nothing.runs == {}
    pd.testing.assert_frame_equal(nothing.lower, nothing.effluent)
    with pytest.raises(ValueError, match="fraction"):
        psdm.model_uncertainty(addesigns_column, tce, infl, capacity=10)


def test_uncertainty_mass_lengthens_bed(addesigns_column):
    from psdm.uncertainty import _perturbed_column

    full = addesigns_column.with_(bed_density=addesigns_column.mass / addesigns_column.bed_volume)
    more = _perturbed_column(full, {"mass": 1.1})
    assert more.length == pytest.approx(1.1 * full.length)
    assert more.bed_porosity == pytest.approx(full.bed_porosity)
    less = _perturbed_column(full, {"mass": 0.9})
    assert less.length == full.length


@pytest.fixture
def ds_data():
    eff = pd.read_csv(REF / "dsfit_effluent.csv", index_col=0)["effluent"]
    return pd.Series([5.0, 5.0], index=[0.0, 174.0]), eff


def test_dsfit_matches_original_epa_code(addesigns_column, tce, ds_data):
    """With the upstream reference ds and window, the same grid point and sums of squares."""
    ref = pd.read_csv(REF / "dsfit.csv").iloc[0]
    ssqs = pd.read_csv(REF / "dsfit_ssqs.csv")
    fit = psdm.fit_surface_diffusivity(
        addesigns_column,
        tce,
        *ds_data,
        conc_unit="mg/L",
        ds_ref=ref["ds_base"],
        until=174,
        nr=8,
        nz=10,
    )
    assert fit.ds == pytest.approx(ref["best_ds"], rel=1e-9)
    assert fit.breakthrough_time == 174
    np.testing.assert_allclose(fit.evaluations.index, ssqs["factor"] * ref["ds_base"], rtol=1e-12)
    # the models differ only numerically (same equations, different solver settings)
    np.testing.assert_allclose(fit.evaluations.to_numpy(), ssqs["ssq"], rtol=0.02)
    assert fit.ssq == fit.evaluations.min()


def test_dsfit_default_grid_and_window(addesigns_column, tce, ds_data):
    """Default: EPA grid of 30 factors, window up to the estimated breakthrough time."""
    fit = psdm.fit_surface_diffusivity(addesigns_column, tce, *ds_data, conc_unit="mg/L")
    np.testing.assert_allclose(fit.evaluations.index / fit.ds_ref, np.linspace(1e-10, 1, 30))
    _, eff = ds_data
    est = psdm.estimate_capacity(addesigns_column, pd.Series(5.0, eff.index), eff, conc_unit="mg/L")
    assert fit.breakthrough_time == est.breakthrough_time
    assert fit.compound.ds == fit.ds
    assert fit.ssq == fit.evaluations.min()
    assert fit.ds == fit.evaluations.idxmin()


def test_dsfit_rejects_fouled_water(addesigns_column, tce, ds_data):
    with pytest.raises(ValueError, match="fouled"):
        psdm.fit_surface_diffusivity(addesigns_column, tce, *ds_data, water="Rhine")
