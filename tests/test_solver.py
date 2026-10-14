import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA, EXAMPLES
from psdm.io import read_excel

SINGLE_CASES = {
    # reference column: (C0 mg/L, water, fouling class, AdDesignS sheet)
    "rhine5": (5.0, "Rhine", "chlorinated hydrocarbon", "5"),
    "rhine50": (50.0, "Rhine", "chlorinated hydrocarbon", "50"),
    "nofoul5": (5.0, "Organic Free", "halogenated alkenes", "5no"),
    "nofoul50": (50.0, "Organic Free", "halogenated alkenes", "50no"),
    "portage50": (50.0, "Portage", "chlorinated hydrocarbon", "50K"),
}


def run_single(column, compound, c0, water, cls):
    influent = pd.DataFrame({"TCE": [c0, c0]}, index=[0.0, 174.0])
    return psdm.simulate(
        column, compound.with_(fouling_class=cls), influent, water=water, conc_unit="mg/L"
    )


@pytest.mark.parametrize("case", SINGLE_CASES)
def test_matches_original_epa_code(case, addesigns_column, tce):
    c0, water, cls, _ = SINGLE_CASES[case]
    ref = pd.read_csv(DATA / "reference" / "addesigns_single.csv", index_col=0)
    result = run_single(addesigns_column, tce, c0, water, cls)
    expected = ref[case].to_numpy() / 1e6  # reference is in ng/L
    assert np.abs(result(ref.index, "TCE") - expected).max() < 5e-3 * c0


@pytest.mark.parametrize("case", SINGLE_CASES)
def test_matches_addesigns(case, addesigns_column, tce):
    c0, water, cls, sheet = SINGLE_CASES[case]
    out = pd.read_excel(DATA / "addesigns_test.xlsx", sheet_name=sheet, index_col=0)["C, mg/L"]
    out = out[out > 0]
    result = run_single(addesigns_column, tce, c0, water, cls)
    assert np.abs(result(out.index / 1440, "TCE") - out.to_numpy()).max() < 5e-3 * c0


@pytest.mark.parametrize(
    ("water", "cls", "ref"),
    [
        ("Organic Free", "halogenated alkenes", "multi_organic_free"),
        ("Rhine", "halogenated alkanes", "multi_rhine"),
    ],
)
def test_multicomponent_matches_original(water, cls, ref):
    ds = read_excel(EXAMPLES / "Example_Multi.xlsx")
    expected = pd.read_csv(DATA / "reference" / f"{ref}.csv", index_col=0)
    result = ds.simulate(ds.compounds(fouling_class=cls), water=water, nr=4, nz=8, ne=2)
    diff = (result(expected.index.to_numpy()) - expected).abs().max()
    assert (diff < 5e-3 * 200).all(), diff


def test_result_shape_and_units(addesigns_column, tce):
    infl = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    ug = psdm.simulate(addesigns_column, tce, infl * 1000, conc_unit="ug/L", nr=8, nz=10)
    mg = psdm.simulate(addesigns_column, tce, infl, conc_unit="mg/L", nr=8, nz=10)
    assert list(ug.effluent.columns) == ["TCE"]
    assert ug.effluent.index[0] == 0 and ug.effluent.index[-1] == pytest.approx(100)
    np.testing.assert_allclose(
        ug(np.arange(100), "TCE") / 1000, mg(np.arange(100), "TCE"), rtol=1e-6, atol=1e-6
    )
    assert set(ug.mass_transfer.index) == {"kf", "dp", "ds"}


def test_hours_equal_days(addesigns_column, tce):
    days = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    hours = pd.Series([50.0, 50.0], index=[0.0, 2400.0])
    rd = psdm.simulate(addesigns_column, tce, days, conc_unit="mg/L", nr=8, nz=10)
    rh = psdm.simulate(
        addesigns_column, tce, hours, conc_unit="mg/L", time_unit="hours", nr=8, nz=10
    )
    np.testing.assert_allclose(rd(np.arange(100), "TCE"), rh(np.arange(100) * 24, "TCE"), atol=1e-3)


def test_mass_transfer_overrides(addesigns_column, tce):
    infl = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    r = psdm.simulate(
        addesigns_column, tce.with_(kf=1e-3, dp=2e-6, ds=3e-10), infl, conc_unit="mg/L", nr=6, nz=8
    )
    assert r.mass_transfer["TCE"].to_dict() == pytest.approx({"kf": 1e-3, "dp": 2e-6, "ds": 3e-10})


def test_more_capacity_delays_breakthrough(addesigns_column, tce):
    infl = pd.Series([50.0, 50.0], index=[0.0, 174.0])
    lo = psdm.simulate(addesigns_column, tce, infl, conc_unit="mg/L", nr=8, nz=10)
    hi = psdm.simulate(
        addesigns_column, tce.with_(k=2 * tce.k), infl, conc_unit="mg/L", nr=8, nz=10
    )
    t = np.arange(174.0)
    # small collocation wiggles at the front are expected; compare at 0.1 % of C0
    assert (hi(t, "TCE") <= lo(t, "TCE") + 0.05).all()
    half_lo, half_hi = (t[r(t, "TCE") > 25][0] for r in (lo, hi))
    assert half_hi > 1.5 * half_lo


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"water": "Lake"}, "water type"),
        ({"correlation": "nope"}, "correlation"),
        ({"conc_unit": "furlongs"}, "concentration unit"),
    ],
)
def test_bad_options(kwargs, match, addesigns_column, tce):
    infl = pd.Series([1.0, 1.0], index=[0.0, 10.0])
    with pytest.raises(ValueError, match=match):
        psdm.simulate(addesigns_column, tce, infl, **kwargs)


def test_influent_validation(addesigns_column, tce):
    with pytest.raises(ValueError, match="time 0"):
        psdm.simulate(addesigns_column, tce, pd.Series([1.0, 1.0], index=[1.0, 10.0]))
    with pytest.raises(ValueError, match="no data"):
        psdm.simulate(addesigns_column, tce, pd.DataFrame({"PCE": [1.0, 1.0]}, index=[0, 1]))
    with pytest.raises(ValueError, match="duplicate"):
        psdm.simulate(addesigns_column, [tce, tce], pd.DataFrame({"TCE": [1.0, 1.0]}))
