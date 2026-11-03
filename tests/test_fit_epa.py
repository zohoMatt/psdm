"""fit_freundlich(method="epa") against the US EPA PSDM.run_all_smart (reference data from
tests/data/reference/generate_fit.py)."""

import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA, EXAMPLES
from psdm.fitting import _fit_epa_smart
from psdm.io import read_excel

REF = DATA / "reference"
NAME = "Trichloroethylene"
CASES = {
    "capacity_organic_free": {"start": "capacity"},
    "capacity_rhine": {"start": "capacity", "water": "Rhine"},
    "given_organic_free": {"start": "compound", "until": 200.0},
}


@pytest.fixture(scope="module")
def data():
    ds = read_excel(EXAMPLES / "Example_TCE.xlsx", data_sheet="data_optimize")
    (tce,) = ds.compounds()
    return ds.column, tce, ds.influent[NAME], ds.effluent[NAME]


def _calls(case):
    calls = pd.read_csv(REF / "fit_smart_calls.csv")
    return calls[calls["case"] == case].reset_index(drop=True)


@pytest.mark.parametrize("case", CASES)
def test_search_replays_epa_path(case, data):
    """Fed the EPA code's own sums of squares, the port visits the same nodes in the same
    order and returns the same K and 1/n: the search logic is equivalent."""
    calls = _calls(case)
    expected = pd.read_csv(REF / "fit_smart.csv", index_col=0).loc[case]

    def replay(k, n_inv):
        hit = calls[np.isclose(calls["k"], k, rtol=1e-7) & np.isclose(calls["n_inv"], n_inv)]
        assert len(hit), f"the port tried K={k}, 1/n={n_inv}, which the EPA code never did"
        return float(hit["ssq"].iloc[0])

    kw = {"pm": 10.0, "num": 11, "des_xn": 0.025, "search_limit": 50, "until": None}
    fit = _fit_epa_smart(*data, **{**kw, **CASES[case]}, callback=None, _ssq_fn=replay)
    ev = fit.evaluations
    assert len(ev) == len(calls)
    np.testing.assert_allclose(ev["k"], calls["k"], rtol=1e-7)
    np.testing.assert_array_equal(ev["n_inv"], calls["n_inv"])
    assert fit.compound.k == pytest.approx(expected["k"], rel=1e-7)
    assert fit.compound.n_inv == expected["n_inv"]
    assert fit.breakthrough_time == expected["brk"]


@pytest.mark.parametrize("case", ["capacity_organic_free", "capacity_rhine"])
def test_end_to_end_matches_epa(case, data):
    """With psdm's own simulations the result is the EPA result."""
    expected = pd.read_csv(REF / "fit_smart.csv", index_col=0).loc[case]
    calls = _calls(case)
    fit = psdm.fit_freundlich(*data, **CASES[case])
    assert fit.method == "epa"
    assert fit.compound.k == pytest.approx(expected["k"], rel=1e-7)
    assert fit.compound.n_inv == expected["n_inv"]
    assert len(fit.evaluations) == len(calls)
    np.testing.assert_allclose(fit.evaluations["k"], calls["k"], rtol=1e-7)
    # the sums of squares differ only by the ~0.3 % of C0 model differences
    np.testing.assert_allclose(fit.evaluations["ssq"], calls["ssq"], rtol=0.03)
    assert fit.capacity.breakthrough_time == expected["brk"]


def test_end_to_end_given_k(data):
    """start="compound" (EPA with k_data). The window runs 25 days past the data, where both
    codes extrapolate their own solver output; the resulting few-% ssq differences flip one
    comparison after 25 identical steps, so the end point is one grid node apart."""
    expected = pd.read_csv(REF / "fit_smart.csv", index_col=0).loc["given_organic_free"]
    calls = _calls("given_organic_free")
    fit = psdm.fit_freundlich(*data, **CASES["given_organic_free"])
    n = 25
    np.testing.assert_allclose(fit.evaluations["k"][:n], calls["k"][:n], rtol=1e-12)
    np.testing.assert_array_equal(fit.evaluations["n_inv"][:n], calls["n_inv"][:n])
    assert fit.compound.n_inv == expected["n_inv"]
    assert fit.compound.k == pytest.approx(expected["k"], rel=0.02)


def test_epa_fit_grid_and_callback(data):
    rec = []
    fit = psdm.fit_freundlich(*data, callback=lambda n, best: rec.append((n, best)))
    assert [r[0] for r in rec] == list(range(1, len(fit.evaluations) + 1))
    grid = fit.ssq_grid
    assert grid.index.name == "k_factor" and grid.columns.name == "n_inv"
    assert grid.notna().sum().sum() == len(fit.evaluations)
    assert fit.ssq == pytest.approx(np.nanmin(grid.to_numpy()))


def test_epa_fit_validation(data):
    with pytest.raises(ValueError, match="needs until"):
        psdm.fit_freundlich(*data, start="compound")
    with pytest.raises(ValueError, match="start must be"):
        psdm.fit_freundlich(*data, start="guess")
    with pytest.raises(ValueError, match="unknown method"):
        psdm.fit_freundlich(*data, method="grid")


def test_bed_density_warning():
    kw = {
        "length": 100.0,
        "diameter": 10.0,
        "mass": 3534.3,  # 0.45 g/mL packed
        "flow_rate": 100.0,
        "particle_radius": 0.05,
        "particle_density": 0.8,
        "particle_porosity": 0.6,
    }
    with pytest.warns(UserWarning, match="bed_density"):
        psdm.Column(**kw, bed_density=0.62)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        psdm.Column(**kw, bed_density=0.45)
        psdm.Column(**kw)
