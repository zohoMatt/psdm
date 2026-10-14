import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA, EXAMPLES
from psdm.capacity import logistic
from psdm.io import read_excel

REF = DATA / "reference"


@pytest.fixture
def tce_column():
    return read_excel(EXAMPLES / "Example_TCE.xlsx").column


def case(name):
    cases = pd.read_csv(REF / "capacity_cases.csv")
    data = cases[cases["case"] == name].set_index("time")
    return data["influent"], data["effluent"]


@pytest.mark.parametrize(
    ("name", "method"),
    [
        ("breakthrough", "breakthrough"),
        ("variable", "breakthrough"),
        ("logistic", "logistic"),
        ("logistic_noisy", "logistic"),
        ("leakage", "linear, single point"),
        ("minimum", "minimum"),
    ],
)
@pytest.mark.parametrize("n_inv", [0.45, 0.7])
def test_matches_original_epa_code(name, method, n_inv, tce_column):
    expected = pd.read_csv(REF / "capacity.csv", index_col=[0, 1]).loc[(name, n_inv)]
    est = psdm.estimate_capacity(tce_column, *case(name), n_inv=n_inv)
    assert est.method == method
    assert est.breakthrough_time == expected["brk"]
    assert est.mean_influent == pytest.approx(expected["AveC"], rel=1e-9)
    assert est.q == pytest.approx(expected["q"], rel=1e-6)
    assert est.k == pytest.approx(expected["K"], rel=1e-6)


def test_k_for_keeps_loading(tce_column):
    est = psdm.estimate_capacity(tce_column, *case("logistic"), n_inv=0.45)
    assert est.k_for(0.45) == pytest.approx(est.k)
    k = est.k_for(np.array([0.3, 0.6]))
    np.testing.assert_allclose(k * est.mean_influent ** np.array([0.3, 0.6]), est.q)


def test_units(tce_column):
    infl, effl = case("breakthrough")
    days = psdm.estimate_capacity(tce_column, infl, effl)
    infl_h, effl_h = infl.copy(), effl.copy()
    infl_h.index = effl_h.index = infl.index * 24
    hours = psdm.estimate_capacity(tce_column, infl_h, effl_h, time_unit="hours")
    assert hours.breakthrough_time == pytest.approx(days.breakthrough_time * 24)
    assert hours.q == pytest.approx(days.q, rel=1e-6)
    in_mg = psdm.estimate_capacity(tce_column, infl / 1e3, effl / 1e3, conc_unit="mg/L")
    assert in_mg.q == pytest.approx(days.q, rel=1e-9)
    assert in_mg.mean_influent == pytest.approx(days.mean_influent / 1e3)


def test_linear_regression_fallback(tce_column):
    t = np.arange(0.0, 101.0, 5.0)
    infl = pd.Series(50.0, index=t)
    effl = pd.Series(np.where(t > 0, 5.0 + 0.02 * t, 0.0), index=t)
    est = psdm.estimate_capacity(tce_column, infl, effl)
    assert est.method == "linear regression"
    assert est.breakthrough_time > t[-1]


def test_mass_balance_of_simulated_breakthrough(addesigns_column, tce):
    """The capacity of a simulated full breakthrough is the Freundlich capacity at C0."""
    infl = pd.Series([50.0, 50.0], index=[0.0, 174.0])
    res = psdm.simulate(addesigns_column, tce, infl, conc_unit="mg/L", nr=8, nz=10)
    t = np.arange(0.0, 175.0)
    est = psdm.estimate_capacity(
        addesigns_column, pd.Series(50.0, index=t), pd.Series(res(t, "TCE"), index=t),
        conc_unit="mg/L", n_inv=tce.n_inv, resolution=0.1,
    )  # fmt: skip
    # the column holds q in the particles plus a little in the pores
    assert est.k == pytest.approx(tce.k, rel=0.05)


def test_logistic():
    assert logistic(0.0, 10.0, 0.0, 1.0) == pytest.approx(5.0)


def test_validation(tce_column):
    t = [0.0, 1.0]
    with pytest.raises(ValueError, match="no mass"):
        psdm.estimate_capacity(tce_column, pd.Series([0.0, 0.0], t), pd.Series([0.0, 0.0], t))
    with pytest.raises(ValueError, match="resolution"):
        psdm.estimate_capacity(
            tce_column, pd.Series([1.0, 1.0], t), pd.Series([0.0, 0.0], t), resolution=0
        )
