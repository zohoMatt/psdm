import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import EXAMPLES
from psdm.io import read_excel


def test_read_excel_converts_units():
    ds = read_excel(EXAMPLES / "Example_TCE.xlsx")
    c = ds.column
    assert c.name == "F400"
    assert c.length == pytest.approx(276.5)
    assert c.diameter == pytest.approx(304.8)
    assert c.mass == pytest.approx(20000 * 453.59237)
    assert c.flow_rate == pytest.approx(566.966 * 3785.411784)
    assert ds.compound_names == ["Trichloroethylene"]
    assert ds.influent.index[0] == 0
    (tce,) = ds.compounds()
    assert (tce.k, tce.n_inv, tce.mw) == (5026.04, 0.43, 131.39)


def test_read_excel_fills_gaps():
    ds = read_excel(EXAMPLES / "Example_TCE.xlsx", data_sheet="data_optimize")
    assert ds.influent.index.equals(ds.effluent.index)
    assert not ds.influent.isna().any().any() and not ds.effluent.isna().any().any()
    assert ds.influent.loc[10.0, "Trichloroethylene"] == pytest.approx(1500 - 250 * 10 / 25)


def test_compound_overrides():
    ds = read_excel(EXAMPLES / "Example_TCE.xlsx", kdata_sheet=None)
    with pytest.raises(ValueError, match="no K"):
        ds.compounds()
    (tce,) = ds.compounds(Trichloroethylene={"k": 100.0, "n_inv": 0.5})
    assert tce.k == 100.0


@pytest.mark.parametrize(
    ("model", "func", "true"),
    [
        ("freundlich", psdm.isotherms.freundlich, (12.0, 0.45)),
        ("langmuir", psdm.isotherms.langmuir, (0.2, 40.0)),
        ("Redlich Peterson", psdm.isotherms.redlich_peterson, (5.0, 2.0, 0.8)),
    ],
)
def test_isotherm_fit_recovers_parameters(model, func, true):
    ce = np.geomspace(0.1, 100, 12)
    fit = psdm.fit_isotherm(ce, func(ce, *true), model)
    np.testing.assert_allclose(list(fit.params.values()), true, rtol=1e-4)
    assert fit.r2 == pytest.approx(1.0)
    conf, pred = fit.bands(ce)
    assert np.all(pred >= conf)


def test_isotherm_from_batch_data():
    q = psdm.isotherms.batch_loading([100, 100], [33, 20.5], [168, 265], volume=1.0)
    np.testing.assert_allclose(q, [67 / 168, 79.5 / 265])
    with pytest.raises(ValueError, match="unknown isotherm"):
        psdm.fit_isotherm([1, 2, 3], [1, 2, 3], "bet")


def test_fit_freundlich_recovers_synthetic_parameters(addesigns_column):
    true = psdm.Compound("TCE", mw=131.39, molar_volume=102, k=5026.04, n_inv=0.43)
    influent = pd.Series([50.0, 50.0], index=[0.0, 174.0])
    kw = {"conc_unit": "mg/L", "nr": 6, "nz": 8}
    observed = psdm.simulate(addesigns_column, true, influent, **kw)
    t_obs = np.arange(0, 175, 7.0)
    effluent = pd.Series(observed(t_obs, "TCE"), index=t_obs)

    start = true.with_(k=true.k * 0.6, n_inv=0.55)
    fit = psdm.fit_freundlich(addesigns_column, start, influent, effluent, max_evaluations=60, **kw)
    assert fit.ssq < 1e-2 * psdm.sum_of_squares(
        psdm.simulate(addesigns_column, start, influent, **kw), effluent
    )
    # with a constant influent only the capacity at that concentration is identifiable
    c_ref = 50_000.0  # ug/L
    q_fit = fit.compound.k * c_ref**fit.compound.n_inv
    assert q_fit == pytest.approx(true.k * c_ref**true.n_inv, rel=0.03)
    assert fit.compound.n_inv == pytest.approx(0.43, abs=0.05)
    assert len(fit.evaluations) > 1


def test_plot_breakthrough(addesigns_column, tce):
    import matplotlib

    matplotlib.use("Agg")
    from psdm.plotting import plot_breakthrough

    influent = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    result = psdm.simulate(addesigns_column, tce, influent, conc_unit="mg/L", nr=6, nz=8)
    ax = plot_breakthrough(result, influent=influent, effluent=influent * 0.5)
    assert len(ax.lines) == 3
