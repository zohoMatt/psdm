import numpy as np
import pandas as pd
import pytest

import psdm
from conftest import DATA, EXAMPLES
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


@pytest.mark.parametrize(
    ("path", "sheet"),
    [
        (EXAMPLES / "Example_TCE.xlsx", "data"),
        (EXAMPLES / "Example_TCE.xlsx", "data_variable"),
        (EXAMPLES / "Example_TCE.xlsx", "data_optimize"),
        (EXAMPLES / "Example_Multi.xlsx", "data"),
        (DATA / "io_missing_t0.xlsx", "data"),
        (DATA / "io_missing_t0.xlsx", "data_infl_t0"),
    ],
)
def test_read_excel_matches_original_epa_reader(path, sheet):
    """Same influent and effluent as upstream ``process_input_file`` (generate_io.py).

    ``io_missing_t0.xlsx`` has no t = 0 row (``data``; upstream adds a row of zeros) or only
    an influent one (``data_infl_t0``; the effluent at t = 0 stays empty), and zero influent
    values (upstream sets them to 0.001).
    """
    ref = pd.read_csv(DATA / "reference" / "io_process_input_file.csv")
    ref = ref[(ref["file"] == path.name) & (ref["sheet"] == sheet)]
    ds = read_excel(path, data_sheet=sheet)
    for kind, got in [("influent", ds.influent), ("effluent", ds.effluent)]:
        want = ref[ref["type"] == kind].pivot(index="time", columns="compound")["concentration"]
        want = want.reindex(columns=got.columns)
        np.testing.assert_array_equal(got.index.to_numpy(), want.index.to_numpy())
        np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=1e-12)


def test_compound_overrides():
    ds = read_excel(EXAMPLES / "Example_TCE.xlsx", kdata_sheet=None)
    with pytest.raises(ValueError, match="no K"):
        ds.compounds()
    (tce,) = ds.compounds(Trichloroethylene={"k": 100.0, "n_inv": 0.5})
    assert tce.k == 100.0


def test_plot_breakthrough(addesigns_column, tce):
    import matplotlib

    matplotlib.use("Agg")
    from psdm.plotting import plot_breakthrough

    influent = pd.Series([50.0, 50.0], index=[0.0, 100.0])
    result = psdm.simulate(addesigns_column, tce, influent, conc_unit="mg/L", nr=6, nz=8)
    ax = plot_breakthrough(result, influent=influent, effluent=influent * 0.5)
    assert len(ax.lines) == 3
