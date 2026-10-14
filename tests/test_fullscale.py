import numpy as np
import pandas as pd
import pytest

import psdm
from psdm import fullscale as fs


@pytest.fixture
def carbon():
    return psdm.Column(
        length=83.3, diameter=5.1, mass=833.8, flow_rate=170.1, particle_radius=0.05,
        particle_density=0.8, particle_porosity=0.6, bed_density=0.49,
    )  # fmt: skip


def test_mesh_diameter():
    assert fs.mesh_diameter("12x40") == pytest.approx(1.05)
    assert fs.mesh_diameter("140 x 200") == pytest.approx(0.0895)
    with pytest.raises(ValueError, match="mesh"):
        fs.mesh_diameter("13x40")


def rssct_values(sc):
    c = sc.column
    return [c.diameter, c.length, 20 * c.particle_radius, sc.ebct, sc.duration,
            sc.loading_rate, c.flow_rate, c.mass, sc.water_volume]  # fmt: skip


@pytest.mark.parametrize(
    ("large", "small", "expected"),
    [
        (  # MWH (Crittenden et al.) example 15-14, test values of the EPA PSDM_extra.py
            {"length": (83.3, "cm"), "diameter": (5.1, "cm"), "flow_rate": (170.1, "ml/min"),
             "particle_diameter": (1.0, "mm"), "bed_density": 0.49},
            {"particle_diameter": (0.21, "mm"), "operation_time": 100},
            [1.1, 17.493, 0.21, 0.4411735945442268, 4.41, 23.790635091937293,
             37.68166089965399, 8.145835355873526, 239.29361937716263],
        ),
        (  # example 15-15
            {"length": (71.4, "cm"), "diameter": (40.64, "cm"), "flow_rate": (18.9016, "l/min"),
             "particle_diameter": (1.62, "mm"), "mass": 40751.92, "bed_density": 0.44},
            {"particle_diameter": (0.212, "mm"), "operation_time": 288},
            [1.1, 9.343703703703705, 0.212, 0.0839148101156238, 4.932126200274347,
             66.8084955980663, 105.8170606465034, 3.9070321625037034, 751.5404600409199],
        ),
    ],
)  # fmt: skip
def test_rssct_matches_original_epa_code(carbon, large, small, expected):
    lc = fs.build_large_scale_column(carbon, **large)
    sc = fs.rssct_design(lc, column_diameter=(1.10, "cm"), **small)
    np.testing.assert_allclose(rssct_values(sc), expected, rtol=1e-9)
    assert sc.column.bed_porosity == pytest.approx(lc.bed_porosity)


def test_rssct_proportional_diffusivity(carbon):
    sc = fs.rssct_design(carbon, column_diameter=1.1, mesh="100x200", power=1.0)
    ratio = (0.1115 / 2 / 10) / carbon.particle_radius
    assert sc.scale_factor == pytest.approx(ratio)
    assert sc.ebct == pytest.approx(carbon.ebct * ratio)
    assert sc.duration is None and sc.water_volume is None
    with pytest.raises(ValueError, match="particle_diameter or mesh"):
        fs.rssct_design(carbon, column_diameter=1.1)


def test_build_large_scale_column(carbon):
    lc = fs.build_large_scale_column(
        carbon, length=(6, "ft"), diameter=(10, "ft"), loading_rate=(5, "gpm/ft2"), mesh="12x40"
    )
    assert lc.loading_rate == pytest.approx(5.0)
    assert lc.particle_radius == pytest.approx(0.0525)
    assert lc.mass / lc.bed_volume == pytest.approx(0.49)
    assert lc.particle_density == carbon.particle_density
    m_h = fs.build_large_scale_column(carbon, length=100, diameter=100, loading_rate=10.0)
    assert m_h.superficial_velocity * 36 == pytest.approx(10.0)
    with pytest.raises(ValueError, match="exactly one"):
        fs.build_large_scale_column(carbon, length=100, diameter=100)
    with pytest.raises(ValueError, match="loading rate unit"):
        fs.build_large_scale_column(carbon, length=100, diameter=100, loading_rate=(1, "mph"))


def test_specific_throughput(addesigns_column, tce):
    st = fs.specific_throughput(
        addesigns_column, tce, c0=50.0, threshold=5.0, ebcts=[5.0, 10.0, 20.0],
        duration=400.0, conc_unit="mg/L", nr=6, nz=8,
    )  # fmt: skip
    assert list(st.columns) == ["TCE"] and list(st.index) == [5.0, 10.0, 20.0]
    # longer contact times use the adsorbent more efficiently
    assert np.all(np.diff(st["TCE"]) > 0)
    # at long EBCT the throughput approaches the equilibrium capacity q/C0
    q = tce.k * 50_000**tce.n_inv / 1e3  # mg/g
    assert st["TCE"].iloc[-1] < q / 50.0
    assert st["TCE"].iloc[-1] > 0.5 * q / 50.0


def test_rescaled_keeps_porosity(addesigns_column):
    for ebct in (0.5 * addesigns_column.ebct, 3 * addesigns_column.ebct):
        col = fs._rescaled(addesigns_column, ebct)
        assert col.ebct == pytest.approx(ebct)
        assert col.bed_porosity == pytest.approx(addesigns_column.bed_porosity)


def test_replacement_interval():
    t = np.arange(0.0, 101.0)
    eff = pd.DataFrame(
        {"PFOA": 0.1 * t, "PFHxS": 0.05 * t, "PFNA": 0.05 * t, "PFBS": 10 * t, "X": t}, index=t
    )
    out = fs.replacement_interval(eff)
    assert out.loc["PFOA", "predicted"] == 40  # exceeds 4 ng/L after day 40
    assert np.isnan(out.loc["PFHxS", "predicted"])  # 5 ng/L at most
    # hazard index 0.005 t + 0.005 t + 0.005 t > 1 after day 66.7
    assert out.loc["hazard index", "predicted"] == 66
    assert "X" not in out.index
    assert out["lower"].isna().all()
    in_ug = fs.replacement_interval(eff / 1e3, conc_unit="ug/L")
    pd.testing.assert_frame_equal(in_ug, out)
    bands = fs.replacement_interval(eff, lower=eff * 0.5, upper=eff * 2, hazard_index=None)
    assert bands.loc["PFOA"].tolist() == [40, 20, 80]
    assert "hazard index" not in bands.index


def test_replacement_interval_with_uncertainty(addesigns_column, tce):
    pfoa = tce.with_(name="PFOA", k=50.0)
    infl = pd.DataFrame({"PFOA": [50.0, 50.0]}, index=[0.0, 3000.0])
    u = psdm.model_uncertainty(addesigns_column, pfoa, infl, nr=6, nz=8, conc_unit="ng/L")
    out = fs.replacement_interval(u.effluent, lower=u.lower, upper=u.upper)
    lo, mid, hi = out.loc["PFOA", ["lower", "predicted", "upper"]]
    assert lo <= mid <= hi and lo < hi


def test_staggered_bed_interval():
    eff = pd.Series(np.arange(0.0, 101.0), index=np.arange(0.0, 101.0))
    # two beds, one replaced per cycle: blend 0.5 (C(tau) + C(i + tau)) <= 10 -> i = 7
    interval, blended = fs.staggered_bed_interval(eff, total_beds=2, beds_per_cycle=1, target=10)
    assert interval == 7
    assert blended.max() == pytest.approx(9.5)
    one, _ = fs.staggered_bed_interval(eff, total_beds=1, beds_per_cycle=1, target=10)
    assert one == 11
    # 3 beds, 2 per cycle: newest group weight 1/3, older group 2/3
    odd, mix = fs.staggered_bed_interval(eff, total_beds=3, beds_per_cycle=2, target=10)
    tau = mix.index.to_numpy()
    np.testing.assert_allclose(mix, tau / 3 + 2 / 3 * (odd + tau))
    assert fs.staggered_bed_interval(eff + 20, total_beds=2, beds_per_cycle=1, target=10)[0] == 0
    with pytest.raises(ValueError):
        fs.staggered_bed_interval(eff, total_beds=2, beds_per_cycle=3, target=10)
