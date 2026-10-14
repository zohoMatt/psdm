import numpy as np
import pandas as pd
import pytest
from scipy import special

import psdm
from psdm import collocation, correlations, fouling, properties, units
from psdm.solver import _build_system


def test_units():
    assert units.convert(1, "m", "length") == 100
    assert units.convert(2, "lb", "mass") == pytest.approx(907.18474)
    assert units.convert(1, "gpm", "flow") == pytest.approx(3785.411784)
    assert units.convert(62.427973725314, "lb/ft3", "density") == pytest.approx(1)
    assert units.factor("concentration", " NG/L ") == 1e-3
    with pytest.raises(ValueError, match="unknown length unit"):
        units.convert(1, "parsec", "length")


def test_water_properties():
    # the AdDesignS correlation is ~1.6 % above the tabulated 1.002 cP at 20 °C
    assert properties.water_viscosity(20) == pytest.approx(0.01002, rel=0.02)
    assert properties.water_density(20) == pytest.approx(0.9982, rel=1e-3)


def test_correlations_are_positive():
    for name in correlations.SHERWOOD:
        kf = correlations.film_transfer(1e-5, 0.05, 2.0, 1000.0, 0.4, name)
        assert np.all(np.asarray(kf) > 0), name


def test_ohashi_regimes_are_continuous_enough():
    sh = correlations.SHERWOOD["Ohashi et al"]
    assert sh(np.array([1.0, 50.0, 1000.0]), 1000.0, 0.4).shape == (3,)


def test_fouling():
    assert fouling.k_multiplier(0, "Organic Free", (1.0, 0.0)) == pytest.approx(1)
    m = fouling.k_multiplier(np.array([0, 1e5, 1e6]), "Rhine", (0.59, 0.41))
    assert m[0] == pytest.approx(1) and np.all(np.diff(m) < 0)
    assert fouling.chemical_coefficients("PFAS", "PFOA") == fouling.PFAS["PFOA"]
    assert fouling.chemical_coefficients("PFAS", "unknown") == fouling.PFAS_AVERAGE
    assert fouling.chemical_coefficients((0.5, 0.5)) == (0.5, 0.5)
    assert fouling.tortuosity(0) == 1
    with pytest.raises(ValueError):
        fouling.chemical_coefficients("unobtainium")


@pytest.mark.parametrize("ne", [1, 3])
def test_axial_operator_differentiates_polynomials(ne):
    nz = 7
    col = collocation.build(4, nz, ne)
    pts = np.concatenate([[0.0], special.roots_sh_legendre(nz - 2)[0], [1.0]])
    z = np.unique(np.concatenate([(pts + k) / ne for k in range(ne)]))
    assert len(z) == col.nz
    np.testing.assert_allclose((col.az @ z**3)[1:], 3 * z[1:] ** 2, atol=1e-8)


def test_radial_weights_integrate_sphere():
    col = collocation.build(8, 5)
    assert col.wr.sum() == pytest.approx(1 / 3)  # ∫ r² dr over the unit sphere / 4π


def test_jacobian_sparsity_covers_numerical_jacobian(addesigns_column):
    """Every entry of a finite-difference Jacobian must be inside the declared pattern."""
    comps = [psdm.Compound("a", 100, 100, 1000, 0.5), psdm.Compound("b", 120, 110, 3000, 0.4)]
    infl = pd.DataFrame({"a": [5.0, 5.0], "b": [5.0, 5.0]}, index=[0.0, 10.0])
    system = _build_system(
        addesigns_column,
        comps,
        infl,
        water="Rhine",
        temperature=20,
        time_unit="days",
        conc_unit="mg/L",
        nr=4,
        nz=5,
        ne=2,
        correlation="Chern and Chien",
    )
    pattern = system.sparsity().toarray()

    rng = np.random.default_rng(0)
    y = rng.uniform(0.1, 1.0, np.prod(system.shape))
    f0 = system.rhs(0.3, y)
    for j in range(len(y)):
        yp = y.copy()
        yp[j] += 1e-7
        changed = np.abs(system.rhs(0.3, yp) - f0) > 1e-12
        assert not np.any(changed & ~pattern[:, j]), f"column {j} has undeclared dependencies"


def test_column_derived_properties(addesigns_column):
    c = addesigns_column
    assert c.ebct == pytest.approx(c.bed_volume / c.flow_rate)
    assert 0 < c.bed_porosity < 1
    with pytest.raises(ValueError, match="bed porosity"):
        c.with_(mass=c.mass * 10)
    with pytest.raises(ValueError, match="positive"):
        c.with_(length=0)


def test_column_from_units():
    c = psdm.Column.from_units(
        length=(1.8, "m"),
        diameter=(3.66, "m"),
        mass=(8500, "kg"),
        flow_rate=(500, "gpm"),
        particle_radius=(0.513, "mm"),
        particle_density=(0.803, "g/mL"),
        particle_porosity=0.641,
    )
    assert c.length == pytest.approx(180) and c.mass == pytest.approx(8.5e6)
    assert c.particle_radius == pytest.approx(0.0513)


def test_compound_validation():
    with pytest.raises(ValueError, match="k must be positive"):
        psdm.Compound("x", 100, 100, k=0, n_inv=0.5)
    with pytest.raises(ValueError, match="kf"):
        psdm.Compound("x", 100, 100, k=1, n_inv=0.5, kf=-1)
