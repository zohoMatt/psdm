from pathlib import Path

import pytest

import psdm

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def addesigns_column():
    """Full-scale column from the AdDesignS verification case shipped with the EPA PSDM."""
    return psdm.Column(
        length=276.5,
        diameter=304.8,
        mass=9072000,
        flow_rate=2146200,
        particle_radius=0.0513,
        particle_density=0.803,
        particle_porosity=0.641,
        bed_density=0.55,
    )


@pytest.fixture
def tce():
    return psdm.Compound("TCE", mw=131.39, molar_volume=102, k=5026.04, n_inv=0.43)
