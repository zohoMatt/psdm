"""Inputs to the model: the adsorber column and the adsorbing compounds."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace

from . import units

BED_DENSITY_TOLERANCE = 0.05
"""Relative difference between ``Column.bed_density`` and mass / bed volume that warns."""


@dataclass(frozen=True)
class Column:
    """A fixed-bed adsorber packed with granular adsorbent.

    All values are in internal units; use :meth:`from_units` to convert.

    Parameters
    ----------
    length
        Bed length, cm.
    diameter
        Bed diameter, cm.
    mass
        Adsorbent mass, g.
    flow_rate
        Volumetric flow rate, mL/min.
    particle_radius
        Adsorbent particle radius, cm.
    particle_density
        Apparent particle density ``rhop``, g/mL.
    particle_porosity
        Particle void fraction ``epor``.
    bed_density
        Bulk (apparent bed) density ``rhof``, g/mL. Not used by the model, which works from
        ``mass`` and the bed volume; it is only checked. A warning is issued when it differs
        from ``mass / bed_volume`` by more than :data:`BED_DENSITY_TOLERANCE`.
    tortuosity
        Pore tortuosity, used when the time-dependent tortuosity correlation is off.
    psdfr
        Surface-to-pore diffusion flux ratio (SPDFR) used to estimate the surface diffusivity:
        ``ds = psdfr * epor * dp * C0 / (rhop * q0)``. The name follows the EPA code.
    shape_factor
        Particle shape factor; multiplies the film transfer coefficient from the correlation,
        as in AdDesignS.
    name
        Label for the adsorbent / column.
    """

    length: float
    diameter: float
    mass: float
    flow_rate: float
    particle_radius: float
    particle_density: float
    particle_porosity: float
    bed_density: float | None = None
    tortuosity: float = 1.0
    psdfr: float = 5.0
    shape_factor: float = 1.0
    name: str = "column"

    def __post_init__(self):
        for attr in (
            "length",
            "diameter",
            "mass",
            "flow_rate",
            "particle_radius",
            "particle_density",
            "tortuosity",
            "psdfr",
            "shape_factor",
        ):
            if not getattr(self, attr) > 0:
                raise ValueError(f"Column.{attr} must be positive, got {getattr(self, attr)!r}")
        if not 0 < self.particle_porosity < 1:
            raise ValueError("Column.particle_porosity must be between 0 and 1")
        if not 0 < self.bed_porosity < 1:
            raise ValueError(
                f"bed porosity is {self.bed_porosity:.3f}; check mass, length, diameter "
                "and particle_density"
            )
        if self.bed_density is not None:
            packed = self.mass / self.bed_volume
            if abs(self.bed_density / packed - 1) > BED_DENSITY_TOLERANCE:
                warnings.warn(
                    f"Column {self.name!r}: bed_density {self.bed_density:.3g} g/mL differs from "
                    f"mass / bed volume = {packed:.3g} g/mL. The model uses mass and volume.",
                    UserWarning,
                    stacklevel=3,
                )

    @classmethod
    def from_units(
        cls,
        *,
        length: tuple[float, str],
        diameter: tuple[float, str],
        mass: tuple[float, str],
        flow_rate: tuple[float, str],
        particle_radius: tuple[float, str],
        particle_density: tuple[float, str],
        bed_density: tuple[float, str] | None = None,
        **kw,
    ) -> Column:
        """Build a column from ``(value, unit)`` pairs.

        >>> Column.from_units(length=(1.8, "m"), diameter=(3.66, "m"), mass=(8500, "kg"),
        ...                   flow_rate=(500, "gpm"), particle_radius=(0.513, "mm"),
        ...                   particle_density=(0.803, "g/mL"), particle_porosity=0.641)
        ... # doctest: +ELLIPSIS
        Column(length=180.0, ...)
        """
        conv = units.convert
        return cls(
            length=conv(*length, "length"),
            diameter=conv(*diameter, "length"),
            mass=conv(*mass, "mass"),
            flow_rate=conv(*flow_rate, "flow"),
            particle_radius=conv(*particle_radius, "length"),
            particle_density=conv(*particle_density, "density"),
            bed_density=None if bed_density is None else conv(*bed_density, "density"),
            **kw,
        )

    def with_(self, **changes) -> Column:
        """Return a copy with some attributes changed."""
        return replace(self, **changes)

    @property
    def area(self) -> float:
        """Cross-sectional area, cm²."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def bed_volume(self) -> float:
        """Bed volume, mL."""
        return self.area * self.length

    @property
    def bed_porosity(self) -> float:
        """Bed void fraction ``ebed``."""
        return 1.0 - self.mass / (self.bed_volume * self.particle_density)

    @property
    def ebct(self) -> float:
        """Empty-bed contact time, min."""
        return self.bed_volume / self.flow_rate

    @property
    def superficial_velocity(self) -> float:
        """Superficial velocity, cm/s."""
        return self.flow_rate / (60.0 * self.area)

    @property
    def residence_time(self) -> float:
        """Fluid residence time ``tau`` in the bed voids, s."""
        return self.ebct * self.bed_porosity * 60.0

    @property
    def loading_rate(self) -> float:
        """Hydraulic loading rate, gpm/ft²."""
        return 0.245423867471 * self.flow_rate / self.area


@dataclass(frozen=True)
class Compound:
    """An adsorbing compound described by a Freundlich isotherm.

    Parameters
    ----------
    name
        Identifier; must match the influent column name.
    mw
        Molecular weight, g/mol.
    molar_volume
        Molar volume at the normal boiling point, cm³/mol.
    k
        Freundlich ``K`` in (ug/g)(L/ug)^(1/n).
    n_inv
        Freundlich exponent ``1/n``.
    kf, dp, ds
        Optional film transfer coefficient (cm/s), pore diffusivity (cm²/s) and surface
        diffusivity (cm²/s). ``None`` uses the built-in correlations.
    fouling_class
        Compound class for NOM fouling, see :func:`psdm.fouling.chemical_coefficients`.
    """

    name: str
    mw: float
    molar_volume: float
    k: float
    n_inv: float
    kf: float | None = None
    dp: float | None = None
    ds: float | None = None
    fouling_class: str | tuple[float, float] = field(default="halogenated alkenes")

    def __post_init__(self):
        for attr in ("mw", "molar_volume", "k", "n_inv"):
            if not getattr(self, attr) > 0:
                raise ValueError(f"Compound {self.name!r}: {attr} must be positive")
        for attr in ("kf", "dp", "ds"):
            value = getattr(self, attr)
            if value is not None and not value > 0:
                raise ValueError(f"Compound {self.name!r}: {attr} must be positive or None")

    def with_(self, **changes) -> Compound:
        """Return a copy with some attributes changed."""
        return replace(self, **changes)
