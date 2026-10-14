"""Pore and Surface Diffusion Model (PSDM) for fixed-bed adsorbers."""

from .fitting import FitResult, fit_freundlich, sum_of_squares
from .isotherms import IsothermFit, fit_isotherm
from .model import Column, Compound
from .solver import Result, simulate

__version__ = "0.1.0.dev0"

__all__ = [
    "Column",
    "Compound",
    "FitResult",
    "IsothermFit",
    "Result",
    "__version__",
    "fit_freundlich",
    "fit_isotherm",
    "simulate",
    "sum_of_squares",
]
