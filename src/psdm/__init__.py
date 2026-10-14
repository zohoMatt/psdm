"""Pore and Surface Diffusion Model (PSDM) for fixed-bed adsorbers."""

from . import fullscale
from .capacity import CapacityEstimate, estimate_capacity
from .fitting import DsFitResult, FitResult, fit_freundlich, fit_surface_diffusivity, sum_of_squares
from .isotherms import IsothermFit, fit_isotherm
from .model import Column, Compound
from .solver import Result, simulate
from .uncertainty import Uncertainty, model_uncertainty

__version__ = "0.1.0"

__all__ = [
    "CapacityEstimate",
    "Column",
    "Compound",
    "DsFitResult",
    "FitResult",
    "IsothermFit",
    "Result",
    "Uncertainty",
    "__version__",
    "estimate_capacity",
    "fit_freundlich",
    "fit_isotherm",
    "fit_surface_diffusivity",
    "fullscale",
    "model_uncertainty",
    "simulate",
    "sum_of_squares",
]
