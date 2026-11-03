"""Pore and Surface Diffusion Model (PSDM) for fixed-bed adsorbers."""

from . import errors
from .model import Column, Compound
from .solver import Result, simulate

__version__ = "0.4.0"

__all__ = [
    "Column",
    "Compound",
    "Result",
    "__version__",
    "errors",
    "simulate",
]
