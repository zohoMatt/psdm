"""Exceptions raised by psdm.

All classes derive from the built-in exception that psdm raised before 0.2, so existing
``except RuntimeError`` and ``except ValueError`` handlers keep working.
"""

from __future__ import annotations


class SolverError(RuntimeError):
    """The ODE integration did not produce a usable solution."""


class SolverDivergedError(SolverError):
    """The integration failed or produced non-finite values."""


class StepSizeTooSmallError(SolverError):
    """The integrator needed a step smaller than floating-point spacing allows.

    Usually a sign of a very stiff problem: extreme parameters or a too-small ``rtol``.
    """


class InputError(ValueError):
    """Invalid input to a psdm function.

    Attributes
    ----------
    field
        Name of the offending argument, e.g. ``"influent"`` or ``"compounds"``.
    """

    def __init__(self, message: str, *, field: str):
        super().__init__(message)
        self.field = field
