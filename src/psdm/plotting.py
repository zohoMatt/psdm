"""Plotting helpers (requires ``matplotlib``; install with ``pip install psdm[plot]``)."""

from __future__ import annotations

import pandas as pd

from .solver import Result


def plot_breakthrough(
    result: Result,
    influent: pd.DataFrame | pd.Series | None = None,
    effluent: pd.DataFrame | pd.Series | None = None,
    ax=None,
):
    """Plot simulated breakthrough curves, optionally with measured data.

    Parameters
    ----------
    result
        Output of :func:`psdm.simulate`.
    influent, effluent
        Measured concentrations (columns named by compound) to overlay.
    ax
        Matplotlib axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    names = list(result.effluent.columns)
    for name in names:
        (line,) = ax.plot(result.effluent.index, result.effluent[name], label=f"{name} model")
        for data, marker, kind in ((influent, "x", "influent"), (effluent, "o", "effluent")):
            if data is None:
                continue
            series = data if isinstance(data, pd.Series) else data.get(name)
            if series is not None:
                ax.plot(
                    series.index,
                    series.to_numpy(),
                    marker,
                    color=line.get_color(),
                    mfc="none",
                    ls="none",
                    label=f"{name} {kind}",
                )
    ax.set_xlabel(f"Time ({result.effluent.index.name or 'time'})")
    ax.set_ylabel("Concentration")
    ax.legend()
    return ax
