"""
plotstyle.py — Figure style standards for the kinematics analysis pipeline.

A single source of truth for figure aesthetics so every notebook produces
consistent, publication-ready, colorblind-safe figures. No domain logic lives
here — only style.

Typical use (once per notebook, after imports):

    from plotstyle import apply_style, PALETTE, style_ax, save_fig
    apply_style()

Then in any plotting cell:

    fig, ax = plt.subplots()
    ax.plot(...)
    style_ax(ax)
    save_fig(fig, "tuning_curve", fig_dir=FIG_DIR, save=SAVE_FIG)

Design notes
------------
- ``apply_style`` sets rcParams globally; call it once per session.
- Colors are the Okabe-Ito colorblind-safe palette, exposed both by hue name
  (OKABE_ITO) and by semantic role (PALETTE). Always reference PALETTE in
  plots so meaning stays consistent across figures.
- ``svg.fonttype = "none"`` keeps text editable in Illustrator/Inkscape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Color palette ─────────────────────────────────────────────────────────────
# Okabe-Ito colorblind-safe qualitative palette (Wong 2011, Nature Methods).
OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Semantic roles — reference these in plots, not raw hexes, so that the same
# concept always gets the same color across every figure in the project.
PALETTE = {
    "pos":      OKABE_ITO["vermillion"],    # positive encoding / positive T
    "neg":      OKABE_ITO["blue"],          # negative encoding / negative T
    "sig":      OKABE_ITO["orange"],        # FDR-significant units
    "not_sig":  "#9b9b9b",                  # non-significant units (neutral grey)
    "baseline": OKABE_ITO["bluish_green"],  # baseline / control window
    "all":      "#333333",                  # all-units reference / pooled
    "accent":   OKABE_ITO["reddish_purple"],
    "neutral":  "#666666",                  # axis guides, reference lines
}


# ── Global style ──────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Apply project-wide matplotlib rcParams.

    Call once per notebook (or session), after importing matplotlib. Sets
    typography, axis spines, ticks, legend, and SVG text handling. Idempotent.
    """
    mpl.rcParams.update({
        # Typography — editable SVG text, single sans-serif family.
        "svg.fonttype":      "none",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        # Axes — despined, ticks outward, grid off by default.
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "axes.grid":          False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Legend — frameless by convention.
        "legend.frameon":     False,
        # Lines.
        "lines.linewidth":    1.5,
        # Output.
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ── Per-axis helpers ──────────────────────────────────────────────────────────

def style_ax(ax: plt.Axes, *, grid: bool = False) -> plt.Axes:
    """Apply per-axis conventions: despine and optional light grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    grid : bool
        If True, draw a light dotted grid (off by default).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, for chaining.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    return ax


def identity_line(ax: plt.Axes, *, color: Optional[str] = None,
                  square: bool = True) -> plt.Axes:
    """Draw a y = x identity line and (optionally) square symmetric limits.

    Useful for T-vs-T comparison scatters. Computes a symmetric limit from the
    current data extent so the diagonal is centered.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis with data already plotted (limits are read from it).
    color : str, optional
        Line color. Defaults to the neutral guide color.
    square : bool
        If True, set equal symmetric x/y limits around 0.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    color = color or PALETTE["neutral"]
    if square:
        lim = max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color=color, alpha=0.6)
    else:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=color, alpha=0.6)
    return ax


# ── Output ────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str, *,
             fig_dir: Optional[Union[str, Path]] = None,
             save: bool = True, formats=("png", "svg")) -> None:
    """Save a figure to fig_dir in multiple formats, honoring a save flag.

    Replaces the duplicated png+svg savefig blocks across the codebase. Does
    nothing when ``save`` is False, so notebooks can gate output behind a single
    SAVE_FIG flag without wrapping each call in an if-block.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Base filename (no extension).
    fig_dir : str or pathlib.Path, optional
        Output directory; created if missing. Required when ``save`` is True.
    save : bool
        If False, return immediately without writing.
    formats : tuple of str
        Extensions to write (default png + svg).
    """
    if not save:
        return
    if fig_dir is None:
        raise ValueError("save_fig: fig_dir is required when save=True")
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(fig_dir / f"{name}.{ext}")
