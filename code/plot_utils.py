"""
plot_utils.py — summary plots shared across notebook series.

Style (rcParams, palette, ``style_ax``, ``save_fig``) stays in ``plotstyle``; this module draws
the recurring summary panels on top of it: mean ± SEM over animals, and strip plots of
per-animal or per-session values. Statistics come from ``stats_utils``.

Contents
--------
:func:`plot_mean_sem`, :func:`plot_etr`, :func:`animal_styles`, :func:`strip_by_measure`,
:func:`strip_by_animal`
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from plotstyle import OKABE_ITO, PALETTE, style_ax
from stats_utils import mean_sem, stars, wilcoxon_animals

#: Per-animal identity: Okabe-Ito in a fixed order (yellow fails contrast on white, black is
#: kept for grand means). The marker repeats the identity so colour is never the only cue.
ANIMAL_HUES = [OKABE_ITO[k] for k in ("orange", "sky_blue", "bluish_green", "blue",
                                      "vermillion", "reddish_purple")]
ANIMAL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def plot_mean_sem(ax, x, arr, color, label: Optional[str] = None, show_rows: bool = False,
                  ls: str = "-", lw: float = 1.6, alpha: float = 0.2, label_n: bool = False):
    """Mean ± SEM across the rows of ``arr`` (usually animals), NaN-aware.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : array_like
        x values, one per column.
    arr : array_like
        ``(n_rows, len(x))``, or a list of equal-length rows.
    color : str
    label : str, optional
        Legend label for the mean line.
    show_rows : bool
        Also draw every row as a thin line.
    ls, lw, alpha : line style, mean line width, band opacity.
    label_n : bool
        Append ``(n=<rows>)`` to the label.
    """
    a = np.asarray(arr, float)
    if a.size == 0:
        return
    a = np.atleast_2d(a)
    m, sem, _ = mean_sem(a)
    if show_rows:
        for r in a:
            ax.plot(x, r, color=color, lw=0.5, alpha=0.35)
    if label is not None and label_n:
        label = "%s (n=%d)" % (label, a.shape[0])
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    if a.shape[0] > 1:
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=alpha, lw=0)


def plot_etr(ax, etr: pd.DataFrame, color, label: Optional[str] = None, value_col: str = "data",
             **kw):
    """Mean ± SEM over events of a tidy event-triggered response.

    ``etr`` is the tidy output of ``aind_dynamic_foraging_data_utils.alignment
    .event_triggered_response`` (or ``fip_utils.peri_event``): columns ``time``, ``event_number``
    and ``value_col``. Keyword arguments go to :func:`plot_mean_sem`.
    """
    if etr is None or etr.empty:
        return
    m = etr.pivot_table(index="event_number", columns="time", values=value_col, dropna=False)
    plot_mean_sem(ax, m.columns.to_numpy(float), m.to_numpy(float), color, label, **kw)


def animal_styles(subjects: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    """``{subject: (colour, marker)}`` in the order given."""
    return {s: (ANIMAL_HUES[i % len(ANIMAL_HUES)], ANIMAL_MARKERS[i % len(ANIMAL_MARKERS)])
            for i, s in enumerate(subjects)}


def strip_by_measure(ax, df: pd.DataFrame, cols: Sequence[str], labels: Sequence[str],
                     colors: Sequence[str], ylabel: str, title: Optional[str] = None,
                     connect: bool = True, zero: bool = True,
                     rng: Optional[np.random.Generator] = None):
    """Per-animal points for several measures, the mean over animals as a bar, and stars.

    ``df`` is indexed by subject with one column per measure. Lines join one animal's points.
    Stars are a Wilcoxon signed-rank test of the animal values against zero.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.arange(len(cols))
    if connect:
        for _, row in df[list(cols)].iterrows():
            ax.plot(x, row.values, color="0.8", lw=0.8, zorder=1)
    for k, c in enumerate(cols):
        v = df[c].to_numpy(float)
        jitter = (rng.random(len(v)) - 0.5) * 0.15
        ax.scatter(x[k] + jitter, v, s=18, color=colors[k], zorder=2, edgecolor="none")
        m, p, _ = wilcoxon_animals(v)
        ax.plot([x[k] - 0.25, x[k] + 0.25], [m, m], color="k", lw=2, zorder=3)
        ax.text(x[k], 1.02, stars(p), transform=ax.get_xaxis_transform(), ha="center",
                fontsize=8)
    if zero:
        ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=16)
    style_ax(ax)


def strip_by_animal(ax, df: pd.DataFrame, col: str, order: Sequence[str], ylabel: str,
                    styles: Optional[Dict[str, Tuple[str, str]]] = None,
                    null_col: Optional[str] = None, null_style: str = "dash",
                    p_col: Optional[str] = None, ref: Optional[float] = 0.0,
                    tick_labels: Optional[Sequence[str]] = None,
                    rng: Optional[np.random.Generator] = None):
    """Sessions grouped by animal: a point per session, a black tick at the animal mean.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per session with ``subject`` and ``col``.
    order : sequence of str
        Animals, left to right.
    styles : dict, optional
        ``{subject: (colour, marker)}``; default :func:`animal_styles` of ``order``.
    null_col : str, optional
        Each session's null value. ``null_style="dash"`` draws it as a grey dash at the session's
        x (for a signed effect, on the session's side of zero); ``"range"`` draws one grey bar
        spanning the animal's sessions.
    p_col : str, optional
        Print the number of sessions with p < 0.05 above each animal.
    ref : float or None
        Dotted reference line.
    tick_labels : sequence of str, optional
        Default ``"<subject> (<n sessions>)"``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    styles = styles or animal_styles(order)
    null_color = PALETTE["not_sig"]
    for x, subj in enumerate(order):
        color, marker = styles[subj]
        g = df[df["subject"] == subj]
        xs = x + (rng.random(len(g)) - 0.5) * 0.35
        if null_col is not None and null_style == "dash":
            thr = g[null_col].to_numpy() * np.sign(g[col].to_numpy())
            ax.scatter(xs, thr, s=40, marker="_", color=null_color, lw=1.5, zorder=1)
        elif null_col is not None:
            lo, hi = g[null_col].min(), g[null_col].max()
            ax.bar(x, hi - lo, bottom=lo, width=0.6, color=null_color, alpha=0.35, lw=0, zorder=0)
        ax.scatter(xs, g[col], s=14, color=color, marker=marker, alpha=0.7, lw=0, zorder=2)
        ax.plot([x - 0.3, x + 0.3], [g[col].mean()] * 2, color="k", lw=2, zorder=3)
        if p_col is not None:
            ax.text(x, 1.01, "%d/%d" % (int((g[p_col] < 0.05).sum()), len(g)), ha="center",
                    fontsize=7, transform=ax.get_xaxis_transform())
    if ref is not None:
        ax.axhline(ref, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(len(order)))
    if tick_labels is None:
        tick_labels = ["%s (%d)" % (s, int((df["subject"] == s).sum())) for s in order]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    style_ax(ax)
