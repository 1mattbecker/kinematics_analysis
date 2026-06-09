"""
encoding_plots.py — Stateless plot functions for encoding analysis results.

All functions:
- accept data arrays/DataFrames plus an optional pre-created ax
- return (fig, ax) or (fig, axes)
- do NOT set rcParams — call plotstyle.apply_style() once per notebook

Primitives (repeated across multiple notebook cells):
    tstat_hist          Per-unit T-stat histogram with FDR-significant overlay
    t_scatter           Paired T-stat scatter with identity line or crosshair
    annotated_heatmap   imshow with per-cell text labels and colorbar
    tuning_band         Population LOWESS tuning curves with SEM bands

Composite (multi-panel figures for specific analyses):
    sign_contingency    3×3 observed counts + Pearson residuals (early × late)
    rt_distribution     log(RT) density + QQ plot
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plotstyle import PALETTE, OKABE_ITO, style_ax, identity_line


# ── internal helpers ──────────────────────────────────────────────────────────

def _ensure_ax(ax: Optional[Axes], figsize: tuple) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _square_symmetric(ax: Axes, all_values: np.ndarray) -> None:
    lim = float(np.nanmax(np.abs(all_values))) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# ── T-stat histogram ──────────────────────────────────────────────────────────

def tstat_hist(
    stats: pd.DataFrame,
    *,
    name: str = "",
    xlabel: str = "T-statistic",
    bins: int = 25,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Per-unit T-stat histogram with FDR-significant overlay.

    Parameters
    ----------
    stats : pd.DataFrame
        AnalysisResult.stats. Must contain columns T and sig_fdr.
    name : str
        Label for the plot title.
    xlabel : str
        X-axis label (use "T-statistic (Wald)" for GLM).
    bins : int
        Number of histogram bins.
    ax : Axes, optional
        Plot into an existing axis; otherwise a new figure is created.
    """
    fig, ax = _ensure_ax(ax, figsize=(5, 4))

    t   = stats["T"].dropna().values
    sig = stats.loc[stats["T"].notna(), "sig_fdr"].values
    n_pos = int(((stats["T"] > 0) & stats["sig_fdr"]).sum())
    n_neg = int(((stats["T"] < 0) & stats["sig_fdr"]).sum())

    edges = np.histogram_bin_edges(t, bins=bins)
    ax.hist(t[~sig], bins=edges, alpha=0.7,  color=PALETTE["not_sig"], label="not sig")
    ax.hist(t[sig],  bins=edges, alpha=0.9,  color=PALETTE["sig"],     label="sig (FDR)")
    ax.axvline(0, lw=1, ls="--", color=PALETTE["neutral"])
    ax.set_title(f"{name}\nn={len(t)}, +{n_pos} / −{n_neg} sig")
    ax.set_xlabel(xlabel)
    ax.legend()
    style_ax(ax)
    return fig, ax


# ── Paired T-stat scatter ─────────────────────────────────────────────────────

def t_scatter(
    t_x: Union[pd.Series, np.ndarray],
    t_y: Union[pd.Series, np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    c=None,
    size: float = 18,
    diagonal: str = "identity",
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Paired T-stat scatter with symmetric limits and a reference line.

    Parameters
    ----------
    t_x, t_y : array-like
        T-stat arrays of equal length (caller must align them first).
    xlabel, ylabel : str
    title : str
    c : color or per-point color array, optional
        Defaults to PALETTE["all"].
    size : float
        Marker size.
    diagonal : "identity" | "crosshair" | None
        "identity"  — y = x diagonal (for method comparisons: OLS vs Spearman).
        "crosshair" — axhline(0) + axvline(0) (for window comparisons).
        None        — no reference; caller handles limits.
    ax : Axes, optional
    """
    fig, ax = _ensure_ax(ax, figsize=(5, 5))
    color = c if c is not None else PALETTE["all"]
    x_arr = np.asarray(t_x, dtype=float)
    y_arr = np.asarray(t_y, dtype=float)

    ax.scatter(x_arr, y_arr, c=color, s=size, alpha=0.5, edgecolors="none")

    if diagonal == "identity":
        identity_line(ax)
    elif diagonal == "crosshair":
        _square_symmetric(ax, np.concatenate([x_arr, y_arr]))
        ax.axhline(0, lw=0.8, ls="--", color=PALETTE["neutral"])
        ax.axvline(0, lw=0.8, ls="--", color=PALETTE["neutral"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    style_ax(ax, grid=True)
    return fig, ax


# ── Annotated heatmap ─────────────────────────────────────────────────────────

def annotated_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    *,
    cmap: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = "{:.1f}",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
    text_threshold: float = 0.3,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """imshow with per-cell text annotations and a colorbar.

    Parameters
    ----------
    mat : 2-D array
        Values to display. NaN cells are left unannotated.
    row_labels, col_labels : list of str
        Axis tick labels (already formatted by the caller).
    cmap : str
        Matplotlib colormap name.
    fmt : str
        Python format string for cell annotations, e.g. "{:.0f}" or "{:.2f}".
    text_threshold : float
        Cells within this fraction of the colormap midpoint use dark text;
        cells far from the midpoint use white text.
    """
    fig, ax = _ensure_ax(ax, figsize=(5, 4))
    mat = np.asarray(mat, dtype=float)
    finite = mat[np.isfinite(mat)]
    _vmin = float(finite.min()) if vmin is None else vmin
    _vmax = float(finite.max()) if vmax is None else vmax

    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="upper", vmin=_vmin, vmax=_vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    mid = (_vmin + _vmax) / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                use_dark = abs(v - mid) < (_vmax - _vmin) * text_threshold
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=7, color="black" if use_dark else "white")
    return fig, ax


# ── Population tuning curves ──────────────────────────────────────────────────

def tuning_band(
    rt_grid: np.ndarray,
    groups: List[dict],
    *,
    vlines: Optional[List[float]] = None,
    title: str = "",
    xlabel: str = "Reaction time (ms)",
    ylabel: str = "Spike count (z-score)",
    xticks: Optional[List[float]] = None,
    figsize: tuple = (8, 5),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Population LOWESS tuning curves: unit-mean ± SEM bands on a log RT axis.

    Parameters
    ----------
    rt_grid : 1-D array
        RT values in seconds. X-axis rendered as log scale with ms labels.
    groups : list of dict, each with:
        mat    : DataFrame (units × grid_points) of per-unit LOWESS curves.
        color  : str
        label  : str
        ls     : str, optional (default "-")
        lw     : float, optional (default 2.0)
        zorder : float, optional
    vlines : list of float, optional
        RT values (seconds) for vertical guide lines.
    xticks : list of float, optional
        RT values (seconds) for x-axis ticks.
    """
    fig, ax = _ensure_ax(ax, figsize=figsize)

    for g in groups:
        mat    = g["mat"]
        color  = g["color"]
        label  = g["label"]
        ls     = g.get("ls", "-")
        lw     = g.get("lw", 2.0)
        zorder = g.get("zorder", None)
        if len(mat) == 0:
            continue
        m = mat.mean(axis=0).values if hasattr(mat, "values") else mat.mean(axis=0)
        s = (mat.sem(axis=0).values  if hasattr(mat, "sem")
             else mat.std(axis=0) / np.sqrt(len(mat)))
        kw = dict(color=color, lw=lw, label=f"{label} (n={len(mat)})")
        if zorder is not None:
            kw["zorder"] = zorder
        ax.fill_between(rt_grid, m - s, m + s, alpha=0.15, color=color)
        ax.plot(rt_grid, m, ls, **kw)

    ax.axhline(0, color=PALETTE["neutral"], lw=0.8, ls=":")
    if vlines:
        for xv in vlines:
            ax.axvline(xv, color=PALETTE["neutral"], lw=0.8, ls=":", alpha=0.6)

    ax.set_xscale("log")
    _xticks = xticks or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ax.set_xticks(_xticks)
    ax.set_xticklabels([f"{v*1000:.0f}" for v in _xticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    style_ax(ax)
    return fig, ax


# ── 3×3 sign contingency ──────────────────────────────────────────────────────

def sign_contingency(
    table_3x3: pd.DataFrame,
    pearson_resid: np.ndarray,
    *,
    tick_labels: Optional[List[str]] = None,
    row_label: str = "Early window sign",
    col_label: str = "Late window sign",
    suptitle: str = "Early × late window sign classification",
) -> Tuple[Figure, np.ndarray]:
    """Two-panel figure: observed 3×3 counts (left) and Pearson residuals (right).

    Parameters
    ----------
    table_3x3 : DataFrame (3, 3)
        Contingency table of observed counts.
    pearson_resid : 2-D array (3, 3)
        Pearson residuals (O − E) / √E from chi-square test.
    tick_labels : list of 3 str, optional
        Defaults to ["neg_sig", "not_sig", "pos_sig"].
    """
    ticks = tick_labels or ["neg_sig", "not_sig", "pos_sig"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    im = ax.imshow(table_3x3.values, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="n units")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(table_3x3.values[i, j]),
                    ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(ticks)
    ax.set_yticks(range(3)); ax.set_yticklabels(ticks)
    ax.set_xlabel(col_label); ax.set_ylabel(row_label)
    ax.set_title("Observed counts (3×3)")

    ax = axes[1]
    _vmax = max(float(np.abs(pearson_resid).max()), 2.1)
    im2 = ax.imshow(pearson_resid, cmap="RdBu_r", aspect="auto", vmin=-_vmax, vmax=_vmax)
    plt.colorbar(im2, ax=ax, shrink=0.8, label="Pearson residual")
    for i in range(3):
        for j in range(3):
            v = float(pearson_resid[i, j])
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(v) > _vmax * 0.6 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(ticks)
    ax.set_yticks(range(3)); ax.set_yticklabels(ticks)
    ax.set_xlabel(col_label); ax.set_ylabel(row_label)
    ax.set_title("Pearson residuals (O−E)/√E\n|residual| > 2 → enriched/depleted")

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig, axes


# ── RT distribution ───────────────────────────────────────────────────────────

def rt_distribution(
    log_rt: np.ndarray,
    *,
    p_da: float,
    sk: float,
    ku: float,
    gm2=None,
    delta_bic: Optional[float] = None,
) -> Tuple[Figure, np.ndarray]:
    """log(RT) density histogram + normal QQ plot.

    Parameters
    ----------
    log_rt : 1-D array
        Log-transformed reaction times.
    p_da : float
        p-value from D'Agostino-Pearson normality test.
    sk, ku : float
        Skewness and excess kurtosis of log_rt.
    gm2 : GaussianMixture, optional
        Fitted 2-component sklearn mixture. If given and delta_bic > 10,
        the mixture density curve is overlaid.
    delta_bic : float, optional
        BIC₁ − BIC₂. Required when gm2 is given.
    """
    from scipy.stats import norm as _norm, probplot

    x_range = np.linspace(log_rt.min(), log_rt.max(), 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(log_rt, bins=40, density=True, alpha=0.65,
            color=PALETTE["neg"], label="Data")
    ax.plot(x_range, _norm.pdf(x_range, log_rt.mean(), log_rt.std()),
            "k--", lw=1.5, label=f"Log-normal fit\np={p_da:.2e}")
    if gm2 is not None and delta_bic is not None and delta_bic > 10:
        y_mix = np.zeros_like(x_range)
        for i in range(2):
            y_mix += gm2.weights_[i] * _norm.pdf(
                x_range, gm2.means_[i, 0], np.sqrt(gm2.covariances_[i, 0, 0]))
        ax.plot(x_range, y_mix, "-", color=PALETTE["pos"], lw=1.5,
                label=f"2-component mixture\nΔBIC={delta_bic:.0f}")

    ms_ticks = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    valid_ticks = [t for t in np.log(ms_ticks) if log_rt.min() <= t <= log_rt.max()]
    ax2.set_xticks(valid_ticks)
    ax2.set_xticklabels([f"{int(np.exp(t)*1000)}" for t in valid_ticks], fontsize=7)
    ax2.set_xlabel("RT (ms)")
    ax.set_xlabel("log(RT)")
    ax.set_ylabel("Density")
    ax.set_title(f"log(RT) distribution\nskew={sk:.2f}  excess kurt={ku:.2f}")
    ax.legend()
    style_ax(ax)

    ax = axes[1]
    res_qq = probplot(log_rt, dist="norm")
    ax.scatter(res_qq[0][0], res_qq[0][1], s=2, alpha=0.3, color=PALETTE["neg"])
    mn, mx = res_qq[0][0].min(), res_qq[0][0].max()
    slope, intercept = res_qq[1][0], res_qq[1][1]
    ax.plot([mn, mx], [slope*mn+intercept, slope*mx+intercept],
            "-", color=PALETTE["pos"], lw=1.5)
    ax.set_xlabel("Theoretical quantiles (normal)")
    ax.set_ylabel("log(RT) quantiles")
    ax.set_title(f"QQ plot: log(RT) vs normal\np={p_da:.3e}")
    style_ax(ax)

    plt.tight_layout()
    return fig, axes
