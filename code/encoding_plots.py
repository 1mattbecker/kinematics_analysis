"""
encoding_plots.py — Stateless plot functions for encoding analysis results.

All functions:
- accept data arrays/DataFrames plus an optional pre-created ax
- return (fig, ax) or (fig, axes)
- do NOT set rcParams — call plotstyle.apply_style() once per notebook

Encoding primitives (reused across multiple notebooks):
    tstat_hist              Per-unit T-stat histogram with FDR-significant overlay
    t_scatter               Paired T-stat scatter with identity line or crosshair
    annotated_heatmap       imshow with per-cell text labels and colorbar
    tuning_band             Population LOWESS tuning curves with SEM bands

Registry visualization (registry keeps store/compare/screen only — no matplotlib):
    registry_summary        T-stat histogram + Wilcoxon/binom tests for one entry
    registry_compare_plot   Scatter + marginals for compare() output DataFrame
    registry_heatmap        Pairwise Spearman rho heatmap across entries
    registry_upset          UpSet plot of significance overlap across entries
    example_unit_scatter    Trial-level scatter for selected (unit, session) pairs
    registry_plot_examples  Auto-select example units from a registry entry and scatter

Analysis-specific figures (used once, in one notebook) belong as local
helper functions inside that notebook — not here.
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


# ── Per-unit example scatter ───────────────────────────────────────────────

def _add_ols_line(ax: Axes, x: np.ndarray, y: np.ndarray,
                  xscale: str = "linear") -> None:
    """Overlay OLS regression line; silently skips if too few points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    x_fit, y_fit = x[mask], y[mask]
    if xscale == "log":
        x_fit = np.log(x_fit)
    z = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_line = np.polyval(z, x_line)
    if xscale == "log":
        x_line = np.exp(x_line)
    ax.plot(x_line, y_line, color=PALETTE["pos"], lw=1.2, alpha=0.8)


def example_unit_scatter(
    df: pd.DataFrame,
    examples: List[Tuple],
    x_col: str,
    x_label: str,
    *,
    y_col: str = "spike_count",
    jitter_x: Union[float, str] = "auto",
    jitter_y: Union[float, str] = "auto",
    jitter_seed: int = 0,
    xscale: str = "linear",
) -> Optional[Figure]:
    """Trial-level scatter for a list of (unit_id, session) example pairs.

    Parameters
    ----------
    df : DataFrame
        Trial-level data. Must contain 'unit_id', 'session', x_col, y_col.
    examples : list of (unit_id, session)
        Pairs to plot; one panel per pair.
    x_col, x_label : str
        Column name and axis label for x.
    y_col : str
        Column for y axis (default 'spike_count').
    jitter_x, jitter_y : float or "auto"
        Jitter σ. "auto" applies jitter when the axis appears integer-valued.
    xscale : str
        "linear" or "log".
    """
    if not examples:
        print("No example units to plot.")
        return None

    rng = np.random.default_rng(jitter_seed)
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (uid, sess) in zip(axes.ravel(), examples):
        g = df[(df["unit_id"] == uid) & (df["session"] == sess)]
        x = g[x_col].to_numpy()
        y = g[y_col].to_numpy()

        if xscale == "log":
            keep = x > 0
            x, y = x[keep], y[keep]

        x_plot, y_plot = x.copy(), y.copy()

        jy = jitter_y
        if jy == "auto":
            jy = 0.2 if (len(y) > 0 and np.all(np.isclose(y, np.round(y), atol=1e-6))) else 0.0
        jx = jitter_x
        if jx == "auto":
            jx = 0.2 if (len(x) > 0 and np.all(np.isclose(x, np.round(x), atol=1e-6))) else 0.0

        if isinstance(jx, (int, float)) and jx > 0:
            x_plot = x_plot + rng.normal(0, jx, size=len(x_plot))
        if isinstance(jy, (int, float)) and jy > 0:
            y_plot = y_plot + rng.normal(0, jy, size=len(y_plot))

        ax.scatter(x_plot, y_plot, s=20, alpha=0.2, edgecolors="none",
                   color=PALETTE["all"])
        _add_ols_line(ax, x, y, xscale=xscale)
        ax.set_xscale(xscale)
        style_ax(ax, grid=True)
        ax.set_title(f"{uid} ({sess}) • n={len(x)}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_col.replace("_", " "))

    fig.suptitle(f"Examples: {y_col} vs {x_label}", y=1.02)
    plt.tight_layout()
    return fig


# ── Registry summary ──────────────────────────────────────────────────────

def registry_summary(
    stats_df: pd.DataFrame,
    name: str = "",
    *,
    bins: int = 30,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """T-stat histogram with Wilcoxon and binomial tests for one registry entry.

    Parameters
    ----------
    stats_df : DataFrame
        Registry entry table from reg.get(name). Must have columns 't', 'sig_fdr'.
    name : str
        Label for the title.
    bins : int
        Number of histogram bins.
    """
    from scipy.stats import wilcoxon, binomtest

    fig, ax = _ensure_ax(ax, figsize=(6, 4))

    t_notna = stats_df["t"].notna()
    t_vals  = stats_df.loc[t_notna, "t"].to_numpy()
    sig_arr = stats_df.loc[t_notna, "sig_fdr"].to_numpy()

    n_pos_sig = int(((t_vals > 0) & sig_arr).sum())
    n_neg_sig = int(((t_vals < 0) & sig_arr).sum())
    n_sig = n_pos_sig + n_neg_sig
    frac_sig = float(sig_arr.mean()) if len(sig_arr) > 0 else 0.0

    w_p = np.nan
    if len(t_vals) >= 10:
        _, w_p = wilcoxon(t_vals)

    binom_p = np.nan
    if n_sig >= 1:
        binom_p = binomtest(n_pos_sig, n_sig, 0.5).pvalue

    bin_edges = np.histogram_bin_edges(t_vals, bins=bins)
    ax.hist(t_vals[~sig_arr], bins=bin_edges, alpha=0.9, color=PALETTE["not_sig"],
            label="not sig", edgecolor="white", linewidth=0.5)
    ax.hist(t_vals[sig_arr],  bins=bin_edges, alpha=0.9, color=PALETTE["sig"],
            label="sig (FDR)", edgecolor="white", linewidth=0.5)
    ax.axvline(0, ls="--", color=PALETTE["neutral"], lw=0.8)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("units")
    ax.set_title(
        f"{name}\nn={len(t_vals)}, sig={n_sig} ({frac_sig:.0%}), "
        f"+{n_pos_sig}/−{n_neg_sig}"
    )

    annot_lines = []
    if np.isfinite(w_p):
        annot_lines.append(f"Wilcoxon p={w_p:.2g}")
    if n_sig >= 1 and np.isfinite(binom_p):
        annot_lines.append(f"Binom +/− p={binom_p:.2g} ({n_pos_sig}/{n_sig})")
    if annot_lines:
        ax.text(0.97, 0.95, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))

    ax.legend()
    style_ax(ax)
    return fig, ax


# ── Registry compare scatter + marginals ──────────────────────────────────

def _compare_scatter(
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    rho: float,
    rho_p: float,
    figsize: Tuple[float, float],
    sig_x: Optional[np.ndarray] = None,
    sig_y: Optional[np.ndarray] = None,
    fisher_or: Optional[float] = None,
    fisher_p: Optional[float] = None,
) -> Tuple[Figure, Axes, Axes, Axes]:
    """Scatter with marginal histograms, colored by significance category.

    Color convention: x-only → pos (warm), y-only → neg (cool), both → accent.
    """
    from matplotlib.gridspec import GridSpec

    x_only_name = f"{x_label} only"
    y_only_name = f"{y_label} only"
    cat_colors = {
        "neither":   PALETTE["not_sig"],
        "both":      PALETTE["accent"],
        x_only_name: PALETTE["pos"],
        y_only_name: PALETTE["neg"],
    }
    color_x_pale = "#f2c4b8"
    color_y_pale = "#b8d4e8"

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    unique_cats = sorted(set(categories))
    draw_order = (
        ["neither"]
        + [c for c in unique_cats if c not in ("neither", "both")]
        + ["both"]
    )
    for cat in [c for c in draw_order if c in unique_cats]:
        mask = np.asarray(categories) == cat
        if not mask.any():
            continue
        ax_main.scatter(x[mask], y[mask],
                        c=cat_colors.get(cat, PALETTE["not_sig"]),
                        label=f"{cat} ({mask.sum()})",
                        s=25, alpha=0.7, edgecolors="none")

    ax_main.axhline(0, ls=":", color=PALETTE["neutral"], lw=0.5)
    ax_main.axvline(0, ls=":", color=PALETTE["neutral"], lw=0.5)
    ax_main.set_xlabel(f"t  ({x_label})")
    ax_main.set_ylabel(f"t  ({y_label})")
    ax_main.legend(fontsize=7, loc="best")

    annot = f"n={len(x)}, ρ={rho:.3f}, p={rho_p:.2g}"
    if fisher_or is not None and fisher_p is not None:
        annot += f"\nFisher OR={fisher_or:.2f}, p={fisher_p:.2g}"
    ax_main.text(0.02, 0.98, annot, transform=ax_main.transAxes,
                 fontsize=8, va="top", ha="left")

    bins_edge = np.linspace(
        min(np.nanmin(x), np.nanmin(y)),
        max(np.nanmax(x), np.nanmax(y)),
        31,
    )
    if sig_x is not None:
        sig_x = np.asarray(sig_x, dtype=bool)
        ax_top.hist(x[~sig_x], bins=bins_edge, color=color_x_pale, alpha=0.8)
        ax_top.hist(x[sig_x],  bins=bins_edge, color=PALETTE["pos"],     alpha=0.8)
    else:
        ax_top.hist(x, bins=bins_edge, color=color_x_pale, alpha=0.8)

    if sig_y is not None:
        sig_y = np.asarray(sig_y, dtype=bool)
        ax_right.hist(y[~sig_y], bins=bins_edge, orientation="horizontal",
                      color=color_y_pale, alpha=0.8)
        ax_right.hist(y[sig_y],  bins=bins_edge, orientation="horizontal",
                      color=PALETTE["neg"], alpha=0.8)
    else:
        ax_right.hist(y, bins=bins_edge, orientation="horizontal",
                      color=color_y_pale, alpha=0.8)

    ax_top.set_ylabel("count")
    ax_right.set_xlabel("count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    style_ax(ax_main)
    style_ax(ax_top)
    style_ax(ax_right)
    fig.suptitle(title, fontsize=11, y=0.98)
    return fig, ax_main, ax_top, ax_right


def registry_compare_plot(
    merged: pd.DataFrame,
    name_a: str,
    name_b: str,
    *,
    figsize: Tuple[float, float] = (7, 7),
    title: Optional[str] = None,
) -> Tuple[Figure, Axes, Axes, Axes]:
    """Scatter + marginals for a PerUnitStatsRegistry.compare() output DataFrame.

    Parameters
    ----------
    merged : DataFrame
        Return value of reg.compare(name_a, name_b). Must have columns
        t_a, t_b, sig_fdr_a, sig_fdr_b, sig_category.
    name_a, name_b : str
        Registry entry names (used for axis labels and sig_category strings).
    """
    from scipy.stats import spearmanr as _spearmanr, fisher_exact

    n = len(merged)
    rho, rho_p = _spearmanr(merged["t_a"], merged["t_b"]) if n >= 3 else (np.nan, np.nan)

    sa = merged["sig_fdr_a"].values
    sb = merged["sig_fdr_b"].values
    contingency = np.array([
        [int((~sa & ~sb).sum()), int((~sa &  sb).sum())],
        [int(( sa & ~sb).sum()), int(( sa &  sb).sum())],
    ])
    fisher_or, fisher_p_val = fisher_exact(contingency)

    return _compare_scatter(
        merged["t_a"].values, merged["t_b"].values,
        merged["sig_category"].values,
        x_label=name_a, y_label=name_b,
        title=title or f"{name_a} vs {name_b}  (n={n})",
        rho=float(rho), rho_p=float(rho_p),
        sig_x=merged["sig_fdr_a"].values,
        sig_y=merged["sig_fdr_b"].values,
        fisher_or=float(fisher_or), fisher_p=float(fisher_p_val),
        figsize=figsize,
    )


# ── Registry heatmap ──────────────────────────────────────────────────────

def registry_heatmap(
    reg,
    entries: Optional[List[str]] = None,
    *,
    source: Optional[str] = None,
    labels: Optional[dict] = None,
    cluster: bool = True,
    annot: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Pairwise Spearman rho heatmap of t-stats across registry entries.

    Parameters
    ----------
    reg : PerUnitStatsRegistry
        Duck-typed: needs .names, .list_by_source(), .get().
    entries : list of str, optional
        Registry keys to include. None → all entries (or filtered by source).
    source : str, optional
        Include only entries from this source when entries is None.
    cluster : bool
        Reorder rows/cols by hierarchical clustering on |rho|.
    """
    from scipy.stats import spearmanr as _spearmanr

    _key = ["session_prefix", "unit"]
    if entries is None:
        entries = reg.list_by_source(source) if source else list(reg.names)
    if len(entries) < 2:
        raise ValueError("Need at least 2 entries for a heatmap")

    n = len(entries)
    rho_mat = np.full((n, n), np.nan)
    p_mat   = np.full((n, n), np.nan)

    for i in range(n):
        rho_mat[i, i] = 1.0
        p_mat[i, i]   = 0.0
        df_i = reg.get(entries[i])
        for j in range(i + 1, n):
            df_j = reg.get(entries[j])
            both = df_i[_key + ["t"]].merge(
                df_j[_key + ["t"]], on=_key, how="inner", suffixes=("_i", "_j")
            ).dropna(subset=["t_i", "t_j"])
            if len(both) >= 3:
                rho, p = _spearmanr(both["t_i"], both["t_j"])
                rho_mat[i, j] = rho_mat[j, i] = float(rho)
                p_mat[i, j]   = p_mat[j, i]   = float(p)

    disp = [labels.get(e, e) if labels else e for e in entries]
    order = np.arange(n)
    if cluster and n >= 3:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = np.nan_to_num(1 - np.abs(rho_mat), nan=1.0)
            np.fill_diagonal(dist, 0.0)
            dist = (dist + dist.T) / 2
            order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
        except Exception:
            pass

    rho_ord = rho_mat[np.ix_(order, order)]
    p_ord   = p_mat[np.ix_(order, order)]
    names_ord = [disp[i] for i in order]

    if figsize is None:
        side = max(6, 0.7 * n + 2)
        figsize = (side, side)
    fig, ax = _ensure_ax(ax, figsize=figsize)

    im = ax.imshow(rho_ord, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names_ord, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names_ord, fontsize=8)

    if annot:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rv, pv = rho_ord[i, j], p_ord[i, j]
                if not np.isfinite(rv):
                    continue
                star = "*" if pv < 0.05 else ""
                color = "white" if abs(rv) > 0.6 else "black"
                ax.text(j, i, f"{rv:.2f}{star}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title(title or "Pairwise t-stat correlations")
    style_ax(ax)
    return fig, ax


# ── Registry UpSet plot ───────────────────────────────────────────────────

def registry_upset(
    reg,
    entries: List[str],
    *,
    labels: Optional[dict] = None,
    min_subset_size: int = 0,
    max_subsets: int = 25,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    sort_by: str = "cardinality",
) -> Tuple[Figure, pd.DataFrame]:
    """UpSet plot of significance overlap across registry entries.

    Parameters
    ----------
    reg : PerUnitStatsRegistry
        Duck-typed: needs .get().
    entries : list of str
        Registry keys to include (2–8 recommended).
    labels : dict, optional
        Mapping from entry name to short display label.

    Returns
    -------
    (fig, membership) where membership is a boolean DataFrame (unit × entry).
    """
    from collections import Counter
    from matplotlib.gridspec import GridSpec

    _key = ["session_prefix", "unit"]
    if len(entries) < 2:
        raise ValueError("Need at least 2 entries for an UpSet plot")

    disp = [labels.get(e, e) if labels else e for e in entries]
    frames = []
    for name, display in zip(entries, disp):
        df = reg.get(name)[["session_prefix", "unit", "sig_fdr"]].copy()
        frames.append(df.rename(columns={"sig_fdr": display}))

    membership = frames[0]
    for f in frames[1:]:
        membership = membership.merge(f, on=_key, how="inner")
    for col in disp:
        membership[col] = membership[col].fillna(False).astype(bool)

    n_total   = len(membership)
    n_any_sig = int(membership[disp].any(axis=1).sum())

    subsets = [
        {"combo": combo, "count": count, "degree": sum(combo)}
        for combo, count in Counter(
            tuple(row) for row in membership[disp].values
        ).items()
        if count >= min_subset_size
    ]
    if sort_by == "degree":
        subsets.sort(key=lambda s: (s["degree"], -s["count"]))
    else:
        subsets.sort(key=lambda s: -s["count"])
    subsets = subsets[:max_subsets]

    n_sub = len(subsets)
    n_set = len(disp)

    if figsize is None:
        figsize = (max(8, n_sub * 0.5 + 2), max(5, n_set * 0.6 + 3))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig,
                  height_ratios=(3, n_set * 0.6), hspace=0.05)
    ax_bars = fig.add_subplot(gs[0])
    ax_dots = fig.add_subplot(gs[1], sharex=ax_bars)

    x = np.arange(n_sub)
    counts = [s["count"] for s in subsets]
    ax_bars.bar(x, counts, color=PALETTE["neg"], edgecolor="white", linewidth=0.5)
    for xi, c in zip(x, counts):
        ax_bars.text(xi, c + max(counts) * 0.02, str(c),
                     ha="center", va="bottom", fontsize=8)
    ax_bars.set_ylabel("units")
    ax_bars.set_xlim(-0.6, n_sub - 0.4)
    ax_bars.set_ylim(0, max(counts) * 1.15)
    plt.setp(ax_bars.get_xticklabels(), visible=False)
    ax_bars.tick_params(axis="x", length=0)
    style_ax(ax_bars)

    dot_on  = PALETTE["all"]
    dot_off = PALETTE["not_sig"]

    for i, s in enumerate(subsets):
        active = [j for j, v in enumerate(s["combo"]) if v]
        for j in range(n_set):
            ax_dots.scatter(i, j, s=80, zorder=3,
                            color=dot_off, edgecolors="none")
        for j in active:
            ax_dots.scatter(i, j, s=80, zorder=4,
                            color=dot_on, edgecolors="none")
        if len(active) > 1:
            ax_dots.plot([i, i], [min(active), max(active)],
                         color=dot_on, linewidth=1.5, zorder=2)

    ax_dots.set_yticks(range(n_set))
    ax_dots.set_yticklabels(disp, fontsize=9)
    ax_dots.set_ylim(-0.5, n_set - 0.5)
    ax_dots.invert_yaxis()
    ax_dots.set_xticks([])
    ax_dots.set_xlim(-0.6, n_sub - 0.4)
    for spine in ax_dots.spines.values():
        spine.set_visible(False)
    ax_dots.tick_params(axis="both", length=0)

    set_sizes = [int(membership[col].sum()) for col in disp]
    for j, sz in enumerate(set_sizes):
        ax_dots.text(n_sub - 0.3, j, f"n={sz}", ha="left", va="center",
                     fontsize=8, color=PALETTE["neutral"])

    suptitle = (f"{title}  (n={n_total})" if title
                else f"Significance overlap (n={n_total}, any sig={n_any_sig})")
    fig.suptitle(suptitle, fontsize=12, y=0.98)
    return fig, membership


# ── Registry plot examples ────────────────────────────────────────────────

def registry_plot_examples(
    reg,
    name: str,
    trial_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    *,
    y_col: str = "spike_count",
    examples: Optional[List[Tuple]] = None,
    n_examples: int = 4,
    select: str = "top_t",
    jitter_x: Union[float, str] = "auto",
    jitter_y: Union[float, str] = "auto",
    jitter_seed: int = 0,
    xscale: str = "linear",
) -> Optional[Figure]:
    """Auto-select example units from a registry entry and plot trial-level scatter.

    Parameters
    ----------
    reg : PerUnitStatsRegistry
        Duck-typed: needs .get() and ._gsp attribute.
    name : str
        Registry entry to draw units from.
    trial_df : DataFrame
        Trial-level data. Must contain 'unit_id', 'session', x_col, y_col.
    select : str
        "top_t" | "random_sig" | "random"
    """
    import warnings

    def _canon(x):
        try:
            return str(int(float(x)))
        except Exception:
            return str(x)

    gsp = getattr(reg, "_gsp", lambda s: s)
    reg_df = reg.get(name)

    if examples is None:
        if select == "top_t":
            cands = reg_df.dropna(subset=["t"]).copy()
            cands["abs_t"] = cands["t"].abs()
            chosen = cands.nlargest(n_examples, "abs_t")[["session_prefix", "unit"]]
        elif select == "random_sig":
            sig = reg_df[reg_df["sig_fdr"]].copy()
            chosen = sig.sample(min(n_examples, len(sig)),
                                random_state=jitter_seed)[["session_prefix", "unit"]]
        else:
            chosen = reg_df.sample(min(n_examples, len(reg_df)),
                                   random_state=jitter_seed)[["session_prefix", "unit"]]

        if len(chosen) == 0:
            print(f"registry_plot_examples('{name}'): no units matched select='{select}'.")
            return None

        tdf = trial_df.copy()
        tdf["_sp"] = tdf["session"].astype(str).map(gsp)
        tdf["_u"]  = tdf["unit_id"].map(_canon)

        examples = []
        for _, row in chosen.iterrows():
            sub = tdf[(tdf["_sp"] == row["session_prefix"]) & (tdf["_u"] == row["unit"])]
            if sub.empty:
                warnings.warn(
                    f"registry_plot_examples: unit {row['unit']} / "
                    f"{row['session_prefix']} not found in trial_df"
                )
                continue
            examples.append((sub["unit_id"].iloc[0], sub["session"].iloc[0]))

    return example_unit_scatter(
        trial_df, examples, x_col, x_label,
        y_col=y_col, jitter_x=jitter_x, jitter_y=jitter_y,
        jitter_seed=jitter_seed, xscale=xscale,
    )
