"""
stats_utils.py — summary statistics shared across notebook series.

The repo's convention for pooling: average within session, then within animal, then across
animals, so every animal weighs the same; tests run on animal means. These helpers implement
that once. Libraries are used where they exist: FDR and Wilson intervals come from statsmodels,
the hierarchical bootstrap from ``aind_hierarchical_bootstrap``.

Contents
--------
Pooling
    :func:`mean_sem`, :func:`group_means`, :func:`animal_means`
Tests
    :func:`wilcoxon_animals`, :func:`corr`, :func:`shift_p`, :func:`shift_z`,
    :func:`cluster_test`, :func:`hier_bootstrap`, :func:`fdr_bh`, :func:`wilson_ci`
Formatting
    :func:`stars`, :func:`fmt_p`
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from signal_utils import contiguous_runs


# ── Pooling ───────────────────────────────────────────────────────────────────

def mean_sem(arr, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware mean, SEM (ddof=1) and n along ``axis``.

    ``n`` is counted per point, so an animal missing part of a trace lowers n only there. SEM is
    NaN where n < 2.
    """
    a = np.asarray(arr, float)
    n = np.sum(np.isfinite(a), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(a, axis=axis)
        sd = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    return m, sem, n


def group_means(values, groups) -> Tuple[list, np.ndarray]:
    """NaN-aware mean of per-session arrays within each group (e.g. animal).

    Parameters
    ----------
    values : sequence of array_like
        One array (scalar, trace, matrix) per session, all the same shape.
    groups : sequence
        Group label per session.

    Returns
    -------
    labels : list
        Sorted group labels.
    means : numpy.ndarray
        ``(n_groups, *value_shape)``; empty when there are no values.
    """
    values = list(values)
    if not values:
        return [], np.empty((0, 0))
    v = np.stack([np.asarray(x, float) for x in values])
    g = np.asarray(list(groups))
    labels = sorted(set(g.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return labels, np.stack([np.nanmean(v[g == lab], axis=0) for lab in labels])


def animal_means(df: pd.DataFrame, cols: Sequence[str], by_session: bool = True) -> pd.DataFrame:
    """Mean within session, then within animal, so every animal and session weighs equally.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have ``subject`` (and ``session`` if ``by_session``).
    cols : sequence of str
    by_session : bool
        If False, rows are already one per session (or are pooled within animal directly).

    Returns
    -------
    pandas.DataFrame
        Indexed by subject.
    """
    if by_session:
        df = df.groupby(["subject", "session"])[list(cols)].mean().reset_index()
    return df.groupby("subject")[list(cols)].mean()


# ── Tests ─────────────────────────────────────────────────────────────────────

def wilcoxon_animals(values, alternative: str = "two-sided") -> Tuple[float, float, int]:
    """Mean, Wilcoxon signed-rank p against zero, and n, over the finite per-animal values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(np.mean(v)) if len(v) else np.nan), np.nan, len(v)
    return float(v.mean()), float(stats.wilcoxon(v, alternative=alternative).pvalue), len(v)


def corr(df: pd.DataFrame, a: str, b: str, method: str = "pearson", min_n: int = 11
         ) -> Tuple[float, float]:
    """Correlation of two columns over their complete rows: ``(r, p)``, NaN below ``min_n`` rows.

    ``method`` is ``"pearson"`` or ``"spearman"``.
    """
    d = df[[a, b]].dropna()
    if len(d) < min_n:
        return np.nan, np.nan
    res = (stats.pearsonr if method == "pearson" else stats.spearmanr)(d[a], d[b])
    return float(res[0]), float(res[1])


def shift_p(obs: float, null) -> float:
    """One-sided p of ``obs`` against a null sample: ``(1 + #{null >= obs}) / (1 + n)``."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return (1 + int(np.sum(null >= obs))) / (1.0 + len(null))


def shift_z(obs: float, null) -> float:
    """``obs`` in SDs of the null above the null mean."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) < 2 or np.std(null) == 0:
        return np.nan
    return float((obs - np.mean(null)) / np.std(null))


def cluster_test(obs, null, freqs, alpha: float = 0.05, threshold_pct: float = 95.0):
    """Cluster-mass permutation test on a spectrum (or any 1-D curve).

    Contiguous runs of ``obs`` above the null's ``threshold_pct`` percentile are scored by their
    area above it and compared with the largest cluster each null draw produces anywhere. Narrow
    instrumental lines are rejected for lack of width; the trade is that a genuinely narrowband
    effect is rejected too.

    Parameters
    ----------
    obs : numpy.ndarray
        Observed curve, ``(n_freq,)``.
    null : numpy.ndarray
        Null curves drawn the same way, ``(n_null, n_freq)``.
    freqs : numpy.ndarray
        x value of each bin.
    alpha : float
        Cluster-level significance.
    threshold_pct : float
        Cluster-forming threshold, percentile of the null at each bin.

    Returns
    -------
    clusters : list of dict
        ``lo_Hz, hi_Hz, n_bins, peak, peak_Hz, mass, p`` for clusters above the critical mass.
    threshold : numpy.ndarray
        The cluster-forming threshold.
    critical_mass : float
    """
    obs = np.asarray(obs, float)
    null = np.asarray(null, float)
    thr = np.percentile(null, threshold_pct, axis=0)

    def masses(curve):
        return [(a, b, float(np.sum(curve[a:b + 1] - thr[a:b + 1])))
                for a, b in contiguous_runs(curve > thr)]

    null_max = np.array([max([m for _, _, m in masses(nl)], default=0.0) for nl in null])
    crit = float(np.percentile(null_max, 100 * (1 - alpha)))
    found = []
    for a, b, mass in masses(obs):
        if mass > crit:
            j = a + int(np.argmax(obs[a:b + 1]))
            found.append({"lo_Hz": float(freqs[a]), "hi_Hz": float(freqs[b]), "n_bins": b - a + 1,
                          "peak": float(obs[j]), "peak_Hz": float(freqs[j]), "mass": mass,
                          "p": (1 + int(np.sum(null_max >= mass))) / (1.0 + len(null))})
    return found, thr, crit


def hier_bootstrap(df: pd.DataFrame, metric: str, levels: Sequence[str] = ("subject",),
                   n_boot: int = 1000, seed: Optional[int] = 0) -> np.ndarray:
    """Bootstrapped means of ``metric``, resampling ``levels`` and then rows within them.

    Wraps ``aind_hierarchical_bootstrap.bootstrap`` (version 4). Its estimate is the pooled mean
    of the resampled rows, so a group with more rows weighs more: with one row per session and
    ``levels=("subject",)`` it is the session-weighted mean, not the mean of animal means.

    Returns
    -------
    numpy.ndarray
        ``n_boot`` bootstrapped means.
    """
    from aind_hierarchical_bootstrap.bootstrap import bootstrap

    d = df[list(levels) + [metric]].dropna().copy()
    for lev in levels:
        d[lev] = d[lev].astype(str)
    if seed is not None:
        np.random.seed(seed)  # the library draws from numpy's global generator
    return np.asarray(bootstrap(d, metric=metric, levels=list(levels), nboots=n_boot)[metric])


def fdr_bh(p) -> np.ndarray:
    """Benjamini–Hochberg q-values (statsmodels); NaN where p is NaN."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def wilson_ci(k, n, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (statsmodels); NaN when n = 0."""
    from statsmodels.stats.proportion import proportion_confint

    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


# ── Formatting ────────────────────────────────────────────────────────────────

def stars(p: float) -> str:
    """``***`` / ``**`` / ``*`` / ``n.s.``; ``n/a`` for a missing p."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def fmt_p(pvalue: float, floor: float = 1e-300) -> str:
    """Format a p-value, reporting underflow as a bound instead of "0".

    SciPy returns exactly 0.0 when a p-value underflows double precision, which reads as a
    computed zero. Returns ``"= 0.0123"`` or ``"< 1e-300"``, to follow ``"p "``.
    """
    if not np.isfinite(pvalue):
        return "= nan"
    return "< {:.0e}".format(floor) if pvalue < floor else "= {:.3g}".format(pvalue)
