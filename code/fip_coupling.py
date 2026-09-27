"""
fip_coupling.py — machinery for ``fip_05_da_ne_rpe_coupling.ipynb`` (DA x NE coupling).

Reads the per-session parquet hierarchy that Rachel's grouped analysis wrapper writes
(``<root>/<subject>/<session>/df_fip.parquet`` + ``df_trials.parquet`` + ``df_events.parquet``,
with ``df_curation_<subject>.csv`` beside the subject folders). Both FIP assets used here share
that layout and were curated when they were built, so every fiber in ``df_fip`` already passed
curation and the event names are the targets (``latNAcc(L)-DA``, ``PL(L)-LCAxonCa``).

Unlike ``fip_utils.py`` this module needs only numpy / pandas / scipy, so the notebook runs
locally as well as on Code Ocean.

Contents
--------
Loading
    :func:`find_asset_root`, :func:`inventory_pairs`, :func:`choose_side`, :func:`load_pairs`
Per-trial measures
    :func:`window_mean`, :func:`peri_event`, :func:`trial_measures`, :func:`fit_rpe_terms`,
    :func:`residualize`
Continuous signals
    :func:`event_design`, :func:`task_residuals`, :func:`band_corr_with_null`,
    :func:`residual_xcorr`
Transients
    :func:`detect_transients`, :func:`nearest_partner`, :func:`label_context`,
    :func:`shift_coincidence_null`
Across animals
    :func:`animal_means`, :func:`wilcoxon_animals`

Clock: every time is session time (s from the first go cue), the clock ``df_fip['timestamps']``
and the ``*_in_session`` trial columns share. Each loaded pair is resampled onto a uniform
``fs``-Hz grid starting at ``t0``; sample ``i`` sits at ``t0 + i / fs``.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.sparse import csr_matrix


# ── Loading ───────────────────────────────────────────────────────────────────

#: Same-side DA and NE labels. ``medNAcc`` is left out so every animal contributes one DA site
#: (lateral NAc); ``-unconfirmed`` PL fibers never survive curation and are excluded anyway.
DA_RE = re.compile(r"^latNAcc\((L|R)\)-DA$")
NE_RE = re.compile(r"^PL\((L|R)\)-LCAxonCa$")

#: Trial columns carried into the cache. ``RPE_all`` / ``Q_*`` come from the behavioral model
#: fit Rachel's wrapper attaches (``model_name`` column, a Q-learning model with choice kernel).
TRIAL_COLS = [
    "trial", "goCue_start_time_in_session", "choice_time_in_session",
    "reward_outcome_time_in_session", "animal_response", "earned_reward", "RPE_all",
    "Q_chosen", "Q_unchosen", "Q_sum", "num_reward_past", "response_time",
]


def find_asset_root(candidates: Sequence[str]) -> Optional[str]:
    """First candidate directory that holds ``df_curation_*.csv`` files.

    Parameters
    ----------
    candidates : sequence of str
        Directories to try in order. For each, the directory itself and its ``data/``
        subfolder are checked (Rachel's wrapper output nests the hierarchy under ``data/``).

    Returns
    -------
    str or None
        The directory that directly contains the subject folders, or None if none match.
    """
    for c in candidates:
        for d in (c, os.path.join(c, "data")):
            if glob.glob(os.path.join(d, "df_curation_*.csv")):
                return d
    return None


def inventory_pairs(asset_roots: Dict[str, str]) -> pd.DataFrame:
    """One row per (session, hemisphere) that has both a DA and an NE fiber.

    Parameters
    ----------
    asset_roots : dict
        ``{asset_name: root}`` with roots from :func:`find_asset_root`.

    Returns
    -------
    pandas.DataFrame
        Columns ``asset, subject, session, side, da_event, ne_event, path``.
    """
    import pyarrow.parquet as pq

    rows = []
    for asset, root in asset_roots.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "df_fip.parquet"))):
            events = set(pq.read_table(f, columns=["event"]).column("event").to_pylist())
            da = {DA_RE.match(e).group(1): e for e in events if DA_RE.match(e)}
            ne = {NE_RE.match(e).group(1): e for e in events if NE_RE.match(e)}
            session_dir = os.path.dirname(f)
            for side in sorted(set(da) & set(ne)):
                rows.append(dict(asset=asset, subject=os.path.basename(os.path.dirname(session_dir)),
                                 session=os.path.basename(session_dir), side=side,
                                 da_event=da[side], ne_event=ne[side], path=session_dir))
    return pd.DataFrame(rows)


def choose_side(inv: pd.DataFrame, min_sessions: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold each animal to one hemisphere and drop animals with too few sessions.

    Parameters
    ----------
    inv : pandas.DataFrame
        From :func:`inventory_pairs`.
    min_sessions : int
        Animals with fewer same-side sessions than this are dropped.

    Returns
    -------
    kept : pandas.DataFrame
        The rows of ``inv`` used downstream.
    table : pandas.DataFrame
        Per animal: sessions per side, the side used, and whether the animal is kept.
    """
    table = pd.crosstab(inv["subject"], inv["side"])
    for s in ("L", "R"):
        if s not in table:
            table[s] = 0
    table["asset"] = inv.groupby("subject")["asset"].first()
    table["side_used"] = np.where(table["L"] >= table["R"], "L", "R")
    table["n_used"] = np.where(table["side_used"] == "L", table["L"], table["R"])
    table["kept"] = table["n_used"] >= min_sessions
    side_of = table["side_used"]
    kept = inv[(inv["side"] == inv["subject"].map(side_of))
               & inv["subject"].map(table["kept"])].reset_index(drop=True)
    return kept, table[["asset", "L", "R", "side_used", "n_used", "kept"]]


def _load_one(row, fs: float, pre_s: float, post_s: float) -> dict:
    fip = pd.read_parquet(os.path.join(row.path, "df_fip.parquet"),
                          columns=["timestamps", "data", "event"])
    trials = pd.read_parquet(os.path.join(row.path, "df_trials.parquet"))
    events = pd.read_parquet(os.path.join(row.path, "df_events.parquet"),
                             columns=["timestamps", "event"])
    cues = trials["goCue_start_time_in_session"].dropna().to_numpy()
    t0 = float(cues.min() - pre_s)
    grid = np.arange(t0, float(cues.max() + post_s), 1.0 / fs)
    traces = {}
    for name, ev in (("da", row.da_event), ("ne", row.ne_event)):
        g = fip[fip["event"] == ev].sort_values("timestamps")
        g = g[np.isfinite(g["data"])]
        traces[name] = np.interp(grid, g["timestamps"].to_numpy(),
                                 g["data"].to_numpy()).astype(np.float32)
    cols = [c for c in TRIAL_COLS if c in trials]
    licks = np.sort(events.loc[events["event"].str.contains("lick"), "timestamps"].to_numpy())
    return dict(asset=row.asset, subject=row.subject, session=row.session, side=row.side,
                t0=t0, fs=fs, da=traces["da"], ne=traces["ne"],
                trials=trials[cols].reset_index(drop=True), licks=licks)


def load_pairs(kept: pd.DataFrame, fs: float = 20.0, pre_s: float = 5.0, post_s: float = 10.0,
               cache_path: Optional[str] = None, verbose: bool = True) -> List[dict]:
    """Load every kept pair onto a uniform grid covering the task period.

    Parameters
    ----------
    kept : pandas.DataFrame
        From :func:`choose_side`.
    fs : float
        Grid rate in Hz (the FIP rate is 20 Hz, so this is a resample onto exact spacing).
    pre_s, post_s : float
        Grid runs from ``pre_s`` before the first go cue to ``post_s`` after the last one.
    cache_path : str, optional
        If given and present, loaded from there; otherwise written there after loading. The cache
        is keyed on the session list, so a changed selection reloads.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict
        One dict per session: ``asset, subject, session, side, t0, fs, da, ne, trials, licks``.
        ``da`` / ``ne`` are dF/F (``data`` column, ``dff-bright_mc-iso-IRLS`` preprocessing),
        not z-scored.
    """
    key = tuple(kept["path"])
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key and cached.get("fs") == fs:
            if verbose:
                print(f"loaded {len(cached['pairs'])} pairs from cache {cache_path}")
            return cached["pairs"]
    pairs = []
    for i, row in enumerate(kept.itertuples()):
        pairs.append(_load_one(row, fs, pre_s, post_s))
        if verbose and (i + 1) % 20 == 0:
            print(f"  loaded {i + 1} / {len(kept)}")
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(dict(key=key, fs=fs, pairs=pairs), fh)
    return pairs


# ── Per-trial measures ────────────────────────────────────────────────────────

def zscore(x) -> np.ndarray:
    """Z-score over the whole array (float64)."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def _idx(times, t0: float, fs: float) -> np.ndarray:
    return np.round((np.asarray(times, dtype=float) - t0) * fs).astype(int)


def window_mean(x: np.ndarray, t0: float, fs: float, times, a: float, b: float) -> np.ndarray:
    """Mean of ``x`` over ``[time + a, time + b)`` for each time; NaN if out of range or NaN time."""
    times = np.asarray(times, dtype=float)
    out = np.full(len(times), np.nan)
    ok = np.isfinite(times)
    i0 = _idx(times[ok] + a, t0, fs)
    i1 = _idx(times[ok] + b, t0, fs)
    c = np.r_[0.0, np.cumsum(x, dtype=float)]
    valid = (i0 >= 0) & (i1 <= len(x)) & (i1 > i0)
    vals = np.full(ok.sum(), np.nan)
    vals[valid] = (c[i1[valid]] - c[i0[valid]]) / (i1[valid] - i0[valid])
    out[ok] = vals
    return out


def peri_event(x: np.ndarray, t0: float, fs: float, times, lags_s: np.ndarray) -> np.ndarray:
    """Event-aligned segments, shape ``(len(times), len(lags_s))``; NaN rows where out of range."""
    times = np.asarray(times, dtype=float)
    lag_i = np.round(np.asarray(lags_s) * fs).astype(int)
    out = np.full((len(times), len(lag_i)), np.nan)
    ok = np.isfinite(times)
    idx = _idx(times[ok], t0, fs)
    inside = (idx + lag_i[0] >= 0) & (idx + lag_i[-1] < len(x))
    rows = np.flatnonzero(ok)[inside]
    out[rows] = x[idx[inside][:, None] + lag_i]
    return out


def trial_measures(pair: dict, windows: Dict[str, Tuple[float, float]],
                   baseline: Tuple[float, float] = (-1.0, 0.0),
                   latency_window: Tuple[float, float] = (0.0, 1.5)) -> pd.DataFrame:
    """Per-trial baseline, outcome-response and peak-latency measures for DA and NE.

    Each signal is z-scored over the session grid first. Response windows are aligned to
    ``reward_outcome_time_in_session`` and baseline-subtracted (``baseline`` is relative to the
    go cue, so the pre-trial level is removed from the response). Peak latency is the time of the
    maximum within ``latency_window`` after the go cue.

    Parameters
    ----------
    pair : dict
        One element of :func:`load_pairs`.
    windows : dict
        ``{name: (a, b)}`` response windows in s relative to the outcome.
    baseline : tuple
        Pre-cue baseline window relative to the go cue.
    latency_window : tuple
        Window after the go cue searched for the peak.

    Returns
    -------
    pandas.DataFrame
        ``pair['trials']`` plus ``{sig}_pre``, ``{sig}_{window}``, ``{sig}_lat``, ``{sig}_peak``
        for ``sig`` in ``da``, ``ne``; ``trial_frac`` (position in session, 0-1); ``rewarded``,
        ``responded``, ``subject``, ``session``.
    """
    tr = pair["trials"].copy()
    t0, fs = pair["t0"], pair["fs"]
    cue = tr["goCue_start_time_in_session"].to_numpy()
    out = tr["reward_outcome_time_in_session"].to_numpy()
    lags = np.arange(int(round(latency_window[0] * fs)), int(round(latency_window[1] * fs))) / fs
    for sig in ("da", "ne"):
        x = zscore(pair[sig])
        pre = window_mean(x, t0, fs, cue, *baseline)
        tr[f"{sig}_pre"] = pre
        for name, (a, b) in windows.items():
            tr[f"{sig}_{name}"] = window_mean(x, t0, fs, out, a, b) - pre
        seg = peri_event(x, t0, fs, cue, lags)
        good = np.isfinite(seg).all(axis=1)
        lat = np.full(len(tr), np.nan)
        peak = np.full(len(tr), np.nan)
        lat[good] = lags[np.argmax(seg[good], axis=1)]
        peak[good] = seg[good].max(axis=1) - pre[good]
        tr[f"{sig}_lat"] = lat
        tr[f"{sig}_peak"] = peak
    tr["responded"] = tr["animal_response"] < 2
    tr["rewarded"] = tr["earned_reward"].astype(bool) & tr["responded"]
    tr["trial_frac"] = np.arange(len(tr)) / max(len(tr) - 1, 1)
    tr["subject"] = pair["subject"]
    tr["session"] = pair["session"]
    return tr


def fit_rpe_terms(df: pd.DataFrame, y: str) -> pd.Series:
    """OLS of a response on reward and on RPE separately within each outcome.

    ``y ~ 1 + rewarded + RPE·rewarded + RPE·(1 − rewarded)``. Within an outcome class RPE is
    ``1 − Q_chosen`` (rewarded) or ``−Q_chosen`` (unrewarded), so the two slopes ask whether the
    response scales with how expected that outcome was. A signed-RPE signal has a positive slope in
    both classes; a pure outcome signal has zero slopes; an unsigned surprise signal has a positive
    slope for rewards and a negative one for omissions.

    Parameters
    ----------
    df : pandas.DataFrame
        Responded trials with ``rewarded``, ``RPE_all`` and column ``y``.
    y : str
        Response column.

    Returns
    -------
    pandas.Series
        ``intercept, reward, rpe_rew, rpe_unrew`` (units of ``y`` per unit RPE), and ``n``.
    """
    d = df[["rewarded", "RPE_all", y]].dropna()
    r = d["rewarded"].to_numpy(float)
    rpe = d["RPE_all"].to_numpy(float)
    X = np.c_[np.ones(len(d)), r, rpe * r, rpe * (1 - r)]
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)
    return pd.Series(dict(intercept=beta[0], reward=beta[1], rpe_rew=beta[2], rpe_unrew=beta[3],
                          n=len(d)))


def residualize(df: pd.DataFrame, y: str, covariates: Sequence[str],
                group: Optional[str] = "session") -> pd.Series:
    """Residual of ``y`` after OLS on ``covariates``, fit separately within each ``group``.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
    covariates : sequence of str
        Numeric columns (booleans are cast to float). Interactions are passed as precomputed columns.
    group : str or None
        Fit within each level (default per session, which also removes session means).

    Returns
    -------
    pandas.Series
        Aligned to ``df.index``; NaN where ``y`` or any covariate is missing.
    """
    out = pd.Series(np.nan, index=df.index)
    groups = [(None, df)] if group is None else df.groupby(group)
    for _, g in groups:
        g = g[[y, *covariates]].dropna()
        if len(g) <= len(covariates) + 2:
            continue
        X = np.c_[np.ones(len(g)), g[list(covariates)].to_numpy(float)]
        beta, *_ = np.linalg.lstsq(X, g[y].to_numpy(float), rcond=None)
        out.loc[g.index] = g[y].to_numpy(float) - X @ beta
    return out


# ── Continuous signals ────────────────────────────────────────────────────────

def event_design(pair: dict, kernel_s: Tuple[float, float] = (-1.0, 4.0)):
    """Sparse FIR design matrix of task events for one session.

    One boxcar regressor per lag in ``kernel_s`` for each of: go cue, outcome on rewarded trials,
    outcome on unrewarded trials, and every lick. Plus an intercept. Kernels are shared across
    trials, so trial-to-trial amplitude variation (RPE, noise) stays in the residual.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_samples, 4 * n_lags + 1)``.
    """
    n = len(pair["da"])
    fs, t0 = pair["fs"], pair["t0"]
    tr = pair["trials"]
    responded = (tr["animal_response"] < 2).to_numpy()
    rewarded = tr["earned_reward"].astype(bool).to_numpy() & responded
    out = tr["reward_outcome_time_in_session"].to_numpy()
    events = [tr["goCue_start_time_in_session"].to_numpy(), out[rewarded],
              out[responded & ~rewarded], pair["licks"]]
    lag_i = np.arange(int(round(kernel_s[0] * fs)), int(round(kernel_s[1] * fs)))
    rows, cols = [], []
    for e, times in enumerate(events):
        times = times[np.isfinite(times)]
        idx = _idx(times, t0, fs)
        for j, k in enumerate(lag_i):
            ii = idx + k
            ii = ii[(ii >= 0) & (ii < n)]
            rows.append(ii)
            cols.append(np.full(len(ii), e * len(lag_i) + j))
    n_col = len(events) * len(lag_i)
    rows.append(np.arange(n))
    cols.append(np.full(n, n_col))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_col + 1))


def task_residuals(pair: dict, kernel_s: Tuple[float, float] = (-1.0, 4.0),
                   ridge: float = 1e-3) -> dict:
    """Split each z-scored signal into a task-evoked fit and a residual.

    Parameters
    ----------
    pair : dict
    kernel_s : tuple
        Kernel span relative to each event, s.
    ridge : float
        Small ridge on the normal equations, for overlapping-lick collinearity.

    Returns
    -------
    dict
        ``{sig: z, sig + '_fit': fitted, sig + '_res': residual, sig + '_r2': fraction explained}``
        for ``sig`` in ``da``, ``ne``.
    """
    X = event_design(pair, kernel_s)
    XtX = (X.T @ X).toarray()
    XtX[np.diag_indices_from(XtX)] += ridge
    out = {}
    for sig in ("da", "ne"):
        y = zscore(pair[sig])
        beta = np.linalg.solve(XtX, X.T @ y)
        fit = X @ beta
        out[sig] = y
        out[sig + "_fit"] = fit
        out[sig + "_res"] = y - fit
        out[sig + "_r2"] = 1.0 - np.var(y - fit) / np.var(y)
    return out


def _circ_corr_all_shifts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r of ``a`` with ``b`` circularly shifted by every k (FFT)."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))) / len(a)


def band_corr_with_null(a: np.ndarray, b: np.ndarray, fs: float,
                        bands: Sequence[Tuple[float, float]], min_shift_s: float = 120.0,
                        n_null: int = 200, rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-lag correlation of two signals within each frequency band, and a circular-shift null.

    Each band is a zero-phase 2nd-order Butterworth band-pass. The null circularly shifts ``b`` by
    at least ``min_shift_s`` (both directions), which keeps each band-passed signal's own
    autocorrelation. At slow bands a 75-min session holds few independent cycles, so single-session
    r is noisy; the null's spread shows how noisy.

    Returns
    -------
    r : ndarray, shape (n_bands,)
    null : ndarray, shape (n_bands, n_null)
    """
    rng = rng or np.random.default_rng(0)
    n = len(a)
    lo = int(min_shift_s * fs)
    shifts = rng.integers(lo, n - lo, size=n_null)
    r = np.full(len(bands), np.nan)
    null = np.full((len(bands), n_null), np.nan)
    for k, (f_lo, f_hi) in enumerate(bands):
        sos = signal.butter(2, [f_lo, f_hi], btype="band", fs=fs, output="sos")
        fa = signal.sosfiltfilt(sos, a)
        fb = signal.sosfiltfilt(sos, b)
        cc = _circ_corr_all_shifts(fa, fb)
        r[k] = cc[0]
        null[k] = cc[shifts]
    return r, null


def residual_xcorr(a: np.ndarray, b: np.ndarray, fs: float, maxlag_s: float = 3.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized cross-correlation; a peak at positive lag means ``b`` follows ``a``.

    Returns
    -------
    lags_s, r : ndarray
        ``r[k] = corr(a[t], b[t + lag_k])``.
    """
    cc = _circ_corr_all_shifts(b, a)  # cc[k] = mean(b[t+k] a[t])
    m = int(round(maxlag_s * fs))
    lags = np.arange(-m, m + 1)
    return lags / fs, cc[lags % len(a)]


# ── Transients ────────────────────────────────────────────────────────────────

def detect_transients(x: np.ndarray, fs: float, prominence: float = 2.0,
                      min_sep_s: float = 0.5) -> pd.DataFrame:
    """Peaks of a z-scored trace with at least ``prominence`` z of prominence.

    Prominence (height above the higher of the two flanking minima) measures a transient against
    its local surround, so a peak riding on a slow elevation is scored by its own size.

    Returns
    -------
    pandas.DataFrame
        ``i`` (sample), ``prominence``, ``height`` (z at the peak).
    """
    i, props = signal.find_peaks(x, prominence=prominence, distance=max(1, int(min_sep_s * fs)))
    return pd.DataFrame(dict(i=i, prominence=props["prominences"], height=x[i]))


def nearest_partner(t_a: np.ndarray, t_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each time in ``t_a``, the index of and signed lag to the nearest time in sorted ``t_b``.

    Returns
    -------
    idx : ndarray of int  (-1 when ``t_b`` is empty)
    lag : ndarray  ``t_b[idx] - t_a`` (positive: the partner comes later)
    """
    if len(t_b) == 0:
        return np.full(len(t_a), -1), np.full(len(t_a), np.inf)
    j = np.searchsorted(t_b, t_a)
    lo = np.clip(j - 1, 0, len(t_b) - 1)
    hi = np.clip(j, 0, len(t_b) - 1)
    pick = np.where(np.abs(t_b[hi] - t_a) < np.abs(t_b[lo] - t_a), hi, lo)
    return pick, t_b[pick] - t_a


def label_context(times: np.ndarray, pair: dict, outcome_win: Tuple[float, float] = (0.0, 1.5),
                  cue_win: Tuple[float, float] = (0.0, 1.5), lick_win_s: float = 1.0) -> np.ndarray:
    """Task context of each transient time.

    In order of precedence: ``rewarded`` / ``unrewarded`` (within ``outcome_win`` of that outcome),
    ``cue, no response`` (within ``cue_win`` of the go cue on an ignored trial), ``licking``
    (a lick within ±``lick_win_s``), ``quiet`` (none of these).
    """
    tr = pair["trials"]
    responded = (tr["animal_response"] < 2).to_numpy()
    rewarded = tr["earned_reward"].astype(bool).to_numpy() & responded
    out = tr["reward_outcome_time_in_session"].to_numpy()
    cue = tr["goCue_start_time_in_session"].to_numpy()

    def within(ev, win):
        ev = np.sort(ev[np.isfinite(ev)])
        if len(ev) == 0:
            return np.zeros(len(times), bool)
        j = np.searchsorted(ev, times, side="right") - 1  # last event at or before t
        ok = j >= 0
        d = np.full(len(times), np.inf)
        d[ok] = times[ok] - ev[j[ok]]
        return (d >= win[0]) & (d <= win[1])

    labels = np.full(len(times), "quiet", dtype=object)
    _, lag = nearest_partner(times, pair["licks"])
    labels[np.abs(lag) <= lick_win_s] = "licking"
    labels[within(cue[~responded], cue_win)] = "cue, no response"
    labels[within(out[responded & ~rewarded], outcome_win)] = "unrewarded"
    labels[within(out[rewarded], outcome_win)] = "rewarded"
    return labels


def shift_coincidence_null(t_a: np.ndarray, t_b: np.ndarray, duration: float, window: float,
                           min_shift_s: float = 30.0, n_null: int = 200,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Fraction of ``t_a`` with a ``t_b`` partner within ``window``, with ``t_b`` circularly shifted.

    Keeps both trains' own rates and clustering and removes only their alignment. Note that task
    structure is also removed, so this null asks whether coincidences exceed what the two event
    rates alone predict, not whether they exceed shared task drive.

    Returns
    -------
    ndarray, shape (n_null,)
    """
    rng = rng or np.random.default_rng(0)
    out = np.full(n_null, np.nan)
    if len(t_a) == 0 or len(t_b) == 0:
        return out
    for k, s in enumerate(rng.uniform(min_shift_s, duration - min_shift_s, n_null)):
        shifted = np.sort((t_b + s) % duration)
        _, lag = nearest_partner(t_a, shifted)
        out[k] = np.mean(np.abs(lag) <= window)
    return out


# ── Across animals ────────────────────────────────────────────────────────────

def animal_means(df: pd.DataFrame, cols: Sequence[str], by_session: bool = True) -> pd.DataFrame:
    """Mean within session, then within animal, so every animal and session weighs equally.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have ``subject`` (and ``session`` if ``by_session``).
    cols : sequence of str
    by_session : bool
        If False, pool rows within animal directly.

    Returns
    -------
    pandas.DataFrame indexed by subject.
    """
    if by_session:
        df = df.groupby(["subject", "session"])[list(cols)].mean().reset_index()
    return df.groupby("subject")[list(cols)].mean()


def wilcoxon_animals(values, alternative: str = "two-sided") -> Tuple[float, float, int]:
    """Mean, Wilcoxon signed-rank p against zero, and n, over finite per-animal values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float(np.mean(v)) if len(v) else np.nan, np.nan, len(v)
    return float(v.mean()), float(stats.wilcoxon(v, alternative=alternative).pvalue), len(v)
