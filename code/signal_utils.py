"""
signal_utils.py — generic time-series helpers (numpy / scipy only).

Nothing here knows about FIP, motion energy or the task. Series-specific code lives in
``fip_utils`` / ``men_utils``; across-animal statistics live in ``stats_utils``.

Two kinds of input:

* **Irregular samples** ``(t, y)``: :func:`threshold_onsets`, :func:`bin_to_grid`.
  (For event alignment of irregular samples use the AIND primitive,
  ``aind_dynamic_foraging_data_utils.alignment.event_triggered_response``, or its wrapper
  ``fip_utils.peri_event``.)
* **A uniform grid** ``x`` with sample ``i`` at ``t0 + i / fs``: everything else.

Lag convention (one for the whole repo): for :func:`norm_xcorr` and :func:`circular_xcorr`,
``out[lag] = mean_t(a[t + lag] * b[t])``, so a peak at **positive lag means ``a`` follows
``b``** (``b`` leads). ``norm_xcorr(ne, da)`` puts "NE later" at positive lags.

Contents
--------
Normalising and events
    :func:`zscore`, :func:`threshold_onsets`, :func:`contiguous_runs`, :func:`detect_transients`
Grids
    :func:`bin_to_grid`, :func:`grid_index`, :func:`peri_event_grid`, :func:`window_mean_grid`,
    :func:`rolling_mean`
Correlation
    :func:`norm_xcorr`, :func:`circular_xcorr`, :func:`xcorr_shift_null`,
    :func:`coherence_shift_null`, :func:`band_corr_with_null`
Event coincidence
    :func:`nearest_partner`, :func:`near_any`, :func:`shift_times`, :func:`coincidence_null`
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal


# ── Normalising and events ────────────────────────────────────────────────────

def zscore(y, ddof: int = 1) -> np.ndarray:
    """Z-score a 1-D array, ignoring NaNs.

    ``ddof=1`` matches ``aind_dynamic_foraging_data_utils.enrich_dfs.zscore_fip``
    (``scipy.stats.zscore(x, ddof=1, nan_policy='omit')``), so a trace z-scored here and the
    same trace's ``data_z`` column agree. Returns the mean-centred array when the SD is zero or
    not finite.
    """
    y = np.asarray(y, float)
    sd = np.nanstd(y, ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        return y - np.nanmean(y)
    return (y - np.nanmean(y)) / sd


def threshold_onsets(
    t,
    y,
    z_thresh: float = 2.5,
    refractory: float = 0.5,
    min_run: int = 1,
    already_z: bool = False,
    min_run_s: Optional[float] = None,
) -> np.ndarray:
    """Causal upward threshold crossings of a z-scored trace (no smoothing).

    An onset is the first sample that rises above ``z_thresh`` SD and then stays above for at
    least ``min_run`` samples. That rejects single-sample spikes without smoothing or shifting
    timing. ``refractory`` (s) keeps one onset per bout.

    Parameters
    ----------
    t, y : array_like
        Timestamps and signal. ``y`` is z-scored here unless ``already_z``.
    z_thresh : float
        Threshold in SD.
    refractory : float
        Minimum spacing between reported onsets, s.
    min_run : int
        Samples that must stay above threshold.
    already_z : bool
        True when ``y`` is already z-scored (e.g. a ``data_z`` column).
    min_run_s : float, optional
        The run length in seconds instead, converted with the median sample interval of ``t``;
        overrides ``min_run``.

    Returns
    -------
    numpy.ndarray
        Onset times, in ``t``'s units.
    """
    z = np.asarray(y, float) if already_z else zscore(y)
    if min_run_s is not None:
        min_run = max(1, int(round(min_run_s / np.median(np.diff(np.asarray(t, float))))))
    above = z > z_thresh  # NaN compares False -> below threshold
    idx = np.where((~above[:-1]) & above[1:])[0] + 1
    if min_run > 1 and len(idx):
        idx = np.array(
            [i for i in idx if i + min_run <= len(above) and above[i:i + min_run].all()],
            dtype=int,
        )
    times = np.asarray(t, float)[idx]
    if len(times):
        times = times[np.insert(np.diff(times) > refractory, 0, True)]
    return times


def contiguous_runs(mask) -> List[Tuple[int, int]]:
    """``[(start, end)]`` index pairs (inclusive) for each run of True in a boolean mask."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def detect_transients(x: np.ndarray, fs: float, prominence: float = 2.0,
                      min_sep_s: float = 0.5) -> pd.DataFrame:
    """Peaks of a z-scored trace with at least ``prominence`` of prominence.

    Prominence (height above the higher of the two flanking minima) measures a transient against
    its local surround, so a peak riding on a slow rise is scored by its own size.

    Returns
    -------
    pandas.DataFrame
        ``i`` (sample), ``prominence``, ``height`` (value at the peak).
    """
    i, props = signal.find_peaks(x, prominence=prominence, distance=max(1, int(min_sep_s * fs)))
    return pd.DataFrame(dict(i=i, prominence=props["prominences"], height=x[i]))


# ── Grids ─────────────────────────────────────────────────────────────────────

def bin_to_grid(t: np.ndarray, y: np.ndarray, t0: float, fs: float, n: int) -> np.ndarray:
    """Average ``y`` within each bin of a uniform grid; NaN where a bin has no samples.

    Bin ``i`` is centred on ``t0 + i / fs`` and spans ``1 / fs``, so taking a fast signal onto a
    slower grid averages it rather than sampling it (500 Hz motion energy onto 20 Hz averages 25
    frames per bin, where ``np.interp`` would keep one and alias the rest).
    """
    k = np.floor((np.asarray(t, float) - t0) * fs + 0.5).astype(np.int64)
    y = np.asarray(y, float)
    ok = (k >= 0) & (k < n) & np.isfinite(y)
    total = np.bincount(k[ok], weights=y[ok], minlength=n)
    count = np.bincount(k[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def grid_index(times, t0: float, fs: float) -> np.ndarray:
    """Nearest grid sample of each time (may fall outside the grid)."""
    return np.round((np.asarray(times, dtype=float) - t0) * fs).astype(np.int64)


def peri_event_grid(x: np.ndarray, t0: float, fs: float, times, lags_s,
                    censor_after=None) -> np.ndarray:
    """Event-aligned segments of a grid signal, shape ``(len(times), len(lags_s))``.

    Samples off the grid, and rows for NaN times, are NaN; a window that runs past either end is
    kept in part.

    Parameters
    ----------
    x : numpy.ndarray
        Signal on the grid.
    t0, fs : float
        Time of sample 0, and sampling rate.
    times : array_like
        Event times.
    lags_s : array_like
        Offsets from each event, s.
    censor_after : array_like, optional
        One time per event; samples at or after it are NaN (e.g. the next go cue, so a cue window
        does not run into the next trial).
    """
    x = np.asarray(x)
    times = np.asarray(times, dtype=float)
    lags_s = np.asarray(lags_s, float)
    lag_i = np.round(lags_s * fs).astype(np.int64)
    out = np.full((len(times), len(lag_i)), np.nan)
    ok = np.isfinite(times)
    if not ok.any():
        return out
    idx = grid_index(times[ok], t0, fs)[:, None] + lag_i
    inside = (idx >= 0) & (idx < len(x))
    seg = np.full(idx.shape, np.nan)
    seg[inside] = x[idx[inside]]
    out[ok] = seg
    if censor_after is not None:
        c = np.asarray(censor_after, float)[:, None]
        with np.errstate(invalid="ignore"):
            out[(times[:, None] + lags_s[None, :]) >= c] = np.nan
    return out


def window_mean_grid(x: np.ndarray, t0: float, fs: float, times, a, b) -> np.ndarray:
    """Mean of a grid signal over ``[time + a, time + b)`` for each time.

    ``a`` and ``b`` are scalars, or one value per time (e.g. ``b`` = each trial's response time
    for a go cue → first lick window). NaN samples inside the window are skipped. The result is
    NaN when the time or a bound is NaN, the window runs off the grid, or it holds no finite sample.
    """
    x = np.asarray(x, float)
    times = np.asarray(times, dtype=float)
    a = np.broadcast_to(np.asarray(a, float), times.shape)
    b = np.broadcast_to(np.asarray(b, float), times.shape)
    out = np.full(len(times), np.nan)
    ok = np.isfinite(times) & np.isfinite(a) & np.isfinite(b)
    i0 = grid_index(times[ok] + a[ok], t0, fs)
    i1 = grid_index(times[ok] + b[ok], t0, fs)
    finite = np.isfinite(x)
    c = np.r_[0.0, np.cumsum(np.where(finite, x, 0.0))]
    cn = np.r_[0, np.cumsum(finite)]
    valid = (i0 >= 0) & (i1 <= len(x)) & (i1 > i0)
    vals = np.full(ok.sum(), np.nan)
    n = cn[i1[valid]] - cn[i0[valid]]
    with np.errstate(invalid="ignore", divide="ignore"):
        vals[valid] = np.where(n > 0, (c[i1[valid]] - c[i0[valid]]) / n, np.nan)
    out[ok] = vals
    return out


def rolling_mean(x: np.ndarray, fs: float, win_s: float, step_s: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Centred running mean of ``x`` over ``win_s`` s, sampled every ``step_s`` s.

    NaN-aware: a window's mean uses its finite samples and is NaN when fewer than half are
    finite. Returns ``(offsets_s, means)``, offsets from the start of ``x``.
    """
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    c = np.r_[0.0, np.cumsum(np.where(finite, x, 0.0))]
    cn = np.r_[0, np.cumsum(finite)]
    half = int(round(win_s * fs / 2))
    centres = np.arange(0, len(x), int(round(step_s * fs)))
    a = np.clip(centres - half, 0, len(x))
    b = np.clip(centres + half, 0, len(x))
    n = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (c[b] - c[a]) / n
    m[n < 0.5 * (2 * half)] = np.nan
    return centres / fs, m


# ── Correlation ───────────────────────────────────────────────────────────────

def norm_xcorr(a, b, fs: float, maxlag: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two equal-length signals on one uniform grid.

    ``out[lag] = mean_t(a[t + lag] * b[t])`` with ``a``, ``b`` z-scored, over the overlap at each
    lag (NaN samples skipped). A peak at **positive lag means ``a`` follows ``b``**.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lags_s, r)``.
    """
    a, b = zscore(a), zscore(b)
    n = len(a)
    k = int(round(maxlag * fs))
    lags = np.arange(-k, k + 1)
    out = np.empty(len(lags), float)
    for i, lag in enumerate(lags):
        if lag >= 0:
            out[i] = np.nanmean(a[lag:] * b[:n - lag])
        else:
            out[i] = np.nanmean(a[:n + lag] * b[-lag:])
    return lags / float(fs), out


def circular_xcorr(a, b) -> np.ndarray:
    """Circular normalised cross-correlation at every shift (FFT).

    ``out[k] = mean_t(a[t + k] * b[t])``, indices mod ``n``, with ``a`` and ``b`` z-scored
    (no NaNs allowed), so index ``k`` matches :func:`norm_xcorr`'s lag ``k``. ``out[0]`` is the
    zero-lag Pearson r, and ``out[s]`` is that r after shifting ``b`` back by ``s`` samples.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = len(a)
    return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n) / n


def _shifts(n: int, fs: float, n_shift: int, min_shift_s: float,
            rng: Optional[np.random.Generator]) -> np.ndarray:
    """``n_shift`` circular shifts (samples) of at least ``min_shift_s`` either way."""
    rng = rng if rng is not None else np.random.default_rng(0)
    lo = int(round(min_shift_s * fs))
    if n - 2 * lo < 1:
        raise ValueError("signal of %d samples is too short for a %.0f-s minimum shift"
                         % (n, min_shift_s))
    return rng.integers(lo, n - lo, size=n_shift)


def xcorr_shift_null(a, b, fs: float, maxlag: float = 3.0, n_shift: int = 200,
                     min_shift_s: float = 60.0, rng: Optional[np.random.Generator] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`norm_xcorr` plus a circular-shift null of the whole curve.

    The null shifts ``b`` against ``a`` by at least ``min_shift_s``, which keeps each signal's
    own autocorrelation and removes only their alignment. One FFT gives every null curve:
    shifting by ``s`` and then lagging by ``L`` is the circular correlation at ``s + L``.

    Returns
    -------
    lags_s, r, null : numpy.ndarray
        ``null`` has shape ``(n_shift, len(lags_s))``.
    """
    lags, r = norm_xcorr(a, b, fs, maxlag)
    circ = circular_xcorr(a, b)
    n = len(circ)
    lag_i = np.round(lags * fs).astype(np.int64)
    shifts = _shifts(n, fs, n_shift, min_shift_s, rng)
    return lags, r, circ[(shifts[:, None] + lag_i[None, :]) % n]


def coherence_shift_null(a, b, fs: float, nperseg: int = 1024, n_shift: int = 200,
                         min_shift_s: float = 60.0, rng: Optional[np.random.Generator] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude-squared coherence (Welch) plus a circular-shift null.

    Returns
    -------
    f, coh, null : numpy.ndarray
        ``null`` has shape ``(n_shift, len(f))`` (float32).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    f, coh = signal.coherence(a, b, fs=fs, nperseg=nperseg)
    shifts = _shifts(len(b), fs, n_shift, min_shift_s, rng)
    null = np.empty((n_shift, len(f)), np.float32)
    for i, s in enumerate(shifts):
        null[i] = signal.coherence(a, np.roll(b, s), fs=fs, nperseg=nperseg)[1]
    return f, coh, null


def band_corr_with_null(a: np.ndarray, b: np.ndarray, fs: float,
                        bands: Sequence[Tuple[float, float]], min_shift_s: float = 120.0,
                        n_null: int = 200, rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-lag correlation of two signals within each frequency band, with a shift null.

    Each band is a zero-phase 2nd-order Butterworth band-pass. The null circularly shifts ``b`` by
    at least ``min_shift_s`` either way, which keeps each band-passed signal's autocorrelation.

    Returns
    -------
    r : numpy.ndarray, shape (n_bands,)
    null : numpy.ndarray, shape (n_bands, n_null)
    """
    shifts = _shifts(len(a), fs, n_null, min_shift_s, rng)
    r = np.full(len(bands), np.nan)
    null = np.full((len(bands), n_null), np.nan)
    for k, (f_lo, f_hi) in enumerate(bands):
        sos = signal.butter(2, [f_lo, f_hi], btype="band", fs=fs, output="sos")
        cc = circular_xcorr(signal.sosfiltfilt(sos, a), signal.sosfiltfilt(sos, b))
        r[k] = cc[0]
        null[k] = cc[shifts]
    return r, null


# ── Event coincidence ─────────────────────────────────────────────────────────

def nearest_partner(t_a, t_b) -> Tuple[np.ndarray, np.ndarray]:
    """For each time in ``t_a``, the index of and signed lag to the nearest time in sorted ``t_b``.

    Returns
    -------
    idx : numpy.ndarray of int
        -1 when ``t_b`` is empty.
    lag : numpy.ndarray
        ``t_b[idx] - t_a`` (positive: the partner comes later); inf when ``t_b`` is empty.
    """
    t_a = np.asarray(t_a, float)
    t_b = np.asarray(t_b, float)
    if len(t_b) == 0:
        return np.full(len(t_a), -1), np.full(len(t_a), np.inf)
    j = np.searchsorted(t_b, t_a)
    lo = np.clip(j - 1, 0, len(t_b) - 1)
    hi = np.clip(j, 0, len(t_b) - 1)
    pick = np.where(np.abs(t_b[hi] - t_a) < np.abs(t_b[lo] - t_a), hi, lo)
    return pick, t_b[pick] - t_a


def near_any(times, refs, lo: float, hi: float) -> np.ndarray:
    """True for each time with at least one ref in ``[time + lo, time + hi]``."""
    refs = np.sort(np.asarray(refs, float))
    times = np.asarray(times, float)
    a = np.searchsorted(refs, times + lo, side="left")
    b = np.searchsorted(refs, times + hi, side="right")
    return b > a


def shift_times(times, span: Tuple[float, float], shift: float) -> np.ndarray:
    """Circularly shift event times by ``shift`` s within ``span``; returned sorted."""
    lo, hi = span
    times = np.asarray(times, float)
    return np.sort(lo + np.mod(times - lo + shift, hi - lo))


def coincidence_null(times, refs, lo: float, hi: float, span: Tuple[float, float],
                     n_shift: int = 200, min_shift_s: float = 30.0,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chance level for :func:`near_any`: the fraction of ``times`` with a ref nearby after
    circularly shifting ``refs`` within ``span`` by at least ``min_shift_s``.

    Keeps the refs' own structure (rate, bouts, clustering) and removes only their alignment to
    ``times``. Task structure shared by both trains is removed too, so the null tests alignment
    beyond event rates, not beyond shared task drive.

    Returns
    -------
    numpy.ndarray
        ``n_shift`` fractions (NaN when ``times`` is empty).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    times = np.asarray(times, float)
    refs = np.asarray(refs, float)
    refs = refs[(refs >= span[0]) & (refs < span[1])]
    out = np.full(n_shift, np.nan)
    if len(times) == 0:
        return out
    for k, s in enumerate(rng.uniform(min_shift_s, span[1] - span[0] - min_shift_s, n_shift)):
        out[k] = near_any(times, shift_times(refs, span, s), lo, hi).mean()
    return out
