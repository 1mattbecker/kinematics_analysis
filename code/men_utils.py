"""
men_utils.py — machinery for the ``men_*`` notebook series (motion energy, behavior only).

The ``men_*`` notebooks describe motion energy (ME) from the bottom and side cameras against the
task: go cues, choices, outcomes and licks. They need no photometry, so this module reads only
``df_trials.parquet`` and ``df_events.parquet`` from the CSV-curated FIP assets (the same
``<root>/<subject>/<session>/`` hierarchy ``fip_coupling`` reads) and ME from the aligned ME table
through :func:`fip_utils.motion_energy_to_session`.

Contents
--------
Loading
    :func:`inventory_sessions`, :func:`bin_to_grid`, :func:`load_session`, :func:`load_sessions`
Licks
    :func:`lick_bouts`, :func:`label_bouts`
ME events
    :func:`detect_me_events`, :func:`near_any`, :func:`shifted_fraction`
Aligning and averaging
    :func:`peri_event`, :func:`window_mean`, :func:`rolling_mean`, :func:`session_trace`,
    :func:`animal_traces`, :func:`plot_mean_sem`

Clock: every time is session time (s from the first go cue), the clock of the ``*_in_session``
trial columns and ``df_events['timestamps']``. Each session's ME is bin-averaged onto a uniform
``fs``-Hz grid; sample ``i`` is the mean ME over ``[t0 + (i - 0.5) / fs, t0 + (i + 0.5) / fs)``.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ── Loading ───────────────────────────────────────────────────────────────────

CAMERAS = ("BottomCamera", "SideCameraRight")

#: Trial columns carried into the cache. ``animal_response``: 0 left, 1 right, 2 no response.
#: ``num_reward_past`` (Rachel's ``enrich_df_trials``) is the run length of consecutive rewarded
#: trials ending at this trial, negated for runs of unrewarded ones; no-response trials count as
#: unrewarded.
TRIAL_COLS = [
    "trial", "goCue_start_time_in_session", "goCue_start_time_raw", "choice_time_in_session",
    "reward_outcome_time_in_session", "animal_response", "earned_reward", "extra_reward",
    "num_reward_past", "response_time", "RPE_earned", "Q_chosen",
]


def inventory_sessions(asset_roots: Dict[str, str], me_index: pd.DataFrame) -> pd.DataFrame:
    """One row per session in the ME table, with where its trial and event tables are.

    Parameters
    ----------
    asset_roots : dict
        ``{asset_name: root}``, roots from :func:`fip_coupling.find_asset_root`.
    me_index : pandas.DataFrame
        ``fip_utils.load_me_index()`` (one row per session x camera, with ``ses_idx``).

    Returns
    -------
    pandas.DataFrame
        ``ses_idx, subject, asset, path`` (``None`` when no asset holds the session) and, per
        camera, ``<camera>_status`` and ``<camera>_action`` from the ME table.
    """
    paths = {}
    for asset, root in asset_roots.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "df_trials.parquet"))):
            session_dir = os.path.dirname(f)
            paths.setdefault(os.path.basename(session_dir), (asset, session_dir))
    wide = me_index.pivot_table(index=["ses_idx", "subject"], columns="camera",
                                values=["status", "action"], aggfunc="first")
    wide.columns = ["%s_%s" % (cam, field) for field, cam in wide.columns]
    inv = wide.reset_index()
    inv["subject"] = inv["subject"].astype(str)
    inv["asset"] = inv["ses_idx"].map(lambda s: paths.get(s, (None, None))[0])
    inv["path"] = inv["ses_idx"].map(lambda s: paths.get(s, (None, None))[1])
    return inv


def bin_to_grid(t: np.ndarray, y: np.ndarray, t0: float, fs: float, n: int) -> np.ndarray:
    """Average ``y`` within each bin of a uniform grid; NaN where a bin has no samples.

    Bin ``i`` is centred on ``t0 + i / fs`` and spans ``1 / fs``. At the camera rate (500 Hz)
    and ``fs = 100`` each bin averages five frames, so this is a low-pass before the
    resample rather than a decimation.
    """
    k = np.floor((np.asarray(t, float) - t0) * fs + 0.5).astype(np.int64)
    ok = (k >= 0) & (k < n) & np.isfinite(y)
    total = np.bincount(k[ok], weights=np.asarray(y, float)[ok], minlength=n)
    count = np.bincount(k[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out.astype(np.float32)


def load_session(row, fs: float = 100.0, pre_s: float = 30.0, post_s: float = 30.0,
                 cameras: Sequence[str] = CAMERAS, data_root: Optional[str] = None) -> dict:
    """Trials, licks and per-camera ME for one session, ME on a uniform grid.

    Parameters
    ----------
    row : namedtuple
        A row of :func:`inventory_sessions` with a ``path``.
    fs : float
        Grid rate in Hz.
    pre_s, post_s : float
        The grid runs from ``pre_s`` before the first go cue to ``post_s`` after the last one.
    cameras : sequence of str
        Cameras to load. A camera the ME table refused is stored as ``None``.
    data_root : str, optional
        Passed to :func:`fip_utils.motion_energy_to_session` (default: Code Ocean's).

    Returns
    -------
    dict
        ``subject, session, asset, t0, fs, trials, licks, lick_side, me`` with ``me`` a dict
        ``{camera: float32 array or None}`` of raw (unnormalised, per-exposure) ME.
    """
    import fip_utils as fu

    trials = pd.read_parquet(os.path.join(row.path, "df_trials.parquet"))
    events = pd.read_parquet(os.path.join(row.path, "df_events.parquet"),
                             columns=["timestamps", "event"])
    trials = trials.sort_values("trial").reset_index(drop=True)
    cues = trials["goCue_start_time_in_session"].dropna().to_numpy()
    t0 = float(cues.min() - pre_s)
    n = int(np.ceil((cues.max() + post_s - t0) * fs)) + 1

    licks = events[events["event"].isin(["left_lick_time", "right_lick_time"])]
    licks = licks.sort_values("timestamps")

    kw = {} if data_root is None else {"data_root": data_root}
    me = {}
    for cam in cameras:
        try:
            t_me, y, _ = fu.motion_energy_to_session(row.ses_idx, trials, camera=cam, **kw)
        except (FileNotFoundError, ValueError):  # MotionEnergyRefused is a ValueError
            me[cam] = None
            continue
        me[cam] = bin_to_grid(t_me, y, t0, fs, n)

    cols = [c for c in TRIAL_COLS if c in trials]
    return dict(subject=str(row.subject), session=row.ses_idx, asset=row.asset, t0=t0, fs=fs,
                trials=trials[cols].copy(), licks=licks["timestamps"].to_numpy(float),
                lick_side=np.where(licks["event"].to_numpy() == "left_lick_time", "L", "R"),
                me=me)


def load_sessions(inv: pd.DataFrame, fs: float = 100.0, cache_path: Optional[str] = None,
                  verbose: bool = True, **kw) -> List[dict]:
    """:func:`load_session` for every inventory row with a ``path``, cached to a pickle.

    The cache is keyed on the session list, ``fs`` and :data:`TRIAL_COLS`, so a changed
    selection or rate reloads.
    """
    rows = inv[inv["path"].notna()]
    key = (tuple(rows["ses_idx"]), fs, tuple(TRIAL_COLS), tuple(sorted(kw.items())))
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            if verbose:
                print("loaded %d sessions from cache %s" % (len(cached["sessions"]), cache_path))
            return cached["sessions"]
    sessions = []
    for i, row in enumerate(rows.itertuples()):
        sessions.append(load_session(row, fs=fs, **kw))
        if verbose and (i + 1) % 10 == 0:
            print("  loaded %d / %d" % (i + 1, len(rows)))
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(dict(key=key, sessions=sessions), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return sessions


def normalise(x: np.ndarray, how: str = "z") -> np.ndarray:
    """Normalise one session's ME so sessions can be pooled.

    ``"z"``: z-score over the grid. ``"median"``: divide by the session median, so 1 is a typical
    frame and values are multiples of it. ``"none"``: unchanged. NaNs are ignored.
    """
    x = np.asarray(x, float)
    if how == "z":
        return (x - np.nanmean(x)) / np.nanstd(x)
    if how == "median":
        return x / np.nanmedian(x)
    if how == "none":
        return x
    raise ValueError("how must be 'z', 'median' or 'none'")


# ── Licks ─────────────────────────────────────────────────────────────────────

def lick_bouts(licks: np.ndarray, max_gap: float = 0.5) -> pd.DataFrame:
    """Group lick times into bouts: a gap longer than ``max_gap`` seconds starts a new bout.

    Returns
    -------
    pandas.DataFrame
        One row per bout: ``onset, offset, n_licks``.
    """
    licks = np.sort(np.asarray(licks, float))
    if len(licks) == 0:
        return pd.DataFrame(columns=["onset", "offset", "n_licks"])
    starts = np.r_[0, np.flatnonzero(np.diff(licks) > max_gap) + 1]
    ends = np.r_[starts[1:] - 1, len(licks) - 1]
    return pd.DataFrame({"onset": licks[starts], "offset": licks[ends],
                         "n_licks": ends - starts + 1})


def label_bouts(bouts: pd.DataFrame, go_cues: np.ndarray, instructed_s: float = 2.0
                ) -> pd.DataFrame:
    """Label each bout by when it starts relative to the go cues.

    * ``instructed``: onset in ``[0, instructed_s]`` s after a go cue.
    * ``uninstructed``: onset more than ``instructed_s`` after the latest go cue and before the
      next one.
    * ``pre-task`` / ``post-task``: before the first go cue, or after the last one (there is no
      next cue to bound it).

    Adds ``label``, ``since_cue`` (s from the latest go cue) and ``to_next_cue`` (s to the next).
    """
    cues = np.sort(np.asarray(go_cues, float))
    out = bouts.copy()
    onset = out["onset"].to_numpy(float)
    j = np.searchsorted(cues, onset, side="right") - 1        # latest cue at or before onset
    prev = np.where(j >= 0, cues[np.clip(j, 0, None)], np.nan)
    nxt = np.where(j + 1 < len(cues), cues[np.clip(j + 1, 0, len(cues) - 1)], np.nan)
    out["since_cue"] = onset - prev
    out["to_next_cue"] = nxt - onset
    label = np.where(out["since_cue"] <= instructed_s, "instructed", "uninstructed")
    label = np.where(j < 0, "pre-task", label)
    label = np.where((j == len(cues) - 1) & (out["since_cue"] > instructed_s), "post-task", label)
    out["label"] = label
    return out


# ── ME events ─────────────────────────────────────────────────────────────────

def detect_me_events(x: np.ndarray, t0: float, fs: float, z_thresh: float = 2.5,
                     min_run_s: float = 0.03, refractory_s: float = 0.5) -> np.ndarray:
    """Onset times of ME events: upward crossings of ``z_thresh`` that stay above it.

    :func:`fip_utils.threshold_onsets` (causal, no smoothing) on the grid; the defaults are
    :data:`fip_utils.ME_ONSET_KW`. ``x`` must already be z-scored.
    """
    import fip_utils as fu

    t = t0 + np.arange(len(x)) / fs
    return fu.threshold_onsets(t, x, z_thresh=z_thresh, refractory=refractory_s,
                               min_run_s=min_run_s, already_z=True)


def near_any(times: np.ndarray, refs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True for each ``time`` with at least one ``ref`` in ``[time + lo, time + hi]``."""
    refs = np.sort(np.asarray(refs, float))
    times = np.asarray(times, float)
    a = np.searchsorted(refs, times + lo, side="left")
    b = np.searchsorted(refs, times + hi, side="right")
    return b > a


def shifted_fraction(times: np.ndarray, refs: np.ndarray, lo: float, hi: float,
                     span: Tuple[float, float], n_shift: int = 200, min_shift: float = 30.0,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chance level for :func:`near_any`: the fraction with a ref nearby after circularly
    shifting ``refs`` within ``span`` by at least ``min_shift`` seconds.

    Keeps the refs' own temporal structure (bouts, rhythm, slow rate changes) and breaks only
    their alignment to ``times``.

    Returns
    -------
    numpy.ndarray
        ``n_shift`` fractions.
    """
    rng = rng or np.random.default_rng(0)
    lo_t, hi_t = span
    length = hi_t - lo_t
    refs = np.asarray(refs, float)
    refs = refs[(refs >= lo_t) & (refs < hi_t)]
    out = np.empty(n_shift)
    for k in range(n_shift):
        d = rng.uniform(min_shift, length - min_shift)
        shifted = lo_t + np.mod(refs - lo_t + d, length)
        out[k] = near_any(times, shifted, lo, hi).mean() if len(times) else np.nan
    return out


# ── Aligning and averaging ────────────────────────────────────────────────────

def _idx(times, t0: float, fs: float) -> np.ndarray:
    return np.round((np.asarray(times, dtype=float) - t0) * fs).astype(np.int64)


def peri_event(x: np.ndarray, t0: float, fs: float, times, lags_s: np.ndarray,
               censor_after=None) -> np.ndarray:
    """Event-aligned segments, shape ``(len(times), len(lags_s))``.

    Samples off the grid are NaN, so a window that runs past either end is kept in part.

    Parameters
    ----------
    censor_after : array_like, optional
        One time per event; samples at or after it are set to NaN (e.g. the next go cue, so a
        go-cue window does not run into the next trial).
    """
    times = np.asarray(times, dtype=float)
    lags_s = np.asarray(lags_s, float)
    lag_i = np.round(lags_s * fs).astype(np.int64)
    out = np.full((len(times), len(lag_i)), np.nan, dtype=np.float32)
    ok = np.isfinite(times)
    if not ok.any():
        return out
    idx = _idx(times[ok], t0, fs)[:, None] + lag_i
    inside = (idx >= 0) & (idx < len(x))
    seg = np.full(idx.shape, np.nan, dtype=np.float32)
    seg[inside] = x[idx[inside]]
    out[ok] = seg
    if censor_after is not None:
        c = np.asarray(censor_after, float)[:, None]
        with np.errstate(invalid="ignore"):
            out[(times[:, None] + lags_s[None, :]) >= c] = np.nan
    return out


def window_mean(x: np.ndarray, t0: float, fs: float, times, a: float, b: float) -> np.ndarray:
    """NaN-aware mean of ``x`` over ``[time + a, time + b)`` for each time."""
    lags = np.arange(int(round(a * fs)), int(round(b * fs))) / fs
    seg = peri_event(x, t0, fs, times, lags)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(seg, axis=1)


def rolling_mean(x: np.ndarray, fs: float, win_s: float, step_s: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Centred running mean of ``x`` over ``win_s`` seconds, sampled every ``step_s`` seconds.

    NaN-aware: a window's mean uses its finite samples, and is NaN when fewer than half are
    finite. Returns ``(sample_offsets_s, means)``, offsets from the start of ``x``.
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


def session_trace(seg: np.ndarray, min_events: int = 1) -> Optional[np.ndarray]:
    """Mean over events (rows) of one session's segments, or None when too few events."""
    if seg.shape[0] < min_events:
        return None
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(seg, axis=0)


def animal_traces(records: pd.DataFrame, group: Dict[str, object]) -> Tuple[np.ndarray, list]:
    """Average session traces within animal for the records matching ``group``.

    Parameters
    ----------
    records : pandas.DataFrame
        One row per session x condition: ``subject``, ``session``, a ``trace`` (1-D array) and
        any condition columns.
    group : dict
        ``{column: value}`` to select, e.g. ``{"camera": "BottomCamera", "cond": "rewarded"}``.

    Returns
    -------
    tuple
        ``(array of shape (n_animals, n_lags), subjects)``. Each session and each animal weighs
        equally.
    """
    sel = records
    for k, v in group.items():
        sel = sel[sel[k] == v]
    subjects, rows = [], []
    for subj, g in sel.groupby("subject"):
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rows.append(np.nanmean(np.stack(g["trace"].to_list()), axis=0))
        subjects.append(subj)
    if not rows:
        return np.empty((0, 0)), []
    return np.stack(rows), subjects


def plot_mean_sem(ax, x: np.ndarray, arr: np.ndarray, color: str, label: Optional[str] = None,
                  show_animals: bool = False, lw: float = 1.6) -> None:
    """Mean ± SEM across rows (animals) of ``arr``; optionally each row as a thin line."""
    if arr.size == 0:
        return
    n = np.sum(np.isfinite(arr), axis=0)
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        m = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)
    if show_animals:
        for r in arr:
            ax.plot(x, r, color=color, lw=0.5, alpha=0.35)
    ax.plot(x, m, color=color, lw=lw,
            label=None if label is None else "%s (n=%d)" % (label, arr.shape[0]))
    if arr.shape[0] > 1:
        ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.25, lw=0)
