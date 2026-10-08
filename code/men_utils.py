"""
men_utils.py — loading for the ``men_*`` notebook series (motion energy, behaviour only).

The ``men_*`` notebooks describe motion energy (ME) from the bottom and side cameras against the
task. They need no photometry, so this module reads only ``df_trials.parquet`` and
``df_events.parquet`` from the CSV-curated FIP assets (the hierarchy ``fip_utils.load_pairs``
reads) and ME from the aligned ME table through :func:`fip_utils.motion_energy_to_session`.

Everything generic lives elsewhere: grids, events and correlation in ``signal_utils``, pooling and
tests in ``stats_utils``, mean ± SEM plots in ``plot_utils``, lick bouts and task context in
``behavior_utils``, the ME onset rule in ``fip_utils.ME_ONSET_KW``.

Contents
--------
:func:`inventory_sessions`, :func:`load_session`, :func:`load_sessions`, :func:`normalise`,
:func:`lick_clock_check`

Clock: every time is session time (s from the first go cue). Each session's ME is bin-averaged
onto a uniform ``fs``-Hz grid; sample ``i`` is the mean over ``[t0 + (i - 0.5) / fs,
t0 + (i + 0.5) / fs)``.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import signal_utils as su

CAMERAS = ("BottomCamera", "SideCameraRight")

#: Trial columns carried into the cache. ``animal_response``: 0 left, 1 right, 2 no response.
#: ``num_reward_past`` (Rachel's ``enrich_df_trials``) is the run length of consecutive rewarded
#: trials ending at this trial, negated for runs of unrewarded ones; no-response trials count as
#: unrewarded.
TRIAL_COLS = [
    "trial", "goCue_start_time_in_session", "goCue_start_time_raw", "choice_time_in_session",
    "reward_outcome_time_in_session", "animal_response", "earned_reward", "extra_reward",
    "num_reward_past", "response_time", "RPE_earned", "Q_chosen", "Q_unchosen", "Q_sum",
]

#: Bumped when the cached session dict changes shape, so an old cache is rebuilt.
CACHE_VERSION = 2


def inventory_sessions(asset_roots: Dict[str, str], me_index: pd.DataFrame) -> pd.DataFrame:
    """One row per session in the ME table, with where its trial and event tables are.

    Parameters
    ----------
    asset_roots : dict
        ``{asset_name: root}``, roots from :func:`fip_utils.find_asset_root`.
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
        Cameras to load. A camera with no ME in the table, or one the table refused, is stored
        as ``None``; any other error is raised.
    data_root : str, optional
        Passed to :func:`fip_utils.motion_energy_to_session` (default: Code Ocean's).

    Returns
    -------
    dict
        ``subject, session, asset, t0, fs, trials, licks, lick_events, me``: ``licks`` the sorted
        lick times (either side), ``lick_events`` their ``df_events`` rows (``timestamps,
        event``), ``me`` a dict ``{camera: float32 array or None}`` of raw per-exposure ME.
    """
    import fip_utils as fu

    trials = pd.read_parquet(os.path.join(row.path, "df_trials.parquet"))
    events = pd.read_parquet(os.path.join(row.path, "df_events.parquet"),
                             columns=["timestamps", "event"])
    trials = trials.sort_values("trial").reset_index(drop=True)
    cues = trials["goCue_start_time_in_session"].dropna().to_numpy()
    t0 = float(cues.min() - pre_s)
    n = int(np.ceil((cues.max() + post_s - t0) * fs)) + 1

    lick_events = (events[events["event"].isin(["left_lick_time", "right_lick_time"])]
                   .sort_values("timestamps").reset_index(drop=True))

    kw = {} if data_root is None else {"data_root": data_root}
    me = {}
    for cam in cameras:
        try:
            t_me, y, _ = fu.motion_energy_to_session(row.ses_idx, trials, camera=cam, **kw)
        except (FileNotFoundError, fu.MotionEnergyRefused):
            me[cam] = None
            continue
        me[cam] = su.bin_to_grid(t_me, y, t0, fs, n).astype(np.float32)

    cols = [c for c in TRIAL_COLS if c in trials]
    return dict(subject=str(row.subject), session=row.ses_idx, asset=row.asset, t0=t0, fs=fs,
                trials=trials[cols].copy(), licks=lick_events["timestamps"].to_numpy(float),
                lick_events=lick_events, me=me)


def load_sessions(inv: pd.DataFrame, fs: float = 100.0, cache_path: Optional[str] = None,
                  verbose: bool = True, **kw) -> List[dict]:
    """:func:`load_session` for every inventory row with a ``path``, cached to a pickle.

    The cache is keyed on the session list, ``fs``, :data:`TRIAL_COLS`, the keyword arguments and
    :data:`CACHE_VERSION`, so a changed selection, rate or layout reloads.
    """
    rows = inv[inv["path"].notna()]
    key = (CACHE_VERSION, tuple(rows["ses_idx"]), fs, tuple(TRIAL_COLS), tuple(sorted(kw.items())))
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
        return su.zscore(x)
    if how == "median":
        return x / np.nanmedian(x)
    if how == "none":
        return x
    raise ValueError("how must be 'z', 'median' or 'none'")


def lick_clock_check(sessions: List[dict], cameras: Sequence[str], fs: float, lags_s: np.ndarray,
                     key: str = "me_n") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lick-triggered ME per session and camera, and the lag of its peak (the clock check).

    Each lick is a tongue protrusion both cameras see, so ME averaged around lick times peaks
    close to the lick; a session whose peak sits far from zero has a misplaced ME clock.

    Parameters
    ----------
    sessions : list of dict
        From :func:`load_sessions`, each with ``s[key]`` = ``{camera: grid trace or None}``.
    cameras : sequence of str
    fs : float
        Grid rate, Hz.
    lags_s : numpy.ndarray
        Offsets from each lick, s.
    key : str
        Which per-camera traces to use (``"me_n"``: normalised, ``"me"``: raw).

    Returns
    -------
    clock : pandas.DataFrame
        One row per session x camera: ``subject, session, camera, n_licks, peak_lag_s, peak``.
    records : pandas.DataFrame
        ``subject, session, camera, trace`` (the lick-triggered mean).
    """
    rows, records = [], []
    for s in sessions:
        for cam in cameras:
            x = s[key][cam]
            if x is None:
                continue
            tr = np.nanmean(su.peri_event_grid(x, s["t0"], fs, s["licks"], lags_s), axis=0)
            rows.append(dict(subject=s["subject"], session=s["session"], camera=cam,
                             n_licks=len(s["licks"]), peak_lag_s=lags_s[np.nanargmax(tr)],
                             peak=np.nanmax(tr)))
            records.append(dict(subject=s["subject"], session=s["session"], camera=cam, trace=tr))
    return pd.DataFrame(rows), pd.DataFrame(records)
