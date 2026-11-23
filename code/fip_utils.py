"""
fip_utils.py — shared code for the ``fip_*`` and ``men_*`` notebooks.

Data locations (:data:`DATA_ROOT`, :data:`CACHE_ROOT`, :data:`FIG_ROOT`) follow the Code Ocean
capsule layout, ``data/`` and ``scratch/`` next to ``code/``, so the same paths work in the capsule
and in a local clone; ``$FIP_DATA_ROOT`` and ``$FIP_CACHE_ROOT`` point elsewhere.
:func:`fip_asset_roots` finds the CSV-curated FIP asset(s) (:data:`FIP_ASSET_SETS`).

* **FIP pairs**: :func:`inventory_pairs`, :func:`choose_side`, :func:`load_pairs` put same-side
  DA/NE pairs on a uniform grid; :func:`add_motion_energy` adds bottom- or side-camera ME on that
  grid (scale from :func:`normalise_me`).
* **Motion energy on the FIP clock** from the aligned ME table: :func:`load_me_index`,
  :func:`load_me`, :func:`motion_energy_to_session`; ``roi=`` picks an ROI trace instead of the
  whole frame (:func:`me_rois` lists them, :func:`load_roi_masks` gives their masks).
* **Per-trial measures and the task model**: :func:`trial_measures`, :func:`fit_rpe_terms`,
  :func:`residualize`, :func:`event_design`, :func:`task_residuals`, :func:`large_transients`.

Generic signal code is in ``signal_utils``, across-animal statistics in ``stats_utils``, task
context and lick bouts in ``behavior_utils``.

This module does **not** own the choice of FIP normalization: notebooks that run Rachel's
pipeline (``aind_dynamic_foraging_data_utils.enrich_dfs``) keep that call visible.

Clock: every time is session time (s from the first go cue), shared by
``df_fip['timestamps']`` and the ``*_in_session`` trial columns. A loaded pair is on a uniform
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
from scipy.sparse import csr_matrix, hstack

import signal_utils as su
from me_results import ROI_MASKS_SUFFIX, me_column, read_roi_masks


# ── Data locations ────────────────────────────────────────────────────────────

#: True on Code Ocean.
IS_CO = os.path.isdir("/root/capsule")


def _find_repo_root(start: str) -> str:
    """``/root/capsule`` on Code Ocean, else the nearest folder above ``start`` that holds ``code/``.

    The module may sit directly in ``code/`` or in a subfolder of it (``code/movement_da_ne/`` in the
    DA/NE paper's capsule), so the depth is not fixed.
    """
    if IS_CO:
        return "/root/capsule"
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "code")):
            return d
        parent = os.path.dirname(d)
        if parent == d:   # no code/ above: fall back to the parent of the module's folder
            return os.path.dirname(os.path.abspath(start))
        d = parent


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Repository root (the folder holding ``code/``). On Code Ocean this is ``/root/capsule``.
REPO_ROOT = _find_repo_root(_MODULE_DIR)

#: Small input tables (the video screen, processed-NWB locations): ``inputs/`` next to this module
#: when there is one, else ``inputs/`` in the repository.
INPUTS_DIR = (os.path.join(_MODULE_DIR, "inputs") if os.path.isdir(os.path.join(_MODULE_DIR, "inputs"))
              else os.path.join(REPO_ROOT, "inputs"))

#: Folder holding the data assets (the FIP asset and the ME table): ``data/`` in the repository,
#: which is Code Ocean's data mount. Locally, ``$FIP_DATA_ROOT`` points elsewhere
#: (``$ME_DATA_ROOT`` is read too, for kernels set up before it existed).
DATA_ROOT = (os.environ.get("FIP_DATA_ROOT") or os.environ.get("ME_DATA_ROOT")
             or os.path.join(REPO_ROOT, "data"))

#: Folder for caches and figures: ``scratch/`` in the repository (Code Ocean's scratch), or
#: ``$FIP_CACHE_ROOT``. Every cache is keyed on what it was built from, so it can be deleted.
CACHE_ROOT = os.environ.get("FIP_CACHE_ROOT") or os.path.join(REPO_ROOT, "scratch")

#: Figures, one folder per series (``fip``, ``men``).
FIG_ROOT = os.path.join(CACHE_ROOT, "figures")

#: The CSV-curated FIP assets under :data:`DATA_ROOT`, in order of preference. Each entry is one
#: complete set ``{name: candidate folder names}``; :func:`fip_asset_roots` uses the first set
#: whose every member is present. Folder names are the Code Ocean mount names, then the names a
#: local download has.
FIP_ASSET_SETS = (
    # Rachel's 3- and 4-channel assets in one, with the ignore-rate, square-pulse and Q-spread
    # curation: asset 15f8fac5-30b9-42a4-a1e8-e6c4be7c1ac7 (2026-11-16); before it the test version
    # 4686d370-a207-4fcf-9653-20754c145eac (2026-11-13)
    {"bilateral": ("DANE_bilateral", "DANE_bilateral_curated_permissive",
                   "DANE_bilateral_curated_permissive_test")},
    # The two assets it combines: 3-channel and the 4-channel rebuild
    {"3ch": ("DANE_3channels_curated", "results-a278f830-9bf2-48e8-9928-a2ff78e4818f"),
     "4ch": ("DANE_4channels_curated", "results-ddcccb0f-f18f-44a6-a2b1-caba680d28a1")},
)


#: Reward/failure streak bins from ``num_reward_past``
#: (positive = consecutive rewards, negative = consecutive failures).
STREAK_BINS = [
    ("fail>=3", lambda v: v <= -3, "#08519c"),
    ("fail2", lambda v: v == -2, "#3182bd"),
    ("fail1", lambda v: v == -1, "#6baed6"),
    ("rew1", lambda v: v == 1, "#fdae6b"),
    ("rew2", lambda v: v == 2, "#fd8d3c"),
    ("rew>=3", lambda v: v >= 3, "#e6550d"),
]


# ── Motion energy ─────────────────────────────────────────────────────────────

#: Aligned motion-energy table under ``data_root`` (asset fa56230e-8633-4e0f-9177-cfbcabb9dba2, 2026-11-20,
#: all 301 curated FIP sessions, 300 with ME; aind-motion-energy 0.4.0 with ROIs; before it e542022e-7d56-48ec-a940-d2dcc7ec4733
#: (the 97 sessions of the earlier analyses) and 90c2d0a7-82e3-4abf-b205-e398c2f7736e; built by ``build_me_table.py``): per session and camera, the corrected Harp time and
#: ``me_clean`` of every video frame, plus ``index.csv`` with each camera's timing status.
#: Tables built from aind-motion-energy 0.4.0 ROI runs add ``me_clean_<roi id>`` per ROI,
#: ``<camera>_roi_masks.npz`` and the ``rois`` / ``roi_masks`` index columns.
ME_TABLE_NAME = "fip_motion_energy_aligned"

#: Folder holding the ME table (:data:`DATA_ROOT`; the name the ME functions' defaults use).
ME_DATA_ROOT = DATA_ROOT

#: Per-camera video screen (timing + image quality) from the video-analysis library's
#: ``screen_sessions`` over the 301 curated FIP sessions (library 0.2.0). ``use`` is False for a
#: camera that failed either check; the ME table's own build applied the timing check only.
VIDEO_SCREEN_CSV = os.path.join(INPUTS_DIR, "video_screen_fip.csv")

_ME_INDEX: Dict[Tuple[str, bool], pd.DataFrame] = {}


class MotionEnergyRefused(ValueError):
    """The camera has no usable ME (``index.csv`` ``status`` is not ``ok``): the ME table build
    refused it, or the video screen excluded it."""


def load_me_index(data_root: str = ME_DATA_ROOT, screen: bool = True) -> pd.DataFrame:
    """``index.csv`` of the aligned ME table: one row per session x camera.

    Adds ``ses_idx`` (``<subject>_<date>``, the form of ``nwb.session_id``) to the table's
    ``session`` (the raw asset name, ``behavior_<subject>_<date>_<time>``). Read once per
    ``data_root`` and cached.

    Parameters
    ----------
    data_root : str
        Folder holding the ME table (see :data:`ME_DATA_ROOT`).
    screen : bool
        Apply :data:`VIDEO_SCREEN_CSV`: a camera the build kept but the screen excludes (image
        quality, e.g. side cameras clipped by overexposure) gets ``status = "excluded"`` and the
        screen's ``reason`` as ``error``, so :func:`me_sessions` leaves it out and :func:`load_me`
        raises :class:`MotionEnergyRefused`. A camera missing from the screen is kept as built.

    Raises
    ------
    FileNotFoundError
        If the ME table asset is not attached under ``data_root``.
    """
    path = os.path.join(data_root, ME_TABLE_NAME, "index.csv")
    if (path, screen) not in _ME_INDEX:
        if not os.path.exists(path):
            raise FileNotFoundError("ME table not attached: %s" % path)
        index = pd.read_csv(path)
        index["ses_idx"] = index["session"].str.split("_").str[1:3].str.join("_")
        if screen:
            verdict = pd.read_csv(VIDEO_SCREEN_CSV).set_index(["session", "camera"])
            key = pd.MultiIndex.from_frame(index[["session", "camera"]])
            use = verdict["use"].reindex(key).to_numpy()
            reason = verdict["reason"].reindex(key).to_numpy()
            excluded = (index["status"] == "ok").to_numpy() & (use == False)  # noqa: E712 (NaN = not screened)
            index.loc[excluded, "status"] = "excluded"
            index["error"] = index["error"].astype(object)  # all NaN (float) when nothing was refused
            index.loc[excluded, "error"] = reason[excluded]
        _ME_INDEX[(path, screen)] = index
    return _ME_INDEX[(path, screen)]


def _table_rois(info: dict) -> List[str]:
    """ROI ids of one ``index.csv`` row (none for a table built before ROIs)."""
    rois = info.get("rois")
    return rois.split(";") if isinstance(rois, str) and rois else []


def resolve_roi(info: dict, roi: Optional[str]) -> Optional[str]:
    """Full ROI id of ``roi`` in one ``index.csv`` row, or None if the camera has no such ROI.

    ``roi`` may be the full id (``SideCameraRight_whiskerpad_box``) or the id without the
    ``<camera>_`` prefix the ROI sets use (``whiskerpad_box``). ``None`` and ``"full_frame"``
    mean the whole frame and return ``"full_frame"``.
    """
    if roi is None or roi == "full_frame":
        return "full_frame"
    available = _table_rois(info)
    for candidate in (roi, "%s_%s" % (info["camera"], roi)):
        if candidate in available:
            return candidate
    return None


def me_rois(camera: str = "SideCameraRight", data_root: str = ME_DATA_ROOT) -> pd.Series:
    """ROI ids in the ME table for ``camera``, with the number of usable sessions holding each.

    Parameters
    ----------
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    data_root : str
        Folder holding the ME table (:data:`ME_DATA_ROOT`).

    Returns
    -------
    pandas.Series
        Sessions per ROI id (refused and screen-excluded cameras not counted); empty for a
        table built before ROIs.
    """
    index = load_me_index(data_root)
    usable = index[(index["camera"] == camera) & (index["status"] == "ok")]
    ids = [r for row in usable.to_dict("records") for r in _table_rois(row)]
    return pd.Series(ids, dtype=object).value_counts().rename("n_sessions")


def me_sessions(
    camera: str = "BottomCamera",
    exclude_actions: Sequence[str] = (),
    data_root: str = ME_DATA_ROOT,
    roi: Optional[str] = None,
) -> List[str]:
    """Session ids (``<subject>_<date>``) with usable motion energy for ``camera``.

    Parameters
    ----------
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    exclude_actions : sequence of str
        Timing actions to leave out, e.g. ``("re-index",)`` for the frame-drop sessions or
        ``("fix glitches",)`` for the Harp-glitch ones. Refused and screen-excluded cameras are always left out.
    data_root : str
        Folder holding the ME table (:data:`ME_DATA_ROOT`).
    roi : str, optional
        Keep only sessions whose camera has this ROI trace (id with or without the camera
        prefix, :func:`resolve_roi`). None: the whole frame, every usable session.

    Returns
    -------
    list of str
        Sorted session ids, matching ``nwb.session_id``.
    """
    index = load_me_index(data_root)
    usable = (
        (index["camera"] == camera)
        & (index["status"] == "ok")
        & ~index["action"].isin(list(exclude_actions))
    )
    if roi is not None:
        usable &= np.array([resolve_roi(row, roi) is not None for row in index.to_dict("records")],
                           dtype=bool)
    return sorted(index.loc[usable, "ses_idx"])


def _me_row(session_id: str, camera: str, data_root: str) -> dict:
    """The usable ``index.csv`` row of one session and camera (errors as :func:`load_me`)."""
    index = load_me_index(data_root)
    rows = index[(index["ses_idx"] == session_id) & (index["camera"] == camera)]
    if len(rows) == 0:
        raise FileNotFoundError("no %s motion energy for %s in the ME table" % (camera, session_id))
    if len(rows) > 1:
        raise ValueError("%d ME table rows for %s %s" % (len(rows), session_id, camera))
    info = rows.iloc[0].to_dict()
    if info["status"] != "ok":
        raise MotionEnergyRefused(
            "%s %s has no usable ME (%s): %s" % (session_id, camera, info["status"], info["error"])
        )
    return info


def load_roi_masks(session_id: str, camera: str = "SideCameraRight",
                   data_root: str = ME_DATA_ROOT) -> Dict[str, np.ndarray]:
    """ROI masks the camera's ME was computed with (the table's ``<camera>_roi_masks.npz``).

    Parameters
    ----------
    session_id : str
        ``<subject>_<date>``, as ``nwb.session_id``.
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    data_root : str
        Folder holding the ME table (:data:`ME_DATA_ROOT`).

    Returns
    -------
    dict
        ROI id -> bool mask ``(H, W)`` in video pixels; empty if the camera has no ROIs.

    Raises
    ------
    FileNotFoundError, MotionEnergyRefused, ValueError
        As :func:`load_me`.
    """
    info = _me_row(session_id, camera, data_root)
    if not _table_rois(info):
        return {}
    return read_roi_masks(os.path.join(data_root, ME_TABLE_NAME, info["session"],
                                       camera + ROI_MASKS_SUFFIX))


def load_me(
    session_id: str, camera: str = "BottomCamera", data_root: str = ME_DATA_ROOT,
    roi: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Motion energy for one camera, normalised per exposure, on an even Harp-time grid.

    Reads the camera's file from the ME table and does two things before anything else sees
    the trace:

    1. **Per-exposure ME.** ``aind-motion-energy`` differences consecutive *saved* frames, so
       after a dropped frame a value spans 2 (or more) exposures and carries about that
       multiple of the motion. Each value is divided by the exposures it spans,
       ``diff(frame_number)``. Not by ``diff(harp_time) / frame interval``: Harp time comes in
       32 µs ticks, so single-exposure steps read 1,984 or 2,016 µs and would scale ordinary
       frames by ±1.6% of noise. Unchanged (divided by 1) on sessions without drops.
    2. **Even sampling.** Linear interpolation on ``harp_time`` onto a grid at the camera
       rate (``1 / frame_interval_s``), so a dropped exposure becomes one interpolated sample
       and code that counts samples (``signal_utils.threshold_onsets``' ``min_run``, Welch on the raw
       rate) sees a uniform series.

    Frame 0 has no ME value (the table stores NaN there), so the grid starts at frame 1.
    This holds for tables built from aind-motion-energy 0.3.0 (N−1 values, the build adds the
    NaN) and 0.4.0 (N values, NaN already at row 0): ``build_me_table.py`` reads both through
    ``me_results``, so the table is frame-aligned either way and nothing is padded here.

    An ROI trace (``roi``) is the mean |frame difference| over the ROI's pixels, on the same
    frames, so both steps apply to it unchanged.

    Parameters
    ----------
    session_id : str
        ``<subject>_<date>``, as ``nwb.session_id``.
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    data_root : str
        Folder holding the ME table (:data:`ME_DATA_ROOT`).
    roi : str, optional
        ROI trace to read instead of the whole frame: the id in the table's ``rois`` column,
        with or without the ``<camera>_`` prefix (``"whiskerpad_box"``). See :func:`me_rois`.

    Returns
    -------
    tuple
        ``(t_harp, me, info)``: grid times on the Harp clock (s), per-exposure ME on that
        grid, and the camera's ``index.csv`` row as a dict (with ``roi``, the full ROI id or
        ``"full_frame"``).

    Raises
    ------
    FileNotFoundError
        If the table has no row for this session and camera, or the camera has no trace for
        ``roi``, so a multi-session loop can skip.
    MotionEnergyRefused
        If the build refused the camera or the video screen excluded it (``info['error']`` says why).
    ValueError
        If the file does not match its index row (a length mismatch is never truncated).
    """
    info = _me_row(session_id, camera, data_root)
    info["roi"] = resolve_roi(info, roi)
    if info["roi"] is None:
        raise FileNotFoundError("%s %s has no ROI %r in the ME table (ROIs: %s)"
                                % (session_id, camera, roi, _table_rois(info) or "none"))
    column = me_column(info["roi"])

    frames = pd.read_parquet(
        os.path.join(data_root, ME_TABLE_NAME, info["session"], "%s.parquet" % camera),
        columns=["harp_time", "frame_number", column],
    )
    if len(frames) != info["n_frames"]:
        raise ValueError(
            "%s %s: %d rows in the file, index.csv says %d"
            % (session_id, camera, len(frames), info["n_frames"])
        )
    harp = frames["harp_time"].to_numpy(float)
    exposures = np.diff(frames["frame_number"].to_numpy())
    if (exposures < 1).any():
        # Camera metadata glitch: a block of rows with offset frame numbers (820691_2026-01-16,
        # 319 rows). The timing QC passed the camera on its corrected harp_time, so count the
        # exposures from its steps instead
        exposures = np.rint(np.diff(harp) / float(info["frame_interval_s"])).astype(int)
        if (exposures < 1).any():
            raise ValueError("%s %s: frame numbers and harp_time do not increase" % (session_id, camera))

    # Per-exposure ME (step 1); row 0 has no value
    me = frames[column].to_numpy(float)[1:] / exposures
    t = harp[1:]
    finite = np.isfinite(me)

    # Even grid at the camera rate (step 2)
    dt = float(info["frame_interval_s"])
    t_harp = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    return t_harp, np.interp(t_harp, t[finite], me[finite]), info


#: Seconds of motion energy kept before the first and after the last go cue. The video runs
#: for minutes outside the task (about 10 min of setup before the first go cue), and movement
#: there (handling, adjustment) would otherwise set the SD of every z-score taken over the
#: trace. Same margin as ``men_utils.load_session``'s grid.
ME_TASK_PAD_S = 30.0

#: Motion-energy onset rule shared by the ``fip_*`` and ``men_*`` notebooks: an upward
#: crossing of 2.5 SD that stays above for 30 ms, at most one per 0.5 s. The run length is in
#: seconds so it means the same at the 500 Hz camera rate and on a resampled grid.
ME_ONSET_KW = {"z_thresh": 2.5, "refractory": 0.5, "min_run_s": 0.03}

#: Percentile of a session's ME taken as its floor by ``normalise_me(..., "floor_sd")``. On the
#: 20 Hz grid it sits ~0.1 session SD below the quiet level of every session, including those
#: whose floor a flickering reflection raises (``men_03``). ``floor_sd`` is the scale for ME traces,
#: levels, responses and regressions since 2026-10-22; ME onsets stay on the z-scored trace.
ME_FLOOR_PCT = 1.0

#: ``normalise_me`` scales: ``zscore`` (x - mean) / SD; ``sd`` x / SD;
#: ``floor_sd`` max(0, x - floor) / SD, with the floor at :data:`ME_FLOOR_PCT`.
ME_NORMS = ("zscore", "sd", "floor_sd")


def normalise_me(x: np.ndarray, how: str = "zscore", floor_pct: float = ME_FLOOR_PCT) -> np.ndarray:
    """One session's ME on a common scale, so sessions can be pooled.

    All three scales divide by the session SD (ddof=1, NaNs ignored), so within a session they
    differ only by an offset (and, for ``floor_sd``, the clipping of the lowest ``floor_pct`` %):
    baseline-subtracted responses, correlations and regression fits with an intercept are the
    same on each. They differ in where stillness sits: below 0 for ``zscore``, at a positive,
    session-dependent level for ``sd``, near 0 for ``floor_sd`` (``men_03``).

    Parameters
    ----------
    x : numpy.ndarray
        Raw ME of one session (e.g. on the 20 Hz grid); NaNs are kept.
    how : str
        One of :data:`ME_NORMS`.
    floor_pct : float
        Percentile taken as the floor for ``floor_sd``.

    Returns
    -------
    numpy.ndarray
        ``x`` on the chosen scale.
    """
    x = np.asarray(x, float)
    if how == "zscore":
        return su.zscore(x)
    sd = np.nanstd(x, ddof=1)
    if how == "sd":
        return x / sd
    if how == "floor_sd":
        return su.floor_scale(x, floor_pct)
    raise ValueError(f"how must be one of {ME_NORMS}")


def motion_energy_to_session(
    session_id: str,
    df_trials: pd.DataFrame,
    go_cue_col: str = "goCue_start_time_raw",
    camera: str = "BottomCamera",
    data_root: str = ME_DATA_ROOT,
    pad_s: Optional[float] = ME_TASK_PAD_S,
    roi: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Motion energy on the session clock (t=0 at the first go cue), evenly sampled.

    ``t_session = harp_time - first_go_cue``: go cues and camera triggers are both on the
    Harp Behavior clock, and ``df_fip['timestamps']`` is zeroed at the same go cue. Uses the
    table's corrected Harp time, so the frame-drop sessions are placed right (the raw CSV
    column put ME up to 280–670 s early there by session end).

    The trace is cut to the task, ``pad_s`` before the first go cue to ``pad_s`` after the
    last, so a z-score over it is not set by movement during setup (see :data:`ME_TASK_PAD_S`).

    ``t_session`` is **not** a position in the video file: after a drop it runs ahead of
    ``row / fps``. To find a frame (clips, BEAST), ``searchsorted`` an event on the
    per-frame ``harp_time`` in the table instead.

    Parameters
    ----------
    session_id : str
        ``<subject>_<date>``, as ``nwb.session_id``.
    df_trials : pandas.DataFrame
        Needs ``trial`` and ``go_cue_col`` (absolute Harp clock).
    go_cue_col : str
        Column holding the raw (unzeroed) go-cue times.
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    data_root : str
        Folder holding the ME table (:data:`ME_DATA_ROOT`).
    pad_s : float or None
        Margin kept around the go cues, in s. ``None`` returns the whole video.
    roi : str, optional
        ROI trace instead of the whole frame (:func:`load_me`).

    Returns
    -------
    tuple
        ``(t_session, me, offset)``, with ``me`` from :func:`load_me` and
        ``offset = first_go_cue - harp_time[0]`` (s from the first frame to the first go cue).

    Raises
    ------
    FileNotFoundError, MotionEnergyRefused, ValueError
        As :func:`load_me`.
    """
    t_harp, me, info = load_me(session_id, camera, data_root, roi=roi)
    go_cues = df_trials.sort_values("trial")[go_cue_col].dropna()  # Harp clock
    first_go_cue = float(go_cues.iloc[0])
    t_session = t_harp - first_go_cue
    if pad_s is not None:
        last = float(go_cues.max()) - first_go_cue
        keep = (t_session >= -pad_s) & (t_session <= last + pad_s)
        t_session, me = t_session[keep], me[keep]
    return t_session, me, first_go_cue - float(info["harp_start"])


# ── CSV-curated assets: loading ───────────────────────────────────────────────

#: Same-side DA and NE labels. ``medNAcc`` is left out so every animal contributes one DA site
#: (lateral NAc); ``-unconfirmed`` PL fibers never survive curation and are excluded anyway.
DA_RE = re.compile(r"^latNAcc\((L|R)\)-DA$")
NE_RE = re.compile(r"^PL\((L|R)\)-LCAxonCa$")

#: Trial columns carried into the cache. ``RPE_*`` / ``Q_*`` come from the behavioral model fit
#: (``QLearning_L1F1_CK1_softmax``, fit per session by the AIND analysis-architecture pipeline) via
#: ``aind_dynamic_foraging_data_utils.enrich_dfs.enrich_df_trials_fm``:
#: ``RPE_earned = earned_reward − Q_chosen``; ``RPE_all`` adds ``extra_reward`` (unearned water).
#: Rachel's analyses (``rachel_analysis_utils.analysis_utils.enrich_df_trials``) use ``RPE_earned``.
TRIAL_COLS = [
    "trial", "goCue_start_time_in_session", "choice_time_in_session",
    "reward_outcome_time_in_session", "animal_response", "earned_reward", "extra_reward",
    "RPE_earned", "RPE_all",
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


def fip_asset_roots(data_root: str = DATA_ROOT, verbose: bool = True) -> Dict[str, str]:
    """Roots of the first complete set in :data:`FIP_ASSET_SETS` found under ``data_root``.

    Parameters
    ----------
    data_root : str
        Folder holding the assets (:data:`DATA_ROOT`).
    verbose : bool
        Print the set used.

    Returns
    -------
    dict
        ``{asset name: root}``, roots as :func:`find_asset_root` returns them; the input of
        :func:`inventory_pairs` and ``men_utils.inventory_sessions``.

    Raises
    ------
    FileNotFoundError
        If no set is complete. A partial set is never used, since it would silently drop the
        animals of the missing asset.
    """
    for asset_set in FIP_ASSET_SETS:
        roots = {name: find_asset_root([os.path.join(data_root, c) for c in cands])
                 for name, cands in asset_set.items()}
        if all(roots.values()):
            if verbose:
                for name, root in roots.items():
                    print(f"FIP asset {name}: {root}")
            return roots
    wanted = " or ".join(" + ".join(c[0] for c in s.values()) for s in FIP_ASSET_SETS)
    raise FileNotFoundError(f"no complete FIP asset set under {data_root}: attach {wanted} "
                            "(or set FIP_DATA_ROOT)")


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
    rows = []
    for asset, root in asset_roots.items():
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "df_fip.parquet"))):
            # pandas picks pyarrow or fastparquet, whichever is installed
            events = set(pd.read_parquet(f, columns=["event"])["event"])
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


def pair_sessions(asset_roots: Dict[str, str], min_sessions: int = 3) -> List[str]:
    """The sessions the ``fip_*`` notebooks use, as ``<subject>_<date>``.

    :func:`inventory_pairs` then :func:`choose_side`: same-side DA/NE pairs, one hemisphere per
    animal, animals with at least ``min_sessions`` sessions. The ME table covers every curated FIP
    session, so the ``men_*`` notebooks restrict its index to these.

    Parameters
    ----------
    asset_roots : dict
        From :func:`fip_asset_roots`.
    min_sessions : int
        As :func:`choose_side`.

    Returns
    -------
    list of str
        Sorted session ids (the ME index's ``ses_idx``).
    """
    kept, _ = choose_side(inventory_pairs(asset_roots), min_sessions)
    return sorted(kept["session"])


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
    key = (tuple(kept["path"]), tuple(TRIAL_COLS))  # a changed column list also reloads
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


def add_motion_energy(pairs: Sequence[dict], kept: pd.DataFrame, camera: str = "BottomCamera",
                      data_root: str = ME_DATA_ROOT, onset_fs: Optional[float] = 100.0,
                      exclude_subjects: Sequence[str] = (), me_norm: str = "floor_sd",
                      keep_raw: bool = False, roi: Optional[str] = None) -> List[dict]:
    """Keep the pairs whose camera passes the ME screen and put DA, NE and ME on their grid.

    ME on the session clock (:func:`motion_energy_to_session`) is averaged into the pair's
    ``fs``-Hz bins (:func:`signal_utils.bin_to_grid`). DA and NE are z-scored over the grid; ME is
    put on the scale ``me_norm`` (:func:`normalise_me`). ME onsets are found on the z-scored trace
    whatever ``me_norm`` is, so the same events enter every analysis.

    Parameters
    ----------
    pairs : sequence of dict
        From :func:`load_pairs`.
    kept : pandas.DataFrame
        From :func:`choose_side`; gives each session's folder.
    camera : str
        ``BottomCamera`` or ``SideCameraRight``.
    data_root : str
        Folder holding the ME table.
    onset_fs : float or None
        Rate of the grid ME onsets are detected on (:data:`ME_ONSET_KW`, on the z-scored
        trace). ``None`` skips onset detection.
    exclude_subjects : sequence of str
        Subjects left out.
    me_norm : str
        ME scale, one of :data:`ME_NORMS` (default ``floor_sd``, SDs above the session's 1st
        percentile; ``zscore`` was the default before 2026-10-22).
    keep_raw : bool
        Also keep the raw ME on the grid as ``me_raw`` (NaN in empty bins).
    roi : str, optional
        Use this ROI's trace (:func:`load_me`) as ``me`` instead of the whole frame; pairs
        whose camera has no such ROI are left out.

    Returns
    -------
    list of dict
        Copies of the kept pairs with ``z`` (``{"da", "ne", "me"}``: DA, NE z-scored, ME on
        ``me_norm``, its empty bins set to the session mean on that scale, 0 for ``zscore``),
        ``lick_events`` (left/right lick rows of ``df_events``), with ``onset_fs`` ``me_onsets``
        (s), and with ``keep_raw`` ``me_raw``.
    """
    path_of = dict(zip(kept["session"], kept["path"]))
    usable = set(me_sessions(camera, data_root=data_root, roi=roi))
    out = []
    for p in pairs:
        if p["session"] not in usable or p["subject"] in exclude_subjects:
            continue
        path = path_of[p["session"]]
        trials_raw = pd.read_parquet(os.path.join(path, "df_trials.parquet"))
        t_me, y, _ = motion_energy_to_session(p["session"], trials_raw, camera=camera,
                                              data_root=data_root, roi=roi)
        n = len(p["da"])
        q = dict(p)
        raw = su.bin_to_grid(t_me, y, p["t0"], p["fs"], n)
        me = normalise_me(raw, me_norm)
        q["z"] = {"da": su.zscore(p["da"]), "ne": su.zscore(p["ne"]),
                  "me": np.where(np.isfinite(me), me, np.nanmean(me) if me_norm != "zscore" else 0.0)}
        if keep_raw:
            q["me_raw"] = raw
        if onset_fs is not None:
            n_on = int(n * onset_fs / p["fs"])
            me_on = su.zscore(su.bin_to_grid(t_me, y, p["t0"], onset_fs, n_on))
            q["me_onsets"] = su.threshold_onsets(p["t0"] + np.arange(n_on) / onset_fs, me_on,
                                                 already_z=True, **ME_ONSET_KW)
        events = pd.read_parquet(os.path.join(path, "df_events.parquet"),
                                 columns=["timestamps", "event"])
        q["lick_events"] = events[events["event"].isin(["left_lick_time", "right_lick_time"])]
        out.append(q)
    return out


# ── Processed NWBs: every stage of the dF/F pipeline ─────────────────────────
#
# The curated df_fip carries one trace per sensor (``dff-bright_mc-iso-IRLS``). The processed NWB
# it came from (``s3_location`` in Rachel's ``df_sess_<mouse>.csv``) holds every channel (G, Iso,
# R) of every fiber at every stage: raw, bleach-corrected (``dff-bright``) and motion-corrected,
# plus the pipeline's ``quality_control.json`` with the fitted F0 terms and the iso coefficient.
# ``fip_18`` (now in the fip-controls repo) reads them from public S3 (``aind-open-data``, zarr over HTTPS).

#: Session → processed NWB (public S3), from the ``df_sess`` tables of ``DANE_bilateral``
#: (2026-11-13; NWBs processed 2026-11-10/11 by aind-fip-dff 19.0). Used when the curated asset at
#: hand lists its NWBs in a private bucket (the 3ch / 4ch assets do).
PROCESSED_NWB_CSV = os.path.join(INPUTS_DIR, "fip_processed_nwb.csv")

#: Stages of a channel in the processed NWB: raw fluorescence (``acquisition``), bleach-corrected
#: dF/F and motion-corrected dF/F (``processing/fiber_photometry``). The NWBs also hold the same two
#: dF/F stages for the ``bright-v2`` F0 fit (``dff-bright-v2``, ``dff-bright-v2_mc-iso-IRLS``) and
#: for ``poly``.
NWB_STAGES = ("raw", "dff-bright", "dff-bright_mc-iso-IRLS")
NWB_CHANNELS = ("G", "Iso", "R")

#: The pipeline's noise filter on the motion-corrected trace: 2nd-order Butterworth, 3 Hz,
#: zero-phase (``motion_correct``, ``cutoff_freq_noise``). Bleach-only traces get the same
#: before any comparison.
NOISE_LOWPASS_HZ = 3.0


def processed_nwb_locations(asset_roots: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Where each session's processed NWB is, from ``df_sess`` tables or the inputs CSV.

    Parameters
    ----------
    asset_roots : dict, optional
        From :func:`fip_asset_roots`. Their ``df_sess_<mouse>.csv`` are read first; sessions whose
        NWB is not on public S3 (``s3://aind-open-data/``) are then taken from
        :data:`PROCESSED_NWB_CSV`.

    Returns
    -------
    pandas.DataFrame
        One row per session: ``session, subject, s3_location, processed`` (date the NWB was
        processed), ``source`` (``asset`` or ``csv``). Only public NWBs.
    """
    rows = []
    for root in (asset_roots or {}).values():
        for f in glob.glob(os.path.join(root, "df_sess_*.csv")) + \
                glob.glob(os.path.join(root, "data", "df_sess_*.csv")):
            d = pd.read_csv(f, usecols=["ses_idx", "subject_id", "s3_location"])
            rows.append(d.rename(columns={"ses_idx": "session", "subject_id": "subject"})
                        .assign(source="asset"))
    csv = pd.read_csv(PROCESSED_NWB_CSV).assign(source="csv")
    rows.append(csv[["session", "subject", "s3_location", "source"]])
    out = pd.concat(rows, ignore_index=True)
    out = out[out["s3_location"].str.startswith("s3://aind-open-data/", na=False)]
    out = out.drop_duplicates("session", keep="first").reset_index(drop=True)
    out["processed"] = out["s3_location"].str.extract(r"processed_(\d{4}-\d{2}-\d{2})")[0]
    out["subject"] = out["subject"].astype(str)
    return out[["session", "subject", "s3_location", "processed", "source"]]


def s3_to_https(s3_location: str) -> str:
    """``s3://bucket/key`` → ``https://bucket.s3.amazonaws.com/key``."""
    bucket, key = s3_location[len("s3://"):].split("/", 1)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def fiber_of_events(path: str) -> Dict[str, int]:
    """Curated event name → fiber (ROI) number, from ``df_fip['patch_cord']`` (``G_1`` → 1)."""
    d = pd.read_parquet(os.path.join(path, "df_fip.parquet"), columns=["event", "patch_cord"])
    d = d.drop_duplicates("event")
    return {e: int(str(pc).split("_")[1]) for e, pc in zip(d["event"], d["patch_cord"])}


def session_time_offset(path: str) -> float:
    """Session time minus the NWB clock, from ``df_fip`` (``timestamps − raw_timestamps``)."""
    d = pd.read_parquet(os.path.join(path, "df_fip.parquet"),
                        columns=["timestamps", "raw_timestamps"]).iloc[:1000]
    off = (d["timestamps"] - d["raw_timestamps"]).to_numpy()
    if np.ptp(off) > 1e-6:
        raise ValueError(f"{path}: session-time offset is not constant")
    return float(off[0])


def load_nwb_stages(s3_location: str, fibers: Sequence[int], channels: Sequence[str] = NWB_CHANNELS,
                    stages: Sequence[str] = NWB_STAGES, cache_dir: Optional[str] = None) -> dict:
    """Every requested channel and stage of the given fibers, from the processed NWB.

    Parameters
    ----------
    s3_location : str
        ``s3://aind-open-data/.../nwb/<name>.nwb`` (a zarr directory).
    fibers : sequence of int
        ROI numbers.
    channels, stages : sequence of str
        From :data:`NWB_CHANNELS`, :data:`NWB_STAGES`.
    cache_dir : str, optional
        Folder for one ``.npz`` per NWB. Series already cached are read from it; only the missing
        ones are fetched, and the cache keeps every series fetched so far.

    Returns
    -------
    dict
        ``t``: timestamps on the NWB clock (the raw series' own); ``(fiber, channel, stage)``:
        the trace. Every series in one NWB shares the timestamps.
    """
    import fsspec  # noqa: F401 — only this reader needs them
    import zarr
    name = os.path.basename(s3_location.rstrip("/"))
    keys = [(f, c, s) for f in fibers for c in channels for s in stages]
    cache = os.path.join(cache_dir, name + ".npz") if cache_dir else None
    cached = {}
    if cache and os.path.exists(cache):
        with np.load(cache) as z:
            cached = {k: z[k] for k in z.files}
    missing = [k for k in keys if f"{k[0]}|{k[1]}|{k[2]}" not in cached]
    if missing or "t" not in cached:
        g = zarr.open_consolidated(fsspec.get_mapper(s3_to_https(s3_location)), mode="r")
        for f, c, s in missing or keys[:1]:
            series = f"acquisition/{c}_{f}" if s == "raw" else f"processing/fiber_photometry/{c}_{f}_{s}"
            if "t" not in cached:
                cached["t"] = g[f"{series}/timestamps"][:]
            cached[f"{f}|{c}|{s}"] = g[f"{series}/data"][:]
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, **cached)
    out = {"t": cached["t"]}
    for f, c, s in keys:
        out[(f, c, s)] = cached[f"{f}|{c}|{s}"]
    return out


def noise_lowpass(x: np.ndarray, fs: float, cutoff_hz: float = NOISE_LOWPASS_HZ) -> np.ndarray:
    """The pipeline's noise filter (2nd-order Butterworth low-pass, zero-phase)."""
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(2, cutoff_hz, fs=fs, output="sos"), x)


MOTION_CUTOFF_HZ = 0.05                     # the pipeline fits β on traces low-passed below this
BETA_FIT_BANDS = {"<0.01": (None, 0.01), "0.01–0.05": (0.01, 0.05)}   # together: the pipeline's fit range


ISO_FLOOR_BAND_HZ = (5.0, 9.0)              # Iso's noise floor is read from its spectrum here


def iso_denoise(iso: np.ndarray, fs: float, floor_band: Tuple[float, float] = ISO_FLOOR_BAND_HZ,
                nperseg: int = 4096, smooth_octaves: float = 0.25) -> Tuple[np.ndarray, dict]:
    """Iso with the part at its own noise floor removed, by a zero-phase Wiener gain from Iso alone.

    Test implementation of the correction proposed in ``fip_isosbestic_todo.md`` §3 item 8 (now fip-controls ``TODO.md``). The
    gain at each frequency is ``max(0, 1 − N / P_iso(f))``, where ``P_iso`` is Iso's Welch spectrum
    (mean over ± ``smooth_octaves`` / 2 octaves) and ``N`` its median over ``floor_band``. Where
    Iso is at its floor the gain is 0; where Iso's power is well above it, near 1. Nothing about
    the sensor channel enters.

    Parameters
    ----------
    iso : np.ndarray
        Bleach-only Iso dF/F, no NaNs.
    fs : float
        Sampling rate (Hz).
    floor_band : (float, float)
        Band (Hz) where Iso is taken to be noise only.
    nperseg : int
        Welch segment length.
    smooth_octaves : float
        Width of the spectral smoothing, in octaves.

    Returns
    -------
    iso_denoised : np.ndarray
        Denoised Iso, mean removed.
    info : dict
        ``floor`` (N), ``f_half`` (lowest frequency above 0.01 Hz where the gain falls below 0.5),
        ``share`` (variance of the denoised over the original Iso), ``f`` and ``gain`` (the gain
        on the Welch grid).
    """
    from scipy.signal import welch
    x = iso - iso.mean()
    f, p = welch(x, fs=fs, nperseg=nperseg)
    half = 2 ** (smooth_octaves / 2)
    p_s = np.array([p[(f >= fi / half) & (f <= fi * half)].mean() if fi > 0 else p[0] for fi in f])
    floor = float(np.median(p[(f >= floor_band[0]) & (f <= floor_band[1])]))
    gain = np.clip(1 - floor / np.maximum(p_s, 1e-30), 0, 1)
    gain[0] = 1.0                                         # the mean is removed anyway
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    x_dn = np.fft.irfft(np.fft.rfft(x) * np.interp(freqs, f, gain), len(x))
    below = np.flatnonzero((f > 0.01) & (gain < 0.5))
    info = dict(floor=floor, f_half=float(f[below[0]]) if len(below) else np.nan,
                share=float(x_dn.var() / x.var()), f=f, gain=gain)
    return x_dn, info


def _asymmetric_tukey(c_pos: float = 2.0, c_neg: float = 4.685):
    """The pipeline's robust norm for β (``AsymmetricTukeyBiweight(2)``, ``aind-fip-dff`` ``c1733bb``),
    copied because that package is not installed here. Positive residuals (G above the fit) lose
    weight from 2 robust SDs, negative ones from 4.685."""
    from statsmodels.robust.norms import RobustNorm

    class AsymmetricTukey(RobustNorm):
        def _c(self, z):
            return np.where(z > 0, c_pos, c_neg)

        def rho(self, z):
            z = np.asarray(z)
            c = self._c(z)
            return np.where(np.abs(z) <= c, c ** 2 / 6 * (1 - (1 - (z / c) ** 2) ** 3), c ** 2 / 6)

        def weights(self, z):
            z = np.asarray(z)
            c = self._c(z)
            return np.where(np.abs(z) <= c, (1 - (z / c) ** 2) ** 2, 0.0)

        def psi(self, z):
            return np.asarray(z) * self.weights(z)

        def psi_deriv(self, z):
            z = np.asarray(z)
            c = self._c(z)
            t2 = (z / c) ** 2
            return np.where(np.abs(z) <= c, (1 - t2) ** 2 - 4 * t2 * (1 - t2) / c ** 2, 0.0)

    return AsymmetricTukey()


def iso_beta_steps(g: np.ndarray, iso: np.ndarray, fs: float,
                   bands: Optional[Dict[str, Tuple[Optional[float], float]]] = None) -> dict:
    """The pipeline's β for one fiber, step by step, next to the slope of G on Iso in each band.

    ``ols_lp`` is least squares on the traces low-passed at :data:`MOTION_CUTOFF_HZ`; it equals the
    band slopes of :data:`BETA_FIT_BANDS` averaged with weights ``iso_share`` (each band's share of
    the low-passed Iso power). ``rlm`` adds the pipeline's robust weights and ``pipeline`` its clip
    at 0, which reproduces the β in ``quality_control.json``.

    Parameters
    ----------
    g, iso : np.ndarray
        Bleach-only (``dff-bright``) sensor and Iso traces of one fiber, full recording, NaNs dropped.
    fs : float
        Sampling rate (Hz).
    bands : dict, optional
        Extra bands ``name → (low or None, high)`` for slopes and correlations.

    Returns
    -------
    dict
        ``ols_lp``, ``rlm``, ``pipeline``, ``rlm_weight_mean``; per band ``beta|<band>`` (slope of
        the band-passed G on the band-passed Iso, zero-phase 2nd-order Butterworth) and ``r|<band>``;
        per fit band ``iso_share|<band>``.
    """
    from scipy.signal import butter, sosfiltfilt
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.tools import add_constant

    def bandpass(x, lo, hi):
        sos = (butter(2, hi, fs=fs, output="sos") if lo is None
               else butter(2, [lo, hi], btype="band", fs=fs, output="sos"))
        return sosfiltfilt(sos, x - x.mean())

    sos = butter(2, MOTION_CUTOFF_HZ, fs=fs, output="sos")
    g_lp, iso_lp = sosfiltfilt(sos, g), sosfiltfilt(sos, iso)
    fit = RLM(g_lp, add_constant(iso_lp), M=_asymmetric_tukey()).fit()
    out = dict(ols_lp=np.polyfit(iso_lp, g_lp, 1)[0], rlm=fit.params[1], pipeline=max(fit.params[1], 0.0),
               rlm_weight_mean=fit.weights.mean())
    fit_power = {}
    for name, (lo, hi) in {**BETA_FIT_BANDS, **(bands or {})}.items():
        g_b, iso_b = bandpass(g, lo, hi), bandpass(iso, lo, hi)
        out[f"beta|{name}"] = (g_b @ iso_b) / (iso_b @ iso_b)
        out[f"r|{name}"] = np.corrcoef(g_b, iso_b)[0, 1]
        if name in BETA_FIT_BANDS:
            fit_power[name] = iso_b @ iso_b
    for name, power in fit_power.items():
        out[f"iso_share|{name}"] = power / sum(fit_power.values())
    return out


def load_pipeline_qc(s3_location: str, method: str = "bright", cache_dir: Optional[str] = None
                     ) -> pd.DataFrame:
    """The dF/F pipeline's QC for one session, one row per (fiber, channel).

    Reads ``quality_control.json`` next to the NWB. Columns: ``fiber, channel``, the F0 terms as
    the method names them (``bright``: ``b_inf, b_slow, b_fast, b_rapid, b_bright, t_slow, t_fast,
    t_rapid, t_bright``; ``bright-v2``: ``b_inf, b1, tau1, b2, tau2, b3, tau3, b_bright,
    tau_bright``; a term the model selection left out is 0 / NaN), ``calibration_ratio`` (median
    negative residual over the Gaussian expectation), ``pre_gocue_dff`` and ``pre_gocue_drift``
    (mean dF/F before the go cue and its drift over the session, bleach-only stage), ``beta`` (iso
    coefficient; G and R only, NaN for Iso), ``noise_inflation`` (noise of the corrected over the
    uncorrected trace) and the evaluation's ``status``.
    """
    import json
    url = s3_to_https(s3_location.rsplit("/nwb/", 1)[0] + "/quality_control.json")
    cache = os.path.join(cache_dir, os.path.basename(s3_location) + ".qc.json") if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            qc = json.load(fh)
    else:
        from s3_utils import read_json
        qc = read_json(url)
        if qc is None:
            raise FileNotFoundError(url)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache, "w") as fh:
                json.dump(qc, fh)
    ev = [e for e in qc["evaluations"] if e["name"] == f"Preprocessing using method '{method}'"]
    if not ev:
        raise KeyError(f"no '{method}' evaluation in {url}")
    ev = ev[0]
    rows: Dict[Tuple[int, str], dict] = {}
    for m in ev["metrics"]:
        roi = re.search(r"ROI (\d+)", m["name"])
        v = m["value"]
        if roi is None or not isinstance(v, dict) or "index" not in v:
            continue
        roi = int(roi.group(1))
        for i, ch in enumerate(v["index"]):
            r = rows.setdefault((roi, ch), dict(fiber=roi, channel=ch))
            if m["name"].startswith("Baseline correction"):
                for k in v:
                    if k != "index":
                        r[k] = v[k][i] if v[k][i] is not None else np.nan
            elif m["name"].startswith("Calibration ratio"):
                r["calibration_ratio"] = v["ratio"][i]
            elif m["name"].startswith("Pre-GoCue") and m["name"].endswith("(dff)"):
                r["pre_gocue_dff"] = v["mean_dff"][i]
                r["pre_gocue_drift"] = v["total_drift"][i]
            elif m["name"].startswith("Motion correction"):
                r["beta"] = v["coefficient"][i]
                r["noise_inflation"] = v["noise_inflation"][i]
    out = pd.DataFrame(list(rows.values())).sort_values(["fiber", "channel"]).reset_index(drop=True)
    out["status"] = ev.get("latest_status")
    return out


# ── CSV-curated assets: per-trial measures ────────────────────────────────────

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
        x = su.zscore(pair[sig])
        pre = su.window_mean_grid(x, t0, fs, cue, *baseline)
        tr[f"{sig}_pre"] = pre
        for name, (a, b) in windows.items():
            tr[f"{sig}_{name}"] = su.window_mean_grid(x, t0, fs, out, a, b) - pre
        seg = su.peri_event_grid(x, t0, fs, cue, lags)
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


def fit_rpe_terms(df: pd.DataFrame, y: str, rpe_col: str = "RPE_earned",
                  covariates: Sequence[str] = ()) -> pd.Series:
    """OLS of a response on reward and on RPE separately within each outcome.

    ``y ~ 1 + rewarded + RPE·rewarded + RPE·(1 − rewarded)``. Within an outcome class RPE is
    ``1 − Q_chosen`` (rewarded) or ``−Q_chosen`` (unrewarded), so the two slopes ask whether the
    response scales with how expected that outcome was. A signed-RPE signal has a positive slope in
    both classes; a pure outcome signal has zero slopes; an unsigned surprise signal has a positive
    slope for rewards and a negative one for omissions.

    Parameters
    ----------
    df : pandas.DataFrame
        Responded trials with ``rewarded``, ``rpe_col`` and column ``y``.
    y : str
        Response column.
    rpe_col : str
        RPE column (default ``RPE_earned``, Rachel's convention).
    covariates : sequence of str
        Extra nuisance columns (e.g. ``response_time``: low-value choices are slower, so more of
        the cue response falls inside a window aligned to the outcome).

    Returns
    -------
    pandas.Series
        ``intercept, reward, rpe_rew, rpe_unrew`` (units of ``y`` per unit RPE), one coefficient
        per covariate under its column name, and ``n``.
    """
    d = df[["rewarded", rpe_col, y, *covariates]].dropna()
    r = d["rewarded"].to_numpy(float)
    rpe = d[rpe_col].to_numpy(float)
    X = np.c_[np.ones(len(d)), r, rpe * r, rpe * (1 - r), d[list(covariates)].to_numpy(float)]
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)
    return pd.Series(dict(intercept=beta[0], reward=beta[1], rpe_rew=beta[2], rpe_unrew=beta[3],
                          **dict(zip(covariates, beta[4:])), n=len(d)))


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


# ── CSV-curated assets: task model ────────────────────────────────────────────

#: Event regressors :func:`task_events` can build. The first four are :func:`event_design`'s
#: default task model. The ``×`` ones are parametric: the event's times, each weighted by a trial
#: value centred within the session over the trials it applies to, so their kernel is the response
#: per unit of the value on top of the plain event's kernel.
TASK_EVENTS = ("go cue", "rewarded outcome", "unrewarded outcome", "lick",
               "go cue × Q_sum", "rewarded outcome × RPE", "unrewarded outcome × RPE")


def task_events(pair: dict, names: Sequence[str] = TASK_EVENTS[:4],
                rng: Optional[np.random.Generator] = None) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Event times (and weights) for :func:`event_design`, by name.

    Parameters
    ----------
    pair : dict
        One element of :func:`load_pairs`.
    names : sequence of str
        From :data:`TASK_EVENTS`. ``go cue × Q_sum`` weights each go cue by that trial's
        ``Q_sum`` (no-response trials, where the model leaves it NaN, get the session mean, i.e.
        weight 0). ``rewarded/unrewarded outcome × RPE`` weight each outcome by ``RPE_earned``,
        centred within that outcome class (where RPE is ``1 − Q_chosen`` or ``−Q_chosen``).
    rng : numpy.random.Generator, optional
        If given, the trial values of the ``×`` events are permuted among their trials (a null
        that keeps event times and the value distribution and breaks the pairing).

    Returns
    -------
    list of (times, weights)
        ``weights`` is None for plain events.
    """
    tr = pair["trials"]
    responded = (tr["animal_response"] < 2).to_numpy()
    rewarded = tr["earned_reward"].astype(bool).to_numpy() & responded
    cue = tr["goCue_start_time_in_session"].to_numpy()
    out = tr["reward_outcome_time_in_session"].to_numpy()

    def centred(v, keep):
        v = np.asarray(v, float)[keep]
        v = np.nan_to_num(v - np.nanmean(v)) if np.isfinite(v).any() else np.zeros(len(v))
        return rng.permutation(v) if rng is not None else v

    every = np.ones(len(tr), bool)
    built = {
        "go cue": (cue, None),
        "rewarded outcome": (out[rewarded], None),
        "unrewarded outcome": (out[responded & ~rewarded], None),
        "lick": (pair["licks"], None),
        "go cue × Q_sum": (cue, centred(tr["Q_sum"], every)),
        "rewarded outcome × RPE": (out[rewarded], centred(tr["RPE_earned"], rewarded)),
        "unrewarded outcome × RPE": (out[responded & ~rewarded],
                                     centred(tr["RPE_earned"], responded & ~rewarded)),
    }
    return [built[nm] for nm in names]


def event_design(pair: dict, kernel_s: Tuple[float, float] = (-1.0, 4.0),
                 events: Optional[Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]] = None):
    """Sparse FIR design matrix of task events for one session.

    One boxcar regressor per lag in ``kernel_s`` for each of: go cue, outcome on rewarded trials,
    outcome on unrewarded trials, and every lick. Plus an intercept. Kernels are shared across
    trials, so trial-to-trial amplitude variation (RPE, noise) stays in the residual.

    Parameters
    ----------
    pair : dict
    kernel_s : tuple
        Kernel span relative to each event, s (end exclusive).
    events : sequence of (times, weights), optional
        Other event regressors, e.g. from :func:`task_events`; ``weights`` (one per time, or
        None for 1) set the regressor's value at each event. Default: the four above.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_samples, n_events * n_lags + 1)``; the intercept is the last column.
    """
    n = len(pair["da"])
    fs, t0 = pair["fs"], pair["t0"]
    if events is None:
        events = task_events(pair)
    lag_i = np.arange(int(round(kernel_s[0] * fs)), int(round(kernel_s[1] * fs)))
    rows, cols, vals = [], [], []
    for e, (times, weights) in enumerate(events):
        times = np.asarray(times, float)
        weights = np.ones(len(times)) if weights is None else np.asarray(weights, float)
        ok = np.isfinite(times)
        idx, w = su.grid_index(times[ok], t0, fs), weights[ok]
        for j, k in enumerate(lag_i):
            ii = idx + k
            on = (ii >= 0) & (ii < n)
            rows.append(ii[on])
            cols.append(np.full(on.sum(), e * len(lag_i) + j))
            vals.append(w[on])
    n_col = len(events) * len(lag_i)
    rows.append(np.arange(n))
    cols.append(np.full(n, n_col))
    vals.append(np.ones(n))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return csr_matrix((np.concatenate(vals), (rows, cols)), shape=(n, n_col + 1))


def lagged_columns(x: np.ndarray, fs: float, lags_s: Tuple[float, float]) -> np.ndarray:
    """Lagged copies of a grid signal as regressor columns.

    Column ``j`` holds ``x`` delayed by ``lag_j`` (row ``t`` is ``x[t - lag_j]``), for every
    sample lag from ``lags_s[0]`` to ``lags_s[1]`` s inclusive. A positive lag lets the response
    follow ``x``; a negative one lets it come first. Samples off the grid and NaN samples are 0,
    so ``x`` should be centred (z-scored) first.

    Returns
    -------
    numpy.ndarray
        ``(len(x), n_lags)``.
    """
    x = np.nan_to_num(np.asarray(x, float))
    n = len(x)
    lags = np.arange(int(round(lags_s[0] * fs)), int(round(lags_s[1] * fs)) + 1)
    out = np.zeros((n, len(lags)))
    for j, k in enumerate(lags):
        if k >= 0:
            out[k:, j] = x[:n - k]
        else:
            out[:n + k, j] = x[-k:]
    return out


def task_residuals(pair: dict, kernel_s: Tuple[float, float] = (-1.0, 4.0),
                   ridge: float = 1e-3, extra: Optional[np.ndarray] = None,
                   task: bool = True, events=None) -> dict:
    """Split each z-scored signal into a task-evoked fit and a residual.

    Parameters
    ----------
    pair : dict
    kernel_s : tuple
        Kernel span relative to each event, s.
    ridge : float
        Small ridge on the normal equations, for overlapping-lick collinearity.
    extra : numpy.ndarray, optional
        ``(n_samples, k)`` further regressors fit with the task events, e.g. lagged motion
        energy from :func:`lagged_columns`.
    task : bool
        Include the task events. ``task=False`` with ``extra`` fits ``extra`` and an intercept only.
    events : sequence of (times, weights), optional
        Passed to :func:`event_design` (default: its four task events).

    Returns
    -------
    dict
        ``{sig: z, sig + '_fit': fitted, sig + '_res': residual, sig + '_r2': fraction explained,
        sig + '_beta': coefficients}`` for ``sig`` in ``da``, ``ne``. With ``task``, ``beta`` holds
        the event kernels (event ``e``, lag ``j`` at ``e * n_lags + j``; events in
        :func:`event_design` order), then the intercept, then one weight per ``extra`` column;
        without, the intercept and then ``extra``.
    """
    n = len(pair["da"])
    blocks = [event_design(pair, kernel_s, events)] if task else [csr_matrix(np.ones((n, 1)))]
    if extra is not None:
        blocks.append(csr_matrix(np.asarray(extra, float)))
    X = hstack(blocks, format="csr")
    XtX = (X.T @ X).toarray()
    XtX[np.diag_indices_from(XtX)] += ridge
    out = {}
    for sig in ("da", "ne"):
        y = su.zscore(pair[sig])
        beta = np.linalg.solve(XtX, X.T @ y)
        fit = X @ beta
        out[sig] = y
        out[sig + "_fit"] = fit
        out[sig + "_res"] = y - fit
        out[sig + "_r2"] = 1.0 - np.var(y - fit) / np.var(y)
        out[sig + "_beta"] = beta
    return out


def large_transients(z_a: np.ndarray, z_b: np.ndarray, t0: float, fs: float,
                     large_prom: float = 2.0, partner_prom: float = 1.0,
                     match_win_s: float = 0.5, min_sep_s: float = 0.5) -> pd.DataFrame:
    """Large transients of one signal and whether the other has a partner (``fip_11`` §1 rule).

    Transients are :func:`signal_utils.detect_transients` peaks. A transient of ``z_a`` is
    *large* at prominence ≥ ``large_prom``; it has a *partner* when ``z_b`` has a peak of
    prominence ≥ ``partner_prom`` within ±``match_win_s`` s, and is *solo* otherwise.

    Returns
    -------
    pandas.DataFrame
        One row per large transient of ``z_a``: ``i`` (sample), ``t`` (s), ``prominence``,
        ``lag`` (s to the nearest ``z_b`` peak, + = later), ``partner`` (bool).
    """
    det_a = su.detect_transients(z_a, fs, prominence=partner_prom, min_sep_s=min_sep_s)
    det_b = su.detect_transients(z_b, fs, prominence=partner_prom, min_sep_s=min_sep_s)
    big = det_a[det_a["prominence"] >= large_prom].reset_index(drop=True)
    big["t"] = t0 + big["i"] / fs
    _, lag = su.nearest_partner(big["t"].to_numpy(), t0 + det_b["i"].to_numpy() / fs)
    big["lag"] = lag
    big["partner"] = np.abs(lag) <= match_win_s
    return big[["i", "t", "prominence", "lag", "partner"]]
