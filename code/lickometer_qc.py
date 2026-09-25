"""
lickometer_qc.py — shared machinery for the lickometer QC notebooks.

``val_04_lickometer_qc.ipynb`` answers dynamic-foraging-processing#96 (licks the video shows
but the lickometer never registers) and ``val_05_lickometer_qc_methods.ipynb`` holds the analyses
behind its choices. Both work on the same two event tables, so the code that builds and scores
them lives here.

The two event tables
--------------------
``pose_events``
    One row per *excursion*: a run of tracked ``tongue_tip_center`` frames within ``d`` pixels of
    the nearer spout. Columns ``t_onset`` (first frame, what the library's ``detect_licks``
    timestamps), ``t_min`` / ``d_min`` (closest approach), ``n_frames``, ``span_s``, ``session``.
    Written before any refractory filter, so downstream code can redo it.
``lickometer_events``
    One row per lickometer event: ``session``, ``t``.

Both are on session time (seconds from the first go cue), the clock ``tongue_kins.parquet``
carries as ``time_in_session`` and ``nwb_df_licks.parquet`` carries as ``timestamps``.

Scoring
-------
:func:`annotate_pose_events` measures each excursion against the lickometer stream (nearest
event, gap to the previous and next event, whether the lickometer was active nearby) and
:func:`classify_events` sorts the surviving excursions into *confirmed contacts* (a lickometer
event within the match window) and *candidate missed licks* (contact-like, lickometer active,
no event within the window). "Contact-like" is calibrated per session from that session's own
confirmed contacts, see :func:`contact_reference`. :func:`summarize_sessions` reduces the
per-event table to one row per session with a Wilson interval, the lickometer-context split,
the stream-alignment statistics and the gates.

Python 3.9-compatible syntax only (see CLAUDE.md). Only ``numpy`` and ``pandas`` are needed to
score exported tables; the library is imported lazily inside :func:`load_session`, which only
runs on Code Ocean.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ── Event detection ───────────────────────────────────────────────────────────

def refractory_mask(times: Sequence[float], t_refractory: float) -> np.ndarray:
    """Which events survive a refractory period, as a positional mask.

    Reproduces the library's ``filter_timestamps_refractory`` on sorted input: an event is kept
    when it follows the last *kept* event by more than ``t_refractory``. Positional rather than
    value-based, so repeated timestamps are handled correctly.

    Parameters
    ----------
    times : sequence of float
        Sorted event times.
    t_refractory : float
        Minimum spacing between kept events, in seconds.

    Returns
    -------
    ndarray of bool
    """
    t = np.asarray(times, dtype=float)
    keep = np.zeros(t.size, dtype=bool)
    last = None
    for i in range(t.size):
        if last is None or (t[i] - last) > t_refractory:
            keep[i] = True
            last = t[i]
    return keep


def pose_excursions(tongue: pd.DataFrame, d: float) -> pd.DataFrame:
    """One row per run of tracked frames within ``d`` pixels of a spout.

    Works on tracked frames with the untracked ones dropped, exactly as the library's
    ``detect_licks`` does, so ``t_onset`` reproduces its output run for run.

    Parameters
    ----------
    tongue : pandas.DataFrame
        Needs ``time`` and ``spout_distance`` (NaN where the tongue is not tracked).
    d : float
        Distance that counts as an excursion, in pixels.

    Returns
    -------
    pandas.DataFrame
        ``t_onset``, ``t_min``, ``d_min``, ``n_frames``, ``span_s``.
    """
    trk = tongue.loc[tongue["spout_distance"].notna(), ["time", "spout_distance"]]
    t = trk["time"].to_numpy()
    dist = trk["spout_distance"].to_numpy()
    below = dist <= d
    cols = ["t_onset", "t_min", "d_min", "n_frames", "span_s"]
    if not below.any():
        return pd.DataFrame(columns=cols)
    starts = np.flatnonzero(below & ~np.concatenate([[False], below[:-1]]))
    ends = np.flatnonzero(below & ~np.concatenate([below[1:], [False]]))
    k_min = np.array([s + int(np.argmin(dist[s:e + 1])) for s, e in zip(starts, ends)])
    return pd.DataFrame({
        "t_onset": t[starts], "t_min": t[k_min], "d_min": dist[k_min],
        "n_frames": (ends - starts + 1).astype(int), "span_s": t[ends] - t[starts],
    })


# ── Code Ocean: raw intermediates → event tables ──────────────────────────────

def load_session(inter, conf: float = 0.8) -> Dict[str, object]:
    """Load one session's tongue keypoints, spout positions and lickometer licks.

    Code Ocean only: reads ``kps_raw_*.parquet``, ``tongue_kins.parquet`` and
    ``nwb_df_licks.parquet`` from a session's ``intermediate_data`` directory.

    Parameters
    ----------
    inter : pathlib.Path
        The session's ``intermediate_data`` directory.
    conf : float
        Keypoint confidence floor for masking.

    Returns
    -------
    dict
        ``tongue`` (masked keypoint table on session time, with ``spout_distance``),
        ``spoutL`` / ``spoutR`` (mean positions, de-mirrored), ``licks`` (sorted lickometer
        times on session time), ``video_offset`` (seconds to add to session time to get video
        time) and ``trial_bounds`` (first go cue, last trial stop; NaN pair when the trials
        table is missing).
    """
    from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (
        mask_keypoint_data,
    )
    from aind_dynamic_foraging_behavior_video_analysis.kinematics.video_clip_utils import (
        get_video_time,
    )

    inter = Path(inter)
    keypoint_dfs = {
        key: pd.read_parquet(inter / "kps_raw_{}.parquet".format(key))
        for key in ["tongue_tip_center", "spout_l", "spout_r"]
    }
    kins = pd.read_parquet(inter / "tongue_kins.parquet", columns=["time", "time_in_session"])

    tongue = mask_keypoint_data(keypoint_dfs, "tongue_tip_center", confidence_threshold=conf)
    tongue["time"] = kins["time_in_session"].values

    # NB the bottom camera mirrors left/right, so spout_r holds the left spout.
    spoutL = np.mean(keypoint_dfs["spout_r"][["x", "y"]], 0)
    spoutR = np.mean(keypoint_dfs["spout_l"][["x", "y"]], 0)

    tracked = tongue[["x", "y"]].notna().all(axis=1)
    xy = tongue.loc[tracked, ["x", "y"]].to_numpy()
    dL = np.linalg.norm(xy - np.array([spoutL["x"], spoutL["y"]]), axis=1)
    dR = np.linalg.norm(xy - np.array([spoutR["x"], spoutR["y"]]), axis=1)
    tongue["spout_distance"] = np.nan
    tongue.loc[tracked, "spout_distance"] = np.minimum(dL, dR)

    return {
        "tongue": tongue, "spoutL": spoutL, "spoutR": spoutR,
        "licks": np.sort(pd.read_parquet(inter / "nwb_df_licks.parquet")["timestamps"].to_numpy()),
        # kps_raw_* `time` is video-relative and tongue_kins carries both bases, so
        # get_video_time at 0 returns the session -> video offset itself.
        "video_offset": float(get_video_time(0.0, kins)),
        "trial_bounds": trial_bounds(inter),
    }


def trial_bounds(inter) -> Tuple[float, float]:
    """First go cue and last trial stop on session time, from ``nwb_df_trials.parquet``.

    Session time is seconds from the first go cue, so the bounds are the raw go cue and stop
    times shifted by the first go cue. Returns a NaN pair when the table or its columns are
    missing.
    """
    path = Path(inter) / "nwb_df_trials.parquet"
    if not path.exists():
        return (np.nan, np.nan)
    trials = pd.read_parquet(path)
    if "goCue_start_time" not in trials.columns:
        return (np.nan, np.nan)
    go = trials["goCue_start_time"].to_numpy(dtype=float)
    t0 = np.nanmin(go)
    stop_col = "stop_time" if "stop_time" in trials.columns else None
    end = np.nanmax(trials[stop_col].to_numpy(dtype=float)) if stop_col else np.nanmax(go)
    return (0.0, float(end - t0))


def lick_coverage(tongue: pd.DataFrame, licks: np.ndarray, halfwidth: float) -> float:
    """Fraction of lickometer events with a tracked tongue frame within ``halfwidth``.

    Conditions on a lick having happened, so it measures how well pose was tracking at the
    moments that matter, independently of how much the animal licked.
    """
    t = tongue.loc[tongue["spout_distance"].notna(), "time"].to_numpy()
    if t.size == 0 or len(licks) == 0:
        return np.nan
    licks = np.asarray(licks, dtype=float)
    lo = np.searchsorted(t, licks - halfwidth, side="left")
    hi = np.searchsorted(t, licks + halfwidth, side="right")
    return float(np.mean(hi > lo))


def build_event_tables(session_dirs: Iterable, d: float, conf: float = 0.8,
                       coverage_halfwidth: float = 0.1, verbose: bool = True
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the pose and lickometer event tables for many sessions (Code Ocean only).

    Parameters
    ----------
    session_dirs : iterable of path
        Session directories, each holding an ``intermediate_data`` folder.
    d : float
        Excursion distance in pixels.
    conf : float
        Keypoint confidence floor.
    coverage_halfwidth : float
        Half-width for :func:`lick_coverage`, in seconds.
    verbose : bool
        Print one line per session.

    Returns
    -------
    (pose_events, lickometer_events, sessions)
        The two event tables and a per-session sidecar with ``lick_coverage``,
        ``trial_start`` / ``trial_end``, ``video_offset``, ``duration_s`` and ``tracked_frac``.
    """
    needed = ["kps_raw_tongue_tip_center.parquet", "kps_raw_spout_l.parquet",
              "kps_raw_spout_r.parquet", "tongue_kins.parquet", "nwb_df_licks.parquet"]
    pose_rows, licko_rows, sess_rows = [], [], []
    session_dirs = list(session_dirs)
    for k, sdir in enumerate(session_dirs, 1):
        sdir = Path(sdir)
        inter = sdir / "intermediate_data"
        if not all((inter / f).exists() for f in needed):
            if verbose:
                print("  [{}/{}] {}: incomplete intermediates, skipped".format(
                    k, len(session_dirs), sdir.name))
            continue
        try:
            sess = load_session(inter, conf=conf)
        except Exception as exc:  # one bad session should not stop the batch
            if verbose:
                print("  [{}/{}] {}: FAILED ({})".format(k, len(session_dirs), sdir.name, exc))
            continue
        tongue = sess["tongue"]
        if len(sess["licks"]) == 0:
            continue
        exc_df = pose_excursions(tongue, d)
        exc_df["session"] = sdir.name
        pose_rows.append(exc_df)
        licko_rows.append(pd.DataFrame({"session": sdir.name, "t": sess["licks"]}))
        t0, t1 = sess["trial_bounds"]
        sess_rows.append({
            "session": sdir.name,
            "lick_coverage": lick_coverage(tongue, sess["licks"], coverage_halfwidth),
            "trial_start": t0, "trial_end": t1,
            "video_offset": sess["video_offset"],
            "duration_s": float(tongue["time"].max() - tongue["time"].min()),
            "tracked_frac": float(tongue["x"].notna().mean()),
        })
        if verbose:
            print("  [{}/{}] {}  excursions {:,}  lickometer {:,}  coverage {:.1%}".format(
                k, len(session_dirs), sdir.name, len(exc_df), len(sess["licks"]),
                sess_rows[-1]["lick_coverage"]))
    pose = pd.concat(pose_rows, ignore_index=True) if pose_rows else pd.DataFrame()
    licko = pd.concat(licko_rows, ignore_index=True) if licko_rows else pd.DataFrame()
    sessions = pd.DataFrame(sess_rows)
    return pose, licko, sessions


# ── Scoring against the lickometer ───────────────────────────────────────────

def nearest_offsets(times: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed offset to the nearest reference event, plus the gaps to the previous and next.

    Parameters
    ----------
    times : ndarray
        Query times (any order).
    ref : ndarray
        Sorted reference times.

    Returns
    -------
    (offset, gap_prev, gap_next)
        ``offset`` is ``ref - time`` for the nearest reference event (positive when the
        reference comes later). ``gap_prev`` / ``gap_next`` are non-negative distances to the
        last reference event at or before, and the first at or after, each query; ``inf``
        where none exists.
    """
    times = np.asarray(times, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if ref.size == 0:
        inf = np.full(times.size, np.inf)
        return inf.copy(), inf.copy(), inf.copy()
    idx = np.searchsorted(ref, times)
    prev_i = np.clip(idx - 1, 0, ref.size - 1)
    next_i = np.clip(idx, 0, ref.size - 1)
    gap_prev = np.where(idx > 0, times - ref[prev_i], np.inf)
    gap_next = np.where(idx < ref.size, ref[next_i] - times, np.inf)
    offset = np.where(gap_next <= gap_prev, gap_next, -gap_prev)
    return offset, gap_prev, gap_next


def lickometer_rhythm(lk: np.ndarray, lo: float = 0.08, hi: float = 0.3) -> float:
    """Modal inter-lick interval of a lickometer stream, in seconds.

    The median of the intervals between ``lo`` and ``hi`` seconds, which brackets the licking
    rhythm (5 to 8 Hz in these animals) and excludes pauses between bouts.
    """
    lk = np.sort(np.asarray(lk, dtype=float))
    ili = np.diff(lk)
    inband = ili[(ili > lo) & (ili < hi)]
    return float(np.median(inband)) if inband.size else np.nan


def annotate_pose_events(pose: pd.DataFrame, licko: pd.DataFrame, t_refractory: float = 0.1,
                         match_window: float = 0.1, active_window: float = 60.0,
                         sibling_window: float = 0.25) -> pd.DataFrame:
    """Measure every pose excursion against its session's lickometer stream.

    Parameters
    ----------
    pose, licko : pandas.DataFrame
        The two event tables (see module docstring).
    t_refractory : float
        Refractory period applied to both streams, in seconds. The lickometer stream is
        filtered before anything is measured against it.
    match_window : float
        A lickometer event within this many seconds of ``t_min`` confirms the excursion.
    active_window : float
        The lickometer counts as active around an excursion when it has an event within this
        many seconds on *both* sides. Excursions outside active stretches (before the first
        trial, after the animal stops, during long pauses) are not scored.
    sibling_window : float
        Half-width for counting neighbouring excursions (before the refractory filter).

    Returns
    -------
    pandas.DataFrame
        ``pose`` with added columns: ``kept`` (survives the refractory filter), ``offset``
        (signed seconds to the nearest lickometer event), ``gap_prev`` / ``gap_next`` / ``gap``
        (seconds), ``has_lk`` (event within ``match_window``), ``lk_active``, ``n_siblings``.
    """
    out = []
    for sess, g in pose.groupby("session", sort=False):
        g = g.sort_values("t_onset").copy()
        lk_raw = np.sort(licko.loc[licko["session"] == sess, "t"].to_numpy(dtype=float))
        lk = lk_raw[refractory_mask(lk_raw, t_refractory)]
        g["kept"] = refractory_mask(g["t_onset"].to_numpy(), t_refractory)
        offset, gap_prev, gap_next = nearest_offsets(g["t_min"].to_numpy(), lk)
        g["offset"] = offset
        g["gap_prev"] = gap_prev
        g["gap_next"] = gap_next
        g["gap"] = np.minimum(gap_prev, gap_next)
        g["has_lk"] = g["gap"] <= match_window
        g["lk_active"] = (gap_prev <= active_window) & (gap_next <= active_window)
        t_all = g["t_onset"].to_numpy()
        g["n_siblings"] = (np.searchsorted(t_all, t_all + sibling_window)
                           - np.searchsorted(t_all, t_all - sibling_window) - 1)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pose.copy()


def contact_reference(pose_annot: pd.DataFrame, depth_quantile: float = 0.5,
                      dwell_quantile: float = 0.25, dwell_max_factor: float = 2.0,
                      dwell_max_quantile: float = 0.95) -> pd.DataFrame:
    """Per-session description of what a confirmed contact looks like in pose.

    A session's confirmed contacts are its kept excursions with a lickometer event in the match
    window. Their closest approach and dwell define "contact-like" for that session, which
    absorbs the differences in camera geometry and pixel scale between sessions.

    Parameters
    ----------
    pose_annot : pandas.DataFrame
        Output of :func:`annotate_pose_events`.
    depth_quantile : float
        Contact-like excursions reach at least as close as this quantile of confirmed
        ``d_min`` (0.5 = the median confirmed contact).
    dwell_quantile : float
        And stay within ``d`` for at least this quantile of confirmed ``n_frames``.
    dwell_max_factor, dwell_max_quantile : float
        And no longer than ``dwell_max_factor`` times this quantile of confirmed ``n_frames``;
        a keypoint parked on the spout for seconds is not a lick.

    Returns
    -------
    pandas.DataFrame
        Indexed by session: ``d_ref``, ``frames_lo``, ``frames_hi``, ``n_confirmed_all``.
    """
    conf = pose_annot[pose_annot["kept"] & pose_annot["has_lk"]]
    ref = conf.groupby("session").agg(
        d_ref=("d_min", lambda s: s.quantile(depth_quantile)),
        frames_lo=("n_frames", lambda s: s.quantile(dwell_quantile)),
        frames_hi=("n_frames", lambda s: dwell_max_factor * s.quantile(dwell_max_quantile)),
        n_confirmed_all=("d_min", "size"),
    )
    return ref


def classify_events(pose_annot: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Label each kept excursion as a confirmed contact, a candidate missed lick, or neither.

    Parameters
    ----------
    pose_annot : pandas.DataFrame
        Output of :func:`annotate_pose_events`.
    ref : pandas.DataFrame
        Output of :func:`contact_reference`.

    Returns
    -------
    pandas.DataFrame
        ``pose_annot`` with ``contact`` (contact-like by the session's reference),
        ``confirmed`` (kept, contact-like, lickometer event in the window) and ``candidate``
        (kept, contact-like, lickometer active, no event in the window).
    """
    p = pose_annot.merge(ref[["d_ref", "frames_lo", "frames_hi"]], left_on="session",
                         right_index=True, how="left")
    p["contact"] = ((p["d_min"] <= p["d_ref"]) & (p["n_frames"] >= p["frames_lo"])
                    & (p["n_frames"] <= p["frames_hi"]))
    scored = p["kept"] & p["contact"] & p["lk_active"]
    p["confirmed"] = scored & p["has_lk"]
    p["candidate"] = scored & ~p["has_lk"]
    return p


def candidate_context(events: pd.DataFrame, rhythm: Dict[str, float], bout_factor: float = 2.5,
                      isolated_gap: float = 1.0, cluster_halfwidth: float = 0.5,
                      cluster_min: int = 3) -> pd.Series:
    """Sort candidate missed licks by what the lickometer was doing around them.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`classify_events`; only rows with ``candidate`` are labelled.
    rhythm : dict
        Session -> modal inter-lick interval in seconds (:func:`lickometer_rhythm`).
    bout_factor : float
        A candidate is *in a bout* when lickometer events sit within ``bout_factor`` modal
        intervals on both sides: the lickometer skipped a beat.
    isolated_gap : float
        A candidate is *isolated* when the nearest lickometer event is further than this, in
        seconds, and it has fewer than ``cluster_min`` candidates around it.
    cluster_halfwidth, cluster_min : float, int
        Candidates with at least ``cluster_min`` candidates (itself included) within
        ``cluster_halfwidth`` seconds form a *silent bout*: rhythmic contact-like excursions
        the lickometer recorded nothing for.

    Returns
    -------
    pandas.Series
        One of ``"in_bout"``, ``"silent_bout"``, ``"isolated"``, ``"bout_edge"`` for each
        candidate row, ``NaN`` elsewhere; aligned to ``events.index``.
    """
    labels = pd.Series(np.nan, index=events.index, dtype=object)
    cand = events[events["candidate"]]
    for sess, g in cand.groupby("session"):
        mode = rhythm.get(sess, np.nan)
        t = g["t_min"].to_numpy()
        order = np.argsort(t)
        ts = t[order]
        n_near = np.searchsorted(ts, ts + cluster_halfwidth) - np.searchsorted(ts, ts - cluster_halfwidth)
        n_near_unsorted = np.empty_like(n_near)
        n_near_unsorted[order] = n_near
        in_bout = (g["gap_prev"] <= bout_factor * mode) & (g["gap_next"] <= bout_factor * mode)
        silent = (n_near_unsorted >= cluster_min) & ~in_bout.to_numpy()
        isolated = (g["gap"] > isolated_gap).to_numpy() & ~silent & ~in_bout.to_numpy()
        lab = np.where(in_bout, "in_bout",
                       np.where(silent, "silent_bout",
                                np.where(isolated, "isolated", "bout_edge")))
        labels.loc[g.index] = lab
    return labels


def skipped_beats(lk: np.ndarray, pose_tmin: np.ndarray, mode: float, exclude: float = 0.06,
                  double: Tuple[float, float] = (1.6, 2.4), single: Tuple[float, float] = (0.8, 1.2)
                  ) -> Dict[str, float]:
    """The lickometer's own rhythm as a witness: how often does a skipped beat hold a tongue?

    Within a bout the lickometer fires at its modal interval. An interval about twice that long
    is a skipped beat: either the animal paused for one cycle or the lickometer missed a lick.
    A pose excursion in the middle of the gap says the tongue was at the spout. The same test
    on normal single intervals gives the false-alarm rate of finding a tongue mid-interval.

    Parameters
    ----------
    lk : ndarray
        Refractory-filtered lickometer times.
    pose_tmin : ndarray
        Closest-approach times of all excursions (before the refractory filter).
    mode : float
        Modal inter-lick interval, seconds.
    exclude : float
        Seconds either side of the flanking lickometer events to ignore, so an excursion
        belonging to a neighbouring lick is not counted.
    double, single : tuple of float
        Interval bands, in multiples of ``mode``.

    Returns
    -------
    dict
        ``n_double``, ``n_double_with_pose``, ``n_single``, ``n_single_with_pose``.
    """
    lk = np.sort(np.asarray(lk, dtype=float))
    pose_tmin = np.sort(np.asarray(pose_tmin, dtype=float))
    ili = np.diff(lk)

    def count(mask):
        js = np.flatnonzero(mask)
        hits = 0
        for j in js:
            lo, hi = lk[j] + exclude, lk[j + 1] - exclude
            if hi <= lo:
                continue
            a, b = np.searchsorted(pose_tmin, [lo, hi])
            hits += int(b > a)
        return js.size, hits

    n_dbl, h_dbl = count((ili > double[0] * mode) & (ili < double[1] * mode))
    n_sgl, h_sgl = count((ili > single[0] * mode) & (ili < single[1] * mode))
    return {"n_double": n_dbl, "n_double_with_pose": h_dbl,
            "n_single": n_sgl, "n_single_with_pose": h_sgl}


def alignment_stats(events: pd.DataFrame, match_window: float = 0.1) -> pd.DataFrame:
    """Per-session offset between the streams, from excursions with a lickometer event nearby.

    Uses ``offset`` (lickometer minus closest approach) for kept excursions with an event inside
    the match window. A median far from the population, or a wide interquartile range, means
    the two clocks disagree and the session's miss counts cannot be trusted.

    Returns
    -------
    pandas.DataFrame
        Indexed by session: ``offset_median_ms``, ``offset_iqr_ms``, ``offset_p5_ms``,
        ``offset_p95_ms``, ``n_offset``.
    """
    m = events[events["kept"] & (events["gap"] <= match_window)]
    def q(s, p):
        return 1000.0 * s.quantile(p)
    return m.groupby("session")["offset"].agg(
        offset_median_ms=lambda s: q(s, 0.5),
        offset_iqr_ms=lambda s: q(s, 0.75) - q(s, 0.25),
        offset_p5_ms=lambda s: q(s, 0.05),
        offset_p95_ms=lambda s: q(s, 0.95),
        n_offset="size",
    )


def dropout_minutes(events: pd.DataFrame, licko_kept: Dict[str, np.ndarray],
                    bounds: Dict[str, Tuple[float, float]], min_contacts: int = 10
                    ) -> pd.Series:
    """Minutes with many contact-like excursions and no lickometer event at all.

    Scans one-minute bins between ``bounds[session]``. A bin with at least ``min_contacts``
    contact-like excursions and zero lickometer events is a dropout minute: the animal was at
    the spout repeatedly and the lickometer recorded nothing.
    """
    out = {}
    for sess, g in events[events["kept"] & events["contact"]].groupby("session"):
        lk = licko_kept.get(sess)
        lo, hi = bounds.get(sess, (np.nan, np.nan))
        if lk is None or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[sess] = np.nan
            continue
        edges = np.arange(lo, hi + 60.0, 60.0)
        hp = np.histogram(g["t_min"], edges)[0]
        hl = np.histogram(lk, edges)[0]
        out[sess] = int(((hp >= min_contacts) & (hl == 0)).sum())
    return pd.Series(out, name="dropout_minutes")


def wilson_ci(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials. NaN pair when ``n`` is 0."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def summarize_sessions(events: pd.DataFrame, licko: pd.DataFrame, ref: pd.DataFrame,
                       t_refractory: float = 0.1, match_window: float = 0.1,
                       sessions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per session: counts, candidate rate with interval, context, alignment, rhythm.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`classify_events`, with a ``context`` column from
        :func:`candidate_context` if the context split is wanted.
    licko : pandas.DataFrame
        The lickometer event table.
    ref : pandas.DataFrame
        Output of :func:`contact_reference`.
    t_refractory, match_window : float
        As used upstream.
    sessions : pandas.DataFrame, optional
        Sidecar from :func:`build_event_tables`; its columns are merged in when given, and
        ``trial_start`` / ``trial_end`` bound the dropout scan. Without it the lickometer's
        first and last event bound the scan.

    Returns
    -------
    pandas.DataFrame
        Indexed by session, sorted by ``miss_rate`` descending.
    """
    rows = []
    licko_kept = {}
    bounds = {}
    rhythm = {}
    for sess, g in licko.groupby("session"):
        lk_raw = np.sort(g["t"].to_numpy(dtype=float))
        lk = lk_raw[refractory_mask(lk_raw, t_refractory)]
        licko_kept[sess] = lk
        rhythm[sess] = lickometer_rhythm(lk)
        bounds[sess] = (float(lk.min()), float(lk.max())) if lk.size else (np.nan, np.nan)
    if sessions is not None and {"trial_start", "trial_end"} <= set(sessions.columns):
        for _, r in sessions.iterrows():
            if np.isfinite(r["trial_start"]) and np.isfinite(r["trial_end"]):
                bounds[r["session"]] = (float(r["trial_start"]), float(r["trial_end"]))

    for sess, g in events.groupby("session"):
        k = g[g["kept"]]
        n_cand = int(k["candidate"].sum())
        n_conf = int(k["confirmed"].sum())
        lo, hi = wilson_ci(n_cand, n_cand + n_conf)
        if np.isfinite(lo):
            # Rounding can push the bound a hair past the point estimate at 0 or 1.
            p_hat = n_cand / float(n_cand + n_conf)
            lo, hi = min(max(lo, 0.0), p_hat), max(min(hi, 1.0), p_hat)
        lk = licko_kept.get(sess, np.array([]))
        sb = skipped_beats(lk, g["t_min"].to_numpy(), rhythm.get(sess, np.nan))
        row = {
            "session": sess, "subject": sess.split("_")[1] if "_" in sess else sess,
            "n_lickometer": int(lk.size), "ili_mode_ms": 1000.0 * rhythm.get(sess, np.nan),
            "n_excursions": int(k.shape[0]), "n_unconfirmed_raw": int((~k["has_lk"]).sum()),
            "raw_disagreement": float((~k["has_lk"]).mean()) if len(k) else np.nan,
            "n_contact": int((k["contact"] & k["lk_active"]).sum()),
            "n_confirmed": n_conf, "n_candidate": n_cand,
            "miss_rate": n_cand / float(n_cand + n_conf) if (n_cand + n_conf) else np.nan,
            "miss_lo": lo, "miss_hi": hi,
            "candidates_per_1000_licks": 1000.0 * n_cand / lk.size if lk.size else np.nan,
        }
        if "context" in k.columns:
            ctx = k.loc[k["candidate"], "context"].value_counts()
            for name in ["in_bout", "silent_bout", "isolated", "bout_edge"]:
                row["n_" + name] = int(ctx.get(name, 0))
        row.update(sb)
        row["skipped_beat_frac"] = (sb["n_double_with_pose"] / float(sb["n_double"])
                                    if sb["n_double"] else np.nan)
        row["single_interval_frac"] = (sb["n_single_with_pose"] / float(sb["n_single"])
                                       if sb["n_single"] else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("session")
    out = out.join(ref, how="left")
    out = out.join(alignment_stats(events, match_window), how="left")
    out = out.join(dropout_minutes(events, licko_kept, bounds), how="left")
    if sessions is not None:
        extra = sessions.set_index("session")
        out = out.join(extra[[c for c in extra.columns if c not in out.columns]], how="left")
    return out.sort_values("miss_rate", ascending=False)


# ── Matching diagnostics (val_05) ─────────────────────────────────────────────

def match_greedy(actual: np.ndarray, predicted: np.ndarray, window: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one greedy matching in time order, as the library's matcher does it.

    Returns boolean masks over the *sorted* ``actual`` and ``predicted`` arrays marking the
    matched events.
    """
    a = np.sort(np.asarray(actual, dtype=float))
    d = np.sort(np.asarray(predicted, dtype=float))
    am = np.zeros(a.size, dtype=bool)
    dm = np.zeros(d.size, dtype=bool)
    i = j = 0
    while i < a.size and j < d.size:
        if abs(d[j] - a[i]) <= window:
            am[i] = dm[j] = True
            i += 1
            j += 1
        elif d[j] < a[i]:
            j += 1
        else:
            i += 1
    return am, dm


def match_optimal(actual: np.ndarray, predicted: np.ndarray, window: float) -> int:
    """Size of an order-preserving maximum matching between two event trains.

    Pairs are allowed when ``|actual_i - predicted_j| <= window`` and each event is used at
    most once. For sorted events on a line the order-preserving optimum is a global optimum,
    so this is an upper bound on any matcher, greedy included.
    """
    a = np.sort(np.asarray(actual, dtype=float))
    d = np.sort(np.asarray(predicted, dtype=float))
    if a.size == 0 or d.size == 0:
        return 0
    merged = np.concatenate([a, d])
    is_a = np.concatenate([np.ones(a.size, bool), np.zeros(d.size, bool)])
    order = np.argsort(merged, kind="mergesort")
    merged, is_a = merged[order], is_a[order]
    a_pos = np.cumsum(is_a) - 1
    d_pos = np.cumsum(~is_a) - 1
    cuts = np.flatnonzero(np.diff(merged) > window) + 1
    bounds = np.concatenate([[0], cuts, [merged.size]])
    total = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        blk = slice(lo, hi)
        ia, jd = a_pos[blk][is_a[blk]], d_pos[blk][~is_a[blk]]
        if ia.size == 0 or jd.size == 0:
            continue
        ok = np.abs(a[ia][:, None] - d[jd][None, :]) <= window
        dp = np.zeros((ia.size + 1, jd.size + 1), dtype=np.int32)
        for i in range(1, ia.size + 1):
            dp[i, 1:] = np.maximum.accumulate(
                np.maximum(dp[i - 1, 1:], dp[i - 1, :-1] + ok[i - 1]))
        total += int(dp[ia.size, jd.size])
    return total
