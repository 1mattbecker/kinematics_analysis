"""
behavior_utils.py — task-behaviour helpers shared by several notebook series.

Lick bouts come from the library (``aind_dynamic_foraging_basic_analysis.licks.annotation``,
whose bout gap is its ``BOUT_THRESHOLD``, 0.7 s), so every series groups licks the same way.
Movement bouts and the within-trial / inter-trial split were in ``ephys_utils`` while ``eph_07``
was their only user; ``men_00`` is the second, so they live here. They are generic enough for
the video-analysis library, which is where they would go next (see CLAUDE.md, "Library vs repo
boundary").

Clock: every time is session time (s from the first go cue), the clock of the
``*_in_session`` trial columns and ``df_events['timestamps']``.

Contents
--------
Lick bouts
    :func:`lick_bouts`
Movement bouts
    :func:`annotate_movement_bouts`, :func:`classify_bout_times`, :func:`get_session_bout_times`
Task context
    :func:`label_context`
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd

from signal_utils import nearest_partner


# ── Lick bouts ────────────────────────────────────────────────────────────────

def lick_bouts(df_events: pd.DataFrame) -> pd.DataFrame:
    """One row per lick bout, grouped by the library's ``annotate_lick_bouts``.

    Parameters
    ----------
    df_events : pandas.DataFrame
        ``timestamps`` and ``event``; rows with ``event`` ``left_lick_time`` /
        ``right_lick_time`` are the licks (other rows are ignored).

    Returns
    -------
    pandas.DataFrame
        ``bout_number, onset, offset, n_licks``.
    """
    from aind_dynamic_foraging_basic_analysis.licks import annotation

    events = df_events.sort_values("timestamps").reset_index(drop=True)
    licks = events[events["event"].isin(["left_lick_time", "right_lick_time"])]
    if licks.empty:
        return pd.DataFrame(columns=["bout_number", "onset", "offset", "n_licks"])
    annotated = annotation.annotate_lick_bouts(SimpleNamespace(df_events=events))
    g = annotated.groupby("bout_number")["timestamps"]
    return pd.DataFrame({"onset": g.min(), "offset": g.max(), "n_licks": g.size()}).reset_index()


# ── Movement bouts ────────────────────────────────────────────────────────────
#
# Ported from tongue_kinematics_ephys_intertrialmovs.ipynb. `annotate_movement_bouts` groups raw
# movements into bouts by inter-movement gap; `classify_bout_times` and `get_session_bout_times`
# implement the within-trial (go-responsive) vs ITI split, with the thresholds as explicit
# parameters. `classify_bout_times` takes any onset times, so it also splits lick bouts.

def annotate_movement_bouts(
    movs: pd.DataFrame,
    gap_threshold_s: float = 0.5,
    time_col: str = "start_time",
) -> pd.DataFrame:
    """
    Annotate tongue movements into bouts based on inter-movement gaps.

    A new bout begins whenever the gap from the previous movement's start
    exceeds `gap_threshold_s`. Bout IDs are assigned sequentially within
    each session.

    Parameters
    ----------
    movs : pd.DataFrame
        Movements dataframe. Must contain `time_col` and (optionally) a
        'session' column — if present, bouts are numbered per session.
    gap_threshold_s : float
        Minimum gap (in seconds) between consecutive movement starts to
        define a new bout.
    time_col : str
        Column name for movement onset time.

    Returns
    -------
    pd.DataFrame
        Copy of `movs` with added columns:
          - mov_bout_id         : int, sequential within session
          - mov_bout_start      : bool, True for first movement in bout
          - mov_bout_end        : bool, True for last movement in bout
          - mov_bout_size       : int, number of movements in the bout
          - mov_bout_position   : int, 1-indexed position within bout
    """
    out = movs.copy()

    group_cols = ["session"] if "session" in out.columns else []
    # stable sort by time within each session
    out = out.sort_values(group_cols + [time_col]).reset_index(drop=False)
    original_index = out["index"]

    def _label(g):
        t = g[time_col].to_numpy()
        gaps = np.diff(t, prepend=t[0] - (gap_threshold_s + 1))
        new_bout = gaps > gap_threshold_s
        bout_id = np.cumsum(new_bout) - 1  # 0-indexed within session

        # position within bout + bout size
        g = g.assign(mov_bout_id=bout_id)
        g["mov_bout_position"] = g.groupby("mov_bout_id").cumcount() + 1
        size_map = g.groupby("mov_bout_id").size()
        g["mov_bout_size"] = g["mov_bout_id"].map(size_map)
        g["mov_bout_start"] = g["mov_bout_position"] == 1
        g["mov_bout_end"]   = g["mov_bout_position"] == g["mov_bout_size"]
        return g

    if group_cols:
        out = out.groupby(group_cols, group_keys=False).apply(_label)
    else:
        out = _label(out)

    # restore original row order
    out = out.set_index("index").loc[original_index.values].reset_index(drop=True)
    return out


def classify_bout_times(
    bout_times: np.ndarray,
    go_cue_times: np.ndarray,
    go_response_window_s: float,
    iti_min_post_cue_s: float,
    iti_min_pre_next_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split bout onset times into within-trial (go-responsive) vs ITI classes
    relative to the surrounding go cues.

    A bout is **go-responsive** if it starts at or after the preceding go cue
    and within `go_response_window_s` of it. A bout is **ITI** if it starts
    more than `iti_min_post_cue_s` after the preceding go cue AND more than
    `iti_min_pre_next_s` before the next go cue. A bout lacking a preceding or
    following go cue, or falling in neither window, is excluded from both
    classes (this is deliberate — the gap between the two windows is a
    no-man's-land, not an ITI bout by a looser definition).

    Parameters
    ----------
    bout_times : array-like of float
        Bout onset times, in session time. Need not be pre-sorted.
    go_cue_times : array-like of float
        Go-cue onset times, in session time. Need not be pre-sorted.
    go_response_window_s : float
        Max time after the preceding go cue for a bout to count as
        go-responsive.
    iti_min_post_cue_s : float
        Min time after the preceding go cue for a bout to count as ITI.
    iti_min_pre_next_s : float
        Min time before the next go cue for a bout to count as ITI.

    Returns
    -------
    (go_responsive_times, iti_times) : tuple of np.ndarray
        Bout onset times in each class, sorted ascending.
    """
    bout_times = np.sort(np.asarray(bout_times, dtype=float))
    go_cues = np.sort(np.asarray(go_cue_times, dtype=float))

    prev_idx = np.searchsorted(go_cues, bout_times, side="right") - 1
    next_idx = prev_idx + 1
    has_prev = prev_idx >= 0
    has_next = next_idx < len(go_cues)

    dt_prev = np.full(len(bout_times), np.nan)
    dt_next = np.full(len(bout_times), np.nan)
    dt_prev[has_prev] = bout_times[has_prev] - go_cues[prev_idx[has_prev]]
    dt_next[has_next] = go_cues[next_idx[has_next]] - bout_times[has_next]

    go_resp_mask = has_prev & (dt_prev >= 0) & (dt_prev <= go_response_window_s)
    iti_mask = (
        has_prev & has_next
        & (dt_prev > iti_min_post_cue_s)
        & (dt_next > iti_min_pre_next_s)
    )

    return bout_times[go_resp_mask], bout_times[iti_mask]


def get_session_bout_times(
    movs: pd.DataFrame,
    trials: pd.DataFrame,
    gap_threshold_s: float = 0.5,
    go_response_window_s: float = 2.0,
    iti_min_post_cue_s: float = 2.0,
    iti_min_pre_next_s: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Movement-bout-derived within-trial (go-responsive) vs ITI event times for
    one session.

    Groups `movs` into bouts via `annotate_movement_bouts`, takes the first
    movement of each bout as the bout onset, then classifies those onsets
    relative to `trials`' go cues via `classify_bout_times`.

    Parameters
    ----------
    movs : pd.DataFrame
        Session movements table. Must contain `start_time`; `session` is
        optional (bouts are numbered per-session if present).
    trials : pd.DataFrame
        Session trials table. Must contain `goCue_start_time_in_session`.
    gap_threshold_s : float
        Passed to `annotate_movement_bouts`.
    go_response_window_s, iti_min_post_cue_s, iti_min_pre_next_s : float
        Passed to `classify_bout_times`.

    Returns
    -------
    (go_responsive_times, iti_times) : tuple of np.ndarray
    """
    movs_annot = annotate_movement_bouts(movs, gap_threshold_s=gap_threshold_s)
    bout_starts = movs_annot.loc[movs_annot["mov_bout_start"], "start_time"].to_numpy()
    go_cues = trials["goCue_start_time_in_session"].dropna().to_numpy()
    return classify_bout_times(
        bout_starts,
        go_cues,
        go_response_window_s=go_response_window_s,
        iti_min_post_cue_s=iti_min_post_cue_s,
        iti_min_pre_next_s=iti_min_pre_next_s,
    )


# ── Task context ──────────────────────────────────────────────────────────────

def label_context(times, trials: pd.DataFrame, licks, outcome_win: Tuple[float, float] = (0.0, 1.5),
                  cue_win: Tuple[float, float] = (0.0, 1.5), lick_win_s: float = 1.0) -> np.ndarray:
    """Task context of each event time.

    In order of precedence: ``rewarded`` / ``unrewarded`` (within ``outcome_win`` after that
    outcome), ``cue, no response`` (within ``cue_win`` after the go cue of an ignored trial),
    ``licking`` (a lick within ±``lick_win_s``), ``quiet`` (none of these).

    Parameters
    ----------
    times : array_like
        Event times (session clock).
    trials : pandas.DataFrame
        ``animal_response`` (2 = no response), ``earned_reward``,
        ``reward_outcome_time_in_session``, ``goCue_start_time_in_session``.
    licks : array_like
        Sorted lick times (either side).
    """
    times = np.asarray(times, float)
    responded = (trials["animal_response"] < 2).to_numpy()
    rewarded = trials["earned_reward"].fillna(0).astype(bool).to_numpy() & responded
    out = trials["reward_outcome_time_in_session"].to_numpy(float)
    cue = trials["goCue_start_time_in_session"].to_numpy(float)

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
    _, lag = nearest_partner(times, licks)
    labels[np.abs(lag) <= lick_win_s] = "licking"
    labels[within(cue[~responded], cue_win)] = "cue, no response"
    labels[within(out[responded & ~rewarded], outcome_win)] = "unrewarded"
    labels[within(out[rewarded], outcome_win)] = "rewarded"
    return labels
