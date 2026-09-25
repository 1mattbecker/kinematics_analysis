"""
Shared ephys analysis utilities for kinematics × ephys notebooks.

Layering: this module sits *on top of* the library's
``ephys.tongue_ephys`` (session lookup, intermediate loading, event tables,
raster/PSTH primitives). What lives here is specific to the LC-NE
RT-encoding question and free to churn: ``AnalysisConfig``, spike counting
into windows, per-trial feature tables, session bundles and
``all_counts_df``. The bout helpers (``annotate_movement_bouts``,
``classify_bout_times``) are generic enough for the library but have one
consumer (``eph_07``); promote them when a second appears. See CLAUDE.md,
"Library vs repo boundary".

Depends on:
  aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys
    (find_session_dir, load_intermediate_data, get_events_dict, build_event_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys import (
    build_event_df,
    find_session_dir,
    get_events_dict,
    load_intermediate_data,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    align_key: str = "goCue"
    count_window_s: Tuple[float, float] = (0.0, 0.100)
    min_trials_per_group: int = 8
    only_cue_response_trials: bool = True
    latency_window_s: Optional[Tuple[float, float]] = None
    baseline_window_s: Optional[Tuple[float, float]] = None


# ── Spike-time utilities ──────────────────────────────────────────────────────

def count_spikes_in_window(
    spike_times_sorted: np.ndarray,
    t0: float,
    window: Tuple[float, float],
) -> int:
    a, b = t0 + window[0], t0 + window[1]
    if not np.isfinite(a) or not np.isfinite(b):
        return 0
    i0 = np.searchsorted(spike_times_sorted, a, side="left")
    i1 = np.searchsorted(spike_times_sorted, b, side="left")
    return int(i1 - i0)


def first_spike_latency_in_window(
    spike_times_sorted: np.ndarray,
    t0: float,
    window: Tuple[float, float],
) -> Tuple[float, bool]:
    """Return (first_spike_latency_s, had_spike). Latency is np.nan if no spike."""
    a, b = t0 + window[0], t0 + window[1]
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan, False
    i0 = np.searchsorted(spike_times_sorted, a, side="left")
    i1 = np.searchsorted(spike_times_sorted, b, side="left")
    if i0 < i1:
        return float(spike_times_sorted[i0] - t0), True
    return np.nan, False


def mannwhitney_summary(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) == 0 or len(y) == 0:
        return dict(p=np.nan, U=np.nan, effect=np.nan)
    stat = mannwhitneyu(x, y, alternative="two-sided")
    return dict(p=float(stat.pvalue), U=float(stat.statistic),
                effect=float(np.mean(x) - np.mean(y)))


# ── Trial feature construction ────────────────────────────────────────────────

def build_trial_features(
    movs: pd.DataFrame,
    licks: pd.DataFrame,
    df_trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return per-trial kinematics features joined from the movements table.
    Requires movs to already be annotated by annotate_movement_timing +
    add_lick_metadata_to_movements.
    """
    tm = movs.copy()

    required_cols = {
        "cue_response_movement_number",
        "movement_latency_from_go",
        "movement_number_in_trial",
        "cue_response",
    }
    missing = required_cols - set(tm.columns)
    if missing:
        raise ValueError(
            f"movs is missing required timing columns {missing}. "
            "Run annotate_movement_timing + add_lick_metadata_to_movements first."
        )

    crmn = tm.groupby("trial")["cue_response_movement_number"].first().astype("Int64")
    rt_first = tm.groupby("trial")["movement_latency_from_go"].min()

    if "cue_response" in tm.columns:
        rt_cr = (
            tm.loc[tm["cue_response"] == True]
              .groupby("trial")["movement_latency_from_go"]
              .first()
        )
    else:
        tmp = tm.copy()
        tmp["movement_number_in_trial"] = pd.to_numeric(
            tmp["movement_number_in_trial"], errors="coerce"
        ).astype("Int64")
        tmp = tmp.join(crmn.rename("crmn"), on="trial")
        rt_cr = (
            tmp.loc[tmp["movement_number_in_trial"] == tmp["crmn"]]
               .groupby("trial")["movement_latency_from_go"]
               .first()
        )

    out = pd.DataFrame({
        "cue_response_movement_number": crmn,
        "reaction_time_firstmove": rt_first,
        "reaction_time_cueresponse": rt_cr,
    })

    kcols = [
        "peak_velocity", "duration", "excursion_angle_deg",
        "endpoint_x", "endpoint_y",
        "out_duration", "out_peak_velocity", "out_mean_velocity", "out_total_distance",
    ]

    first_moves = (
        tm.loc[tm["movement_number_in_trial"] == 1, ["trial", *kcols]]
          .drop_duplicates("trial", keep="first")
          .set_index("trial")
          .rename(columns={c: f"first_move_{c}" for c in kcols})
    )
    cue_resp_moves = (
        tm.loc[tm["cue_response"] == True, ["trial", *kcols]]
          .drop_duplicates("trial", keep="first")
          .set_index("trial")
          .rename(columns={c: f"cue_response_{c}" for c in kcols})
    )

    return out.join(first_moves, how="left").join(cue_resp_moves, how="left").sort_index()


# ── Movement bouts ────────────────────────────────────────────────────────────
#
# Ported from tongue_kinematics_ephys_intertrialmovs.ipynb (the sole copy in the
# repo prior to this port; see TODO.md's eph_07 item for why this lives here
# rather than in a new bout_utils.py module). `annotate_movement_bouts` groups
# raw movements into bouts by inter-movement gap; `classify_bout_times` and
# `get_session_bout_times` implement the within-trial (go-responsive) vs ITI
# split that the source notebook copy-pasted into four near-identical
# `get_session_bout_times` cells with drifting thresholds — consolidated here
# as one function with the thresholds as explicit parameters.

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


# ── Session bundle ────────────────────────────────────────────────────────────

def make_session_bundle(
    session: str,
    cfg: AnalysisConfig,
    base_dirs: List[Path],
) -> dict:
    """
    Load all intermediate data for a session and return an alignment bundle.

    Returns dict with keys:
        session, Ev, align_times, trial_features, session_offset
    """
    sdir = find_session_dir(session, roots=base_dirs)
    data = load_intermediate_data(sdir)
    movs, trials, licks, kins, evnts = (
        data["movs"], data["trials"], data["licks"], data["kins"], data["events"]
    )

    session_offset = evnts[evnts["event"] == "goCue_start_time"]["raw_timestamps"].iloc[0]

    events_dict = get_events_dict(trials, licks, kins)
    E = build_event_df(events_dict)
    if cfg.align_key not in E.columns:
        raise KeyError(f"align_key '{cfg.align_key}' not in events: {list(E.columns)}")

    Ev = E.dropna(subset=[cfg.align_key])
    align_times = Ev[cfg.align_key].astype(float)
    trial_features = build_trial_features(movs, licks, trials)

    if cfg.only_cue_response_trials:
        cr_trials = trial_features.index[trial_features["cue_response_movement_number"].notna()]
        keep = align_times.index.intersection(cr_trials)
        align_times = align_times.loc[keep]
        Ev = Ev.loc[keep]
        trial_features = trial_features.loc[keep]

    return {
        "session": session,
        "Ev": Ev,
        "align_times": align_times,
        "trial_features": trial_features,
        "session_offset": session_offset,
    }


# ── All-counts table ──────────────────────────────────────────────────────────

def build_all_counts_df(
    units_with_spikes: pd.DataFrame,
    cfg: "AnalysisConfig",
    base_dirs: List[Path],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the trial × unit spike counts + kinematics table.

    This is the primary intermediate used by encoding analyses.
    Loops over every session × unit in units_with_spikes, builds
    a session bundle (alignment times + trial features), then extracts
    spike counts for each trial.

    Parameters
    ----------
    units_with_spikes : DataFrame from data_loading.load_units_with_spike_times
    cfg : AnalysisConfig — controls spike-count window, alignment key, etc.
    base_dirs : list of Path — roots to search for session intermediate data
    verbose : print per-session progress

    Returns
    -------
    DataFrame sorted by (session, unit_id, trial) with columns:
        session, unit_id, trial, spike_count, spike_rate_hz,
        first_spike_latency_s, baseline_spike_count, baseline_spike_rate_hz,
        delta_spike_count, reaction_time_firstmove, reaction_time_cueresponse,
        first_move_* and cue_response_* kinematics columns.
    """
    bundle_cache: dict = {}
    all_counts: list = []
    n_ok = n_skip = 0

    for u in units_with_spikes.itertuples(index=False):
        session = u.session
        if session not in bundle_cache:
            try:
                bundle_cache[session] = make_session_bundle(session, cfg, base_dirs=base_dirs)
            except Exception as e:
                if verbose:
                    print(f"[skip session] {session}: {e}")
                n_skip += 1
                continue

        try:
            unit_counts = analyze_unit_for_session(
                pd.Series(u._asdict()), bundle_cache[session], cfg
            )
            all_counts.append(unit_counts)
            n_ok += 1
        except Exception as e:
            if verbose:
                print(f"[skip unit] {getattr(u, 'unit_id', '?')} in {session}: {e}")
            n_skip += 1

    if verbose:
        print(f"build_all_counts_df: {n_ok} units OK, {n_skip} skipped")

    if not all_counts:
        return pd.DataFrame()

    return (
        pd.concat(all_counts, ignore_index=True)
        .sort_values(["session", "unit_id", "trial"])
        .reset_index(drop=True)
    )


# ── Per-unit analysis ─────────────────────────────────────────────────────────

def analyze_unit_for_session(
    unit_row: pd.Series,
    bundle: dict,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """
    Count spikes and extract first-spike latency for every trial in bundle.
    Returns a trial-level DataFrame with spike counts joined to trial_features.
    """
    session = bundle["session"]
    unit_id = unit_row["unit_id"]
    spikes = np.asarray(unit_row["spike_times"], dtype=float) - bundle["session_offset"]

    t0_map = bundle["align_times"].to_dict()
    win_count = cfg.count_window_s
    win_base = cfg.baseline_window_s
    win_lat = cfg.latency_window_s if cfg.latency_window_s is not None else cfg.count_window_s
    if not (np.isfinite(win_lat[0]) and np.isfinite(win_lat[1]) and win_lat[0] < win_lat[1]):
        win_lat = win_count

    dur = win_count[1] - win_count[0]
    dur_b = win_base[1] - win_base[0] if win_base is not None else np.nan

    recs = []
    for tr in bundle["align_times"].index:
        t0 = t0_map.get(tr, np.nan)
        if not np.isfinite(t0):
            continue
        n = count_spikes_in_window(spikes, t0, win_count)
        n_base = count_spikes_in_window(spikes, t0, win_base) if win_base is not None else np.nan
        fsl, had = first_spike_latency_in_window(spikes, t0, win_lat)
        recs.append({
            "unit_id": unit_id,
            "session": session,
            "trial": int(tr),
            "align_key": cfg.align_key,
            "win_start_s": win_count[0],
            "win_stop_s": win_count[1],
            "spike_count": int(n),
            "spike_rate_hz": (n / dur) if dur > 0 else np.nan,
            "baseline_spike_count": n_base,
            "baseline_spike_rate_hz": (n_base / dur_b) if np.isfinite(dur_b) else np.nan,
            "delta_spike_count": (n - n_base) if np.isfinite(n_base) else np.nan,
            "lat_win_start_s": win_lat[0],
            "lat_win_stop_s": win_lat[1],
            "first_spike_latency_s": fsl,
            "had_spike_in_latency_win": had,
        })

    unit_counts = pd.DataFrame(recs).set_index("trial")
    unit_counts = unit_counts.join(bundle["trial_features"], how="left")
    return unit_counts.reset_index()
