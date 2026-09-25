"""
fip_utils.py — shared setup for the ``fip_*`` notebook series.

One home for the load → curate → session-setup preamble that
``fip_00_explore.ipynb``, ``fip_01_movement_value_coding.ipynb`` and
``fip_02_ne_only_events.ipynb`` previously each carried their own copy of
(~490 duplicated lines across the three).

Scope
-----
This module owns *plumbing*: locating and loading the saved parquet hierarchy,
applying curation, picking a session and its example channels, putting motion
energy on the FIP clock, and the small signal helpers (z-score, onset
detection, peri-event alignment, cross-correlation).

It deliberately does **not** own the choice of FIP normalization. The three
``aind_dynamic_foraging_data_utils.enrich_dfs`` entry points are three
*different* normalizations, not three steps of one pipeline:

* ``zscore_fip``              -> whole-session ``data_z``
* ``enrich_fip_in_df_trials`` -> re-cuts signals into per-trial windows
                                 (z-scores internally; a prior ``zscore_fip``
                                 would double-process)
* ``remove_tonic_df_fip``     -> per-trial ``data_z_*_baseline`` / ``_norm``

Which one a notebook runs defines what "elevated" means for
:func:`threshold_onsets` and what its AUC figures measure, so that call stays
visible in each notebook. What this module provides is the bridge into that
pipeline — :func:`attach_me_to_df_fip`, which takes **raw** motion energy
precisely because those functions z-score ``data`` themselves.

Usage (notebooks run with ``code/`` as cwd on Code Ocean)::

    %load_ext autoreload
    %autoreload 2
    import fip_utils as fu

    nwb_list = fu.load_curated_sessions()
    nwb, df_fip, df_trials = fu.select_session(nwb_list, 0)
    meta = fu.build_meta(df_fip)
    examples = fu.pick_examples(meta, df_fip)

``autoreload`` matters here: a session load is tens of GB and several minutes,
so without it every edit to this file costs a kernel restart and a full reload.

Data lives on Code Ocean; nothing in this module runs against local data.
Python 3.9-compatible syntax only (see CLAUDE.md).
"""

from __future__ import annotations

import gc
import glob
import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import fastparquet  # noqa: F401  # load_nwb_list uses pd.read_parquet(engine="fastparquet")

# Rachel's lab utilities (installed / mounted in the Code Ocean capsule).
from rachel_analysis_utils import nwb_utils as r_utils

# AIND event-alignment primitive (peri-event windowing). df_fip['timestamps'] and the
# df_trials '*_in_session' columns share one clock: t=0 is the first go cue of the session.
from aind_dynamic_foraging_data_utils import alignment


# ── Constants ─────────────────────────────────────────────────────────────────

#: Root of the attached asset's saved parquet hierarchy, as load_nwb_list wants it:
#: ``plot_loc/<subject>/<session>/df_fip.parquet``. Asset 6babbf3d-6970-4456-aab5-d730ed57c269.
DEFAULT_PLOT_LOC = "/root/capsule/data/DA_NE_4channels/"

#: Curation JSON shipped inside the rachel_analysis_utils package; carries ``correct_mapping``
#: and the per-session ``misconnect_fixes`` that annotate ``df_fip['intended_measurement']``.
DEFAULT_CURATION_FILE = "DA_NE_4channel_datacuration_firstpass"

#: Where Code Ocean mounts attached data assets.
DEFAULT_DATA_ROOT = "/root/capsule/data"

#: Go-cue times on the df_fip clock (first-go-cue-zeroed) drive every peri-event alignment.
ALIGN_COL = "goCue_start_time_in_session"

#: (label, region substring, plot colour) for each example signal type.
EXAMPLE_SPECS = [
    ("NAc DA (dLight)", "dLight", "#2ca02c"),
    ("PL (GCaMP)", "Gcamp", "#1f77b4"),
    ("NAc ACh (rAch)", "rAch", "#d62728"),
]

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


# ── Loading + curation ────────────────────────────────────────────────────────

def discover_plot_locs(data_root: str = DEFAULT_DATA_ROOT) -> List[str]:
    """Rediscover candidate ``plot_loc`` roots by globbing for df_fip.parquet.

    Fallback for when the asset layout changes and :data:`DEFAULT_PLOT_LOC` no longer
    resolves; normally unnecessary.

    Parameters
    ----------
    data_root : str
        Directory to search recursively.

    Returns
    -------
    list of str
        Sorted, de-duplicated ``plot_loc`` roots.
    """
    fip_files = glob.glob(os.path.join(data_root, "**", "df_fip.parquet"), recursive=True)
    locs = sorted({os.path.dirname(os.path.dirname(os.path.dirname(f))) for f in fip_files})
    print("%d session(s) found across %d root(s): %s" % (len(fip_files), len(locs), locs))
    return locs


def find_curation_json(curation_file: str = DEFAULT_CURATION_FILE) -> str:
    """Locate a curation JSON inside the installed rachel package, with a /src fallback.

    Parameters
    ----------
    curation_file : str
        Basename without the ``.json`` extension.

    Returns
    -------
    str
        Absolute path to the curation JSON.
    """
    try:
        import rachel_analysis_utils

        pkg_dir = os.path.dirname(rachel_analysis_utils.__file__)
        cand = os.path.join(pkg_dir, "data_curation", curation_file + ".json")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    hits = glob.glob("/src/**/data_curation/" + curation_file + ".json", recursive=True)
    assert hits, "Could not locate " + curation_file + ".json"
    return hits[0]


def patch_curation_helpers() -> None:
    """Inject the private helpers ``apply_curation_nwb_list`` calls but never imports.

    Upstream bug: ``data_curation_helpers.apply_curation_nwb_list`` calls
    ``_parse_session_id``, ``_actual_map_for_session``, ``_get_df_trials_col_mapping``,
    ``_map_event_to_intended_measurement`` and ``_apply_channel_drops_to_nwb`` without
    importing them. They live in ``nwb_utils`` and work fine there.

    Skipping curation is *not* a safe shortcut instead: several subjects (e.g. 808054) have
    per-session ``misconnect_fixes`` overriding the global ``correct_mapping``, so applying
    ``correct_mapping`` alone would silently mislabel most of their sessions' regions.

    Gated on ``hasattr`` so it becomes a no-op once the helpers are imported upstream.
    """
    from rachel_analysis_utils import data_curation_helpers as r_curation

    for helper in (
        "_parse_session_id",
        "_actual_map_for_session",
        "_get_df_trials_col_mapping",
        "_map_event_to_intended_measurement",
        "_apply_channel_drops_to_nwb",
    ):
        if not hasattr(r_curation, helper):
            setattr(r_curation, helper, getattr(r_utils, helper))


def load_curated_sessions(
    plot_loc: str = DEFAULT_PLOT_LOC,
    curation_file: str = DEFAULT_CURATION_FILE,
    use_curation: bool = True,
    return_tables: bool = False,
    verbose: bool = True,
):
    """Load the saved parquet hierarchy and apply curation, releasing the raw copy.

    Memory note: the pre-curation ``nwb_list_raw`` and the unused second curated list
    are locals here, so both are released when this returns. Curation deep-copies, so
    holding the raw list alongside the curated one roughly doubles peak RSS — which
    matters at this dataset's ~32 GB per copy.

    ``drop_borderline`` is pinned to False: it only controls whether a *second* list
    (``curated_with_borderline``) gets built. ``nwb_list`` itself is identical either way,
    but with True ``apply_curation_nwb_list`` deep-copies and processes every session again
    to build a list nothing here uses.

    Parameters
    ----------
    plot_loc : str
        Root of the saved parquet hierarchy.
    curation_file : str
        Curation JSON basename, without extension.
    use_curation : bool
        When False, return the uncurated list (no ``intended_measurement`` region labels,
        so :func:`pick_example` will not work).
    return_tables : bool
        Also return the ``df_sess`` / ``df_slope`` / ``df_bg`` side tables, which
        ``load_nwb_list`` returns but the ``fip_*`` notebooks do not use.
    verbose : bool
        Print progress and per-session summaries.

    Returns
    -------
    list
        ``nwb_list``, or ``(nwb_list, df_sess, df_slope, df_bg)`` when ``return_tables``.
    """
    assert os.path.isdir(plot_loc), (
        "%s not found. Attach asset 6babbf3d-6970-4456-aab5-d730ed57c269 "
        "(saved parquet hierarchy) to this capsule, or fix the path." % plot_loc
    )
    if verbose:
        print("plot_loc =", plot_loc)

    # load_nwb_list returns df_bg (background-coefficient csvs, bg_coef*.csv) as a 4th
    # value; unused here (same as df_sess/df_slope) but must be unpacked.
    nwb_list_raw, df_sess, df_slope, df_bg = r_utils.load_nwb_list(plot_loc, load_fip=True)
    if verbose:
        print("Loaded %d session(s)" % len(nwb_list_raw))

    if not use_curation:
        if verbose:
            print("No curation applied; using raw nwb_list.")
        nwb_list = nwb_list_raw
    else:
        from rachel_analysis_utils import data_curation_helpers as r_curation

        patch_curation_helpers()

        json_path = find_curation_json(curation_file)
        with open(json_path, "r") as fh:
            df_curation = json.load(fh)
        if verbose:
            print("Using curation:", json_path)

        nwb_list, _unused_with_borderline = r_curation.apply_curation_nwb_list(
            nwb_list_raw, df_curation, drop_borderline=False
        )
        if verbose:
            print("Curated -> %d session(s) kept." % len(nwb_list))

        del nwb_list_raw, _unused_with_borderline
        gc.collect()

    if return_tables:
        return nwb_list, df_sess, df_slope, df_bg
    return nwb_list


# ── Session setup ─────────────────────────────────────────────────────────────

def select_session(
    nwb_list: Sequence,
    idx: int = 0,
    require_trial_zero_gocue: bool = False,
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """Pick one session and validate the assumptions downstream code relies on.

    Asserts (rather than sorts) that each series is already time-ordered, so
    :func:`get_trace` slices and ``np.interp`` stay valid — this fails loudly if upstream
    ever delivers unsorted rows.

    Parameters
    ----------
    nwb_list : sequence
        Curated sessions from :func:`load_curated_sessions`.
    idx : int
        Index into ``nwb_list``.
    require_trial_zero_gocue : bool
        Additionally assert ``goCue_start_time_in_trial == 0`` for every trial. Needed only
        by notebooks that run ``enrich_fip_in_df_trials``, whose per-trial window math
        assumes the go cue is trial-local time zero.

    Returns
    -------
    tuple
        ``(nwb, df_fip, df_trials)``.
    """
    nwb = nwb_list[idx]
    df_fip = nwb.df_fip
    assert (df_fip.groupby("event")["timestamps"].diff().dropna() >= 0).all(), (
        "df_fip timestamps not sorted within an event — sort before use"
    )
    df_trials = getattr(nwb, "df_trials", None)

    assert df_trials is not None and ALIGN_COL in df_trials.columns, (
        "Need '%s' on df_trials (first-go-cue-zeroed clock). Available: %s"
        % (ALIGN_COL, None if df_trials is None else list(df_trials.columns))
    )

    if require_trial_zero_gocue:
        assert "goCue_start_time_in_trial" in df_trials.columns, (
            "Need 'goCue_start_time_in_trial'"
        )
        assert np.allclose(df_trials["goCue_start_time_in_trial"].dropna(), 0.0), (
            "goCue_start_time_in_trial is not all zero -- enrich_fip_in_df_trials's per-trial "
            "window math assumes the go cue is trial-local time zero. Re-check before proceeding."
        )

    print(
        "session_id:", nwb.session_id, "| df_fip", df_fip.shape, "| trials", df_trials.shape
    )
    return nwb, df_fip, df_trials


def parse_event(name: str) -> Tuple[str, str, str]:
    """Split a df_fip event name into (channel, fiber, variant).

    ``'G_1_dff-poly' -> ('G', '1', 'dff-poly')``; ``'R_0' -> ('R', '0', 'raw')``.
    """
    parts = name.split("_")
    channel = parts[0]
    fiber = parts[1] if len(parts) > 1 else "?"
    variant = "_".join(parts[2:]) if len(parts) > 2 else "raw"
    return channel, fiber, variant


def get_trace(
    df_fip: pd.DataFrame, event_name: str, data_col: str = "data"
) -> Tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` arrays for one df_fip series (asserted time-sorted per event)."""
    sub = df_fip[df_fip["event"] == event_name]
    return sub["timestamps"].to_numpy(), sub[data_col].to_numpy()


def build_meta(df_fip: pd.DataFrame) -> pd.DataFrame:
    """Signal inventory for one session's df_fip.

    Drops ``pearsonR`` series (signal-signal correlations, not photometry) and reads each
    event's region label from whichever curation format the asset carries:

    * JSON curation applied at load time (:func:`load_curated_sessions`, ``DA_NE_4channels``):
      events keep their patch-cord names (``G_1_dff-...``) and the region is in
      ``intended_measurement``.
    * CSV curation applied when the asset was built (rachel-analysis-utils ``b7b7487`` onward,
      e.g. ``DANE_3channels_curated``): ``apply_curation_df_fip`` has already replaced each
      event with its target (``latNAcc(L)-DA``), moved the patch cord to ``patch_cord`` and the
      variant to ``preprocessing``, and there is no ``intended_measurement``. Load these with
      ``use_curation=False``.

    Returns
    -------
    pandas.DataFrame
        Columns ``event, channel, fiber, variant`` and, if available, ``region``.
    """
    events = [e for e in sorted(df_fip["event"].unique()) if "pearson" not in e.lower()]
    if "intended_measurement" not in df_fip.columns and "patch_cord" in df_fip.columns:
        # Curated at build time: the event name is the region label.
        first = df_fip.groupby("event")[["patch_cord", "preprocessing"]].first()
        rows = []
        for e in events:
            channel, fiber, _ = parse_event(str(first.loc[e, "patch_cord"]))
            rows.append((e, channel, fiber, str(first.loc[e, "preprocessing"]), e))
        return pd.DataFrame(rows, columns=["event", "channel", "fiber", "variant", "region"])

    meta = pd.DataFrame(
        [(e,) + parse_event(e) for e in events],
        columns=["event", "channel", "fiber", "variant"],
    )
    if "intended_measurement" in df_fip.columns:  # curation -> region labels
        ev2region = (
            df_fip.dropna(subset=["intended_measurement"])
            .groupby("event")["intended_measurement"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        )
        meta["region"] = meta["event"].map(ev2region)
    return meta


def pick_example(
    meta: pd.DataFrame,
    df_fip: pd.DataFrame,
    region_substr: str,
    prefer_variant: str = "dff",
) -> Dict[str, str]:
    """Choose one df_fip 'event' whose region label matches ``region_substr``.

    Prefers a dff variant, then the series with the most finite samples.

    ``regex=False`` on both matches is required, not cosmetic: region labels contain
    literal parentheses (``'NAc(R)-dLight'``), and under pandas' default ``regex=True``
    a substring like ``'NAc(R)'`` is read as a capture group and matches nothing.

    Parameters
    ----------
    meta : pandas.DataFrame
        From :func:`build_meta`; must carry a ``region`` column.
    df_fip : pandas.DataFrame
        Used to count finite samples per candidate.
    region_substr : str
        Case-insensitive literal substring of the region label, e.g. ``'dLight'``,
        ``'Gcamp'``, ``'NAc(R)'``.
    prefer_variant : str
        Preferred preprocessing variant substring.

    Returns
    -------
    dict
        ``{'event', 'region', 'channel', 'variant'}``.
    """
    if "region" not in meta.columns or meta["region"].dropna().empty:
        raise RuntimeError(
            "No region labels on `meta` -- load with use_curation=True so df_fip carries "
            "intended_measurement."
        )
    cand = meta[meta["region"].fillna("").str.contains(region_substr, case=False, regex=False)]
    if cand.empty:
        raise ValueError(
            "No FIP series with region matching %r. Have: %s"
            % (region_substr, sorted(meta["region"].dropna().unique()))
        )
    pref = cand[cand["variant"].str.contains(prefer_variant, case=False, regex=False)]
    pool = pref if not pref.empty else cand
    best = max(pool["event"], key=lambda ev: int(np.isfinite(get_trace(df_fip, ev)[1]).sum()))
    row = pool[pool["event"] == best].iloc[0]
    return {
        "event": best,
        "region": row["region"],
        "channel": row["channel"],
        "variant": row["variant"],
    }


def pick_examples(
    meta: pd.DataFrame, df_fip: pd.DataFrame, specs: Optional[Sequence] = None
) -> List[Dict[str, str]]:
    """Run :func:`pick_example` over ``specs``, attaching each spec's label and colour.

    Parameters
    ----------
    meta, df_fip : pandas.DataFrame
        As for :func:`pick_example`.
    specs : sequence of (label, region_substr, colour), optional
        Defaults to :data:`EXAMPLE_SPECS`.

    Returns
    -------
    list of dict
        One :func:`pick_example` result per spec, plus ``label`` and ``color``.
    """
    if specs is None:
        specs = EXAMPLE_SPECS
    examples = []
    for label, substr, color in specs:
        info = pick_example(meta, df_fip, substr)
        info["label"], info["color"] = label, color
        examples.append(info)
    return examples


def iter_region_signals(sessions: Sequence[dict], region_substr: str) -> Iterator[Tuple[dict, str]]:
    """Yield ``(session_dict, event)`` for every FIP series whose region matches, across sessions.

    ``regex=False`` for the same reason as :func:`pick_example`.
    """
    for s in sessions:
        meta = s["meta"]
        if "region" not in meta.columns:
            continue
        hit = meta[meta["region"].fillna("").str.contains(region_substr, case=False, regex=False)]
        for ev in hit["event"]:
            yield s, ev


def _enrich_trials_fallback(df_trials: pd.DataFrame) -> pd.DataFrame:
    """3.9-safe reimplementation of the two ``enrich_df_trials`` columns the notebooks need.

    ``num_reward_past`` (run length of consecutive rewarded trials, negated for consecutive
    unrewarded ones) and ``RPE-binned3``, matching upstream's exact logic. Used only when
    ``rachel_analysis_utils.analysis_utils`` cannot be imported.

    Unlike upstream this degrades rather than raises: sessions without the reward columns
    come back unchanged, and ``RPE-binned3`` is skipped when ``RPE_earned`` is absent.
    """
    df = df_trials.copy()
    if "earned_reward" not in df.columns:
        return df  # streak bins stay off for this session

    extra = df["extra_reward"] if "extra_reward" in df.columns else 0
    reward_all = df["earned_reward"].astype(float) + (
        extra if np.isscalar(extra) else extra.astype(float)
    )
    df["reward_all"] = reward_all

    # Group by ses_idx when present so streaks never run across a session boundary.
    if "ses_idx" in df.columns:
        prev = reward_all.groupby(df["ses_idx"]).shift(1)
    else:
        prev = reward_all.shift(1)
    df["rewarded_prev"] = prev

    run_id = (prev != reward_all).cumsum()             # new id each time reward state flips
    num = df.groupby(run_id).cumcount() + 1            # 1-indexed position within the run
    df["num_reward_past"] = num.where(reward_all != 0, -num)  # negate runs of no-reward

    if "RPE_earned" in df.columns:
        labels = [str(np.round(i, 2)) for i in np.arange(-1, 0.99, 1 / 3)]
        bins = np.arange(-1, 1.01, 1 / 3)
        bins[-1] = 1.001
        df["RPE-binned3"] = pd.cut(df["RPE_earned"], bins=bins, right=True, labels=labels)
    return df


def enrich_trials(df_trials: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add ``num_reward_past`` / ``RPE-binned3`` etc., preferring Rachel's own function.

    Calls ``rachel_analysis_utils.analysis_utils.enrich_df_trials`` when it imports, so
    the notebooks stay on the upstream definition, and falls back to
    :func:`_enrich_trials_fallback` otherwise.

    Note that upstream is all-or-nothing and needs more columns than the fallback
    (``RPE_earned``, ``Q_chosen``, ``animal_response``, ``choice_time_in_trial``). Callers
    looping over many sessions should wrap this in try/except so one incomplete session
    does not fail the whole loop — see :func:`process_session`.

    Parameters
    ----------
    df_trials : pandas.DataFrame
        Trial table for one session (or several, keyed by ``ses_idx``).
    verbose : bool
        Print which path was taken.

    Returns
    -------
    pandas.DataFrame
        Enriched trial table.
    """
    try:
        from rachel_analysis_utils import analysis_utils as r_analysis

        out = r_analysis.enrich_df_trials(df_trials)
        if verbose:
            print("Used rachel_analysis_utils.analysis_utils.enrich_df_trials directly.")
        return out
    except Exception as e:
        if verbose:
            print(
                "enrich_df_trials unavailable (%s: %s); using local fallback for "
                "num_reward_past + RPE-binned3." % (type(e).__name__, e)
            )
        return _enrich_trials_fallback(df_trials)


def streak_go_cues(df_trials: pd.DataFrame, bin_fn) -> Optional[np.ndarray]:
    """Go-cue times (session clock) for trials whose ``num_reward_past`` satisfies ``bin_fn``."""
    if "num_reward_past" not in df_trials.columns:
        return None
    sub = df_trials.dropna(subset=[ALIGN_COL, "num_reward_past"])
    mask = sub["num_reward_past"].map(bin_fn).astype(bool)
    return sub.loc[mask, ALIGN_COL].to_numpy()


# ── Motion energy ─────────────────────────────────────────────────────────────

_VA = None


def _get_va():
    """Import ``video_alignment`` lazily and cache it.

    Deferred so that merely importing ``fip_utils`` cannot trigger the pip-install
    fallback as a side effect. ``video_alignment`` is on the package's ``main`` branch and
    the Dockerfile installs @main, so the plain import normally succeeds; the fallback only
    covers a from-scratch env that has not run postInstall yet.
    """
    global _VA
    if _VA is None:
        import importlib
        import subprocess
        import sys

        try:
            from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va
        except ModuleNotFoundError:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/AllenNeuralDynamics/"
                "aind-dynamic-foraging-behavior-video-analysis.git@main",
            ])
            importlib.invalidate_caches()
            from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va
        _VA = va
    return _VA


def locate_me_assets(
    session_id: str, data_root: str = DEFAULT_DATA_ROOT
) -> Tuple[str, str]:
    """``(video_csv, me_path)`` for a session.

    Both assets are matched by ``subject_date``; the behavior folders carry an
    acquisition-time suffix (e.g. ``behavior_808054_2025-09-02_10-38-37``), so we glob.

    Raises
    ------
    FileNotFoundError
        If either the behavior-video or the motion-energy asset is missing, so a
        multi-session loop can skip the session.
    """
    subject, date = session_id.split("_")[:2]

    beh = [
        d
        for d in sorted(glob.glob(os.path.join(data_root, "behavior_%s_%s_*" % (subject, date))))
        if "motionenergy" not in d
    ]
    if not beh:
        raise FileNotFoundError("no behavior-video asset for %s_%s" % (subject, date))
    video_csv = os.path.join(beh[0], "behavior-videos", "bottom_camera.csv")
    if not os.path.exists(video_csv):
        raise FileNotFoundError("missing %s" % video_csv)

    me_dirs = sorted(
        glob.glob(os.path.join(data_root, "behavior_%s_%s_*motionenergy*" % (subject, date)))
    )
    if not me_dirs:
        raise FileNotFoundError("no motion-energy asset for %s_%s" % (subject, date))
    hits = glob.glob(
        os.path.join(me_dirs[0], "**", "bottom_camera_motion_energy_clean.npy"), recursive=True
    )
    if not hits:
        raise FileNotFoundError("me .npy not found under %s" % me_dirs[0])
    return video_csv, hits[0]


def motion_energy_to_session(
    me_path: str,
    video_csv: str,
    df_trials: pd.DataFrame,
    go_cue_col: str = "goCue_start_time_raw",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-frame motion energy on the session clock (t=0 at the first go cue).

    After padding (below) ``me[i]`` is 1-to-1 with row ``i`` of the camera CSV, so
    ``session_time = behavior_time(frame) - first_go_cue_raw``, which is exactly how
    ``df_fip['timestamps']`` is built. Routed through the ``video_alignment`` helpers so
    the offset semantics and the CSV-layout detection stay in one place, shared with
    aind-dynamic-foraging-behavior-video-analysis.

    Parameters
    ----------
    me_path : str
        ``bottom_camera_motion_energy_clean.npy``.
    video_csv : str
        The camera CSV for the same session.
    df_trials : pandas.DataFrame
        Needs ``trial`` and ``go_cue_col`` (absolute behaviour clock).
    go_cue_col : str
        Column holding the raw (unzeroed) go-cue times.

    Returns
    -------
    tuple
        ``(t_session, me, offset)``.
    """
    va = _get_va()

    me = np.load(me_path)
    # Old/flat camera CSVs (e.g. bottom_camera.csv) ship headerless in DEFAULT_COLUMNS order;
    # read_video_csv applies that only when needed and auto-detects the AIND header layout
    # otherwise, so we don't have to know which layout THIS session's CSV uses.
    cam = va.read_video_csv(video_csv, columns=list(va.DEFAULT_COLUMNS))
    time_col = next(c for c in va.TIME_COLUMN_ALIASES if c in cam.columns)

    # aind-motion-energy emits a consecutive-frame difference: one fewer value than frames
    # (no value for frame 0). Pad a leading 0 so me[i] is the motion AT frame i, 1-to-1 with
    # camera rows. Decided from the ME metadata (n_me_frames vs n_frames_decoded) so this
    # auto-disables once the library pads upstream; fall back to the row count if absent.
    meta_path = me_path.replace("_motion_energy_clean.npy", "_me_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            me_meta = json.load(fh)
        pad = me_meta["n_me_frames"] == me_meta["n_frames_decoded"] - 1
    else:
        pad = (len(cam) - len(me)) == 1
    if pad:
        me = np.insert(me, 0, 0.0)

    if len(me) != len(cam):  # should match now; warn only on a genuine mismatch
        n = min(len(me), len(cam))
        print("WARNING: %d ME vs %d CSV rows after pad; truncating to %d" % (len(me), len(cam), n))
        me, cam = me[:n], cam.iloc[:n]

    first_go_cue = float(df_trials.sort_values("trial")[go_cue_col].iloc[0])  # absolute clock
    offset = va.compute_video_session_offset(video_csv, first_go_cue)
    first_frame = va.get_first_frame_behavior_time(video_csv)
    video_t = va.behavior_time_to_video_time(cam[time_col].to_numpy(float), first_frame)
    t_session = va.video_time_to_session_time(video_t, offset)
    return t_session, np.asarray(me, float), offset


def attach_me_to_df_fip(
    df_fip: pd.DataFrame,
    t_me: np.ndarray,
    me: np.ndarray,
    ses_idx: str,
    event_name: str = "ME",
) -> pd.DataFrame:
    """Append **raw** motion energy to df_fip as a pseudo-channel (``event='ME'``).

    This is the bridge into the upstream FIP machinery: ``plot_fip``'s PSTH functions and
    the ``enrich_dfs`` normalizations all select a signal via ``df_fip['event'] == channel``,
    so once ME is a channel they treat it exactly like photometry.

    Pass **raw** ME, not z-scored: ``zscore_fip`` / ``enrich_fip_in_df_trials`` /
    ``remove_tonic_df_fip`` all z-score ``data`` themselves, so pre-z-scored input would be
    silently double-processed. To plot z-scored ME afterwards, run ``zscore_fip`` and ask
    for ``data_column="data_z"``.

    Parameters
    ----------
    df_fip : pandas.DataFrame
        Tidy FIP table to append to.
    t_me, me : numpy.ndarray
        Motion energy on the session clock, from :func:`motion_energy_to_session`.
    ses_idx : str
        Session id, matching df_fip's ``ses_idx``.
    event_name : str
        Channel name for the pseudo-channel.

    Returns
    -------
    pandas.DataFrame
        ``df_fip`` with the ME rows appended.
    """
    me_rows = pd.DataFrame({
        "timestamps": np.asarray(t_me, float),
        "data": np.asarray(me, float),
        "event": event_name,
        "intended_measurement": "motion_energy",
        "ses_idx": ses_idx,
    })
    return pd.concat([df_fip, me_rows], ignore_index=True)


# ── Signal helpers ────────────────────────────────────────────────────────────

def zscore(y) -> np.ndarray:
    """Z-score a 1D array, ignoring NaNs.

    Uses ``ddof=1`` to match ``aind_dynamic_foraging_data_utils.enrich_dfs.zscore_fip``
    (``scipy.stats.zscore(x, ddof=1, nan_policy='omit')``), so a trace z-scored here and
    the same trace's ``data_z`` column agree. Prefer ``data_z`` when it is available; this
    is for array-only paths where df_fip has not been enriched.

    Returns the mean-centred array unchanged when the standard deviation is zero or
    non-finite, rather than dividing by it.
    """
    y = np.asarray(y, float)
    sd = np.nanstd(y, ddof=1)
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
) -> np.ndarray:
    """Causal upward threshold crossings of a z-scored trace (no smoothing).

    An onset is the first sample that rises above ``z_thresh`` SD and then stays above for
    at least ``min_run`` **samples** (not seconds). That rejects single-sample spikes
    without smoothing or shifting timing, so the onset time is the true first crossing.
    ``refractory`` (seconds) collapses repeats so one bout is counted once.

    Parameters
    ----------
    t : array_like
        Timestamps, same length as ``y``.
    y : array_like
        Signal. Already z-scored when ``already_z``, else z-scored internally by
        :func:`zscore`.
    z_thresh : float
        Threshold in SD.
    refractory : float
        Minimum spacing between reported onsets, in seconds.
    min_run : int
        Number of consecutive samples that must stay above threshold.
    already_z : bool
        Set True when passing an upstream ``data_z`` column, so it is not z-scored twice.

    Returns
    -------
    numpy.ndarray
        Onset times, in ``t``'s units.
    """
    z = np.asarray(y, float) if already_z else zscore(y)
    above = z > z_thresh  # NaN compares False -> treated as below threshold
    idx = np.where((~above[:-1]) & above[1:])[0] + 1
    if min_run > 1 and len(idx):  # require the crossing to persist (causal spike rejection)
        idx = np.array(
            [i for i in idx if i + min_run <= len(above) and above[i:i + min_run].all()],
            dtype=int,
        )
    times = np.asarray(t, float)[idx]
    if len(times):  # enforce refractory period
        times = times[np.insert(np.diff(times) > refractory, 0, True)]
    return times


def peri_event(
    t,
    y,
    event_times,
    t_before: float = 1.0,
    t_after: float = 3.0,
    fs: int = 20,
    censor: bool = True,
    censor_times=None,
) -> pd.DataFrame:
    """Align signal ``(t, y)`` to ``event_times`` via the AIND ETR primitive.

    Returns a tidy DataFrame ``[time, event_number, event_time, data]``, ready to
    ``groupby('time')``.

    Parameters
    ----------
    t, y : array_like
        Signal and its timestamps. NaNs are dropped before alignment.
    event_times : array_like
        Alignment times, on the same clock as ``t``.
    t_before, t_after : float
        Window around each event, in seconds.
    fs : int
        Output sampling rate for the common time grid.
    censor : bool
        Blank samples spilling past the next event. Use True for go cues; pass False for
        closely-spaced events such as movement onsets, where censoring would erase most
        of the window.
    censor_times : array_like, optional
        The full set of neighbouring events to censor against when ``event_times`` is a
        subset (e.g. go cues for one reward-streak bin). Without it the primitive censors
        only against the sparse subset and lets intervening trials bleed in.

    Returns
    -------
    pandas.DataFrame
        Tidy event-triggered response.
    """
    s = pd.DataFrame({
        "timestamps": np.asarray(t, float),
        "data": np.asarray(y, float),
    }).dropna()
    return alignment.event_triggered_response(
        data=s,
        t="timestamps",
        y="data",
        event_times=np.asarray(event_times, float),
        t_before=t_before,
        t_after=t_after,
        output_sampling_rate=fs,
        output_format="tidy",
        censor=censor,
        censor_times=censor_times,
    )


def norm_xcorr(a, b, fs, maxlag: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two equal-length, uniformly-sampled signals.

    ``out[lag] = mean(a[t+lag] * b[t])`` with ``a``, ``b`` z-scored. A peak at POSITIVE lag
    means **b leads a** (b precedes a by ``lag`` seconds).

    Parameters
    ----------
    a, b : array_like
        Equal-length signals on a common uniform grid.
    fs : float
        Sampling rate of that grid, in Hz.
    maxlag : float
        Maximum lag to evaluate, in seconds.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lags_in_seconds, correlation)``.
    """
    a, b = zscore(a), zscore(b)
    n = len(a)
    K = int(round(maxlag * fs))
    lags = np.arange(-K, K + 1)
    out = np.empty(len(lags), float)
    for i, L in enumerate(lags):
        if L >= 0:
            out[i] = np.nanmean(a[L:] * b[:n - L])
        else:
            out[i] = np.nanmean(a[:n + L] * b[-L:])
    return lags / float(fs), out


def window_mean(series: pd.Series, lo: float, hi: float) -> float:
    """Mean of a time-indexed Series over ``[lo, hi)`` seconds (NaN-safe).

    Operates on a per-session ETR *mean trace* (time-indexed), which is why this is not
    ``aind_dynamic_foraging_basic_analysis.metrics.trial_metrics.get_average_signal_window``
    — that one adds a per-trial column to ``df_trials`` instead.
    """
    idx = series.index.to_numpy(float)
    m = (idx >= lo) & (idx < hi)
    return float(np.nanmean(series.values[m])) if m.any() else np.nan


# ── Multi-session ─────────────────────────────────────────────────────────────

def process_session(nwb, data_root: str = DEFAULT_DATA_ROOT, onset_kw: Optional[dict] = None) -> dict:
    """Run the single-session pipeline and return a dict for cross-session analyses.

    Mirrors the single-session path exactly, reusing :func:`build_meta`,
    :func:`locate_me_assets` and :func:`motion_energy_to_session`.

    Parameters
    ----------
    nwb : object
        One curated session.
    data_root : str
        Where Code Ocean mounts attached assets.
    onset_kw : dict, optional
        Passed to :func:`threshold_onsets` for the motion-energy onsets. Defaults to
        ``{'z_thresh': 2.5, 'refractory': 0.5, 'min_run': 3}``.

    Returns
    -------
    dict
        ``session_id, subject_id, nwb, df_fip, df_trials, meta, t_me, me, me_z,
        me_onsets, go_cues``.

    Raises
    ------
    ValueError, FileNotFoundError
        When df_trials or the ME/video assets are missing, so the caller can skip.
    """
    if onset_kw is None:
        onset_kw = {"z_thresh": 2.5, "refractory": 0.5, "min_run": 3}

    df_trials = getattr(nwb, "df_trials", None)
    if df_trials is None or ALIGN_COL not in df_trials.columns:
        raise ValueError("missing df_trials / %s" % ALIGN_COL)

    # Upstream enrich_df_trials is all-or-nothing and needs more columns than the streak
    # fallback, so one incomplete session must not fail the whole loop.
    try:
        df_trials_enr = enrich_trials(df_trials.copy(), verbose=False)
    except Exception as e:
        print(
            "  [%s] enrich_trials failed (%s); streak bins off for this session"
            % (nwb.session_id, type(e).__name__)
        )
        df_trials_enr = df_trials

    video_csv, me_path = locate_me_assets(nwb.session_id, data_root)
    t_me, me, _offset = motion_energy_to_session(me_path, video_csv, nwb.df_trials)
    me_z = zscore(me)
    me_onsets = threshold_onsets(t_me, me_z, already_z=True, **onset_kw)

    return {
        "session_id": nwb.session_id,
        "subject_id": nwb.session_id.split("_")[0],
        "nwb": nwb,
        "df_fip": nwb.df_fip,
        "df_trials": df_trials_enr,
        "meta": build_meta(nwb.df_fip),
        "t_me": t_me,
        "me": me,
        "me_z": me_z,
        "me_onsets": me_onsets,
        "go_cues": df_trials_enr[ALIGN_COL].dropna().to_numpy(),
    }
