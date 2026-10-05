"""
fip_utils.py — shared code for the ``fip_*`` notebook series (and ``men_*``'s trial tables).

Two data paths, one per kind of FIP asset:

* **NWB-list assets with JSON curation** (``DA_NE_4channels``; ``fip_00``–``fip_04``):
  :func:`load_curated_sessions` reads the saved parquet hierarchy with Rachel's
  ``load_nwb_list`` and applies the JSON curation; :func:`select_session`, :func:`build_meta`,
  :func:`pick_example` set up one session; :func:`process_session` adds motion energy.
* **CSV-curated assets read directly** (``DANE_3channels_curated``, the 4-channel rebuild;
  ``fip_05``, ``fip_07``, ``men_00``): :func:`find_asset_root`, :func:`inventory_pairs`,
  :func:`choose_side`, :func:`load_pairs` put same-side DA/NE pairs on a uniform grid;
  :func:`trial_measures`, :func:`fit_rpe_terms`, :func:`residualize`, :func:`task_residuals`
  measure them. (This was ``fip_coupling.py``.)

Motion energy on the FIP clock (:func:`load_me`, :func:`motion_energy_to_session`) serves both.
Generic signal code is in ``signal_utils``, across-animal statistics in ``stats_utils``, task
context and lick bouts in ``behavior_utils``.

This module does **not** own the choice of FIP normalization. The three
``aind_dynamic_foraging_data_utils.enrich_dfs`` entry points are three different
normalizations, not three steps of one pipeline:

* ``zscore_fip``              -> whole-session ``data_z``
* ``enrich_fip_in_df_trials`` -> re-cuts signals into per-trial windows (z-scores internally;
                                 a prior ``zscore_fip`` would double-process)
* ``remove_tonic_df_fip``     -> per-trial ``data_z_*_baseline`` / ``_norm``

Which one a notebook runs defines what "elevated" means for onset detection and what its AUC
figures measure, so that call stays visible in each notebook. :func:`attach_me_to_df_fip`
bridges motion energy into that pipeline and takes **raw** motion energy because those
functions z-score ``data`` themselves.

Usage (notebooks run with ``code/`` as cwd)::

    %load_ext autoreload
    %autoreload 2
    import fip_utils as fu

    nwb_list = fu.load_curated_sessions()
    nwb, df_fip, df_trials = fu.select_session(nwb_list, 0)

``autoreload`` matters for the NWB path: a session load is tens of GB and several minutes.

Clock: every time is session time (s from the first go cue), shared by
``df_fip['timestamps']`` and the ``*_in_session`` trial columns. A loaded pair is on a uniform
``fs``-Hz grid starting at ``t0``; sample ``i`` sits at ``t0 + i / fs``.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import pickle
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

import signal_utils as su

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




def enrich_trials(df_trials: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add ``num_reward_past``, ``RPE-binned3`` and the rest of Rachel's trial columns.

    Calls ``rachel_analysis_utils.analysis_utils.enrich_df_trials``. Upstream is all-or-nothing
    and needs ``RPE_earned``, ``Q_chosen``, ``animal_response`` and ``choice_time_in_trial``, so a
    loop over many sessions should catch its errors (see :func:`process_session`).

    Parameters
    ----------
    df_trials : pandas.DataFrame
        Trial table for one session (or several, keyed by ``ses_idx``).
    verbose : bool
        Unused; kept so existing calls keep working.

    Returns
    -------
    pandas.DataFrame
        Enriched trial table.
    """
    from rachel_analysis_utils import analysis_utils as r_analysis

    return r_analysis.enrich_df_trials(df_trials)


def streak_go_cues(df_trials: pd.DataFrame, bin_fn) -> Optional[np.ndarray]:
    """Go-cue times (session clock) for trials whose ``num_reward_past`` satisfies ``bin_fn``."""
    if "num_reward_past" not in df_trials.columns:
        return None
    sub = df_trials.dropna(subset=[ALIGN_COL, "num_reward_past"])
    mask = sub["num_reward_past"].map(bin_fn).astype(bool)
    return sub.loc[mask, ALIGN_COL].to_numpy()


# ── Motion energy ─────────────────────────────────────────────────────────────

#: Aligned motion-energy table under ``data_root`` (asset 90c2d0a7-82e3-4abf-b205-e398c2f7736e,
#: built by ``build_me_table.py``): per session and camera, the corrected Harp time and
#: ``me_clean`` of every video frame, plus ``index.csv`` with each camera's timing status.
ME_TABLE_NAME = "fip_motion_energy_aligned"

#: Folder holding the ME table: Code Ocean's data mount, or ``$ME_DATA_ROOT`` when set (a local
#: copy, e.g. ``../../data`` from ``code/``).
ME_DATA_ROOT = os.environ.get("ME_DATA_ROOT", DEFAULT_DATA_ROOT)

#: Per-camera video screen (timing + image quality) from the video-analysis library's
#: ``screen_sessions`` over the 301 curated FIP sessions (library 0.2.0). ``use`` is False for a
#: camera that failed either check; the ME table's own build applied the timing check only.
VIDEO_SCREEN_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "inputs", "video_screen_fip.csv")

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
            index.loc[excluded, "error"] = reason[excluded]
        _ME_INDEX[(path, screen)] = index
    return _ME_INDEX[(path, screen)]


def me_sessions(
    camera: str = "BottomCamera",
    exclude_actions: Sequence[str] = (),
    data_root: str = ME_DATA_ROOT,
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
    return sorted(index.loc[usable, "ses_idx"])


def load_me(
    session_id: str, camera: str = "BottomCamera", data_root: str = ME_DATA_ROOT
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
    tuple
        ``(t_harp, me, info)``: grid times on the Harp clock (s), per-exposure ME on that
        grid, and the camera's ``index.csv`` row as a dict.

    Raises
    ------
    FileNotFoundError
        If the table has no row for this session and camera, so a multi-session loop can skip.
    MotionEnergyRefused
        If the build refused the camera or the video screen excluded it (``info['error']`` says why).
    ValueError
        If the file does not match its index row (a length mismatch is never truncated).
    """
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

    frames = pd.read_parquet(
        os.path.join(data_root, ME_TABLE_NAME, info["session"], "%s.parquet" % camera),
        columns=["harp_time", "frame_number", "me_clean"],
    )
    if len(frames) != info["n_frames"]:
        raise ValueError(
            "%s %s: %d rows in the file, index.csv says %d"
            % (session_id, camera, len(frames), info["n_frames"])
        )
    harp = frames["harp_time"].to_numpy(float)
    exposures = np.diff(frames["frame_number"].to_numpy())
    if (exposures < 1).any():
        raise ValueError("%s %s: frame numbers do not increase" % (session_id, camera))

    # Per-exposure ME (step 1); row 0 has no value
    me = frames["me_clean"].to_numpy(float)[1:] / exposures
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


def motion_energy_to_session(
    session_id: str,
    df_trials: pd.DataFrame,
    go_cue_col: str = "goCue_start_time_raw",
    camera: str = "BottomCamera",
    data_root: str = ME_DATA_ROOT,
    pad_s: Optional[float] = ME_TASK_PAD_S,
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
    t_harp, me, info = load_me(session_id, camera, data_root)
    go_cues = df_trials.sort_values("trial")[go_cue_col].dropna()  # Harp clock
    first_go_cue = float(go_cues.iloc[0])
    t_session = t_harp - first_go_cue
    if pad_s is not None:
        last = float(go_cues.max()) - first_go_cue
        keep = (t_session >= -pad_s) & (t_session <= last + pad_s)
        t_session, me = t_session[keep], me[keep]
    return t_session, me, first_go_cue - float(info["harp_start"])


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
        idx = su.grid_index(times, t0, fs)
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
                   task: bool = True) -> dict:
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

    Returns
    -------
    dict
        ``{sig: z, sig + '_fit': fitted, sig + '_res': residual, sig + '_r2': fraction explained}``
        for ``sig`` in ``da``, ``ne``.
    """
    n = len(pair["da"])
    blocks = [event_design(pair, kernel_s)] if task else [csr_matrix(np.ones((n, 1)))]
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
    return out


def large_transients(z_a: np.ndarray, z_b: np.ndarray, t0: float, fs: float,
                     large_prom: float = 2.0, partner_prom: float = 1.0,
                     match_win_s: float = 0.5, min_sep_s: float = 0.5) -> pd.DataFrame:
    """Large transients of one signal and whether the other has a partner (``fip_07`` Fig 2 rule).

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


# ── Signal helpers ────────────────────────────────────────────────────────────


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

def process_session(
    nwb,
    data_root: str = DEFAULT_DATA_ROOT,
    onset_kw: Optional[dict] = None,
    camera: str = "BottomCamera",
) -> dict:
    """Run the single-session pipeline and return a dict for cross-session analyses.

    Mirrors the single-session path exactly, reusing :func:`build_meta` and
    :func:`motion_energy_to_session`.

    Parameters
    ----------
    nwb : object
        One curated session.
    data_root : str
        Where Code Ocean mounts attached assets (the ME table among them).
    onset_kw : dict, optional
        Passed to ``signal_utils.threshold_onsets`` for the motion-energy onsets. Defaults to
        :data:`ME_ONSET_KW`.
    camera : str
        Camera whose motion energy to use.

    Returns
    -------
    dict
        ``session_id, subject_id, nwb, df_fip, df_trials, meta, t_me, me, me_z,
        me_onsets, go_cues``.

    Raises
    ------
    ValueError, FileNotFoundError
        When df_trials is missing, the session has no ME, or the ME table refused the camera
        (:class:`MotionEnergyRefused`, a ValueError), so the caller can skip.
    """
    if onset_kw is None:
        onset_kw = ME_ONSET_KW

    df_trials = getattr(nwb, "df_trials", None)
    if df_trials is None or ALIGN_COL not in df_trials.columns:
        raise ValueError("missing df_trials / %s" % ALIGN_COL)

    # Upstream enrich_df_trials is all-or-nothing, so one session missing a column must not fail
    # the whole loop.
    try:
        df_trials_enr = enrich_trials(df_trials.copy(), verbose=False)
    except Exception as e:
        print(
            "  [%s] enrich_trials failed (%s); streak bins off for this session"
            % (nwb.session_id, type(e).__name__)
        )
        df_trials_enr = df_trials

    t_me, me, _offset = motion_energy_to_session(
        nwb.session_id, nwb.df_trials, camera=camera, data_root=data_root
    )
    me_z = su.zscore(me)
    me_onsets = su.threshold_onsets(t_me, me_z, already_z=True, **onset_kw)

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
