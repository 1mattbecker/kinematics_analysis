"""
Shared data loading utilities for kinematics × ephys notebooks.

Three stages:
  1. load_session_quality_filter  → filtered_session_paths
  2. filter_ephys_units           → filtered_ephys (QC + session filter)
  3. load_units_with_spike_times  → units_with_spikes (DataFrame with spike_times column)
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import (
    get_session_name_from_path,
    session_already_done,
)
from aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys import get_session_prefix


# ── Stage 1: session quality filter ──────────────────────────────────────────

DEFAULT_COVERAGE_MIN   = 90.0
DEFAULT_DURATION50_MIN = 0.06   # seconds

def load_session_quality_filter(
    base_dirs: List[Path],
    coverage_min: float = DEFAULT_COVERAGE_MIN,
    duration50_min: float = DEFAULT_DURATION50_MIN,
) -> List[Path]:
    """
    Scan session subdirectories for tongue_quality_stats.json and return
    the paths of sessions meeting coverage and duration thresholds.
    """
    rows_pass, rows_fail = [], []

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for subdir in base_dir.iterdir():
            if not (subdir.is_dir() and session_already_done(subdir)):
                continue
            json_file = subdir / "tongue_quality_stats.json"
            try:
                with open(json_file, "r") as f:
                    d = json.load(f)
            except Exception as e:
                print(f"[skip] {subdir.name}: {e}")
                continue

            cov   = float(d.get("coverage_pct", 0.0))
            dur50 = float(d.get("percentiles", {}).get("duration", {}).get("0.5", 0.0))
            row   = {
                "session_path": subdir,
                "session_id":   d.get("session_id", subdir.name),
                "coverage_pct": cov,
                "duration_p50": dur50,
            }
            (rows_pass if cov > coverage_min and dur50 > duration50_min else rows_fail).append(row)

    passed = pd.DataFrame(rows_pass).sort_values(
        ["coverage_pct", "duration_p50"], ascending=False
    ) if rows_pass else pd.DataFrame()

    print(f"Sessions passing quality filter: {len(passed)} "
          f"(coverage>{coverage_min}%, median dur>{duration50_min}s)")
    for _, r in passed.iterrows():
        print(f"  {r['session_path'].name}  "
              f"(cov={r['coverage_pct']:.1f}%, dur50={r['duration_p50']:.3f}s)")

    return passed["session_path"].tolist() if not passed.empty else []


# ── Stage 2: unit QC + session filter ────────────────────────────────────────

DEFAULT_CRITERIA: Dict = {
    "isi_violations": {"bounds": [0.0,  0.1]},
    "p_max":          {"bounds": [0.5,  1.0]},
    "lat_max_p":      {"bounds": [0.005, 0.02]},
    "eu":             {"bounds": [0.0,  0.25]},
    "corr":           {"bounds": [0.95, 1.0]},
    "qc_pass":        {"items":  [True]},
    "peak":           {"bounds": [-1000, 0]},
    "trial_count":    {"bounds": [100, 2000]},
    "in_df":          {"items":  [True]},
}


def filter_by_criteria(df: pd.DataFrame, criteria: Dict = None) -> pd.DataFrame:
    if criteria is None:
        criteria = DEFAULT_CRITERIA
    mask = pd.Series(True, index=df.index)
    for col, rule in criteria.items():
        if "bounds" in rule:
            lo, hi = rule["bounds"]
            mask &= df[col].between(lo, hi, inclusive="both")
        if "items" in rule:
            mask &= df[col].isin(rule["items"])
    return df.loc[mask].copy()


def filter_ephys_units(
    combined_ephys_data: pd.DataFrame,
    filtered_session_paths: List[Path],
    criteria: Dict = None,
) -> pd.DataFrame:
    """
    Apply QC criteria and session allowlist to combined_ephys_data.
    Returns filtered_ephys with a session_prefix column added.
    """
    session_order   = [get_session_name_from_path(str(p)) for p in filtered_session_paths]
    prefix_allow    = set(get_session_prefix(s) for s in session_order)

    df = combined_ephys_data.copy()
    df["session_prefix"] = df["session"].map(get_session_prefix)

    qc_pass  = filter_by_criteria(df, criteria)
    filtered = qc_pass.loc[qc_pass["session_prefix"].isin(prefix_allow)].copy()

    print(f"Filtered units: {len(filtered)} / {len(combined_ephys_data)}")
    return filtered


# ── Stage 3: load spike times ─────────────────────────────────────────────────

def _get_animal_id(session: str) -> str:
    m = re.match(r'^behavior_(\d+)_', session)
    if not m:
        raise ValueError(f"Cannot parse animal id from: {session}")
    return m.group(1)


def _find_summary_pkl(root: str, session: str) -> Optional[Path]:
    animal = _get_animal_id(session)
    exact  = (Path(root) / animal / session / "ephys" / "opto" / "curated"
              / f"{session}_curated_soma_opto_tagging_summary.pkl")
    if exact.exists():
        return exact
    pref       = get_session_prefix(session)
    candidates = list((Path(root) / animal).glob(
        f"{pref}_*/ephys/opto/curated/*_curated_soma_opto_tagging_summary.pkl"
    ))
    return candidates[0] if candidates else None


def load_units_with_spike_times(
    filtered_ephys: pd.DataFrame,
    root_scratch: str,
) -> pd.DataFrame:
    """
    For each session in filtered_ephys, locate the curated opto-tagging pkl
    and extract rows matching the QC-filtered unit IDs.

    Returns a DataFrame with columns [session, unit_id, spike_times, ...].
    """
    records = []
    for session, subdf in filtered_ephys.groupby("session"):
        pkl_path = _find_summary_pkl(root_scratch, session)
        if pkl_path is None:
            print(f"[skip] no summary pkl for {session}")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception:
                print(f"[skip] pkl not a DataFrame for {session}")
                continue
        unit_ids   = subdf["unit"].unique()
        ephys_data = data[data["unit_id"].isin(unit_ids)].copy().assign(session=session)
        print(f"[ok] {session}: {len(ephys_data)}/{len(unit_ids)} units")
        records.append(ephys_data)

    if records:
        out = pd.concat(records, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["session", "unit_id", "spike_times"])

    print(f"units_with_spikes shape: {out.shape}")
    return out
