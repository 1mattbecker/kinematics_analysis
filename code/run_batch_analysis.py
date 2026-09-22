"""
run_batch_analysis.py

This is the capsule's Reproducible Run entrypoint (see `code/run`).

2026-09-22 re-point: re-runs the per-session pipeline with the migrated
library (`compute_outbound_metrics` now native in `aggregate_tongue_movements`
— TODO.md "Fold outbound metrics...", CHANGELOG.md 2026-09-17/22) into a fresh
save_root, `session_analysis_fall2026`, leaving the existing
`session_analysis_mlk` output untouched for comparison. Pools the result into
a freshly dated parquet at the end.

Session list is reconstructed from `session_analysis_mlk` itself — the
`pred_csv` path recorded in each session's existing `tongue_quality_stats.json`
— rather than trusting `pred_csv_list_20250113.json` (dated January 2025;
several sessions currently in the pooled data, e.g. the 791691/784803/784806/
763590 series, postdate it and would be silently dropped by reusing that
file). This reproduces the exact same session set as an apples-to-apples
re-run, not a resync to a newer raw-data list.

`extract_clips=False`: the mlk run already has example clips for these
sessions and clip extraction re-decodes video per session, which is the
slowest part of the batch for no benefit here. Set it to True below if you
also want fresh clips under the new save_root.
"""
from pathlib import Path
import datetime

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import (
    run_batch_analysis,
    session_already_done,
    load_tongue_quality_stats,
)

from build_all_tongue_movements import build_all_tongue_movements

OLD_SAVE_ROOT = Path("/root/capsule/scratch/session_analysis_mlk")
SAVE_ROOT = Path("/root/capsule/scratch/session_analysis_fall2026")
DATA_ROOT = Path("/root/capsule/data")

EXTRACT_CLIPS = False

DATE_TAG = datetime.date.today().strftime("%m%d%Y")  # e.g. "09222026"
POOLED_OUT = Path(f"/root/capsule/scratch/temp/all_tongue_movements_{DATE_TAG}.parquet")


def discover_pred_csv_list(old_save_root):
    """
    Reconstruct the batch's input list from the `pred_csv` already recorded
    per session under `old_save_root`'s `tongue_quality_stats.json`, rather
    than a separately-maintained JSON list that can go stale. Includes every
    previously-processed session, pass or fail on the quality gate — that
    gate is re-evaluated downstream by `build_all_tongue_movements`, not here.
    """
    pred_csv_list = []
    skipped = []
    for session_dir in sorted(old_save_root.glob("behavior_*")):
        if not session_dir.is_dir() or not session_already_done(session_dir):
            continue
        stats = load_tongue_quality_stats(session_dir)
        pred_csv = stats.get("pred_csv")
        if not pred_csv:
            skipped.append(session_dir.name)
            continue
        pred_csv_list.append(pred_csv)
    if skipped:
        print(f"[warn] {len(skipped)} sessions have no recorded pred_csv, skipped: {skipped}")
    return pred_csv_list


if __name__ == "__main__":
    pred_csv_list = discover_pred_csv_list(OLD_SAVE_ROOT)
    print(f"Re-running {len(pred_csv_list)} sessions (from {OLD_SAVE_ROOT.name}) -> {SAVE_ROOT}")

    run_batch_analysis(pred_csv_list, DATA_ROOT, SAVE_ROOT, extract_clips=EXTRACT_CLIPS)

    all_tongue_movements = build_all_tongue_movements(base=SAVE_ROOT, out=POOLED_OUT)
    print(f"\nDone. New pooled parquet: {POOLED_OUT}")
