"""
build_all_tongue_movements.py

Builds the combined all_tongue_movements parquet: loop over per-session
intermediates, keep sessions that pass the tongue-tracking quality filter,
concatenate their movement tables. The tongue_movs.parquet files carry the
outbound (out_*) metrics: natively once the library's
aggregate_tongue_movements has been re-run on the sessions, otherwise via
the add_outbound.ipynb backfill.

Run as a script (`python build_all_tongue_movements.py`) or via `%run` in a
notebook (both set __name__ == "__main__"); a plain `import
build_all_tongue_movements` loads the helpers without running the batch job
— it previously ran the full loop and wrote OUT as a side effect of import.
"""

from pathlib import Path

import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import (
    get_quality_summary,
    load_tongue_quality_stats,
    session_already_done,
)

BASE = Path("/root/capsule/data/keypoint_tracking_bottomview_LCrecordings_20260403")
OUT = Path("/root/capsule/scratch/temp/all_tongue_movements_04022026.parquet")

COVERAGE_MIN = 90.0    # percent of frames with a confident tongue keypoint
DURATION50_MIN = 0.06  # median movement duration, seconds


def passes_quality(session_dir):
    """True if the session's tongue_quality_stats.json clears the thresholds."""
    if not session_already_done(session_dir):
        return False
    q = get_quality_summary(load_tongue_quality_stats(session_dir))
    return q["coverage_pct"] > COVERAGE_MIN and q["duration_p50"] > DURATION50_MIN


def build_all_tongue_movements(base=None, out=None):
    """
    Run the filter + concatenate pass and write the pooled parquet.

    Parameters
    ----------
    base : Path, optional
        Root folder of per-session `behavior_*` output dirs. Defaults to `BASE`
        (the canonical `session_analysis_mlk`-derived pipeline output).
    out : Path, optional
        Where to write the pooled parquet. Defaults to `OUT`.

    Returns
    -------
    pd.DataFrame
        The combined table (also written to `out`).
    """
    base = Path(base) if base is not None else BASE
    out = Path(out) if out is not None else OUT

    chunks = []
    for session_dir in sorted(base.glob("behavior_*")):
        if not passes_quality(session_dir):
            print("skip %s" % session_dir.name)
            continue
        movs = pd.read_parquet(session_dir / "intermediate_data" / "tongue_movs.parquet")
        movs["session"] = session_dir.name
        chunks.append(movs)
        print("ok   %s: %d movements" % (session_dir.name, len(movs)))

    all_tongue_movements = pd.concat(chunks, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_tongue_movements.to_parquet(out, index=False)
    print("\nsaved %d movements from %d sessions -> %s" % (len(all_tongue_movements), len(chunks), out))
    return all_tongue_movements


if __name__ == "__main__":
    all_tongue_movements = build_all_tongue_movements()
