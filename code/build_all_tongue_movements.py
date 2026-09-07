"""
build_all_tongue_movements.py

Builds the combined all_tongue_movements parquet: loop over per-session
intermediates, keep sessions that pass the tongue-tracking quality filter,
concatenate their movement tables. The tongue_movs.parquet files already
contain the outbound (out_*) metrics.
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path("/root/capsule/data/keypoint_tracking_bottomview_LCrecordings_20260403")
OUT = Path("/root/capsule/scratch/temp/all_tongue_movements_04022026.parquet")

COVERAGE_MIN = 90.0    # percent of frames with a confident tongue keypoint
DURATION50_MIN = 0.06  # median movement duration, seconds


def passes_quality(session_dir):
    """True if the session's tongue_quality_stats.json clears the thresholds."""
    stats = session_dir / "tongue_quality_stats.json"
    if not stats.exists():
        return False
    d = json.loads(stats.read_text())
    cov = float(d.get("coverage_pct", 0.0))
    dur50 = float(d.get("percentiles", {}).get("duration", {}).get("0.5", 0.0))
    return cov > COVERAGE_MIN and dur50 > DURATION50_MIN


chunks = []
for session_dir in sorted(BASE.glob("behavior_*")):
    if not passes_quality(session_dir):
        print("skip %s" % session_dir.name)
        continue
    movs = pd.read_parquet(session_dir / "intermediate_data" / "tongue_movs.parquet")
    movs["session"] = session_dir.name
    chunks.append(movs)
    print("ok   %s: %d movements" % (session_dir.name, len(movs)))

all_tongue_movements = pd.concat(chunks, ignore_index=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
all_tongue_movements.to_parquet(OUT, index=False)
print("\nsaved %d movements from %d sessions -> %s" % (len(all_tongue_movements), len(chunks), OUT))
