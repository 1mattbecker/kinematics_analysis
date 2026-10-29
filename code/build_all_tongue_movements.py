"""
build_all_tongue_movements.py

Builds the pooled tongue-movement tables from the library's per-session batch output
(``kin_utils.KINEMATICS_ROOT/<session>/``): keep sessions that pass the quality filter
(``data_loading.passes_quality``), concatenate their ``movements.parquet`` and ``landmarks.parquet``.

Each movement also gets its trial's ``cue_response_movement_number``, ``lick_latency`` and
``reward_before_choice`` from ``trials.parquet`` (one row per trial since library 0.5.0), so per-trial
filters work on the pooled table as they did on the pre-0.5.0 one. Positions (``start_*``, ``end_*``,
``extent_*``, the ``extremes`` columns) are relative to the session's jaw landmark; the landmark
table gives them back in camera pixels.

Run as a script (`python build_all_tongue_movements.py`) or via `%run` (both set
__name__ == "__main__"); a plain import loads the helpers without running the batch.
"""

from pathlib import Path

import pandas as pd

from data_loading import passes_quality, session_quality
from kin_utils import KINEMATICS_ROOT, POOLED_LANDMARKS, POOLED_MOVEMENTS

OUT_DIR = Path("/root/capsule/scratch/all_tongue_movements")

TRIAL_COLUMNS = ["cue_response_movement_number", "lick_latency", "reward_before_choice"]


def session_movements(session_dir):
    """One session's movements with its trials' :data:`TRIAL_COLUMNS` merged on ``trial``."""
    session_dir = Path(session_dir)
    movs = pd.read_parquet(session_dir / "movements.parquet")
    trials = pd.read_parquet(session_dir / "trials.parquet")
    cols = [c for c in TRIAL_COLUMNS if c in trials.columns]
    return movs.merge(trials[["trial", *cols]], on="trial", how="left", validate="many_to_one")


def build_all_tongue_movements(base=None, out_dir=None):
    """
    Run the filter + concatenate pass and write the pooled parquets.

    Parameters
    ----------
    base : Path, optional
        Root folder of per-session ``behavior_*`` output dirs. Defaults to ``KINEMATICS_ROOT``.
    out_dir : Path, optional
        Where to write ``POOLED_MOVEMENTS`` and ``POOLED_LANDMARKS``. Defaults to ``OUT_DIR``.

    Returns
    -------
    movements, landmarks : pd.DataFrame
        The pooled tables (also written to ``out_dir``).
    """
    base = Path(base) if base is not None else KINEMATICS_ROOT
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR

    movement_chunks, landmark_chunks = [], []
    for session_dir in sorted(base.glob("behavior_*")):
        if not passes_quality(session_quality(session_dir)):
            print("skip %s" % session_dir.name)
            continue
        movs = session_movements(session_dir)
        movement_chunks.append(movs)
        landmark_chunks.append(pd.read_parquet(session_dir / "landmarks.parquet"))
        print("ok   %s: %d movements" % (session_dir.name, len(movs)))

    movements = pd.concat(movement_chunks, ignore_index=True)
    landmarks = pd.concat(landmark_chunks, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    movements.to_parquet(out_dir / POOLED_MOVEMENTS, index=False)
    landmarks.to_parquet(out_dir / POOLED_LANDMARKS, index=False)
    print("\nsaved %d movements from %d sessions -> %s"
          % (len(movements), len(movement_chunks), out_dir / POOLED_MOVEMENTS))
    return movements, landmarks


if __name__ == "__main__":
    all_tongue_movements, all_landmarks = build_all_tongue_movements()
