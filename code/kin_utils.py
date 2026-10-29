"""
kin_utils.py — where the tongue-kinematics data live, and small helpers for the ``kin_*`` series.

numpy / pandas only, so the ``kin_*`` notebooks keep running locally on ``data/for_local``
without the video-analysis library.

Per-session tables are written by the library's batch
(``kinematics.tasks.dynamic_foraging_tongue.run_batch``) into ``KINEMATICS_ROOT/<session>/``:
``movements``, ``submovements``, ``bouts``, ``trials``, ``licks``, ``events``, ``landmarks``,
``kinematics`` (frames), ``pose`` (raw keypoints) and ``record.json``. Read them with
``pd.read_parquet`` or the library's ``pipeline.read``. ``build_all_tongue_movements.py`` pools the
movements of the sessions that pass the quality filter into ``POOLED_MOVEMENTS`` and their
landmarks into ``POOLED_LANDMARKS``.

Contents
--------
:data:`KINEMATICS_ROOT`, :data:`POOLED_MOVEMENTS`, :data:`POOLED_LANDMARKS`,
:func:`pooled_path`, :func:`coerce_bool`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Library v0.5.3 batch output (one folder per session). Change here when it moves to a data asset.
KINEMATICS_ROOT = Path("/root/capsule/scratch/kinematics_v050")

# Pooled tables written by build_all_tongue_movements.py; the 04022026 file is the pre-0.5.0 table,
# kept for comparison.
POOLED_MOVEMENTS = "all_tongue_movements_v053.parquet"
POOLED_LANDMARKS = "all_landmarks_v053.parquet"
POOLED_MOVEMENTS_OLD = "all_tongue_movements_04022026.parquet"

_CO_POOLED_DIR = Path("/root/capsule/scratch/all_tongue_movements")
_LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "for_local"


def pooled_path(name: str = POOLED_MOVEMENTS) -> Path:
    """Path of a pooled table: Code Ocean scratch if present, else ``data/for_local``.

    The pre-0.5.0 table (``POOLED_MOVEMENTS_OLD``) sits in its own folder on Code Ocean.
    """
    co_dirs = [_CO_POOLED_DIR, Path("/root/capsule/scratch") / Path(name).stem]
    for d in co_dirs:
        if (d / name).exists():
            return d / name
    return _LOCAL_DIR / name


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce an object-dtype True/False/None column to real booleans.

    Plain ``astype(bool)`` on an object column turns any non-empty string (including the string
    "False") into True. This treats only True / 1 / "True" as true and everything else, None
    included, as false. (The library's tables have real booleans since 0.5.0; this is kept for
    the pre-0.5.0 pooled table.)
    """
    return series.isin([True, 1, "True"])
