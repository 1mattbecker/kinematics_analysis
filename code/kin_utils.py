"""
kin_utils.py — small helpers shared by the ``kin_*`` notebook series (tongue kinematics).

numpy / pandas only, so the ``kin_*`` notebooks keep running locally on ``data/for_local``
without the video-analysis library. Keypoint I/O that the batch pipeline needs lives in the
library (``tongue_kinematics_utils``); these read the intermediates it already wrote.

Contents
--------
:func:`coerce_bool`, :func:`load_kps_raw`, :func:`jaw_position`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce an object-dtype True/False/None column to real booleans.

    Plain ``astype(bool)`` on an object column turns any non-empty string (including the string
    "False") into True. This treats only True / 1 / "True" as true and everything else, None
    included, as false.
    """
    return series.isin([True, 1, "True"])


def load_kps_raw(intermediate_dir, prefix: str = "kps_raw_") -> Dict[str, pd.DataFrame]:
    """A session's saved keypoint parquet files, ``{label: per-frame table}``.

    Parameters
    ----------
    intermediate_dir : str or pathlib.Path
        Directory holding files named like ``kps_raw_jaw.parquet``.
    prefix : str
        Filename prefix used when the keypoint tables were written.
    """
    intermediate_dir = Path(intermediate_dir)
    return {f.stem[len(prefix):]: pd.read_parquet(f)
            for f in intermediate_dir.glob("{}*.parquet".format(prefix))}


def jaw_position(session: str, session_dir) -> Optional[Tuple[float, float]]:
    """``(jaw_x, jaw_y)``: the jaw keypoint's mean position over a session, in camera pixels.

    Reads ``<session_dir>/<session>/intermediate_data/kps_raw_jaw.parquet``; returns None when the
    file is absent (running locally, or an unprocessed session).
    """
    path = Path(session_dir) / session / "intermediate_data" / "kps_raw_jaw.parquet"
    if not path.exists():
        return None
    kps_jaw = pd.read_parquet(path, columns=["x", "y"])
    return float(kps_jaw["x"].mean()), float(kps_jaw["y"].mean())
