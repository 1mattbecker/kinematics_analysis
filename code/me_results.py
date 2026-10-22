"""
me_results.py — read aind-motion-energy result files from either output version.

Used by the motion-energy table scripts (``build_me_table.py``, ``build_me_asset_map.py``,
``check_leading_lost_frames.py``). Results from aind-motion-energy 0.3.0 and earlier
(no ``output_version`` in ``<camera>_me_metadata.json``) save ``<camera>_motion_energy_clean.npy``
with one value fewer than frames, and readers add the leading NaN. From 0.4.0
(``output_version`` 2) the library writes ``<camera>_motion_energy.npz`` with one row per
frame and the NaN already at row 0, so padding again would shift ME one frame late.
:func:`read_me_clean` returns the same frame-aligned trace for both.

Contents
--------
:func:`is_frame_aligned`, :func:`me_file_suffix`, :func:`read_me_clean`
"""

from __future__ import annotations

import numpy as np

# output_version written by aind-motion-energy >= 0.4.0
FRAME_ALIGNED_VERSION = 2
FULL_FRAME = "full_frame"


def is_frame_aligned(meta):
    """True if the result was written by aind-motion-energy >= 0.4.0 (already padded)."""
    return meta.get("output_version", 1) >= FRAME_ALIGNED_VERSION


def me_file_suffix(meta):
    """Suffix of the file holding the cleaned trace, after ``<camera>``."""
    return "_motion_energy.npz" if is_frame_aligned(meta) else "_motion_energy_clean.npy"


def read_me_clean(path, meta):
    """Cleaned whole-frame ME with one value per decoded frame (row i is frame i, row 0 NaN).

    Parameters
    ----------
    path : Path
        Local copy of the file named by :func:`me_file_suffix`.
    meta : dict
        The camera's ``_me_metadata.json``.

    Raises
    ------
    ValueError
        If the trace length doesn't match ``n_frames_decoded``.
    """
    n_frames = int(meta["n_frames_decoded"])
    if is_frame_aligned(meta):
        with np.load(path, allow_pickle=False) as f:
            ids = [str(i) for i in f["roi_ids"]]
            me = f["me_clean"][:, ids.index(FULL_FRAME)]
    else:
        me = np.load(path)
        if len(me) != n_frames - 1:
            raise ValueError(f"0.3.0 ME file has {len(me)} values for {n_frames} frames, expected N-1")
        me = np.concatenate([[np.nan], me])
    if len(me) != n_frames:
        raise ValueError(f"ME file has {len(me)} rows, its metadata says {n_frames} frames")
    return me.astype("float32")
