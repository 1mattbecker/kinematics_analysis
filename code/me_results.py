"""
me_results.py — read aind-motion-energy result files from either output version.

Used by the motion-energy table scripts (``build_me_table.py``, ``build_me_asset_map.py``,
``check_leading_lost_frames.py``) and, for the ROI column names and masks, ``fip_utils``.
Results from aind-motion-energy 0.3.0 and earlier (no ``output_version`` in
``<camera>_me_metadata.json``) save ``<camera>_motion_energy_clean.npy`` with one value fewer
than frames, and readers add the leading NaN. From 0.4.0 (``output_version`` 2) the library
writes ``<camera>_motion_energy.npz`` with one row per frame and the NaN already at row 0, so
padding again would shift ME one frame late. :func:`read_me_traces` and :func:`read_me_clean`
return the same frame-aligned traces for both (checked on 800886 2025-08-28: the 0.4.0
``full_frame`` column equals the 0.3.0 table's ``me_clean`` at offset 0, both cameras).

ROIs (0.4.0): the NPZ's ``me_clean`` has one column per ``roi_ids`` entry, ``full_frame``
first; every ROI column is a mean |frame difference| over its mask's pixels, on the same rows
(frames) and with the same keyframe cleaning as ``full_frame``. When the run was given masks, the
library saves a copy as ``<camera>_roi_masks.npz`` (``masks`` (H, W, K), ``roi_ids`` (K,)).

In the aligned ME table the whole frame stays ``me_clean`` and each ROI is ``me_clean_<roi id>``
(:func:`me_column`).

Contents
--------
:func:`is_frame_aligned`, :func:`me_file_suffix`, :func:`read_me_traces`, :func:`read_me_clean`,
:func:`me_column`, :func:`read_roi_masks`
"""

from __future__ import annotations

import numpy as np

# output_version written by aind-motion-energy >= 0.4.0
FRAME_ALIGNED_VERSION = 2
FULL_FRAME = "full_frame"
# Copy of the run's ROI masks, after ``<camera>``, in 0.4.0 results given --roi-masks
ROI_MASKS_SUFFIX = "_roi_masks.npz"
# Table column of the whole-frame trace; an ROI's column adds "_<roi id>"
ME_COLUMN = "me_clean"


def is_frame_aligned(meta):
    """True if the result was written by aind-motion-energy >= 0.4.0 (already padded)."""
    return meta.get("output_version", 1) >= FRAME_ALIGNED_VERSION


def me_file_suffix(meta):
    """Suffix of the file holding the cleaned trace, after ``<camera>``."""
    return "_motion_energy.npz" if is_frame_aligned(meta) else "_motion_energy_clean.npy"


def read_me_traces(path, meta):
    """Cleaned ME of the whole frame and every ROI, one value per decoded frame.

    Row i is frame i and row 0 is NaN in every trace, for both output versions.

    Parameters
    ----------
    path : Path
        Local copy of the file named by :func:`me_file_suffix`.
    meta : dict
        The camera's ``_me_metadata.json``.

    Returns
    -------
    dict
        ROI id -> float32 trace, in file order with ``full_frame`` first. A 0.3.0 result has
        ``full_frame`` only.

    Raises
    ------
    ValueError
        If a trace's length doesn't match ``n_frames_decoded``, a 0.4.0 file has no
        ``full_frame`` column, or a 0.3.0 result was computed on a ``--roi`` box (its one trace
        is not the whole frame).
    """
    n_frames = int(meta["n_frames_decoded"])
    if is_frame_aligned(meta):
        with np.load(path, allow_pickle=False) as f:
            ids = [str(i) for i in f["roi_ids"]]
            me = f["me_clean"]
        if FULL_FRAME not in ids:
            raise ValueError(f"ME file has no {FULL_FRAME!r} column (roi_ids {ids})")
        traces = {roi_id: me[:, k] for k, roi_id in enumerate(ids)}
    else:
        if meta.get("roi"):
            raise ValueError(f"0.3.0 ME was computed on the box {meta['roi']}, not the whole frame")
        me = np.load(path)
        if len(me) != n_frames - 1:
            raise ValueError(f"0.3.0 ME file has {len(me)} values for {n_frames} frames, expected N-1")
        traces = {FULL_FRAME: np.concatenate([[np.nan], me])}
    for roi_id, trace in traces.items():
        if len(trace) != n_frames:
            raise ValueError(f"ME file has {len(trace)} rows for {roi_id}, its metadata says {n_frames} frames")
    return {roi_id: trace.astype("float32") for roi_id, trace in traces.items()}


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
        As :func:`read_me_traces`.
    """
    return read_me_traces(path, meta)[FULL_FRAME]


def me_column(roi=None):
    """Aligned-table column holding the whole frame (``roi`` None or ``full_frame``) or one ROI."""
    if roi is None or roi == FULL_FRAME:
        return ME_COLUMN
    return f"{ME_COLUMN}_{roi}"


def read_roi_masks(path):
    """ROI masks saved with a 0.4.0 result (the annotator's ``*_masks.npz`` contract).

    The same contract as ``aind_motion_energy.rois.load_roi_masks``, read here so this repo does
    not need the library installed.

    Parameters
    ----------
    path : Path
        NPZ with ``masks`` (``(H, W, K)``, 0/1) and ``roi_ids`` (``(K,)``).

    Returns
    -------
    dict
        ROI id -> bool mask of shape ``(H, W)``, in file order.
    """
    with np.load(path, allow_pickle=False) as f:
        masks, ids = f["masks"], [str(i) for i in f["roi_ids"]]
    if masks.ndim != 3 or masks.shape[2] != len(ids):
        raise ValueError(f"{path}: masks {masks.shape} do not match {len(ids)} roi_ids")
    return {roi_id: masks[:, :, k].astype(bool) for k, roi_id in enumerate(ids)}
