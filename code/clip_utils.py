"""
clip_utils.py — video clips and frames at event times, found by frame index.

Row ``i`` of a session's per-frame tables (``kps_raw_*``, ``tongue_kins``, the ME table) is frame
``i`` of its video, also when frames were dropped: the library's timing QC corrects each frame's
Harp time and keeps every frame. So an event is put on the video by looking its time up in those
per-frame times (``video_alignment.event_frame_ranges``), and the video is cut by frame index
(``video_clips.cut_clips``), which seeks with the MP4's own sample timestamps.

Seconds into the MP4 are not used anywhere. Frame ``i`` sits at about ``i / fps`` in the file, but
its Harp time also counts every frame dropped before it, so session time plus a constant offset
(``get_video_time``, ``session_time_to_video_time``) lands late by the time of those frames.

Needs the video-analysis library >= v0.3.0 with its ``video-clips`` extra and ``ffmpeg`` on PATH;
both are imported when a function is called, so importing this module needs neither.

Contents
--------
:func:`event_clips`, :func:`frame_pngs`
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def event_clips(video, event_times, frame_times, before, after, out_dir, prefix) -> pd.DataFrame:
    """Cut one frame-exact clip around each event.

    Parameters
    ----------
    video : str or pathlib.Path
        The session's MP4 (raw or labeled; row ``i`` of ``frame_times`` must be its frame ``i``).
    event_times : array-like
        Event times, on the same clock as ``frame_times``.
    frame_times : array-like
        Time of every video frame, increasing: e.g. ``tongue_kins["time_in_session"]`` for
        session-time events, or ``tongue_kins["time_raw"]`` for Harp-time events.
    before, after : float
        Seconds before and after each event.
    out_dir : str or pathlib.Path
        Folder for the clips (created if needed).
    prefix : str
        Start of each clip name; clips are ``<prefix>_f<start_frame:07d>.mp4``.

    Returns
    -------
    pandas.DataFrame
        One row per event: ``event_time``, ``start_frame``, ``n_frames``, ``in_video``,
        ``clip_path`` (NaN when skipped) and ``status`` (``cut``, ``exists`` or ``skipped: ...``).
    """
    from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va
    from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc

    ranges = va.event_frame_ranges(np.asarray(event_times, dtype=float),
                                   np.asarray(frame_times, dtype=float), before, after)
    return vc.cut_clips(video, ranges, out_dir, prefix)


def frame_pngs(video, frames, out_dir, prefix) -> list:
    """Write video frames, by index, as PNGs.

    Each seek is the frame's own presentation time from the MP4's sample tables, so the PNG is
    that frame however many frames were dropped before it.

    Parameters
    ----------
    video : str or pathlib.Path
        The MP4.
    frames : array-like of int
        Frame indices (row numbers of the session's per-frame tables).
    out_dir : str or pathlib.Path
        Folder for the PNGs (created if needed).
    prefix : str
        Start of each name; PNGs are ``<prefix>_f<frame:07d>.png``.

    Returns
    -------
    list of pathlib.Path
        The PNGs, in the order of ``frames``.
    """
    from aind_video_utils import read_mp4_frame_index

    index = read_mp4_frame_index(video)
    if not index.is_frame_addressing_safe():
        raise ValueError(f"{video}: edit list is not safe for addressing frames by index")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in np.asarray(frames, dtype=int):
        out_png = out_dir / f"{prefix}_f{frame:07d}.png"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-accurate_seek",
             "-ss", f"{index.presentation_seconds(int(frame)):.9f}", "-i", str(video),
             "-frames:v", "1", "-y", str(out_png)],
            check=True,
        )
        paths.append(out_png)
    return paths
