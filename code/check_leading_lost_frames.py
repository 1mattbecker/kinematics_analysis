"""
check_leading_lost_frames.py — which end of the trigger log has no frames?

In ``behavior_818586_2026-01-16_09-19-39`` the Harp trigger log
(``Event_94.bin``) has 1,326 (bottom) / 1,328 (side) more events than the
camera frame numbers span, so ``video_timing_qc.correct_video_timing`` refuses
both cameras. The CSV pairs frames with triggers in arrival order, so it gives
row 0 the first trigger either way; the log count alone cannot say whether the
unmatched triggers are at the start or the end of the session.

The video can. The bottom and side cameras see the tongue, and lick times come
from the lickometer on the same Harp clock, so lick-triggered motion energy
peaks at lag 0 only when the frame times are right. This script compares the
two assignments (frames on the first ``n`` triggers, as the CSV has it, or on
the last ``n``) against a clean session from the same mouse and rig
(``818586_2026-01-15``), and draws the log and the saved frames at the start
and end of the session. Figures go to ``docs/video_timing_qc/``.

Reads only public S3 objects over HTTPS (camera CSVs, trigger log, lick
times, register writes, ME ``.npy``); downloads are cached under ``--cache``.

Usage (from ``code/``)::

    python check_leading_lost_frames.py [--cache <dir>]
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

import plotstyle as ps

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "docs" / "video_timing_qc"
OPEN_DATA_URL = "https://aind-open-data.s3.amazonaws.com"
SCRATCH_URL = "https://aind-scratch-data.s3.amazonaws.com/matt.becker/motion_energy"
ME_ASSETS_CSV = REPO / "inputs" / "me_assets_fip.csv"

SESSION = "behavior_818586_2026-01-16_09-19-39"
CONTROL = "behavior_818586_2026-01-15_09-18-12"  # same mouse and rig, log count matches
CAMERAS = {"BottomCamera": "bottom_camera", "SideCameraRight": "side_camera_right"}  # both flat

LAGS_S = np.arange(-4.0, 4.0, 0.002)  # one camera frame
EDGE_S = 10.0  # licks this close to the first/last frame are left out

COLOR_FIRST = ps.OKABE_ITO["vermillion"]  # frames on the first n triggers (CSV pairing)
COLOR_LAST = ps.OKABE_ITO["blue"]  # frames on the last n triggers
COLOR_CONTROL = ps.OKABE_ITO["black"]


def fetch(url, cache):
    """Download ``url`` once into ``cache``; return the local path."""
    path = cache / url.split("amazonaws.com/", 1)[1].replace("/", "__")
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def read_register(url, cache):
    """Harp time of the single message in a one-message register file."""
    message = np.fromfile(fetch(url, cache), dtype=np.uint8)
    seconds = message[5:9].view("<u4")[0]
    ticks = message[9:11].view("<u2")[0]
    return seconds + ticks * vtq.HARP_TICK_S


def load_session(session, cache):
    """Trigger log, lick times, camera-control writes and per-camera timing + ME."""
    harp = f"{OPEN_DATA_URL}/{session}/behavior/raw.harp"
    me_name = pd.read_csv(ME_ASSETS_CSV).set_index("raw_session").loc[session, "me_asset_name"]
    data = {
        "log": vtq.read_harp_trigger_log(fetch(f"{harp}/BehaviorEvents/Event_94.bin", cache)),
        # Write_78 / Write_79: start / stop cameras, written by the rig software
        "start": read_register(f"{harp}/BehaviorEvents/Write_78.bin", cache),
        "stop": read_register(f"{harp}/BehaviorEvents/Write_79.bin", cache),
        "licks": np.sort(
            np.concatenate(
                [
                    pd.read_csv(fetch(f"{harp}/ToBonsaiOSC2/{side}LickTime.csv", cache))["Item3"]
                    for side in ("Left", "Right")
                ]
            )
        ),
        "cameras": {},
    }
    for camera, source in CAMERAS.items():
        timing = vtq.load_video_timing(
            fetch(f"{OPEN_DATA_URL}/{session}/behavior-videos/{source}.csv", cache)
        )
        me = np.load(fetch(f"{SCRATCH_URL}/{me_name}/{source}_motion_energy_clean.npy", cache))
        # ME has no value for frame 0; pad so row i is frame i
        data["cameras"][camera] = {"timing": timing, "me": np.concatenate([[np.nan], me])}
    return data


def lick_triggered(frame_times, me, licks):
    """Mean ME around licks, z-scored against its own ±4 s window.

    Each lick + lag is placed on the first frame at or after it (2 ms frames).
    """
    inside = (licks > frame_times[0] + EDGE_S) & (licks < frame_times[-1] - EDGE_S)
    rows = np.searchsorted(frame_times, licks[inside, None] + LAGS_S[None, :])
    mean = np.nanmean(me[np.clip(rows, 0, len(me) - 1)], axis=0)
    return (mean - np.median(mean)) / np.std(mean), int(inside.sum())


def figure_lick_triggered(session_data, control_data):
    """Lick-triggered ME: control session, then each camera under both assignments."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row")
    zoom = np.abs(LAGS_S) <= 0.15
    peaks = {}

    control = control_data["cameras"]["BottomCamera"]
    z, n_licks = lick_triggered(
        control["timing"]["harp_time_raw"].to_numpy(), control["me"], control_data["licks"]
    )
    for row, mask in enumerate([slice(None), zoom]):
        axes[row, 0].plot(LAGS_S[mask], z[mask], color=COLOR_CONTROL, lw=1.5)
    axes[0, 0].set_title(f"Control: 818586_2026-01-15, bottom\nlog count = exposures, {n_licks} licks", fontsize=10)
    peaks["control"] = LAGS_S[np.argmax(z)]

    log = session_data["log"]
    for col, camera in enumerate(CAMERAS, start=1):
        cam = session_data["cameras"][camera]
        n = len(cam["timing"])
        extra = len(log) - n
        # Arrival-order pairing (the CSV): frames on the first n triggers; log[:n] == CSV Harp column
        assert np.abs(log[:n] - cam["timing"]["harp_time_raw"].to_numpy()).max() <= vtq.HARP_TICK_S
        for label, times, color in [
            (f"first {n:,} triggers (CSV)", log[:n], COLOR_FIRST),
            (f"last {n:,} triggers", log[extra:], COLOR_LAST),
        ]:
            z, n_licks = lick_triggered(times, cam["me"], session_data["licks"])
            peaks[(camera, label.split()[0])] = (LAGS_S[np.argmax(z)], LAGS_S[zoom][np.argmax(z[zoom])], z[zoom].max())
            axes[0, col].plot(LAGS_S, z, color=color, lw=1.5)
            axes[1, col].plot(LAGS_S[zoom], z[zoom], color=color, lw=1.5, label=f"frames on {label}")
        shift = extra * vtq.frame_interval(log)
        axes[0, col].axvline(-shift, color=ps.OKABE_ITO["black"], lw=0.8, ls=":")
        axes[0, col].annotate(
            f"−{extra:,} frames ({shift:.3f} s)", (-shift, 0.02), xycoords=("data", "axes fraction"),
            ha="right", va="bottom", fontsize=8, xytext=(-4, 0), textcoords="offset points",
        )
        axes[0, col].set_title(
            f"818586_2026-01-16, {camera}\nlog has {extra:,} more triggers than frames, {n_licks} licks",
            fontsize=10,
        )
        axes[1, col].legend(loc="center", fontsize=8, frameon=False)

    for ax in axes.flat:
        ax.axvline(0, color="#999999", lw=0.8, zorder=0)
        ps.style_ax(ax)
    for ax in axes[1]:
        ax.set_xlabel("Time from lick (s)")
    axes[0, 0].set_ylabel("Lick-triggered ME (z, ±4 s)")
    axes[1, 0].set_ylabel("Same, ±150 ms")
    fig.suptitle(
        "Motion at each lick lands 2.65 s early when frames take the first triggers, and at 0 s when they take the last",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, peaks


def figure_timeline(session_data):
    """Trigger log and saved frames at the start and end of the session (bottom camera)."""
    log = session_data["log"]
    n = len(session_data["cameras"]["BottomCamera"]["timing"])
    extra = len(log) - n
    t0 = session_data["start"]
    rows = [
        ("Trigger log (Event_94)", log[0], log[-1], ps.OKABE_ITO["black"]),
        (f"Frames on first {n:,} triggers (CSV)", log[0], log[n - 1], COLOR_FIRST),
        (f"Frames on last {n:,} triggers", log[extra], log[-1], COLOR_LAST),
    ]
    fig, (ax_start, ax_end) = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
    windows = [(ax_start, -0.5, 5.0), (ax_end, log[-1] - t0 - 5.0, log[-1] - t0 + 0.5)]
    for ax, lo, hi in windows:
        for y, (label, first, last, color) in enumerate(rows):
            ax.plot([first - t0, last - t0], [y, y], color=color, lw=6, solid_capstyle="butt")
        for t, name in [(session_data["start"], "start cameras\n(Write_78)"), (session_data["stop"], "stop cameras\n(Write_79)")]:
            if lo <= t - t0 <= hi:
                ax.axvline(t - t0, color="#999999", lw=0.8, ls="--")
                ax.annotate(name, (t - t0, -1.25), ha="center", va="top", fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(2.6, -1.35)
        ps.style_ax(ax)
    ax_start.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax_start.set_xlabel("Harp time from start cameras (s)")
    ax_end.set_xlabel("Harp time from start cameras (s)")
    ax_start.set_title("Session start", fontsize=10)
    ax_end.set_title("Session end", fontsize=10)
    fig.suptitle(
        f"818586_2026-01-16, bottom camera: {len(log):,} triggers, {n:,} saved frames — which {extra:,} triggers have no frame?",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main():
    """Load both sessions, draw the figures and print the numbers behind them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "leading_lost_frames")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    ps.apply_style()
    session_data = load_session(SESSION, args.cache)
    control_data = load_session(CONTROL, args.cache)

    fig, peaks = figure_lick_triggered(session_data, control_data)
    ps.save_fig(fig, "lick_triggered_me", fig_dir=FIG_DIR, formats=("png",))
    ps.save_fig(figure_timeline(session_data), "trigger_log_timeline", fig_dir=FIG_DIR, formats=("png",))

    log = session_data["log"]
    print(f"start cameras -> first trigger: {(log[0] - session_data['start']) * 1e3:.3f} ms")
    print(f"last trigger -> stop cameras:   {(session_data['stop'] - log[-1]) * 1e3:.3f} ms")
    print(f"control peak: {peaks.pop('control') * 1e3:+.0f} ms")
    for (camera, which), (wide, near, z_near) in peaks.items():
        print(f"{camera:15s} frames on {which:5s} triggers: peak {wide * 1e3:+.0f} ms (±4 s), "
              f"{near * 1e3:+.0f} ms within ±150 ms (z {z_near:.1f})")


if __name__ == "__main__":
    main()
