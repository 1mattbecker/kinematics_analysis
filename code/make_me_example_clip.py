"""
make_me_example_clip.py — frame-exact video clip of a stretch of a FIP session, both cameras.

Cuts the same window from the bottom and side camera videos, puts them side by side, labels the
task events (go cue, reward, no reward, lick) as they happen, and adds the bottom camera's motion
energy (ME) with a moving cursor. Used for the example window of ``men_01`` (Figure 1).

With ``--roi`` (e.g. ``whiskerpad_box``) the ROI is outlined on the side camera and the strip shows
three traces: bottom whole frame, side whole frame and the side ROI. ``--marks-csv`` adds labelled
times to the strip (``time`` in session s, ``label``, optional ``colour``), e.g. ``men_06`` §5's
whisker-pad ME events.
``--playback-fps`` below ``500 / step`` plays the clip in slow motion.

Timing, step by step (nothing assumes a frame rate):

1. Session time (s from the first go cue, the clock of every ``men_*``/``fip_*`` notebook) plus the
   first go cue's raw Harp time gives Harp time.
2. Harp time -> frame index through the ME table's corrected per-frame ``harp_time`` (video timing
   QC: dropped frames re-indexed), with the library's
   ``video_alignment.behavior_time_to_frame_index``. The old ``get_video_time`` (one constant
   offset) is wrong once frames are dropped and is not used.
3. Frame index -> a clip of exactly those frames, with the library's ``video_clips.cut_clip``: it
   seeks to the frame's presentation time read from the MP4's sample tables, since the MP4 timeline
   is not a uniform grid (encoder seams) and frame / fps can be off by several frames.

Check: ME recomputed from 2 s of decoded frames around the window's first go cue (mean absolute
difference of consecutive frames) is correlated with the table's ``me_clean`` at frame offsets
-10..+10; the best offset must be 0, or the script stops.

Environment: ``.venv-fip`` (the video-analysis library with ``video-clips``, ``aind-video-utils``
and PyAV), plus Pillow; ffmpeg >= 5.1 on the PATH.

Usage (from ``code/``)::

    python make_me_example_clip.py --session 809491_2025-10-23 --start 2342.9 --duration 60
    python make_me_example_clip.py --session 809491_2025-10-30 --start <s> --duration 30 \
        --roi whiskerpad_box --marks-csv <men_06 marks> --step 2 --playback-fps 50 --prefix men06_pad_events
"""

import argparse
import glob
import os
import subprocess
import tempfile
from fractions import Fraction

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis.video_clips import cut_clip  # noqa: E402
from aind_video_utils import read_mp4_frame_index  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import cue_me_maps as cmm  # noqa: E402  # video_url: S3 URL from the ME index, either raw layout
import fip_utils as fu  # noqa: E402  # data locations
import me_results as mr  # noqa: E402  # ME column names, ROI masks

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
EVENTS = [  # label, colour, how long the label stays lit (s)
    ("GO CUE", (0, 0, 0), 0.3),
    ("REWARD", (230, 159, 0), 0.5),
    ("NO REWARD", (110, 110, 110), 0.5),
    ("LICK", (0, 158, 115), 0.1),
]
TRACE_COLOURS = {"bottom": "0.25", "side": "#56B4E9", "roi": "#D55E00"}  # as men_06
ROI_RGB = (213, 94, 0)
MARK_COLOURS = ["#CC79A7", "#009E73", "#0072B2", "#E69F00"]  # marks labels without a colour, in order of appearance


def session_tables(ses_idx, data_roots):
    """Trials and lick/event table of a session from the CSV-curated FIP assets."""
    for root in data_roots:
        hits = glob.glob(os.path.join(root, "**", ses_idx, "df_trials.parquet"), recursive=True)
        if hits:
            d = os.path.dirname(hits[0])
            trials = pd.read_parquet(os.path.join(d, "df_trials.parquet")).sort_values("trial")
            events = pd.read_parquet(os.path.join(d, "df_events.parquet"), columns=["timestamps", "event"])
            return trials, events
    raise FileNotFoundError(f"no trial table for {ses_idx}")


def check_alignment(url, check_frame, me_clean, n=1000):
    """Best frame offset between ME recomputed from the video and the table's me_clean.

    Decodes ``n + 1`` full-resolution frames from ``check_frame`` (place it where the mouse moves,
    e.g. just before a go cue; a still stretch is mostly noise and cannot tell offsets apart).
    """
    seek = read_mp4_frame_index(url).presentation_seconds(check_frame)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-i", url, "-frames:v", str(n + 1),
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(n + 1, 540, 720).astype(np.float32)
    me = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))      # ME of frames check_frame+1 ...
    offsets = np.arange(-10, 11)
    r = [np.corrcoef(me, me_clean[check_frame + 1 + k:check_frame + 1 + k + len(me)])[0, 1] for k in offsets]
    return int(offsets[int(np.argmax(r))]), float(np.max(r)), float(np.sort(r)[-2])


def me_strip(traces, events_rel, marks_rel, duration, width_px, height_px):
    """Pre-rendered ME strip (RGB array) and the x pixel of t = 0 and t = duration.

    ``traces``: ``[(label, t_rel, z, colour)]``, stacked top to bottom when there are several.
    ``marks_rel``: ``{label: (times, colour)}``, drawn as triangles above the traces.
    """
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.05, 0.22, 0.93, 0.7])
    offset = 0.0
    for label, t_rel, z, colour in traces:
        ax.plot(t_rel, z + offset, color=colour, lw=0.6, label=label)
        offset -= np.nanpercentile(z, 99.5) - np.nanmin(z) + 0.5
    for (label, colour, _), times in zip(EVENTS, events_rel):
        y = ax.get_ylim()[1] if label != "LICK" else ax.get_ylim()[0]
        ax.plot(times, np.full(len(times), y), "|", color=np.array(colour) / 255, ms=8, mew=1.2)
    y_mark = ax.get_ylim()[1]
    for label, (times, colour) in marks_rel.items():
        ax.plot(times, np.full(len(times), y_mark), "v", color=colour, ms=5, label=label)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time in clip (s)", fontsize=9)
    ax.set_ylabel("ME (z)" if len(traces) > 1 else f"{traces[0][0]} ME (z)", fontsize=9)
    if len(traces) > 1:
        ax.set_yticks([])
        ax.legend(fontsize=7, ncol=len(traces) + len(marks_rel), loc="lower right", bbox_to_anchor=(1, 0.98),
                  frameon=False)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0 = ax.transData.transform((0, 0))[0]
    x1 = ax.transData.transform((duration, 0))[0]
    plt.close(fig)
    return img, x0, x1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--session", default="809491_2025-10-23")
    p.add_argument("--start", type=float, default=2342.8998720000964, help="session time, s from first go cue")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--step", type=int, default=10, help="keep every step-th 500-Hz frame (10 = 50 fps, real time)")
    p.add_argument("--me-root", default=fu.DATA_ROOT, help="folder holding the ME table and the FIP asset(s)")
    p.add_argument("--fip-roots", nargs="+", default=None, help="FIP asset roots (default: fu.fip_asset_roots)")
    p.add_argument("--out-dir", default=os.path.join(fu.FIG_ROOT, "men"))
    p.add_argument("--roi", default=None, help="side-camera ROI to outline and plot, e.g. whiskerpad_box")
    p.add_argument("--marks-csv", default=None, help="CSV of time (session s) and label to mark on the strip")
    p.add_argument("--playback-fps", type=float, default=None, help="output frame rate (default 500 / step: real time)")
    p.add_argument("--prefix", default="men01_example_clip", help="output file name before _<session>.mp4")
    args = p.parse_args()
    args.fip_roots = args.fip_roots or list(fu.fip_asset_roots(args.me_root).values())

    index = pd.read_csv(os.path.join(args.me_root, "fip_motion_energy_aligned", "index.csv"))
    index["ses_idx"] = index["session"].str.split("_").str[1:3].str.join("_")
    rows = index[(index.ses_idx == args.session) & (index.status == "ok")].set_index("camera")
    cameras = [c for c in ("BottomCamera", "SideCameraRight") if c in rows.index]
    trials, events = session_tables(args.session, args.fip_roots)
    first_cue = float(trials["goCue_start_time_raw"].dropna().iloc[0])          # Harp s
    t0, t1 = args.start, args.start + args.duration                             # session s

    responded = trials["animal_response"] != 2
    rewarded = responded & (trials["earned_reward"].fillna(0) > 0)
    outc = trials["reward_outcome_time_in_session"]
    licks = events.loc[events.event.isin(["left_lick_time", "right_lick_time"]), "timestamps"]
    ev_session = [trials["goCue_start_time_in_session"], outc[rewarded], outc[responded & ~rewarded], licks]
    ev_rel = []
    for e in ev_session:
        e = e.dropna().to_numpy(float)
        ev_rel.append(np.sort(e[(e >= t0) & (e < t1)] - t0))
    marks_rel = {}
    if args.marks_csv:
        marks = pd.read_csv(args.marks_csv)
        for k, (label, g) in enumerate(marks.groupby("label", sort=False)):
            m = g["time"].to_numpy(float)
            colour = g["colour"].iloc[0] if "colour" in g else MARK_COLOURS[k % len(MARK_COLOURS)]
            marks_rel[label] = (np.sort(m[(m >= t0) & (m < t1)] - t0), colour)

    work = tempfile.mkdtemp(prefix="me_clip_")
    cuts, frame_times, traces, roi_box = {}, {}, [], None
    for cam in cameras:
        r = rows.loc[cam]
        roi_id = None
        if args.roi and cam == "SideCameraRight":
            rois = str(r.rois).split(";")
            roi_id = next((c for c in (args.roi, f"{cam}_{args.roi}") if c in rois), None)
            if roi_id is None:
                raise ValueError(f"{cam} has no ROI {args.roi!r} (ROIs: {rois})")
            mask = mr.read_roi_masks(os.path.join(args.me_root, "fip_motion_energy_aligned", r.session,
                                                  cam + mr.ROI_MASKS_SUFFIX))[roi_id]
            ys, xs = np.nonzero(mask)
            roi_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        columns = ["harp_time", "me_clean"] + ([mr.me_column(roi_id)] if roi_id else [])
        table = pd.read_parquet(os.path.join(args.me_root, "fip_motion_energy_aligned", r.session, f"{cam}.parquet"),
                                columns=columns)
        harp = table["harp_time"].to_numpy()
        i0, i1 = va.behavior_time_to_frame_index(np.array([t0, t1]) + first_cue, harp)
        url = cmm.video_url(r)
        check_frame = int(va.behavior_time_to_frame_index(t0 + ev_rel[0][0] - 0.2 + first_cue, harp))
        offset, rmax, rnext = check_alignment(url, check_frame, table["me_clean"].to_numpy())
        print(f"{cam}: frames {i0}–{i1}; ME check from frame {check_frame}: best offset {offset} frames "
              f"(r = {rmax:.3f}, next best {rnext:.3f})")
        if offset != 0:
            raise RuntimeError(f"{cam}: cut is {offset} frames off the ME table; not writing the clip")
        out = os.path.join(work, f"{cam}.mp4")
        cut_clip(url, int(i0), int(i1 - i0), out)                                         # every frame, exact
        print(f"  cut frames {i0}–{i1} of {url}")
        cuts[cam] = out
        frame_times[cam] = harp[np.arange(i0, i1, args.step)] - first_cue            # session s per output frame
        cam_name = "bottom" if cam == "BottomCamera" else "side"
        sources = [(cam_name, "me_clean", cam_name)] if (cam == "BottomCamera" or args.roi) else []
        if roi_id:
            sources.append((f"side, {args.roi}", mr.me_column(roi_id), "roi"))
        m = slice(int(i0), int(i1))
        for label, col, colour in sources:
            me = table[col].to_numpy()[m]
            me = me[: len(me) // 5 * 5].reshape(-1, 5).mean(1)                           # 100 Hz
            me_t = harp[m][::5][: len(me)] - first_cue - t0
            traces.append((label, me_t, (me - np.nanmean(me)) / np.nanstd(me), TRACE_COLOURS[colour]))
    order = {"side, " + str(args.roi): 0, "side": 1, "bottom": 2}                       # ROI on top
    traces.sort(key=lambda tr: order.get(tr[0], 3))

    # Compose: cameras side by side, header, event labels, ME strip with cursor
    readers = [av.open(cuts[c]) for c in cameras]
    w, h = 720 * len(cameras), 540
    header, strip_h = 44, 200
    strip_h = strip_h + 60 * (len(traces) - 1)
    strip, sx0, sx1 = me_strip(traces, ev_rel, marks_rel, args.duration, w, strip_h)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    out_path = os.path.join(args.out_dir, f"{args.prefix}_{args.session}.mp4")
    os.makedirs(args.out_dir, exist_ok=True)
    container = av.open(out_path, "w")
    fps_out = args.playback_fps or 500 / args.step
    stream = container.add_stream("libx264", rate=Fraction(fps_out).limit_denominator(1000))
    stream.width, stream.height, stream.pix_fmt = w, header + h + strip_h, "yuv420p"
    stream.options = {"crf": "18"}
    times = frame_times["BottomCamera"] if "BottomCamera" in frame_times else frame_times[cameras[0]]
    decoded = zip(*(r.decode(video=0) for r in readers))
    for k, frames in enumerate(f for i, f in enumerate(decoded) if i % args.step == 0):
        canvas = Image.new("RGB", (w, header + h + strip_h), (255, 255, 255))
        for j, fr in enumerate(frames):
            canvas.paste(fr.to_image().convert("RGB"), (720 * j, header))
        draw = ImageDraw.Draw(canvas)
        t_ses = times[min(k, len(times) - 1)]
        draw.text((12, 10), f"{args.session}   t = {t_ses - t0:5.2f} s   (session {t_ses:.2f} s)", fill=(0, 0, 0), font=font)
        for j, cam in enumerate(cameras):
            label = cam.replace("Camera", " camera").replace("Right", "").lower()
            draw.rectangle([720 * j + 4, header + 4, 720 * j + 16 + draw.textlength(label, font=small), header + 30],
                           fill=(0, 0, 0))
            draw.text((720 * j + 10, header + 6), label, fill=(255, 255, 255), font=small)
            if cam == "SideCameraRight" and roi_box is not None:
                x0, y0, x1, y1 = roi_box
                draw.rectangle([720 * j + x0, header + y0, 720 * j + x1, header + y1], outline=ROI_RGB, width=2)
        x = w - 12
        for (label, colour, hold), times_rel in zip(EVENTS[::-1], ev_rel[::-1]):
            lit = np.any((t_ses - t0 >= times_rel) & (t_ses - t0 < times_rel + hold))
            tw = draw.textlength(label, font=font)
            x -= tw + 16
            draw.rectangle([x - 6, 6, x + tw + 6, header - 6], fill=colour if lit else (235, 235, 235))
            draw.text((x, 10), label, fill=(255, 255, 255) if lit else (180, 180, 180), font=font)
        canvas.paste(Image.fromarray(strip), (0, header + h))
        cx = sx0 + (t_ses - t0) / args.duration * (sx1 - sx0)
        draw.line([(cx, header + h + 8), (cx, header + h + strip_h - 40)], fill=(213, 94, 0), width=2)
        for packet in stream.encode(av.VideoFrame.from_image(canvas)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
