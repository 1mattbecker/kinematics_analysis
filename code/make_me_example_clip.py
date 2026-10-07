"""
make_me_example_clip.py — frame-exact video clip of a stretch of a FIP session, both cameras.

Cuts the same window from the bottom and side camera videos, puts them side by side, labels the
task events (go cue, reward, no reward, lick) as they happen, and adds the bottom camera's motion
energy (ME) with a moving cursor. Used for the example window of ``men_01`` (Figure 1).

Timing, step by step (nothing assumes a frame rate):

1. Session time (s from the first go cue, the clock of every ``men_*``/``fip_*`` notebook) plus the
   first go cue's raw Harp time gives Harp time.
2. Harp time -> frame index through the ME table's corrected per-frame ``harp_time`` (video timing
   QC: dropped frames re-indexed), with the library's
   ``video_alignment.behavior_time_to_frame_index``. The old ``get_video_time`` (one constant
   offset) is wrong once frames are dropped and is not used.
3. Frame index -> the frame's presentation time in the MP4, read from the file's sample tables with
   ``aind_video_utils.read_mp4_frame_index(...).presentation_seconds``, which ffmpeg then seeks to
   exactly (``-ss`` before ``-i``, accurate seek). The MP4 timeline is not a uniform grid (encoder
   seams), so frame / fps can be off by several frames.

Check: ME recomputed from 2 s of decoded frames around the window's first go cue (mean absolute
difference of consecutive frames) is correlated with the table's ``me_clean`` at frame offsets
-10..+10; the best offset must be 0, or the script stops.

Environment: the video-analysis library with its ``video-qc`` extra (brings ``aind-video-utils``
and PyAV), plus pandas/pyarrow/matplotlib/Pillow; ffmpeg on the PATH. Not ``.venv-fip``.

Usage (from ``code/``)::

    python make_me_example_clip.py --session 809491_2025-10-23 --start 2342.9 --duration 60
"""

import argparse
import glob
import os
import subprocess
import tempfile

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va  # noqa: E402
from aind_video_utils import read_mp4_frame_index  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

VIDEO_URL = "https://aind-open-data.s3.us-west-2.amazonaws.com/{raw}/behavior-videos/{camera}.mp4"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
EVENTS = [  # label, colour, how long the label stays lit (s)
    ("GO CUE", (0, 0, 0), 0.3),
    ("REWARD", (230, 159, 0), 0.5),
    ("NO REWARD", (110, 110, 110), 0.5),
    ("LICK", (0, 158, 115), 0.1),
]


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


def cut(url, first_frame, n_frames, step, out_path):
    """Frames first_frame, first_frame + step, ... (n_frames // step of them), exact, to out_path."""
    seek = read_mp4_frame_index(url).presentation_seconds(first_frame)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.6f}", "-i", url,
                    "-vf", f"select=not(mod(n\\,{step})),setpts=N/{500 // step}/TB",
                    "-frames:v", str(n_frames // step), "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", out_path], check=True)
    return seek


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


def me_strip(t_rel, me_z, events_rel, duration, width_px, height_px):
    """Pre-rendered ME strip (RGB array) and the x pixel of t = 0 and t = duration."""
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.05, 0.22, 0.93, 0.7])
    ax.plot(t_rel, me_z, color="0.2", lw=0.6)
    for (label, colour, _), times in zip(EVENTS, events_rel):
        y = ax.get_ylim()[1] if label != "LICK" else ax.get_ylim()[0]
        ax.plot(times, np.full(len(times), y), "|", color=np.array(colour) / 255, ms=8, mew=1.2)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time in clip (s)", fontsize=9)
    ax.set_ylabel("bottom ME (z)", fontsize=9)
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
    p.add_argument("--me-root", default=os.environ.get("ME_DATA_ROOT", "../../data"))
    p.add_argument("--fip-roots", nargs="+", default=["../../data/results-a278f830-9bf2-48e8-9928-a2ff78e4818f",
                                                      "../../data/results-ddcccb0f-f18f-44a6-a2b1-caba680d28a1"])
    p.add_argument("--out-dir", default="../data/figures/men")
    args = p.parse_args()

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

    work = tempfile.mkdtemp(prefix="me_clip_")
    cuts, frame_times = {}, {}
    for cam in cameras:
        r = rows.loc[cam]
        table = pd.read_parquet(os.path.join(args.me_root, "fip_motion_energy_aligned", r.session, f"{cam}.parquet"),
                                columns=["harp_time", "me_clean"])
        harp = table["harp_time"].to_numpy()
        i0, i1 = va.behavior_time_to_frame_index(np.array([t0, t1]) + first_cue, harp)
        url = VIDEO_URL.format(raw=r.session, camera=r.source_camera)
        check_frame = int(va.behavior_time_to_frame_index(t0 + ev_rel[0][0] - 0.2 + first_cue, harp))
        offset, rmax, rnext = check_alignment(url, check_frame, table["me_clean"].to_numpy())
        print(f"{cam}: frames {i0}–{i1}; ME check from frame {check_frame}: best offset {offset} frames "
              f"(r = {rmax:.3f}, next best {rnext:.3f})")
        if offset != 0:
            raise RuntimeError(f"{cam}: cut is {offset} frames off the ME table; not writing the clip")
        out = os.path.join(work, f"{cam}.mp4")
        seek = cut(url, int(i0), int(i1 - i0), args.step, out)
        print(f"  seek {seek:.6f} s into {url}")
        cuts[cam] = out
        frame_times[cam] = harp[np.arange(i0, i1, args.step)] - first_cue            # session s per output frame
        if cam == "BottomCamera":
            m = slice(int(i0), int(i1))
            me = table["me_clean"].to_numpy()[m]
            me = me[: len(me) // 5 * 5].reshape(-1, 5).mean(1)                           # 100 Hz
            me_t = harp[m][::5][: len(me)] - first_cue - t0
            me_z = (me - np.nanmean(me)) / np.nanstd(me)

    # Compose: cameras side by side, header, event labels, ME strip with cursor
    readers = [av.open(cuts[c]) for c in cameras]
    w, h = 720 * len(cameras), 540
    header, strip_h = 44, 200
    strip, sx0, sx1 = me_strip(me_t, me_z, ev_rel, args.duration, w, strip_h)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    out_path = os.path.join(args.out_dir, f"men01_example_clip_{args.session}.mp4")
    os.makedirs(args.out_dir, exist_ok=True)
    container = av.open(out_path, "w")
    stream = container.add_stream("libx264", rate=500 // args.step)
    stream.width, stream.height, stream.pix_fmt = w, header + h + strip_h, "yuv420p"
    stream.options = {"crf": "18"}
    times = frame_times["BottomCamera"] if "BottomCamera" in frame_times else frame_times[cameras[0]]
    for k, frames in enumerate(zip(*(r.decode(video=0) for r in readers))):
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
