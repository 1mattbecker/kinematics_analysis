"""
make_me_noise_clips.py — bottom-camera clips of sessions with low and high ME floors, side by side.

For each session, 20 s around a go cue in the middle of the task (12 s before, 8 s after) is cut
frame-exact from the MP4 on public S3 with the library's ``video_clips.cut_clip`` (window from
``video_alignment.event_frame_ranges`` on the ME table's corrected ``harp_time``; all 500-Hz frames).
Each clip is checked against the table: ME recomputed from its frames must match ``me_clean`` at
frame offset 0 (``cue_me_maps.check_alignment``), or the script stops. Written to ``--out-dir``,
named by ``--name``:

- ``<name>_comparison.mp4``: the sessions side by side at 50 fps (real time); per session the frame,
  ``|frame - previous frame|`` (what ME averages; x8 gain), and the window's raw ME on a shared axis
  with a cursor.
- ``<name>_frames.png``: a frame per session and the mean ``|frame difference|`` over the quietest
  1 s of the window (lowest table ME), on one colour scale.
- ``<name>_brightness.png``: mean brightness at 500 Hz (whole frame and spout region), first 1 s of
  the clip, and the spectra of the session's ME and of the clip's frame brightness.
- ``<name>_frame_types.csv``: the encoder's frame-type pattern of each source MP4 (``ffprobe`` over
  the first 1,500 frames): the I-frame spacing and the first 20 frame types.

The clips are re-encoded (CRF 18), which adds a little difference noise of their own; the
sessions' differences are far larger. Used by ``men_01`` §6.

Environment: ``.venv-fip`` (video-analysis library with ``video-clips``, PyAV); ffmpeg/ffprobe on
the PATH.

Usage (from ``code/``)::

    python make_me_noise_clips.py --set across
    python make_me_noise_clips.py --set 818586
    python make_me_noise_clips.py --name mine --sessions 809491_2025-10-06:low 808054_2025-09-05:high
"""

import argparse
import os
import subprocess

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import signal  # noqa: E402

import cue_me_maps as cmm  # noqa: E402
import fip_utils as fu  # noqa: E402
from make_me_example_clip import FONT, session_tables  # noqa: E402

#: Session sets: across sessions (lowest, median and highest floor) and 818586 in January (two
#: days with a low floor, two with a high one, same rig).
SETS = {
    "across": [("809491_2025-10-06", "low"), ("818586_2025-12-24", "median"), ("808054_2025-09-05", "high")],
    "818586": [("818586_2026-01-14", "low"), ("818586_2026-01-22", "low"),
               ("818586_2026-01-15", "high"), ("818586_2026-01-27", "high")],
}
CAMERA = "BottomCamera"
FS = 500.0
BEFORE, AFTER = 12.0, 8.0     # s around the go cue
STEP = 10                     # keep every 10th frame: 50 fps, real time
GAIN = 8.0                    # difference image gain
SPOUT_ROI = (0.35, 0.75, 0.6, 1.0)   # fraction of height (top, bottom) and width (left, right)


def frame_types(url, n=1500):
    """I-frame spacing and the first 20 frame types of an MP4 (``ffprobe``)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+#{n}",
                          "-show_entries", "frame=pict_type", "-of", "csv=p=0", url],
                         check=True, capture_output=True, text=True).stdout
    types = "".join(t.strip().strip(",") for t in out.splitlines() if t.strip())
    i_pos = [k for k, t in enumerate(types) if t == "I"]
    spacing = np.unique(np.diff(i_pos)).tolist() if len(i_pos) > 1 else []
    return dict(first_20=types[:20], i_frame_spacing=" ".join(map(str, spacing)), n_read=len(types))


def decode(clip):
    """Every STEP-th frame and |difference|, ME per difference, brightness per frame (whole, spout)."""
    raw, diff, me, bright, bright_roi = [], [], [], [], []
    prev = None
    with av.open(clip) as c:
        for k, fr in enumerate(c.decode(video=0)):
            g = fr.to_ndarray(format="gray").astype(np.int16)
            h, w = g.shape
            bright.append(g.mean())
            bright_roi.append(g[int(SPOUT_ROI[0] * h):int(SPOUT_ROI[1] * h), int(SPOUT_ROI[2] * w):].mean())
            if prev is not None:
                d = np.abs(g - prev)
                me.append(d.mean())
            if k % STEP == 0:
                raw.append(g.astype(np.uint8))
                d = np.abs(g - prev) if prev is not None else np.zeros_like(g)
                diff.append(np.clip(d * GAIN, 0, 255).astype(np.uint8))
            prev = g
    return raw, diff, np.array(me), np.array(bright), np.array(bright_roi)


def quiet_difference(clip, table_me, n=500):
    """Mean |frame difference| image over the quietest n frames (lowest table ME) and its last frame."""
    end = int(np.nanargmin(pd.Series(table_me).rolling(n).mean().to_numpy()))
    acc, prev = None, None
    with av.open(clip) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k < end - n:
                continue
            if k > end:
                break
            g = fr.to_ndarray(format="gray").astype(np.float32)
            if prev is not None:
                acc = np.abs(g - prev) if acc is None else acc + np.abs(g - prev)
            prev = g
    return acc / (n - 1), prev, end


def me_strip(p, ymax, width, height):
    """The window's raw ME (100 Hz) as an image, and the x pixels of the window's ends."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    me100 = p["me"][: len(p["me"]) // 5 * 5].reshape(-1, 5).mean(1)
    ax.plot(p["t"][::5][: len(me100)], me100, color="0.2", lw=0.6)
    ax.axhline(np.median(p["me"]), color="#CC79A7", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.5)
    ax.set(ylim=(0, ymax), xlim=(-BEFORE, AFTER))
    ax.set_xlabel("s from go cue", fontsize=7)
    ax.set_ylabel("raw ME", fontsize=7)
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.27, top=0.95)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0, x1 = ax.transData.transform([(-BEFORE, 0), (AFTER, 0)])[:, 0]
    plt.close(fig)
    return img, x0, x1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", choices=sorted(SETS), default="across")
    ap.add_argument("--sessions", nargs="+", help="session:label ... (overrides --set)")
    ap.add_argument("--name", help="output prefix (default: the set's name)")
    ap.add_argument("--me-root", default=os.environ.get("ME_DATA_ROOT", "../../data"))
    ap.add_argument("--fip-roots", nargs="+", default=["../../data/results-a278f830-9bf2-48e8-9928-a2ff78e4818f",
                                                      "../../data/results-ddcccb0f-f18f-44a6-a2b1-caba680d28a1"])
    ap.add_argument("--out-dir", default="../data/figures/men/noise_clips")
    args = ap.parse_args()
    chosen = [tuple(s.split(":", 1)) for s in args.sessions] if args.sessions else SETS[args.set]
    name = args.name or args.set
    os.makedirs(args.out_dir, exist_ok=True)
    index = fu.load_me_index(args.me_root)

    per, ftypes = [], []
    for ses, label in chosen:
        row = index[(index.ses_idx == ses) & (index.camera == CAMERA)].iloc[0]
        harp, me_clean = cmm.load_frame_table(args.me_root, row.session, CAMERA)
        trials, _ = session_tables(ses, args.fip_roots)
        cues = trials["goCue_start_time_raw"].dropna().to_numpy()
        cue = cues[len(cues) // 2]
        win = va.event_frame_ranges(np.array([cue]), harp, before=BEFORE, after=AFTER).iloc[0]
        start, n = int(win["start_frame"]), int(win["n_frames"])
        url = cmm.video_url(row)
        clip = os.path.join(args.out_dir, f"{ses}_bottom_f{start:07d}.mp4")
        if not os.path.exists(clip):
            vc.cut_clip(url, start, n, clip)
        raw, diff, clip_me, bright, bright_roi = decode(clip)
        shifts, r = cmm.check_alignment(clip_me[None, :], me_clean, [start])
        best = int(shifts[np.argmax(r)])
        print(f"{ses}: frames {start}–{start + n}; clip ME vs table best offset {best} (r = {r.max():.3f})")
        if best != 0:
            raise RuntimeError(f"{ses}: clip is {best} frames off the ME table")
        table_me = me_clean[start:start + n]
        q_diff, q_frame, _ = quiet_difference(clip, table_me)
        f, P = signal.welch(np.nan_to_num(me_clean - np.nanmean(me_clean)), fs=FS, nperseg=4096)
        fb, Pb = signal.welch(bright - bright.mean(), fs=FS, nperseg=1024)
        per.append(dict(ses=ses, label=label, raw=raw, diff=diff, me=table_me, t=harp[start:start + n] - cue,
                        q_diff=q_diff, q_frame=q_frame, bright=bright, bright_roi=bright_roi,
                        f=f, P=P, fb=fb, Pb=Pb))
        ftypes.append(dict(session=ses, label=label, **frame_types(url)))
    pd.DataFrame(ftypes).to_csv(os.path.join(args.out_dir, f"{name}_frame_types.csv"), index=False)

    # frames and quiet difference images, one colour scale
    vmax = max(np.percentile(p["q_diff"], 99.5) for p in per)
    fig, axes = plt.subplots(2, len(per), figsize=(5 * len(per), 8.2), squeeze=False)
    for j, p in enumerate(per):
        axes[0, j].imshow(p["q_frame"], cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(f"{p['label']}: {p['ses']}", loc="left", fontsize=10)
        im = axes[1, j].imshow(p["q_diff"], cmap="magma", vmin=0, vmax=vmax)
        axes[1, j].set_title(f"mean |frame difference|, quietest 1 s: {p['q_diff'].mean():.3f}", loc="left", fontsize=9)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="grey levels")
    fig.savefig(os.path.join(args.out_dir, f"{name}_frames.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # brightness at 500 Hz and spectra
    fig, axes = plt.subplots(2, len(per), figsize=(5 * len(per), 7), squeeze=False)
    for j, p in enumerate(per):
        t = np.arange(500) / FS
        ax = axes[0, j]
        ax.plot(t, p["bright_roi"][:500] - p["bright_roi"][:500].mean(), lw=0.6, label="spout region")
        ax.plot(t, p["bright"][:500] - p["bright"][:500].mean(), lw=0.6, label="whole frame")
        ax.set_title(f"{p['label']}: {p['ses']}\nmean brightness, first 1 s (500 Hz)", loc="left", fontsize=9)
        ax.set_xlabel("s")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        ax.semilogy(p["f"], p["P"] / p["P"].max(), lw=0.8, label="ME, whole session")
        ax.semilogy(p["fb"], p["Pb"] / p["Pb"].max(), lw=0.8, label="frame brightness, clip")
        ax.set_xlabel("Hz")
        ax.set_title("power (normalised)", loc="left", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{name}_brightness.png"), dpi=110)
    plt.close(fig)

    # comparison video
    h0, w0 = per[0]["raw"][0].shape
    cw, ch = 400, int(round(h0 * 400 / w0))
    header, strip_h = 54, 170
    W, H = cw * len(per), header + 2 * ch + strip_h
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 14)
    ymax = max(np.nanpercentile(p["me"], 99.5) for p in per) * 1.05
    strips = [me_strip(p, ymax, cw, strip_h) for p in per]
    out = os.path.join(args.out_dir, f"{name}_comparison.mp4")
    container = av.open(out, "w")
    stream = container.add_stream("libx264", rate=int(FS) // STEP)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "18"}
    for k in range(min(len(p["raw"]) for p in per)):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for j, p in enumerate(per):
            x = cw * j
            canvas.paste(Image.fromarray(p["raw"][k]).resize((cw, ch)).convert("RGB"), (x, header))
            canvas.paste(Image.fromarray(p["diff"][k]).resize((cw, ch)).convert("RGB"), (x, header + ch))
            img, sx0, sx1 = strips[j]
            canvas.paste(Image.fromarray(img), (x, header + 2 * ch))
            tk = p["t"][min(k * STEP, len(p["t"]) - 1)]
            cx = x + sx0 + (tk + BEFORE) / (BEFORE + AFTER) * (sx1 - sx0)
            draw.line([(cx, header + 2 * ch + 6), (cx, header + 2 * ch + strip_h - 45)], fill=(213, 94, 0), width=2)
            draw.text((x + 8, 6), p["label"], fill=(0, 0, 0), font=font)
            draw.text((x + 8, 30), f"{p['ses']}   t = {tk:+5.2f} s", fill=(90, 90, 90), font=small)
            draw.rectangle([x + 4, header + ch + 4, x + 176, header + ch + 24], fill=(0, 0, 0))
            draw.text((x + 8, header + ch + 6), f"|frame diff| x{GAIN:.0f}", fill=(255, 255, 255), font=small)
        for pkt in stream.encode(av.VideoFrame.from_image(canvas)):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
