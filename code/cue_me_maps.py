"""
cue_me_maps.py — where in the image the motion energy (ME) right after the go cue comes from.

The session-average ME map (``aind-motion-energy``'s ``_motion_energy_map.npy``) cannot separate one
moment after the cue from another. :func:`cue_maps` cuts a frame-exact clip around each of N go cues
and averages the per-pixel absolute frame difference at every lag, so each 2 ms step after the cue
gets its own map. Used by ``men_04``; also runs as a script for one session.

Steps in :func:`cue_maps`:

1. Go-cue Harp times -> frame windows with the library's ``video_alignment.event_frame_ranges`` on
   the ME table's corrected per-frame ``harp_time``. Windows that span dropped frames (fewer frames
   than the rest) are left out.
2. Clips cut from the session's MP4 on public S3 with ``video_clips.cut_clips`` (frame-exact seek
   from the MP4 sample tables; clips re-encoded near-losslessly, CRF 18; already-cut clips reused).
3. Each clip decoded (luma plane) with PyAV; ``|frame[k+1] - frame[k]|`` summed per lag over cues.
   The lag of a difference is the Harp time of its later frame minus the go cue, as in the ME table.
4. Check: the mean of each difference image is correlated with the table's ``me_clean`` at frame
   offsets -5..+5 (pooled over clips); the best offset must be 0, or it raises.

Then :func:`smooth_encoder_cycle` (removes the re-encoding cycle of the clips), :func:`region_masks`
(the pixels that move most after the cue, split by position), :func:`highpass` and
:func:`highpass_peaks` (15 Hz zero-phase Butterworth, and its peaks) and :func:`band_power_map`
(per-pixel power of the high-passed part).

Environment: the video-analysis library (>= v0.3.0) with its ``video-clips`` extra, plus PyAV; ffmpeg
on the PATH. Both are installed in ``.venv-fip``.

Usage (from ``code/``)::

    python cue_me_maps.py --session 809487_2025-10-10 --n-cues 100
"""

import argparse
import glob
import json
import os

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va
from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

OPEN_DATA_URL = "https://aind-open-data.s3.us-west-2.amazonaws.com"
CO_DATA_ROOT = "/root/capsule/data/"
ME_TABLE_NAME = "fip_motion_energy_aligned"
# Lag windows (ms after the cue) drawn as maps, minus the pre-cue map
MAP_WINDOWS_MS = [(10, 16), (18, 30), (34, 46), (50, 62), (66, 80), (100, 150)]
REGION_COLORS = {"left edge": "#009E73", "upper half": "#D55E00", "lower half": "#0072B2",
                 "rest of frame": "#999999"}


def video_url(index_row):
    """Public S3 URL of a camera's MP4, from a row of the ME table's ``index.csv``.

    Built from ``source_video`` (the path the table was built from, under the capsule's data
    folder), so it is right for both raw layouts: ``behavior-videos/bottom_camera.mp4`` (flat) and
    ``behavior-videos/BottomCamera/video.mp4`` (nested).
    """
    path = index_row["source_video"]
    if not path.startswith(CO_DATA_ROOT):
        raise ValueError(f"source_video not under {CO_DATA_ROOT}: {path}")
    return f"{OPEN_DATA_URL}/{path[len(CO_DATA_ROOT):]}"


def load_frame_table(me_root, raw_session, camera):
    """Per-frame ``harp_time`` (corrected) and ``me_clean`` of one camera from the ME table."""
    path = os.path.join(me_root, ME_TABLE_NAME, raw_session, f"{camera}.parquet")
    table = pd.read_parquet(path, columns=["harp_time", "me_clean"])
    return table["harp_time"].to_numpy(), table["me_clean"].to_numpy()


def clip_luma(path):
    """Luma planes of every frame of a clip, (n, H, W) int16."""
    with av.open(str(path)) as container:
        frames = []
        for frame in container.decode(video=0):
            plane = frame.planes[0]
            y = np.frombuffer(memoryview(plane), np.uint8).reshape(plane.height, plane.line_size)
            frames.append(y[:, : frame.width].astype(np.int16))
    return np.stack(frames)


def check_alignment(clip_me, table_me, starts, max_shift=5):
    """Pooled correlation of clip ME with the table's ``me_clean`` at frame offsets ±max_shift.

    Difference ``k`` of a clip starting at source frame ``s`` is frame ``s+k+1`` minus frame
    ``s+k``: the table's row ``s+k+1`` (the table's ME is padded with a leading 0).

    Returns
    -------
    shifts, r : numpy.ndarray
    """
    shifts = np.arange(-max_shift, max_shift + 1)
    n_diff = clip_me.shape[1]
    r = []
    for d in shifts:
        a, b = [], []
        for x, s in zip(clip_me, starts):
            idx = s + 1 + np.arange(n_diff) + d
            if idx[0] < 0 or idx[-1] >= len(table_me):
                continue
            a.append(x - x.mean())
            b.append(table_me[idx] - table_me[idx].mean())
        a, b = np.concatenate(a), np.concatenate(b)
        r.append(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return shifts, np.array(r)


def cue_maps(url, harp_time, me_clean, cue_times, n_cues=100, before=0.06, after=0.16, seed=0,
             clip_dir="clips", prefix="clip", cache_path=None):
    """Mean per-pixel |frame difference| at each lag around a random sample of go cues.

    Parameters
    ----------
    url : str
        The camera's MP4 (local path or URL), frame-aligned with ``harp_time``.
    harp_time, me_clean : numpy.ndarray
        Per-frame corrected Harp time and ME from the ME table (:func:`load_frame_table`).
    cue_times : array-like
        Go-cue times on the Harp clock (``goCue_start_time_raw``).
    n_cues : int
        Cues sampled (without replacement, ``seed``) from those whose window has every frame.
    before, after : float
        Seconds before and after each cue.
    clip_dir, prefix : str
        Where the clips go and how they are named (``video_clips.cut_clips``); existing ones are reused.
    cache_path : str, optional
        ``.npz`` holding the result; reused when its parameters match.

    Returns
    -------
    dict
        ``lags_s`` (n_lags,), lag of each difference from the cue; ``maps`` (n_lags, H, W) mean
        |difference| (luma units); ``reference`` (H, W) mean frame; ``clip_me`` (n_cues, n_lags)
        per-cue ME of the clip; ``cues``, ``start_frames``; ``align_shifts``, ``align_r``.
    """
    params = dict(url=str(url), n_cues=int(n_cues), before=float(before), after=float(after),
                  seed=int(seed))
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as f:
            if json.loads(str(f["params"])) == params:
                return {k: f[k] for k in f.files if k != "params"}

    ranges = va.event_frame_ranges(np.asarray(cue_times, float), harp_time, before, after)
    n_full = int(ranges.loc[ranges.in_video, "n_frames"].mode()[0])
    ranges = ranges[ranges.in_video & (ranges.n_frames == n_full)]       # no drops inside a window
    ranges = ranges.sample(min(n_cues, len(ranges)), random_state=seed).sort_values("start_frame")
    clips = vc.cut_clips(url, ranges, clip_dir, prefix)
    clips = clips[clips.clip_path.notna()]

    acc, ref_acc, clip_me, starts, cues = None, None, [], [], []
    for row in clips.itertuples():
        y = clip_luma(row.clip_path)
        if len(y) != n_full:
            continue
        d = np.abs(np.diff(y, axis=0)).astype(np.float32)
        acc = d if acc is None else acc + d
        ref_acc = y.mean(0) if ref_acc is None else ref_acc + y.mean(0)
        clip_me.append(d.mean(axis=(1, 2)))
        starts.append(int(row.start_frame))
        cues.append(row.event_time)
    clip_me, starts, cues = np.array(clip_me), np.array(starts), np.array(cues)
    lags_s = np.mean([harp_time[s + 1: s + n_full] - c for s, c in zip(starts, cues)], axis=0)
    shifts, r = check_alignment(clip_me, me_clean, starts)
    if shifts[np.argmax(r)] != 0:
        raise RuntimeError(f"{url}: clips are {shifts[np.argmax(r)]} frames off the ME table")

    out = dict(lags_s=lags_s, maps=(acc / len(clip_me)).astype(np.float32),
               reference=(ref_acc / len(clip_me)).astype(np.float32), clip_me=clip_me, cues=cues,
               start_frames=starts, align_shifts=shifts, align_r=r)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, params=json.dumps(params), **out)
    return out


def smooth_encoder_cycle(maps, n_frames=4):
    """Mean over ``n_frames`` consecutive lags, to remove the clips' re-encoding cycle.

    ``video_clips`` re-encodes every clip with x264 (CRF 18, default B-frames), and every clip starts
    on the same frame type, so clip ME carries a cycle of 4 frames (10–25% deep) that is the same in
    every clip and survives averaging over cues. A running mean over 4 frames cancels any 4- or
    2-frame pattern; at 500 Hz it keeps ~91% of the amplitude at 30 Hz and ~80% at 45 Hz. The edges
    use the end value.
    """
    return uniform_filter1d(np.asarray(maps, np.float32), n_frames, axis=0, mode="nearest")


def excess_maps(maps, lags_s, windows_ms=MAP_WINDOWS_MS):
    """Mean |difference| in each lag window minus the pre-cue mean, (n_windows, H, W)."""
    lags_ms = lags_s * 1000
    base = maps[lags_ms < 0].mean(0)
    return np.stack([maps[(lags_ms >= a) & (lags_ms < b)].mean(0) - base for a, b in windows_ms])


def region_masks(maps, lags_s, window_ms=(10, 80), top_pct=3.0, left_frac=0.12, blur_px=2.0):
    """Pixels that move most after the cue, split by where they are in the frame.

    The top ``top_pct`` % of pixels by mean excess |difference| over ``window_ms`` (minus before the
    cue, blurred by a Gaussian of ``blur_px`` so a region is an area, not single pixels), split into
    the left edge of the frame (x < ``left_frac`` of the width), the upper half and
    the lower half; ``rest of frame`` is every other pixel. In the bottom camera the left edge is the
    face/body edge and the two halves hold the whiskers on either side of the snout.

    Returns
    -------
    dict
        ``{name: bool mask (H, W)}``, in :data:`REGION_COLORS` order.
    """
    excess = gaussian_filter(excess_maps(maps, lags_s, [window_ms])[0], blur_px)
    h, w = excess.shape
    hot = excess > np.percentile(excess, 100 - top_pct)
    yy, xx = np.mgrid[:h, :w]
    return {"left edge": hot & (xx < left_frac * w),
            "upper half": hot & (xx >= left_frac * w) & (yy < h / 2),
            "lower half": hot & (xx >= left_frac * w) & (yy >= h / 2),
            "rest of frame": ~hot}


def highpass(x, dt, cutoff_hz=15.0, order=2, axis=0):
    """Zero-phase Butterworth high-pass (``filtfilt``): the fast part of a cue-triggered trace.

    Run forwards and backwards, so peaks keep their times, but a fast rise is spread over
    ~±25 ms: before a sharp onset the output dips (and bulges before that) although the input
    does not. Read the output only from ~20 ms after the cue.
    """
    b, a = butter(order, cutoff_hz, "high", fs=1 / dt)
    return filtfilt(b, a, np.asarray(x, float), axis=axis)


def highpass_peaks(lags_s, x, cutoff_hz=15.0, window_ms=(20, 110), rel_prominence=0.1):
    """Peaks of the high-passed trace within ``window_ms`` after the cue.

    Returns
    -------
    times_ms, prominence : numpy.ndarray
        Peak times, and each peak's prominence as a fraction of the high-passed trace's range in
        the window. A peak counts if that fraction exceeds ``rel_prominence``.
    """
    lags_ms = np.asarray(lags_s) * 1000
    hp = highpass(x, float(np.median(np.diff(lags_s))), cutoff_hz)
    w = (lags_ms >= window_ms[0]) & (lags_ms <= window_ms[1])
    span = np.ptp(hp[w])
    idx, props = find_peaks(hp[w], prominence=rel_prominence * span)
    return lags_ms[w][idx], props["prominences"] / span


def band_power_map(maps, lags_s, window_ms=(20, 110), band_hz=(22, 45), cutoff_hz=15.0, nfft=128):
    """Per-pixel power in ``band_hz`` of the high-passed post-cue |difference| (:func:`highpass`
    over the whole clip, then ``window_ms``), (H, W)."""
    lags_ms = lags_s * 1000
    dt = float(np.median(np.diff(lags_s)))
    win = (lags_ms >= window_ms[0]) & (lags_ms <= window_ms[1])
    hp = highpass(maps.reshape(len(lags_s), -1), dt, cutoff_hz, axis=0)
    fast = hp[win]
    freqs = np.fft.rfftfreq(nfft, dt)
    power = np.abs(np.fft.rfft(fast, nfft, axis=0)) ** 2
    return power[(freqs >= band_hz[0]) & (freqs <= band_hz[1])].sum(0).reshape(maps.shape[1:])


def draw_regions(ax, reference, masks, alpha=0.45, left_frac=0.12):
    """Mean frame with each region filled in its colour, and the region boundaries as lines."""
    ax.imshow(reference, cmap="gray")
    h, w = reference.shape
    for name, m in masks.items():
        if name == "rest of frame":
            continue
        rgba = np.zeros((h, w, 4))
        rgba[m] = (*plt.matplotlib.colors.to_rgb(REGION_COLORS[name]), alpha)
        ax.imshow(rgba)
    ax.axvline(left_frac * w, color="w", ls=":", lw=0.8)
    ax.plot([left_frac * w, w - 1], [h / 2, h / 2], color="w", ls=":", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Script ────────────────────────────────────────────────────────────────────

def _load_trials(ses_idx, data_roots):
    for root in data_roots:
        hits = glob.glob(os.path.join(root, "**", ses_idx, "df_trials.parquet"), recursive=True)
        if hits:
            return pd.read_parquet(hits[0]).sort_values("trial")
    raise FileNotFoundError(f"no trial table for {ses_idx}")


def _plot(res, title, path):
    lags_ms = res["lags_s"] * 1000
    exc = excess_maps(res["maps"], res["lags_s"])
    vmax = np.nanpercentile(np.abs(exc), 99.8)
    masks = region_masks(res["maps"], res["lags_s"])
    fig, axes = plt.subplots(2, 5, figsize=(21, 7.6))
    axes = axes.ravel()
    draw_regions(axes[0], res["reference"], masks)
    axes[0].set_title("mean frame, regions")
    for ax, (a, b), e in zip(axes[1:], MAP_WINDOWS_MS, exc):
        ax.imshow(e, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{a}–{b} ms minus before")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[7]
    ax.imshow(band_power_map(res["maps"], res["lags_s"]), cmap="magma")
    ax.set_title("power 22–45 Hz, high-passed, 20–110 ms")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[8]
    for name, m in masks.items():
        trace = res["maps"][:, m].mean(1)
        ax.plot(lags_ms, trace - trace[lags_ms < 0].mean(), color=REGION_COLORS[name], label=name)
    ax.set(xlabel="time from go cue (ms)", ylabel="|diff| minus before", title="ME by region")
    ax.legend(fontsize=7)
    ax = axes[9]
    for x in res["clip_me"][:15]:
        ax.plot(lags_ms, x, color="0.75", lw=0.6)
    ax.plot(lags_ms, res["clip_me"].mean(0), "k", lw=1.5, label=f"mean of {len(res['clip_me'])} cues")
    ax.set(xlabel="time from go cue (ms)", ylabel="mean |diff| (luma)", title="ME of the clips")
    ax.legend(fontsize=7)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--session", default="809487_2025-10-10")
    p.add_argument("--camera", default="BottomCamera")
    p.add_argument("--n-cues", type=int, default=100)
    p.add_argument("--before", type=float, default=0.06, help="s before the cue")
    p.add_argument("--after", type=float, default=0.16, help="s after the cue")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--me-root", default=os.environ.get("ME_DATA_ROOT", "../../data"))
    p.add_argument("--fip-roots", nargs="+", default=["../../data/results-a278f830-9bf2-48e8-9928-a2ff78e4818f",
                                                      "../../data/results-ddcccb0f-f18f-44a6-a2b1-caba680d28a1"])
    p.add_argument("--clip-dir", default="../../data/cue_me_clips")
    p.add_argument("--out-dir", default="../data/figures/men")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    index = pd.read_csv(os.path.join(args.me_root, ME_TABLE_NAME, "index.csv"))
    index["ses_idx"] = index["session"].str.split("_").str[1:3].str.join("_")
    rows = index[(index.ses_idx == args.session) & (index.camera == args.camera) & (index.status == "ok")]
    if rows.empty:
        raise SystemExit(f"{args.session} {args.camera}: not in the ME table with status ok")
    row = rows.iloc[0]
    harp, me = load_frame_table(args.me_root, row["session"], args.camera)
    cues = _load_trials(args.session, args.fip_roots)["goCue_start_time_raw"].dropna().to_numpy()
    stem = f"cue_me_maps_{args.session}_{args.camera}"
    res = cue_maps(video_url(row), harp, me, cues, args.n_cues, args.before, args.after, args.seed,
                   args.clip_dir, f"{args.session}_{args.camera}",
                   cache_path=os.path.join(args.clip_dir, stem + ".npz"))
    res["maps"] = smooth_encoder_cycle(res["maps"])
    print("alignment r by offset: " + ", ".join(f"{s:+d}:{v:.3f}" for s, v in
                                                 zip(res["align_shifts"], res["align_r"])))
    os.makedirs(args.out_dir, exist_ok=True)
    _plot(res, f"{args.session} {args.camera}: per-pixel |frame difference| after the go cue "
               f"({len(res['clip_me'])} cues)", os.path.join(args.out_dir, stem + ".png"))
    print("wrote", os.path.join(args.out_dir, stem + ".png"))


if __name__ == "__main__":
    main()
