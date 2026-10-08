"""
cue_me_maps.py — where in the image the motion energy (ME) right after the go cue comes from.

The session-average ME map (``aind-motion-energy``'s ``_motion_energy_map.npy``) cannot separate one
moment after the cue from another. This cuts a frame-exact clip around each of N go cues and averages
the per-pixel absolute frame difference at every lag, so each 2 ms step after the cue gets its own map.
Written for the ~30 Hz ringing in ``men_00`` §9b (first 150 ms after the cue, bottom camera).

Steps:

1. Go-cue Harp times (``goCue_start_time_raw`` of the CSV-curated FIP trial table) -> frame windows
   with the library's ``video_alignment.event_frame_ranges`` on the ME table's corrected per-frame
   ``harp_time``. Windows that span dropped frames (fewer frames than the rest) are left out.
2. Clips cut from the session's MP4 on public S3 with ``video_clips.cut_clips`` (frame-exact seek from
   the MP4 sample tables; clips re-encoded near-losslessly, CRF 18).
3. Each clip decoded (luma plane) with PyAV; ``|frame[k+1] - frame[k]|`` summed per lag over cues. The
   lag of a difference is the Harp time of its later frame minus the go cue, as in the ME table.
4. Check: per clip, the mean of each difference image is correlated with the table's ``me_clean`` at
   frame offsets -5..+5 (summed over clips); the best offset must be 0, or the script stops.

Writes ``<out-dir>/cue_me_maps_<session>_<camera>.npz`` (``lags_s``, ``maps`` [lag, H, W] mean
|difference|, ``reference`` mean frame, ``clip_me`` [cue, lag] per-clip ME, ``cues``) and two figures:
the maps (``.png``) and ME by image region with the map of the fast part (``_regions.png``).

Environment: the video-analysis library with its ``video-clips`` extra, plus PyAV, pandas, pyarrow and
matplotlib; ffmpeg on the PATH. Not ``.venv-fip``.

Usage (from ``code/``)::

    python cue_me_maps.py --session 809487_2025-10-10 --n-cues 100
"""

import argparse
import glob
import os

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va  # noqa: E402
from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc  # noqa: E402

VIDEO_URL = "https://aind-open-data.s3.us-west-2.amazonaws.com/{raw}/behavior-videos/{camera}.mp4"
# Lag windows (ms after the cue) drawn as maps, minus the pre-cue map
MAP_WINDOWS_MS = [(10, 16), (18, 30), (34, 46), (50, 62), (66, 80), (100, 150)]


def load_trials(ses_idx, data_roots):
    """Trial table of a session from the CSV-curated FIP assets."""
    for root in data_roots:
        hits = glob.glob(os.path.join(root, "**", ses_idx, "df_trials.parquet"), recursive=True)
        if hits:
            return pd.read_parquet(hits[0]).sort_values("trial")
    raise FileNotFoundError(f"no trial table for {ses_idx}")


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
    """Summed correlation of clip ME with the table's me_clean at frame offsets -max_shift..max_shift."""
    shifts = np.arange(-max_shift, max_shift + 1)
    n_diff = clip_me.shape[1]
    r = []
    for d in shifts:
        a, b = [], []
        for x, s in zip(clip_me, starts):
            # difference k of a clip is frame s+k+1 minus frame s+k: the table's row s+k+1
            idx = s + 1 + np.arange(n_diff) + d
            if idx[0] < 0 or idx[-1] >= len(table_me):
                continue
            a.append(x - x.mean())
            b.append(table_me[idx] - table_me[idx].mean())
        a, b = np.concatenate(a), np.concatenate(b)
        r.append(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    r = np.array(r)
    return shifts, r


def plot(lags_s, maps, reference, clip_me, title, path):
    lags_ms = lags_s * 1000
    base = maps[lags_ms < 0].mean(0)
    excess = [maps[(lags_ms >= a) & (lags_ms < b)].mean(0) - base for a, b in MAP_WINDOWS_MS]
    vmax = np.nanpercentile(np.abs(np.stack(excess)), 99.8)
    n = len(MAP_WINDOWS_MS) + 2
    fig, axes = plt.subplots(2, (n + 1) // 2 + 1, figsize=(4.2 * ((n + 1) // 2 + 1), 7.6))
    axes = axes.ravel()
    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("mean frame")
    im = axes[1].imshow(base, cmap="magma")
    axes[1].set_title("|diff| before the cue (−60–0 ms)")
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    for ax, (a, b), e in zip(axes[2:], MAP_WINDOWS_MS, excess):
        im = ax.imshow(e, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{a}–{b} ms minus before")
    fig.colorbar(im, ax=axes[2 + len(excess) - 1], fraction=0.04)
    for ax in axes[: 2 + len(excess)]:
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[2 + len(excess)]
    for x in clip_me[:15]:
        ax.plot(lags_ms, x, color="0.75", lw=0.6)
    ax.plot(lags_ms, clip_me.mean(0), "k", lw=1.5, label=f"mean of {len(clip_me)} cues")
    ax.axvline(0, ls="--", color="0.5", lw=0.8)
    ax.set(xlabel="time from go cue (ms)", ylabel="mean |diff| (luma)", title="ME of the clips")
    ax.legend(fontsize=8)
    for ax in axes[3 + len(excess):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_regions(lags_s, maps, reference, title, path, smooth_ms=16, band_hz=(22, 45)):
    """ME by image region, and where the fast (~30 Hz) part of the post-cue ME lies.

    Regions: the top 3% of pixels by mean excess |difference| 10–80 ms after the cue, split into the
    left edge of the frame (x < 12%), the upper half and the lower half (in the bottom camera: the
    face/body edge and the whiskers on either side of the snout); the rest of the frame is the fourth.
    The fast part is each trace minus its ``smooth_ms`` running mean; its power in ``band_hz`` over
    15–110 ms is drawn per pixel.
    """
    lags_ms = lags_s * 1000
    dt = np.median(np.diff(lags_s))
    base = maps[lags_ms < 0].mean(0)
    excess = maps[(lags_ms >= 10) & (lags_ms < 80)].mean(0) - base
    h, w = excess.shape
    hot = excess > np.percentile(excess, 97)
    yy, xx = np.mgrid[:h, :w]
    regions = {"left edge": hot & (xx < 0.12 * w),
               "upper half": hot & (xx >= 0.12 * w) & (yy < h / 2),
               "lower half": hot & (xx >= 0.12 * w) & (yy >= h / 2),
               "rest of frame": ~hot}
    colors = ["tab:green", "tab:red", "tab:blue", "0.5"]
    kernel = np.ones(int(round(smooth_ms / 1000 / dt))) / int(round(smooth_ms / 1000 / dt))

    win = (lags_ms >= 15) & (lags_ms <= 110)
    x = maps[win].reshape(win.sum(), -1)
    x = x - x.mean(0)
    fast = x - np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"), 0, x)
    freqs = np.fft.rfftfreq(128, dt)
    power = (np.abs(np.fft.rfft(fast, 128, axis=0)) ** 2)[(freqs >= band_hz[0]) & (freqs <= band_hz[1])]
    power = power.sum(0).reshape(h, w)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    axes[0].imshow(reference, cmap="gray")
    for (name, m), c in zip(regions.items(), colors):
        if name != "rest of frame":
            axes[0].contour(m, levels=[0.5], colors=[c], linewidths=0.8)
    axes[0].set_title("regions: top 3% of excess |diff|, 10–80 ms")
    axes[1].imshow(power, cmap="magma", vmax=np.percentile(power, 99.7))
    axes[1].set_title(f"power {band_hz[0]}–{band_hz[1]} Hz, 15–110 ms")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    for (name, m), c in zip(regions.items(), colors):
        trace = maps[:, m].mean(1)
        pre = trace[lags_ms < 0].mean()
        axes[2].plot(lags_ms, (trace - pre) / (trace.max() - pre), color=c, label=f"{name} ({m.sum()} px)")
        if name != "rest of frame":
            axes[3].plot(lags_ms, trace - np.convolve(trace, kernel, "same"), color=c, label=name)
    axes[2].set(xlabel="time from go cue (ms)", ylabel="|diff|, 0 = before the cue, 1 = peak",
                title="ME by region")
    axes[3].set(xlabel="time from go cue (ms)", ylabel="|diff| (luma)", xlim=(10, 150),
                title=f"by region, minus {smooth_ms} ms running mean")
    for ax in axes[2:]:
        ax.axvline(0, ls="--", color="0.5", lw=0.8)
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
    p.add_argument("--trials", choices=["all", "responded", "ignored"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--me-root", default=os.environ.get("ME_DATA_ROOT", "../../data"))
    p.add_argument("--fip-roots", nargs="+", default=["../../data/results-a278f830-9bf2-48e8-9928-a2ff78e4818f",
                                                      "../../data/results-ddcccb0f-f18f-44a6-a2b1-caba680d28a1"])
    p.add_argument("--clip-dir", default="../../data/cue_me_clips")
    p.add_argument("--out-dir", default="../data/figures/men")
    args = p.parse_args()

    index = pd.read_csv(os.path.join(args.me_root, "fip_motion_energy_aligned", "index.csv"))
    index["ses_idx"] = index["session"].str.split("_").str[1:3].str.join("_")
    row = index[(index.ses_idx == args.session) & (index.camera == args.camera) & (index.status == "ok")]
    if row.empty:
        raise SystemExit(f"{args.session} {args.camera}: not in the ME table with status ok")
    row = row.iloc[0]
    table = pd.read_parquet(os.path.join(args.me_root, "fip_motion_energy_aligned", row.session,
                                         f"{args.camera}.parquet"), columns=["harp_time", "me_clean"])
    harp = table["harp_time"].to_numpy()

    trials = load_trials(args.session, args.fip_roots)
    keep = {"all": np.ones(len(trials), bool),
            "responded": (trials["animal_response"] != 2).to_numpy(),
            "ignored": (trials["animal_response"] == 2).to_numpy()}[args.trials]
    cues = trials.loc[keep, "goCue_start_time_raw"].dropna().to_numpy(float)
    ranges = va.event_frame_ranges(cues, harp, args.before, args.after)
    n_full = int(ranges.loc[ranges.in_video, "n_frames"].mode()[0])
    ranges = ranges[ranges.in_video & (ranges.n_frames == n_full)]       # no drops inside a window
    ranges = ranges.sample(min(args.n_cues, len(ranges)), random_state=args.seed).sort_values("start_frame")
    print(f"{args.session} {args.camera}: {len(ranges)} cues ({args.trials}), {n_full} frames each")

    url = VIDEO_URL.format(raw=row.session, camera=row.source_camera)
    clips = vc.cut_clips(url, ranges, args.clip_dir, f"{args.session}_{args.camera}")
    clips = clips[clips.clip_path.notna()]
    print(clips.status.str.split(":").str[0].value_counts().to_dict())

    acc, ref_acc, clip_me, starts, cue_times = None, None, [], [], []
    for r in clips.itertuples():
        y = clip_luma(r.clip_path)
        if len(y) != n_full:
            print(f"  {r.clip_path}: {len(y)} frames, expected {n_full}; skipped")
            continue
        d = np.abs(np.diff(y, axis=0)).astype(np.float32)
        acc = d if acc is None else acc + d
        ref_acc = y.mean(0) if ref_acc is None else ref_acc + y.mean(0)
        clip_me.append(d.mean(axis=(1, 2)))
        starts.append(int(r.start_frame))
        cue_times.append(r.event_time)
    clip_me, starts = np.array(clip_me), np.array(starts)
    n = len(clip_me)
    maps, reference = acc / n, ref_acc / n
    # lag of difference k: Harp time of frame start+k+1 minus the cue, averaged over cues
    lags_s = np.mean([harp[s + 1: s + n_full] - c for s, c in zip(starts, cue_times)], axis=0)

    shifts, r = check_alignment(clip_me, table["me_clean"].to_numpy(), starts)
    best = shifts[np.argmax(r)]
    print("alignment r by offset: " + ", ".join(f"{s:+d}:{v:.3f}" for s, v in zip(shifts, r)))
    if best != 0:
        raise RuntimeError(f"clips are {best} frames off the ME table")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"cue_me_maps_{args.session}_{args.camera}" + ("" if args.trials == "all" else f"_{args.trials}")
    np.savez_compressed(os.path.join(args.out_dir, stem + ".npz"), lags_s=lags_s, maps=maps.astype(np.float32),
                        reference=reference.astype(np.float32), clip_me=clip_me, cues=np.array(cue_times))
    plot(lags_s, maps, reference, clip_me,
         f"{args.session} {args.camera}: per-pixel |frame difference| after the go cue ({n} cues, {args.trials})",
         os.path.join(args.out_dir, stem + ".png"))
    plot_regions(lags_s, maps, reference, f"{args.session} {args.camera}: ME by region ({n} cues, {args.trials})",
                 os.path.join(args.out_dir, stem + "_regions.png"))
    print("wrote", os.path.join(args.out_dir, stem + ".png"), "and _regions.png")


if __name__ == "__main__":
    main()
