# Plan: aligned motion-energy table for FIP sessions

_Started 2026-09-29 on branch `fip-motion-energy`._

**Goal:** one derived data asset holding, for every FIP session used in fip_05/fip_07 and
each camera, per-frame Harp time plus the cleaned motion-energy trace, so this capsule
attaches one asset instead of ~200 raw-behavior and ME assets.

## Status

| Item | State |
|---|---|
| ME computed for all 97 sessions (`.mp4`, full length) | 87 done 2026-09-29; 10 June 808054 sessions rerunning from `.mp4` (batch run `342a45f7`) |
| Video-CSV QC (thresholds, Harp slip) | **In progress — settle before building the table** (see below) |
| Move 31 leftover test ME results to `motion_energy/test/` | To do |
| `code/build_me_table.py` | Not started |
| `fip_utils` loader switch | Not started |

## Decisions

- **Source video: `.mp4` only.** Older flat-layout sessions store each camera as both
  `.avi` and `.mp4`. The 10 June 808054 results were computed from the `.avi` (older
  library build); they are being rerun from `.mp4` so all 97 match. The 14 nested-layout
  sessions only have `.mp4`.
- **Store `me_clean` only** (what `fip_utils` uses), not the raw `me`.
- **Padding: NaN.** The library emits one fewer value than frames (no value for frame 0).
  The table pads a leading **NaN** so row *i* is frame *i*; the current `fip_utils` pads 0.
- **Harp time only, no session clock.** The session offset (first go cue) needs the NWB
  trials table, so it stays in the analysis.

## Inputs

- **Sessions:** rows of `metadata/me_sessions_fip_curated.csv` with `used_in_fip05_07`
  (97); `raw_session` is the raw asset name.
- **ME result per session — explicit, never "newest by tag":** from the batch launcher's
  manifests (`aind-motion-energy-batch-capsule`, runs `ce719ee8`, `2b3a9315`, and the
  June rerun `342a45f7`). ~31 leftover test results share the `motion-energy` tag,
  including 5000-frame runs for 808054 09-02 to 09-04.

## Reading data without attaching

- **Timestamp CSVs:** `s3://aind-open-data/<raw_session>/behavior-videos/`
  (public bucket). Flat layout: `bottom_camera.csv`, `side_camera_right.csv`, headerless
  `Behav_Time, Frame, Camera_Time`. Nested layout: `BottomCamera/metadata.csv`,
  `SideCameraRight/metadata.csv`, header `ReferenceTime, CameraFrameNumber, CameraFrameTime`.
- **ME files:** `s3://aind-scratch-data/matt.becker/motion_energy/<ME asset name>/`
  (`<cam>_motion_energy_clean.npy`, `<cam>_me_metadata.json`). Readable from Code Ocean with
  the capsule's AWS role.
- Only ~130 MB per camera is read (CSV + `.npy`); videos are never touched.

## Build steps per session × camera

1. Read the timestamp CSV (`video_alignment.read_video_csv` handles both layouts); keep
   Harp time and camera frame number.
2. Run the video-CSV QC (below) and record its results.
3. Load `motion_energy_clean.npy` and `me_metadata.json`; pad a leading NaN when
   `n_me_frames == n_frames_decoded - 1`.
4. Check: ME length == CSV rows after padding; `end_frame` is null; `video_path` is `.mp4`;
   MP4 decoded frames == CSV rows (catches the six-frame transcode loss at the second
   keyframe from a default ffmpeg transcode).
5. Normalize camera names: `BottomCamera` → `bottom_camera`,
   `SideCameraRight` → `side_camera_right`.

Checks are recorded per row in the index, not raised, so one bad session doesn't stop the build.

## Output layout

```
fip_motion_energy_aligned/
  index.csv            # session, subject, camera, n_frames, fps, harp_start, harp_end,
                       # source_video, raw_asset_id, me_asset_id, padded, QC columns, status
  <session>/bottom_camera.parquet       # frame, harp_time (float64), camera_frame (int64),
  <session>/side_camera_right.parquet   # me_clean (float32, NaN at frame 0), is_keyframe (bool)
```

Roughly 20–30 MB per camera compressed, ~4–6 GB for 97 sessions × 2 cameras.

## Where it runs

`code/build_me_table.py` in this repo, one Code Ocean run (this capsule already has
`video_alignment` and the `aind-codeocean-user` AWS role), results saved as one data asset.
Re-runnable: skips sessions already built.

## `fip_utils` changes

- New `load_me(session_id, camera="bottom_camera")` reading the Parquet file.
- `motion_energy_to_session` computes the offset from `harp_time` with the same
  `video_alignment` helpers; `locate_me_assets` and the padding logic go away.
- Then detach the 10 ME assets and, if nothing else needs them, the raw behavior assets.

## Video-CSV QC

Current checks live in `integrate_keypoints_with_video_time`
(`aind-dynamic-foraging-behavior-video-analysis`, `kinematics/tongue_kinematics_utils.py`):
frame numbers must step by exactly 1, and `|ΔHarp − ΔCamera| > 2 × interval` flags clock
disagreement (then interpolates isolated bad frames).

**Why it needs to change (from a colleague's review):** `foraging.bonsai` pairs frames with
Harp events using `rx:Zip`, which pairs in arrival order. When the host drops a frame, every
later frame gets the Harp time meant for the frame before it, and each further drop shifts it
one more. Frame number and camera time record each drop; Harp time doesn't, and its error
never resets. A *k*-frame drop makes the clocks disagree by *k* intervals, so 2× misses every
single-frame drop and about half the two-frame drops; 0.5× catches every drop. Neither
threshold flags the frames *between* drops, whose Harp times are also wrong.
`Aind.Behavior.JustFrames` uses the same pairing.

**Goal here: identify affected sessions, not correct them.**

**Findings on an 8-session sample (both cameras):**

- 7 sessions clean: no frame gaps, no backward steps, no flags at 2× or 0.5×. Normal
  Harp/camera jitter is ~0.05 ms (p99), max ~0.09 ms, so 0.5× (1 ms) raises no false flags.
- `behavior_816212_2025-12-05_13-47-41`: ~187,000 single-frame drops per camera (~6.8%).
  2× flagged none; 0.5× flagged all, none off a frame gap. Camera clock ran ~374 s longer
  than Harp over the session, i.e. Harp times ~6 min early by the end.
- Clean sessions still show 16–37 frames (30–75 ms) of end-to-end clock difference from
  clock-rate drift, so a slip check needs a tolerance, not zero.
- MP4 decoded frames == CSV rows for every sampled camera (no transcode loss).

**Proposed QC per camera (to confirm on all 97 sessions):**

| Metric | Flag when |
|---|---|
| `frame_gaps`, `frames_dropped` (frame-number steps ≠ 1) | > 0 |
| backward steps in frame / Harp / camera time | > 0 |
| `flag_0.5x`: frames with `|ΔHarp − ΔCamera| > 0.5 × interval` | > 0 |
| `clock_slip_frames`: (camera span − Harp span) / interval | beyond normal drift (tolerance TBD from the full run) |
| MP4 decoded frames − CSV rows | ≠ 0 |

Affected sessions are marked in `index.csv` (and excluded or handled in the analysis);
no correction is applied.
