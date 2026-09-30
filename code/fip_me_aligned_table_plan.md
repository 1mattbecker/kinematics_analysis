# Plan: aligned motion-energy table for FIP sessions

_Started 2026-09-29 on branch `fip-motion-energy`. Revised 2026-09-30 for `video_timing_qc`
(`aind-dynamic-foraging-behavior-video-analysis`, revision 5, commit `84937d6`)._

**Goal:** one derived data asset holding, for every FIP session used in fip_05/fip_07 and
each camera, per-frame corrected Harp time plus the cleaned motion-energy trace, so this
capsule attaches one asset instead of ~200 raw-behavior and ME assets.

## Status

| Item | State |
|---|---|
| ME computed for all 97 sessions (`.mp4`, full length) | 87 done 2026-09-29; 10 June 808054 sessions rerun from `.mp4` (batch run `342a45f7`); manifest not yet pulled |
| Video timing QC and correction | Done in the library (`video_timing_qc`, on branch `plan/video-timing-qc`, about to merge to `main`). The build calls it; nothing is reimplemented here |
| Scratch QC (`metadata/video_csv_qc_fip.csv`, `qc_class`) | Superseded by `video_timing_qc`; kept as a record of the 2026-09-29 survey |
| 31 leftover test ME results | Kept in place (not moved or deleted); excluded by the explicit session → result mapping |
| `code/build_me_table.py` | Not started |
| `fip_utils` loader switch | Not started |

## Decisions

- **Source video: `.mp4` only.** All 97 sessions' ME comes from `.mp4` once the June rerun is
  in (the earlier June results were from `.avi`).
- **Store `me_clean` only** (what `fip_utils` uses), not the raw `me`.
- **Padding: NaN.** The library emits one fewer value than frames (no value for frame 0). The
  table pads a leading NaN so row *i* is frame *i*; the current `fip_utils` pads 0.
- **Harp time only, no session clock.** The session offset (first go cue) needs the NWB
  trials table, so it stays in the analysis.
- **Timing: `video_timing_qc`, trigger log when present** (the library's default in the LP
  pipeline). Log missing → CSV alone. Log present but not matching (event count ≠ exposures,
  or ≠ CSV Harp column by > 1 tick) → the module refuses and the camera is marked refused; no
  silent fallback to CSV. Refusals get looked at before deciding anything else.
- **Drop sessions (18): included, corrected, flagged** (index columns below). The analysis
  can exclude them from `index.csv`.
- **ME stays frame-to-frame differences.** Neither the ME capsule nor this build uses timing
  QC. After a drop, a row's ME spans more than one frame interval; the analysis accounts for
  that from the real time difference (`diff(harp_time)`) and frame numbers, not the build.
- **Library pinned by commit.** The build installs `aind-dynamic-foraging-behavior-video-analysis`
  at the `main` commit that includes `video_timing_qc` and records that SHA in the output.

## Inputs

- **Sessions:** rows of `metadata/me_sessions_fip_curated.csv` with `used_in_fip05_07`
  (97); `raw_session` is the raw asset name.
- **ME result per session — explicit, never "newest by tag":** from the batch launcher's
  manifests (`aind-motion-energy-batch-capsule`, runs `ce719ee8`, `2b3a9315`, and the June
  rerun `342a45f7`). ~31 leftover test results share the `motion-energy` tag, including
  5000-frame runs for 808054 09-02 to 09-04.

## Reading data without attaching

All from S3; only CSVs, trigger logs and `.npy` files are read, never videos.

| File | Location |
|---|---|
| Timestamp CSV, flat layout (166 cameras) | `s3://aind-open-data/<raw_session>/behavior-videos/{bottom_camera,side_camera_right}.csv` (headerless) |
| Timestamp CSV, nested layout (28 cameras) | `s3://aind-open-data/<raw_session>/behavior-videos/{BottomCamera,SideCameraRight}/metadata.csv` (header) |
| Trigger log (one per session, shared by all cameras) | `s3://aind-open-data/<raw_session>/behavior/raw.harp/BehaviorEvents/Event_94.bin` (same path in both layouts; checked on 3 sessions) |
| ME | `s3://aind-scratch-data/matt.becker/motion_energy/<ME asset name>/<cam>_motion_energy_clean.npy`, `<cam>_me_metadata.json` |

`aind-open-data` is public; the scratch bucket is readable from Code Ocean with the capsule's
AWS role. About 170 MB per session.

## Build steps per session

1. Read `Event_94.bin` once (`video_timing_qc.read_harp_trigger_log`); if missing,
   `trigger_times = None`.
2. Per camera:
   1. `timing = load_video_timing(csv)` (both layouts).
   2. Load `me_clean.npy` and `me_metadata.json`. Require `video_path` ends in `.mp4` and
      `end_frame` is null.
   3. `checks = check_video_timing(timing, video_frame_count=n_frames_decoded)`;
      `action = timing_action(checks)`.
   4. Refuse if `video_frame_count` fails: `timing_action` does not look at it, so the build
      must (it catches the six-frame transcode loss; 0 of 194 cameras had it).
   5. `fixed = correct_video_timing(timing, trigger_times=trigger_times)`. A `ValueError` →
      camera refused, error text recorded.
   6. Pad a leading NaN when `n_me_frames == n_frames_decoded - 1`; require ME length == rows.
   7. Write the per-frame file and the index row.
3. Normalize camera names: `BottomCamera` → `bottom_camera`, `SideCameraRight` →
   `side_camera_right`.

Nothing raises past a camera: one bad camera does not stop the build. `check_session` is not
used (it checks neither the log nor the video frame count).

## Output layout

```
fip_motion_energy_aligned/
  index.csv
  build_info.json                        # library SHA, build date, ME run IDs
  <session>/bottom_camera.parquet
  <session>/side_camera_right.parquet
```

**Per-frame file** (one row per saved video frame; row *i* is video frame *i*):

| Column | Type | Note |
|---|---|---|
| `harp_time` | float64 | Corrected Harp time of the frame's exposure (s) |
| `harp_source` | category | From `correct_video_timing`: `trigger_log`, `glitch_interpolated` (log mode); `original`, `reindexed`, `estimated_camera_fit`, `glitch_interpolated` (CSV mode) |
| `frame_number` | int64 | Camera exposure counter; `diff` > 1 marks a row after a drop |
| `camera_time` | float64 | Camera clock (s) |
| `me_clean` | float32 | NaN at row 0 |

In log mode `harp_source` does not say which rows were re-indexed; `frame_number` does (row *i*
was moved when `frame_number[i] − frame_number[0] ≠ i`). Refused cameras get no file.

**`index.csv`** (one row per session × camera):

- identity: `session`, `subject`, `camera`, `layout`, `raw_asset_id`, `me_asset_id`, `source_video`
- size: `n_frames`, `fps` (from corrected `harp_time`), `harp_start`, `harp_end`
- timing: `timing_source` (`trigger_log` / `csv`), `action` (from `timing_action`),
  `status` (`ok` / `refused`), `error`, `failed_checks`, `frames_lost`, `glitch_rows`,
  `clock_rate_ppm` (the `clock_rates_agree` count), `video_frame_count_diff`
- per-row source counts: `n_<harp_source>` for each value present
- ME: `padded`

The "flag" for drop sessions is `action == "re-index"` (equivalently `frames_lost > 0`).

Roughly 25–35 MB per camera compressed, ~5–7 GB in total.

## Effects of `video_timing_qc` on ME and video alignment

- **Row alignment is unchanged.** ME has one value per MP4 frame, and MP4 frames = CSV rows
  (all 194 cameras). Correction changes only the time on each row, never the rows.
- **What changes, by session group** (counts from the 2026-09-29 survey; the build confirms them):

  | Group | Sessions | Expected action | Effect on ME times |
  |---|---|---|---|
  | clean | 64 | `use harp as written` | None (log mode: identical to the CSV within one 32 µs tick) |
  | Harp glitch | 15 | `fix glitches` | 1–3 rows move by up to ~983 ms, back into line |
  | frame drops | 18 | `re-index` | Nearly every row moves later, by up to 280–670 s at session end; times now step by 2 IFI after each drop |

- **Session clock.** `motion_energy_to_session` computes `harp_time − first go cue` for each
  row: it never turns a row number into a time with fps, so the subtraction caveat in
  `video_alignment` does not apply. Row 0's time is the same raw and corrected (row 0 is
  never re-indexed, and a glitch on the first row is refused), so the offset helpers still
  agree. FIP and go-cue times are on the same Harp clock, so corrected ME lines up with
  photometry in drop sessions; uncorrected, it drifted by up to 670 s by the end.
- **Uneven sampling.** Corrected times have gaps at drops. `peri_event` interpolates onto a
  20 Hz grid; `norm_xcorr` takes signals already on a common grid, so it needs resampled
  input (as now). `threshold_onsets`' `min_run` counts samples, not time: in drop sessions
  a run of *n* samples is slightly longer. Minor; note it in the analysis.
- **ME after a drop.** The value spans the real interval `harp_time[i] − harp_time[i−1]`
  (2 IFI after a single drop; ~10% of rows in drop sessions). Handled in analysis, not here.
- **Video positions** (clips, not this table): after correction, event → frame by
  `searchsorted` on `harp_time`, then frame / fps. Not needed for ME.

## `fip_utils` changes (after the build)

- New `load_me(session_id, camera="bottom_camera")` reading the Parquet file and the index row.
- `motion_energy_to_session` takes `harp_time` from the table: `t_session = harp_time −
  first_go_cue`; `locate_me_assets` and the padding logic go away.
- Account for the real time step in ME (e.g. divide by `diff(harp_time)` or by
  `diff(frame_number)`); to be settled in the analysis, not the build.
- Session selection option: exclude cameras by `action` / `status` from `index.csv`.
- Then detach the ME assets and, if nothing else needs them, the raw behavior assets.

## Where it runs

`code/build_me_table.py` in this repo, one Code Ocean run, results saved as one data asset.
Needs an environment rebuild to pin the library commit. Re-runnable: skips sessions whose
files exist unless `--force`.

## Before building

1. Merge `plan/video-timing-qc` to `main`; pin that commit in this capsule's environment.
2. Pull the June rerun manifest (`342a45f7`) and check all 10 completed from `.mp4`.
3. Dry pass on all 97 (timing only, no ME, no writes): confirm the expected actions above,
   that the trigger log's event count equals exposures for every camera (open question 3 in
   the library plan; held on 7 sessions so far), and that `clock_rates_agree` passes. Review
   any refusal before the full build.

## Open questions

1. How the analysis corrects ME for the real time step (above).
2. If the dry pass finds log/CSV mismatches: refuse (current decision) or allow CSV alone for
   those cameras.
