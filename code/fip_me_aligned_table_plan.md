# Plan: aligned motion-energy table for FIP sessions

_Started 2026-09-29 on branch `fip-motion-energy`. Revised 2026-09-30 for `video_timing_qc`
(`aind-dynamic-foraging-behavior-video-analysis` `v0.1.0`, commit `41e5b59`)._

**Goal:** one derived data asset holding, for every FIP session used in fip_05/fip_07 and
each camera, per-frame corrected Harp time plus the cleaned motion-energy trace, so this
capsule attaches one asset instead of ~200 raw-behavior and ME assets.

## Status

| Item | State |
|---|---|
| ME computed for all 97 sessions (`.mp4`, full length) | 87 done 2026-09-29 (runs `ce719ee8`, `2b3a9315`). The 10 808054 sessions from 2025-09-02 to 2025-09-15 (ME first computed June 2026, from `.avi`) rerun from `.mp4` (run `342a45f7`); manifest not yet pulled |
| Video timing QC and correction | In the library: `video_timing_qc`, released in `v0.1.0` (`41e5b59`, merged to `main`). The build calls it; nothing is reimplemented here |
| Library pin | Done 2026-09-30: `wild` merged into `fip-motion-energy`, which now pins `v0.1.0` (`41e5b59`). Environment not yet rebuilt |
| Scratch QC (`metadata/video_csv_qc_fip.csv`, `qc_class`) | Superseded by `video_timing_qc`; kept as a record of the 2026-09-29 survey |
| 31 leftover test ME results | Kept in place (not moved or deleted); excluded by the explicit session → result mapping |
| Session → ME asset mapping | Done 2026-09-30: `metadata/me_assets_fip.csv` (97 rows) from the three run manifests, by `code/build_me_asset_map.py`; every `me_metadata.json` is `.mp4`, full length, N−1 values |
| `code/build_me_table.py` | Written 2026-09-30; tested locally on 4 sessions; dry pass on all 97 done (below). Code Ocean run not yet done |
| `fip_utils` loader switch | Not started |

## Decisions

- **Source video: `.mp4` only.** All 97 sessions' ME comes from `.mp4` once the 808054
  2025-09-02 to 09-15 rerun is in.
- **Store `me_clean` only** (what `fip_utils` uses), not the raw `me`.
- **Padding: NaN.** The library emits one fewer value than frames (no value for frame 0). The
  table pads a leading NaN so row *i* is frame *i*; the current `fip_utils` pads 0.
- **Camera names: modern convention, `BottomCamera` and `SideCameraRight`,** in file names
  and the `camera` column, for both layouts. Flat-layout sources (`bottom_camera`,
  `side_camera_right`) are mapped on read.
- **Harp time only, no session clock.** The session offset (first go cue) needs the NWB
  trials table, so it stays in the analysis.
- **Timing: `video_timing_qc`, trigger log when present** (the library's default in the LP
  pipeline). Log missing → CSV alone. Log present but not matching (event count ≠ exposures,
  or ≠ CSV Harp column by > 1 tick) → the module refuses and the camera is marked refused; no
  silent fallback to CSV. Refusals get looked at before deciding anything else.
- **Drop sessions (18): included, corrected, flagged** (index columns below). The analysis
  can exclude them from `index.csv`. They include `808054_2025-09-26` (161,072 / 166,628 frames lost, bottom / side),
  which is in the fip05/07 set and has ME (run `ce719ee8`).
- **Video frame count must equal CSV rows, or the camera is refused.** A mismatch shifts
  every later frame (the transcode problem loses 6 frames ~0.5 s in), and truncating at the
  end, as `motion_energy_to_session` does today, hides that. Refused in the build and
  asserted again in the loader. No workaround is worth it: the lost frames' position is
  not recorded, so they cannot be removed from the CSV safely; the fix is re-transcoding
  with `aind-video-utils` ≥ 0.7.0 and rerunning ME. 0 of 194 cameras mismatched.
- **ME stays frame-to-frame differences in the table.** Neither the ME capsule nor this build
  uses timing QC. Normalisation for the real step happens in the loader (below).
- **ME normalised by the real step, in the loader** (see "ME after a drop"): divide by the
  number of exposures between consecutive saved frames, `diff(frame_number)`.
- **Even sampling before any alignment.** The loader returns ME on an even time grid;
  nothing downstream gets the uneven per-frame samples (see "Uneven sampling").
- **Library pinned by commit:** `v0.1.0` (`41e5b59`); the SHA is recorded in the output.

## Dry pass (2026-09-30)

All 97 sessions have a trigger log, so every camera is timed from it. 178 cameras accepted,
16 refused (9 sessions). Accepted cameras all pass `clock_rates_agree` (+6 to +16 ppm), match
the video frame count, have exactly one log event per exposure, and pad one leading NaN.

| Survey group | Sessions | Result |
|---|---|---|
| clean | 64 | 64 `use harp as written` |
| Harp glitch | 15 | 10 `fix glitches`; 5 refused (clock step) |
| frame drops | 18 | 17 `re-index` (2 with the bottom camera refused, trigger log); 1 refused (clock step) |

**Excluded for now, as the QC refuses them (decided 2026-09-30).** The build already refuses
them; nothing is overridden.

- Harp clock step (`harp_evenly_spaced`, both cameras): `800886_2025-08-26`,
  `808054_2025-09-09`, `809491_2025-11-13`, `815334_2025-11-04`, `816212_2025-12-10`,
  `818585_2025-12-10`. One or two rows per session where Harp advances −1.3 to +0.7 ms
  instead of 2 ms and stays 1.3–3.3 ms behind; paired steps are ~640,005 rows (~1,280 s)
  apart. This contradicts "≤ 2 ms, only seen before Sep 2025" above. `808054_2025-09-09`
  (expected `fix glitches`) is one of the 10 sessions fip_05/07 already use ME from.
  `809491_2025-11-13` also has a 444 s pause in triggers at row 70.
- Trigger log has more events than exposures: `818586_2026-01-16` (both cameras; 1,326 /
  1,328 extra) and the bottom camera of `816212_2025-11-13` and `816212_2025-12-02` (1 extra;
  their side cameras are accepted). In `818586_2026-01-16` the unmatched triggers are at the
  **start**: lick-triggered ME peaks at −2.65 s with frames on the first *n* triggers (the
  CSV's pairing) and at 0 with frames on the last *n*
  (`code/check_leading_lost_frames.py`, figures in `docs/video_timing_qc/`). The CSV alone
  would misplace every frame by 2.65 s there, and the survey called it clean. For the two
  816212 cameras the end cannot be told (one frame, 2 ms). Reported to the library as an issue.

## Inputs

- **Sessions:** rows of `metadata/me_sessions_fip_curated.csv` with `used_in_fip05_07`
  (97); `raw_session` is the raw asset name.
- **ME result per session — explicit, never "newest by tag":** from the batch launcher's
  manifests (`aind-motion-energy-batch-capsule`, runs `ce719ee8`, `2b3a9315`, and the rerun
  `342a45f7`). ~31 leftover test results share the `motion-energy` tag, including
  5000-frame runs for 808054 09-02 to 09-04.

## Reading data without attaching

All from S3; only CSVs, trigger logs and `.npy` files are read, never videos.

| File | Location |
|---|---|
| Timestamp CSV, flat layout (166 cameras) | `s3://aind-open-data/<raw_session>/behavior-videos/{bottom_camera,side_camera_right}.csv` (headerless) |
| Timestamp CSV, nested layout (28 cameras) | `s3://aind-open-data/<raw_session>/behavior-videos/{BottomCamera,SideCameraRight}/metadata.csv` (header) |
| Trigger log (one per session, shared by all cameras) | `s3://aind-open-data/<raw_session>/behavior/raw.harp/BehaviorEvents/Event_94.bin` (same path in both layouts; checked on 3 sessions) |
| ME | `s3://aind-scratch-data/matt.becker/motion_energy/<ME asset name>/<source cam>_motion_energy_clean.npy`, `<source cam>_me_metadata.json`; `<source cam>` follows the video layout: `bottom_camera` / `side_camera_right` (flat), `BottomCamera` / `SideCameraRight` (nested) |

`aind-open-data` is public; the scratch bucket is readable from Code Ocean with the capsule's
AWS role. About 170 MB per session.

## Build steps per session

1. Read `Event_94.bin` once (`video_timing_qc.read_harp_trigger_log`); if missing,
   `trigger_times = None`.
2. Per camera (`BottomCamera`, `SideCameraRight`; source name per layout as above):
   1. `timing = load_video_timing(csv)` (both layouts).
   2. Load `me_clean.npy` and `me_metadata.json`. Require `video_path` ends in `.mp4` and
      `end_frame` is null.
   3. `checks = check_video_timing(timing, video_frame_count=n_frames_decoded)`;
      `action = timing_action(checks)`.
   4. Refuse if `video_frame_count` fails: `timing_action` does not look at it, so the build
      must.
   5. `fixed = correct_video_timing(timing, trigger_times=trigger_times)`. A `ValueError` →
      camera refused, error text recorded.
   6. Pad a leading NaN when `n_me_frames == n_frames_decoded - 1`; require ME length == rows
      (refuse otherwise; never truncate).
   7. Write the per-frame file and the index row.

Nothing raises past a camera: one bad camera does not stop the build. `check_session` is not
used (it checks neither the log nor the video frame count).

## Output layout

```
fip_motion_energy_aligned/
  index.csv
  build_info.json                        # library SHA, build date, ME run IDs
  <session>/BottomCamera.parquet
  <session>/SideCameraRight.parquet
```

**Per-frame file** (one row per saved video frame; row *i* is video frame *i*):

| Column | Type | Note |
|---|---|---|
| `harp_time` | float64 | Corrected Harp time of the frame's exposure (s) |
| `harp_source` | category | From `correct_video_timing`: `trigger_log`, `glitch_interpolated` (log mode); `original`, `reindexed`, `estimated_camera_fit`, `glitch_interpolated` (CSV mode) |
| `frame_number` | int64 | Camera exposure counter; `diff` > 1 marks a row after a drop |
| `camera_time` | float64 | Camera clock (s) |
| `me_clean` | float32 | As computed (frame-to-frame difference, not normalised); NaN at row 0 |

In log mode `harp_source` does not say which rows were re-indexed; `frame_number` does (row *i*
was moved when `frame_number[i] − frame_number[0] ≠ i`). Refused cameras get no file.

**`index.csv`** (one row per session × camera):

- identity: `session`, `subject`, `camera` (`BottomCamera` / `SideCameraRight`),
  `source_camera` (name in the raw asset), `layout`, `raw_asset_id`, `me_asset_id`, `source_video`
- size: `n_frames`, `frame_interval_s` (`video_timing_qc.frame_interval` of the raw Harp
  column), `harp_start`, `harp_end`
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

- **Session clock: what `motion_energy_to_session` computes.** With `h` the per-row Harp
  array it is given and `g` the first go cue (both on the Harp clock):

  ```
  first_frame = h_csv[0]                    # get_first_frame_behavior_time (raw CSV row 0)
  offset      = g − h_csv[0]                # compute_video_session_offset
  video_t     = h − h_csv[0]                # behavior_time_to_video_time
  t_session   = video_t − offset = h − g    # video_time_to_session_time
  ```

  `h_csv[0]` cancels, so each row's session time is its own Harp time minus one constant.
  Nothing assumes even spacing, so **the result is exactly as right as `h`**:

  - **Today `h` is the raw CSV column, which is wrong in the 18 drop sessions.** Arrival-order
    pairing gives row *n* trigger *n*'s time, but row *n* was exposed later (at trigger
    `frame_number[n] − frame_number[0]`). ME is placed too early by (frames lost so far) × IFI,
    growing through the session to 280–670 s at the end. Subtracting the go cue carries the
    error through unchanged. Checked on `816214_2025-12-02` bottom (172,631 frames lost): the
    current path is 86 / 172 / 259 / 345 s early at ¼ / ½ / ¾ / end, and the end value equals
    frames lost × IFI exactly. Clean and glitch-only sessions are right today except on the
    1–3 glitch rows. The 10 808054 sessions from 2025-09-02 to 09-15, the only ones
    fip_05/07 have used ME from so far, have no drops (checked from their CSVs: normal clock
    drift); 09-04 and 09-09 have Harp glitches, so they are off on 1–3 rows today. `808054_2025-09-26` is a drop session in the same set; it needs the corrected
    time before it is used.
  - **With corrected `harp_time` as `h`, it is right.** Same session, same four calls: 0.000 µs
    from `harp_time − g`. This holds even if `first_frame` and `offset` still come from the
    raw CSV, because `h_csv[0]` cancels. (It also equals corrected row 0: exactly in CSV
    mode, within one 32 µs tick in log mode (0 µs on every session checked); row 0 is never
    re-indexed, and a glitch on it is refused.)
  - **Clock caveat.** Go cues and camera triggers come from the Harp Behavior board.
    Photometry may come through separate, synchronised Harp devices; whether those step
    with the board is the library's open clock-step question (≤ 2 ms per step, only seen
    before Sep 2025). At photometry sample intervals (~50 ms) that does not matter.
- **`video_t` is not a file position after correction.** `video_alignment` calls `h − h[0]`
  "seconds into the video", i.e. row × IFI (the file position when the MP4's frame rate is
  the recording rate). With raw Harp in a drop session it matches row × IFI (raw Harp
  advances one IFI per saved row; off only on glitch rows), but it places events on the
  wrong frames. With corrected Harp it advances two IFI across each drop while the file
  advances one frame, so it runs ahead of row × IFI by (frames lost so far) × IFI: 345 s at
  the end of the session above. Inside `motion_energy_to_session` this cancels and does no
  harm; nothing may reuse corrected `video_t` (or `offset` + session time) to seek a video.
  For that (clips, BEAST): event → frame index by `searchsorted` on `harp_time`, then
  index / fps. `offset` is only printed in the FIP notebooks, never used to seek.
- **ME after a drop: normalise by the real step.** `aind-motion-energy` differences
  consecutive *saved* frames, so after a drop the value spans 2 (or more) exposures and
  carries roughly that multiple of the motion; 5–12% of rows in drop sessions. The loader
  divides by the number of exposures spanned:

  ```
  me_norm[i] = me_clean[i] / (frame_number[i] − frame_number[i−1])
  ```

  This is the real step: on every accepted camera it equals `round(Δharp_time / IFI)`
  (re-indexed times must match camera steps within half a frame). **Not** `Δharp_time / IFI`
  directly: Harp timestamps come in 32 µs ticks, so single-exposure steps read 1,984 or
  2,016 µs at random (camera steps over the same rows: 1,996–2,004 µs, correlation with the
  Harp step 0.0, on `816214_2025-12-02`). Dividing by them would scale ordinary rows by
  0.976–1.024 with noise that is not in the frames. On clean sessions `me_norm == me_clean`.
  Assumes ME grows linearly with the interval, which holds for small steps (2 vs 4 ms).
- **Uneven sampling: resample onto an even grid before anything else.** Corrected times have
  gaps at drops (steps of 1 or 2 IFI). Code that counts samples rather than using timestamps
  is off in drop sessions: `threshold_onsets`' `min_run` (samples), `norm_xcorr` (needs a
  common grid), the `fip_03` Welch cell (`fps = rows / span` gives 467.6 instead of 500 on
  the session above, and assumes even spacing), and possibly filters, rolling windows or
  index-based resampling in `enrich_dfs` / PSTH code (not checked yet). So the loader returns
  normalised ME on an even grid at the camera rate (`1 / frame_interval_s`), by linear
  interpolation on `harp_time`; a dropped exposure becomes one interpolated sample between
  its neighbours. Downsampling to analysis rates happens after that, as now. Only
  `peri_event` (interpolates on timestamps) was already safe.

## `fip_utils` changes (after the build)

- New `load_me(session_id, camera="BottomCamera")` reading the Parquet file and the index
  row, refusing cameras with `status != "ok"`, and returning ME normalised by
  `diff(frame_number)` and resampled onto an even grid at `1 / frame_interval_s`.
- `motion_energy_to_session` computes `t_session = harp_time − first_go_cue` from the
  table's corrected `harp_time`, directly (no `video_t` in between, so it cannot be reused
  as a file position), and returns `offset = first_go_cue − harp_time[0]` as before;
  `locate_me_assets`, the raw-CSV read, the padding and the truncate-on-mismatch logic go
  away (a length mismatch raises). Until this lands, the current function misplaces ME in
  the 18 drop sessions: do not use them with it.
- `attach_me_to_df_fip`, `threshold_onsets`, `norm_xcorr` and the `fip_03` Welch cell take
  the evenly resampled ME.
- Session selection option: exclude cameras by `action` / `status` from `index.csv`.
- Then detach the ME assets and, if nothing else needs them, the raw behavior assets.

## Where it runs

`code/build_me_table.py` in this repo, one Code Ocean run, results saved as one data asset.
Needs an environment rebuild for the `v0.1.0` pin (already on this branch).
Re-runnable: skips sessions whose files exist unless `--force`.

## Before building

1. ~~Merge `wild` into `fip-motion-energy` to get the `v0.1.0` pin~~ (done); rebuild the environment.
2. Pull the rerun manifest (`342a45f7`) and check all 10 completed from `.mp4`.
3. Dry pass on all 97 (timing only, no ME, no writes): confirm the expected actions above,
   that the trigger log's event count equals exposures for every camera (open question 3 in
   the library plan; held on 7 sessions so far), and that `clock_rates_agree` passes. Review
   any refusal before the full build.
4. Check `enrich_dfs` / PSTH code for sample-count assumptions (none should see uneven ME
   once the loader resamples, but confirm the loader's grid is what they expect).

## Open questions

1. ~~If the dry pass finds log/CSV mismatches: refuse or allow CSV alone.~~ Refuse (2026-09-30):
   the CSV alone is wrong by 2.65 s in `818586_2026-01-16` (see "Dry pass").
