# Handoff: moving kinematics_analysis to the library's v0.5.2 outputs

_Written 2026-10-28. Library: `aind-dynamic-foraging-behavior-video-analysis` tag `v0.5.2`.
The batch outputs below were made with v0.5.1, so rerun with
`force=True` (§2.1 and §2.10 change `movements` and `trials`; command in `TODO.md`)._

## Status (2026-10-29, branch `kinematics-v052`)

The code side is done; the data side runs on Code Ocean (`TODO.md`, "Move to the library's v0.5.x
kinematics outputs", steps 1-8). How each item below was handled:

| Item | Done |
|---|---|
| §2.1 spanning movements | Rule kept (no trial). Pooled cue-response counts: `kin_05`/`kin_06` group on `(session, trial)` and drop movements without a trial before merging |
| §2.2 `firstMove`, cue response | `ephys_utils` passes `movements` to `get_events_dict`; `build_trial_features` merges `cue_response_movement_number` from `trials`. Old vs new per trial: `verify_kinematics_v053` §5 |
| §2.3 `reward_before_choice` | Trials kept; dropped only in `kin_07` (value/RPE), after the latents are attached. Pooled table carries the flag |
| §2.4 sessions, coverage | Threshold 0.90 on `lick_coverage` (`data_loading`, one place); old vs new pass/fail: `verify_kinematics_v053` §2. 3 of 32 ephys sessions are not in the new batch |
| §2.5 jaw, `max_x_distance` | Library 0.5.3 adds the `extremes` feature group, so `max_x_distance` = \|`x_max`\|, `max_y_distance` = \|`extent_y`\|, `max_x_from_jaw_y` = `y_at_x_max` exactly. `kin_06` no longer recovers the jaw; `kin_07` uses the relative columns |
| §2.5 `out_duration` filters | Counted old vs new in `verify_kinematics_v053` §3 |
| §2.6 speeds | Measured in `verify_kinematics_v053` §4 (matched movements) |
| §2.7 dropped columns | Renamed throughout; `has_lick` = `n_licks > 0` and `movement_before_cue_response` derived where used; `kin_07`'s absolute-pixel columns and `time_to_endpoint` dropped from its screen; `kin_04` keeps `rewarded` NA on lick-less movements |
| §2.8 frame table | `kinematics.parquet` read by keypoint and time window; video frames from the `frame` column; `lickometer_qc` on `pose.parquet` + `signal.mask` |
| §2.9 bouts | `eph_07` keeps the repo rule (`behavior_utils.annotate_movement_bouts`) |
| §3 step 1 image | Pinned v0.5.3 (`c904311`, PR #19, 2026-10-29); rebuild the image |
| §3 steps 2-5 | Shared modules, pooled builder, one session root (`kin_utils.KINEMATICS_ROOT`), every notebook in the table; the "file names only" notebooks and other pre-0.5.0 tools archived (`code/archive/README.md`) |

Each migrated notebook gave the same numbers as before when run locally on the old pooled table
renamed to the new columns; exceptions are listed in `CHANGELOG.md` (2026-10-29).

## Background

The library replaced the old tongue pipeline (`tongue_analysis`, `tongue_kinematics_utils`) with a
general kinematics core plus a dynamic-foraging task module. New outputs for the 53-session list
are in `/root/capsule/scratch/kinematics_v050/<session>/` (48 processed; see §4).

New column names are accepted as they are. This doc is about the places where **what a column
means, or which rows it covers, changed**, so that an analysis quietly computes something different.
Each item says how big the change is (measured where possible) and what to do.

References, in the library repo:
- `KINEMATICS_PLAN.md` § "Old → new": file, function and column map.
- `KINEMATICS_GUIDE.md`: how to use the core, plus the output tables.

## 1. What did not change (verified, no need to re-check)

Full batch, all 30 sessions both versions processed (old: `session_analysis_fall2026`), using
`scripts/compare_old_new_kinematics.py`:

- **Movements**: identical in every session, about 190,000 movements with the same start, end and
  ids. Rule: tracked runs at confidence ≥ 0.9, bridging up to 10 missing frames.
- **Lick → movement**: the same rule (lick inside the movement, within half a frame). 100% of licks
  agree in 28 sessions and 99.98% in 2 (one lick each, at a movement edge).
- **Clock**: session time, seconds from the first go cue (`nwb.trials.goCue_start_time[0]`).
- **`cue_response`, `rewarded`** on licks: the same basic-analysis `annotate_licks`. A movement
  takes them from its first lick, as before.
- **`number_in_trial`** (old `movement_number_in_trial`) and **`latency_from_go`**: same order and
  values for movements that start inside a trial and end before the next go cue.
- **Endpoint position**: identical (p99 difference 0.8%) once the jaw reference is accounted for.

## 2. Definition changes

### 2.1 Movements that span a go cue: no trial, as before

0.5.0–0.5.1 gave these a trial. **0.5.2 restores the old rule**: `trial`, `go_cue_time`,
`latency_from_go` and `number_in_trial` are NA, and the new `spans_go_cue = True` marks them.
`dropna(subset=["trial"])` drops them, as it always did.

One difference remains: their `cue_response`, `rewarded`, `lick_time` and `lick_side` describe
their first lick. Before 0.5.0 they were NA. A per-trial step drops them anyway, since `trial` is
NA. A pooled count of `cue_response` movements includes them; filter `~spans_go_cue` to match.

**Size**, in the old pooled table: about 1% of trials in typical sessions (median 0.01 spanning
movements per trial). Up to 0.24 per trial in 761038 (04-16, 04-18) and 0.07–0.15 in 784803.
`TODO.md` → `kin_08_go_cue_spanning_movements` is about what these movements are.

### 2.2 Cue response and first movement per trial

- **What a cue response is**, unchanged and defined upstream:
  - basic-analysis `annotate_cue_response`: the first lick that *starts a lick bout* (no lick in
    the previous 0.7 s) within (go cue, go cue + 1 s].
  - So `lick_latency` is at most 1 s. A lick that continues licking from before the cue is never
    a cue response.
  - data-utils' `choice_time` is a different measure: the trial's first lick, set to NaN when
    later than `response_duration` + 5 ms on unrewarded trials.
- **Trial summaries** (`trials.parquet`) count only movements with a trial. A trial can no longer
  take its cue response from a movement that began before the next go cue. That was a 0.5.0–0.5.1
  bug, fixed in 0.5.2.
- **`firstMove`** (`get_events_dict`, used for the `eph_*` alignment):
  - Old: built from frames that each carried a trial label. When a movement spanned the go cue,
    its frames after the cue counted, so `firstMove` was the first frame after the cue, about
    2 ms after it.
  - New: the first movement that *starts* after the go cue, excluding spanning movements.

  In the roughly 1% of trials with a spanning movement, the event moves from about 2 ms to the
  next movement's start. Re-check any reaction-time distribution near 0.
- **First movement per trial** (`kin_02`, `kin_04`, `build_trial_features` from
  `number_in_trial == 1`): unchanged.

### 2.3 Trials in the 2025 sessions that data-utils now rejects

The old pooled table (`all_tongue_movements_04022026`) has `trial = NA` for more movements than
there are trials in 11 sessions. These are 752014 (03-26, 03-27), 758018 (03-19, 03-20, 03-21),
761038 (04-17), 782394 (04-23, 04-24), 784806 (06-17, 06-18) and 791691 (06-27), with 0.8 to 9.7
NA movements per trial. That's too many to be spanning movements. All 11 are among the sessions
where current data-utils raises "Reward before choice time". v0.5.1 processes them with
`ignore_trial_errors=True`, so they now have a trial for every movement.

- **Expect large changes in per-trial results for these sessions.** Their old per-trial numbers
  were mostly missing.
- New column `trials.reward_before_choice` marks the trials data-utils objects to: an earned reward
  timed before the choice. There are 1 to 5 per session in the sessions checked. **Decide** whether
  to drop those trials. Count per session: `record.json → quality.n_trials_reward_before_choice`.

### 2.4 Session set and quality filter

- **Sessions.** Pooled old: 44. New batch: 48 processed.
  - Lost (new timing checks fail): 754897 (03-14), 758018 (03-20), 761038 (04-15). The first has
    Harp clock steps; the other two have Harp steps or a short trigger log. The old pipeline used
    their frame times anyway.
  - In the new batch but not the old pooled table: 717259, 717263, 752014 (03-25),
    754897 (03-13), 763590 (05-02), 791691 (06-24, 06-26). These probably failed the old quality
    filter; check that they fail the new one.
- **Coverage scale**: was 0–100 (`coverage_pct`), now 0–1 (`quality.lick_coverage`). Change
  `COVERAGE_MIN = 90.0` to `0.90` in both `build_all_tongue_movements.py` and `data_loading.py`.
  The comment there ("percent of frames with a confident tongue keypoint") is wrong; it is the
  fraction of lickometer licks inside a tracked movement.
- **Coverage, which licks are counted**:
  - Old: averaged over licks that had a trial label, so licks before the first go cue were left
    out (and, in §2.3's sessions, many more).
  - New: all licks.
  - Small in clean sessions; the comparison CSV has both, in the `quality lick_coverage` rows.
    Sessions near 0.90 can switch between pass and fail.
- **Median duration**: was a nearest-rank percentile, now `median()`. Negligible, but it can flip
  a session sitting exactly at 0.06 s.
- **Do**: list pass/fail under old and new numbers side by side (§5 check B), then accept the new
  list deliberately.

### 2.5 Jaw reference, spouts, and positions measured from the jaw

- **Jaw**: was the mean of every jaw point, including low-confidence ones. Now it's the median of
  points with likelihood ≥ 0.9, in `landmarks.parquet`. It shifted about 2–6 px per session, and
  `angle_deg` moved about 3° (2.6–5° median per session in the sessions checked).
- **Spouts**: the same rule. Note: `spout_l` is the **animal's right** spout (the bottom camera
  mirrors). `lick_side` comes from the lickometer and is the animal's side.
- **Positions**: `start_x/y`, `end_x/y`, `extent_x/y` are already relative to that jaw. Don't
  subtract a jaw again. `kin_06` and `kin_07` currently subtract `jaw_position()` or `kps_raw`
  means; use `landmarks.parquet` once, or the relative columns directly.
  `estimate_jaw_position` (the algebraic fallback in `kin_06`) is no longer needed.
- **`out_*`**: the farthest frame can move by a frame when the jaw moves, so `out_duration`
  changes for some movements, and some that were exactly 0 no longer are. This matters for
  filters like `out_duration > 0.01` / `> 0.05` (`kin_00`, `kin_01`); check how many rows they keep.
- **`max_x_distance` is not `abs(extent_x)`**: the library plan's map was wrong here (corrected in 0.5.2).
  - Old: the distance of the *most positive* x offset from the jaw.
  - New: `extent_x` is the signed offset farthest from the jaw *in either direction*.
  - On the old pooled table the two differ in 12.1% of movements (9.6% of lick movements), where
    the movement reaches farther in −x than +x.
  - `max_y_distance` does equal `abs(extent_y)`.
  - `max_x_from_jaw_y` (y at the most-positive-x frame, used by `kin_07`) has no equivalent.
  - If you need these exactly, compute them from `kinematics.parquet`
    (`groupby(movement_id).x.max()` minus the jaw) or add a feature group. Otherwise switch to
    `extent_x` / `end_x` deliberately.

### 2.6 Speeds near movement edges

Smoothing now runs per tracked run, so a movement is no longer smoothed together with its
neighbours across the gaps. Interior speeds are identical and edge frames differ. Peak speed falls
on an edge frame in 49–96% of movements, so `peak_speed` differs by a median of about 8% (p99
about 50%). `path_length` differs by about 1%, and `mean_speed` by less. The library's simulation
puts the new edge error at about half the old. Fixed cutoffs on peak speed or velocity were set
on the old values; re-check them.

### 2.7 Dropped columns

| Old | Instead |
|---|---|
| `min_x … max_yv` | absolute pixels, dropped. `kin_07`'s value screen lists them; drop them, or recompute from frames |
| `time_to_endpoint` | `out_duration` (they were equal) |
| `dropped_frames_n/pct` | `n_frames − n_valid`, `/ n_frames` |
| `first_lick_index`, `lick_count`, `has_lick` | `licks.movement_id`, `n_licks`, `n_licks > 0` |
| `movement_before_cue_response` | `number_in_trial < cue_response_movement_number`, with the latter merged from `trials` on `trial` |
| `lick_latency`, `cue_response_movement_number` on each movement | on `trials.parquet` now (one row per trial; merge on `session, trial`). See §2.2 |
| `goCue_start_time_in_session` on movements | `go_cue_time` |
| `event` | `lick_side` ('left'/'right') |

Booleans: `cue_response` and `rewarded` are real booleans, and False on movements without a lick
(old: None). `== True` is unaffected; `.isna()` / `fillna(False)` checks behave differently.

### 2.8 The frame table

`kinematics.parquet` replaces `tongue_kins.parquet`:
- **Long format**: one row per frame per keypoint. Filter `keypoint == "tongue_tip_center"`.
  Only the tongue is processed now, but don't rely on that.
- **`time` is session time.** It was Harp time in `tongue_kins.time`; `time_in_session` is gone.
  Code that read `time` expecting Harp time now silently gets session time. Harp time is
  `time + record["clock"]["time_zero"]`.
- **No per-frame `trial`, `lick`, `lick_index`, `has_lick`.** Join from `movements` on
  `movement_id`, or label frames with `intervals.interval_of` against the trials' go cues.
- `x, y` are NaN on frames with no confident point, as before. New: `valid`, `radial_velocity`,
  `submovement_id`, `bout_id`.
- The raw keypoints (old `kps_raw_*.parquet`) are now `pose.parquet` (long format: `keypoint`,
  `likelihood`). `lickometer_qc.load_session` masks at its own 0.8 confidence; do that on
  `pose.parquet` with `signal.mask(pose[pose.keypoint == kp], 0.8)`.

### 2.9 Bouts: two different rules

| | Repo (`behavior_utils.annotate_movement_bouts`) | Library (`bouts.parquet`) |
|---|---|---|
| New bout when | start − previous **start** > 0.5 s | start − previous **end** > 0.5 s |
| Single movements | bouts of size 1 | not bouts (`min_count = 2`) |

The library rule gives longer bouts. Keep `annotate_movement_bouts` for `eph_07` so results stay
comparable, or switch deliberately.

### 2.10 The last trial ends at the longest go-cue interval

Before 0.5.2, the last trial ran to the end of the video. In the old pooled table that is a
median of about 18 minutes past the last go cue. Now it ends one *longest go-cue-to-go-cue
interval* of the session after its go cue (`dft.trial_windows`). Movements after that get no
trial, and the last trial's `n_movements`, `n_licks` and coverage stop there. The old pipeline
gave all of those movements to the last trial.

## 3. Order of work

1. **Environment.** Pin `v0.5.2` in `environment/Dockerfile` (currently the v0.4.0 commit
   `a6bac04`) and rebuild. This is a deliberate `/environment` change.
2. **Shared modules.** They import functions that no longer exist, so they fail on import until
   updated:
   - `data_loading.py`, `build_all_tongue_movements.py`: `quality_summary` / `pipeline.is_done`,
     coverage 0.90 (§2.4).
   - `kin_utils.py`: `jaw_position` and `load_kps_raw` → `landmarks.parquet` / `pose.parquet` (§2.5).
   - `ephys_utils.py`:
     - Pass `movs`, not `kins`, to `get_events_dict`.
     - `build_trial_features`: merge `cue_response_movement_number` from `trials` and
       rename the `kcols` (§2.2, §2.7).
   - `lickometer_qc.py`: `pose.parquet` + `signal.mask`; session time from `pose.time` (§2.8).
   - `run_batch_analysis*.py`: `dft.run_batch(..., ignore_trial_errors=True)`.
3. **Pooled table.** Rebuild `all_tongue_movements` from `kinematics_v050` with the new filter, as
   a new dated file; keep `04022026` for comparison. Save `kinematics_v050` as a data asset, so
   it isn't just in scratch.
4. **Session folders.** Everything reads `SCRATCH / "session_analysis_mlk"` (79 references).
   Point those at the new root. First confirm it covers every session `session_analysis_mlk` has,
   including ephys sessions, which may need their own batch run.
5. **Notebooks**, one at a time, each checked against its old figures:

| Notebook | Definition items |
|---|---|
| `kin_00_movement_qc` | §2.5 `out_duration` filter, §2.7 dropped frames, §2.8 frame table |
| `kin_01_population` | §2.5 `out_duration` filter, §2.6 speeds |
| `kin_02_latency` | §2.2, §2.7 per-trial columns, §2.8 |
| `kin_03_umap` | §2.5, §2.6 (the feature set changes) |
| `kin_04_timecourse` | §2.2 first movement, §2.3 sessions |
| `kin_05_nonlick_movements` | §2.1 (pooled lick counts), §2.7 `movement_before_cue_response`, §2.8 |
| `kin_06_lick_geometry_choice` | §2.5 (jaw, spouts, `max_x_distance`) |
| `kin_07_value_encoding` | §2.5 (`max_x_distance`, `max_x_from_jaw_y`), §2.7 absolute columns |
| `eph_00`–`eph_09` | through `ephys_utils`: §2.2 `firstMove`, cue-response trials, §2.4 session list |
| `eph_07_bout_encoding` | §2.9 |
| `val_02`–`val_05`, `model_quality` | through `lickometer_qc`; §2.8 |
| `env_00`, `tongue_movements_all`, `verify_library_migration`, `test_session_*` | file names only; or archive them |

Not affected: `fip_*` and `men_*` (motion energy, not tongue tables).

## 4. Batch status (2026-10-28)

48 of 53 sessions done. Five stopped on timing checks, listed in `kinematics_v050/errors.txt`:
four Harp clock steps (758017 02-04, 751181 02-26, 754897 03-14, 758018 03-20) and one
trigger log one event short (761038 04-15). The 30 sessions from the first run lack the
`trials.reward_before_choice` column; rerun them with `force=True` to add it.

## 5. Checks to run on Code Ocean

Run with `PYTHONPATH=/root/capsule/scratch/kinematics_compare/dfbva_v050/src` until the image is
rebuilt.

**A. Spanning movements (§2.1)**

```python
import pandas as pd
from pathlib import Path
root = Path("/root/capsule/scratch/kinematics_v050")
mv = pd.concat([pd.read_parquet(p) for p in root.glob("*/movements.parquet")])
print((mv.groupby("session").spans_go_cue.sum() / mv.groupby("session").trial.max()).describe())
assert mv.loc[mv.spans_go_cue, "trial"].isna().all()  # true from 0.5.2
```

**B. Quality filter, old vs new (§2.4)**

```python
import json
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tasks import dynamic_foraging_tongue as dft
old_root = Path("/root/capsule/scratch/session_analysis_fall2026")
rows = []
for d in sorted(root.glob("behavior_*/record.json")):
    s, q = d.parent.name, dft.quality_summary(d.parent)
    o = old_root / s / "tongue_quality_stats.json"
    o = json.loads(o.read_text()) if o.exists() else {}
    rows.append({"session": s, "cov_new": q["lick_coverage"], "dur_new": q["movement_duration_p50"],
                 "cov_old": o.get("coverage_pct", float("nan")) / 100,
                 "dur_old": (o.get("percentiles", {}).get("duration", {}) or {}).get("0.5")})
t = pd.DataFrame(rows)
t["pass_new"] = (t.cov_new > 0.90) & (t.dur_new > 0.06)
t["pass_old"] = (t.cov_old > 0.90) & (t.dur_old.astype(float) > 0.06)
print(t.to_string(index=False))
```

**C. How many rows the `out_duration` filters keep (§2.5)**: on the old and new pooled tables,
count `out_duration > 0.01`, `> 0.05` and `< 5` per session.

## 6. Decisions for you

Settled 2026-10-28: spanning movements get no trial (§2.1), and the last trial ends at the
longest go-cue interval (§2.10).

1. Drop trials with `reward_before_choice` (§2.3)?
2. Keep exact `max_x_distance` / `max_x_from_jaw_y` (compute from frames), or move to
   `extent_x` / `end_*` (§2.5)?
3. The bout rule for `eph_07` (§2.9).
4. Whether to keep the absolute-pixel columns in `kin_07`'s value screen (§2.7).
