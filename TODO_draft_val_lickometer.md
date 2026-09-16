# DRAFT — Repair and modernize `tongue_lickometer.ipynb` → `val_02_lick_detection.ipynb`

_Drafted 2026-09-16 on `wild`. Staging file — see "Where this belongs" at the bottom for
which parts go to `TODO.md`, which to `REORG.md`, and which append to the existing
library/repo boundary item._

---

## Why

`tongue_lickometer.ipynb` (21 cells, last substantive edit 2025-03-07 `f969e15`; the
2025-07-14 commit was a bulk capsule mirror) is the repo's **only** measurement of
lick-detection **precision**. Everything else that touches this question measures recall
only. It is classified KEEP in `REORG.md` and it does not run.

Four separate problems, verified below. Two are cleanup; two change numbers.

### 1. It does not run — every domain import is a dead bare name

`tongue_kinematics_utils` and `tongue_lickometer_utils` were promoted into
`aind-dynamic-foraging-behavior-video-analysis` and are no longer in `code/`. Current
locations (both under `…behavior_video_analysis.kinematics`):

| Cell | Imported name | Lives now in |
|---|---|---|
| 4 | `load_keypoints_from_csv` | `tongue_kinematics_utils` (L1388) **and** `tongue_lickometer_utils` (L227) |
| 4 | `mask_keypoint_data` | both (`tku` L716 / `tlu` L202) |
| 6, 7 | `detect_licks` | both (`tku` L185 / `tlu` L158) — **different signatures** |
| 6, 7 | `filter_timestamps_refractory` | both (`tku` L61 / `tlu` L42) |
| 6, 7, 14 | `calculate_metrics` | both (`tku` L140 / `tlu` L116) — **different return arity** |
| 17 | `calculate_metrics_witheventkeys` | both (`tku` L79 / `tlu` L60) — **different return arity** |
| 20 | `extract_clips_ffmpeg_encode` | `tongue_lickometer_utils` (L6) only |

The package ships no `__init__.py` under `kinematics/` or `ephys/`, so the full dotted
path is the only import form — which is what every modern notebook in `code/` already
uses.

### 2. The library duplication is a correctness bug, not just cleanup

Diffed all six pairs on the library's `main` (`b21eac0`). **Two are byte-identical**
(`filter_timestamps_refractory`, `mask_keypoint_data`). **Four have drifted**, and three
of those drifts break call sites:

- **`detect_licks` — signature change.** `tlu`: `(tongue_df, spoutL, spoutR, threshold)`,
  reading times from `tongue_df['time']`. `tku`: `(tongue_df, timestamps, spoutL, spoutR,
  threshold)`, times from a separate series. Every call in cells 6, 7, 12, 13, 17 passes
  four positional args → `TypeError` against `tku`. Fails loudly, not silently.
- **`detect_licks` — ~225× performance gap.** `tlu` vectorizes the distance computation;
  `tku` is a pure-Python `tongue_df.iloc[i]` row loop. Benchmarked on a synthetic
  1.8 M-frame session: **0.2 s (`tlu`) vs 51.8 s (`tku`)**. This decides which definition
  survives.
- **`calculate_metrics` — return arity.** `tlu` returns `(tp, fp, fn)`; `tku` returns
  `(tp, fp, fn, tn)`. Cells 6, 7, 14 unpack three → `ValueError` against `tku`. The `tku`
  `tn` is meaningless anyway: `total_observations = max(last event time)` is a **time in
  seconds**, and `tn = total_observations - (tp+fp+fn)` subtracts counts from it. Do not
  propagate it.
- **`calculate_metrics_witheventkeys` — return arity.** 5-tuple (`tlu`) vs 6-tuple
  (`tku`), same bogus `tn` inserted at position 3. Cell 17 unpacks five.
- **`load_keypoints_from_csv` — `tku` is strictly better**: reads `dtype=str` then
  `pd.to_numeric(errors='coerce')`, so mixed-type header rows don't poison columns.
  `tlu`'s version does not. This is the one function where `tlu` is the worse copy.

Blast radius is small: across the library **and** `code/`, the only consumers of these six
names are `tongue_lickometer.ipynb` and `tongue_kinematics.ipynb` (HOLD). **Nothing in the
library's own pipeline calls `detect_licks`** — `run_batch_analysis` →
`generate_tongue_dfs` uses segmentation + `annotate_licks_in_kinematics` instead.

### 3. The fp/fn convention is inverted, and inconsistent within the notebook

`calculate_metrics(ground_truth, detected_events, time_window)`. Cells 6, 7, 14 and 17
call it as `calculate_metrics(LP_licks, all_licks, …)` — **video as ground truth,
lickometer as the detector**. That inverts every asymmetric rate the notebook prints:

| Notebook variable | What it actually counts |
|---|---|
| `fp` | lickometer licks the **video missed** (conventionally FN) |
| `fn` | video detections with **no lickometer lick** (conventionally FP) |
| `recall = tp/(tp+fn)` | **precision** of the video detector |
| `precision = tp/(tp+fp)` | **recall** of the video detector |
| `false_negative_rate` | FDR |
| `false_discovery_rate` | FNR |

From the notebook's own stored output, cell 16 (30 px / 0.1 s / 0.1 s): `tp=5411,
fp=435, fn=180`, printed as `recall 0.9678, precision 0.9256`. Read correctly, the video
detector's **recall is 0.926** and its **precision is 0.968** — the two are swapped. Cell
8's top-5 table has the same transposition.

**`f1_score = 2tp/(2tp+fp+fn)` is symmetric in fp/fn, so every F1 heatmap, the facet grid,
and the 30–35 px / 0.1 s / 0.1 s parameter conclusion are unaffected.** The headline result
stands; the rate labels on it do not.

Cells 19–20 make it worse by being inconsistent with themselves: `FP_times` is read from
`licko_licks_classified` (the *lickometer* frame) and fed to
`extract_clips_ffmpeg_encode` writing into `…/false_positive/`. Those clips are lickometer
licks the video **missed**. Cell 5's markdown ("sensitivity and specificity of lickometer
wrt LP licks") suggests the inversion may have been deliberate at some point; cell 20's
output directory says otherwise. Either way it needs pinning by assertion, not by comment.

Secondary, lower-confidence: `calculate_metrics` uses a greedy two-pointer match that
advances the ground-truth index on mismatch. With a 100 ms window and real ILIs of
100–150 ms, an interleaved run can be matched sub-optimally. Not obviously wrong, but it
is not mutual-nearest-neighbour and the notebook never says so.

### 4. It predates every current convention

- Cells 0–1 are bare `pip install` (`aind_dynamic_foraging_basic_analysis`,
  `aind-dynamic-foraging-data-utils@main`). Both are Dockerfile-pinned now
  (`environment/Dockerfile`, the `-e git+…@main` block).
- No ENV block. Cell 3 hardcodes `/root/capsule/data/foraging_nwb_bonsai`; cell 4
  hardcodes `/root/capsule/data/matt_test_DLC_LP_results_20240920/…` and a raw
  `behavior-videos/bottom_camera.csv`; cell 20 hardcodes
  `/root/capsule/data/video_preds_labeltest/…`. Compare `kin_00` cell 3 / `kin_06` cell 3.
- No `plotstyle.py`. Raw `sns.heatmap` / `sns.FacetGrid` defaults (cells 9, 10), `plt.grid()`,
  hand-picked `'YlGnBu'`/`'b'`/`'r'`/`'g'`.
- Cell 3 selects the session by walking a `glob` of NWB filenames and matching
  `anim_name = '716325'` + `session_date = '05-31-2024'`. The date comparison compares a
  `datetime` to the string `'05-31-2024'` and is always `False` — it only works because
  the animal filter already leaves one row.
- Cell 4 reimplements `trim_kinematics_timebase_to_match` inline; the library's
  `integrate_keypoints_with_video_time` (`tku` L1172) owns this now.
- **Keypoint and threshold drift.** Cell 4 masks `'tongue_tip'` at `confidence_threshold=0.8`.
  The current pipeline (`generate_tongue_dfs`, and `analyze_tongue_movement_quality`'s
  `keypt`) uses **`'tongue_tip_center'` at `0.90`**. The 30 px / 0.1 s parameters were
  fitted to a different keypoint from a 2024 model at a looser confidence gate.

That last point is not cosmetic. `detect_licks` emits a lick on the **rising edge** of
"within threshold of a spout", and NaN frames are skipped **without resetting the
`is_licking` state** (true in both copies). So the detector needs the tongue to be tracked
during *retraction* in order to re-arm. Tightening confidence 0.8 → 0.90 drops exactly
those low-confidence retraction frames, which merges adjacent licks and inflates ILIs.
Verified as a failure mode, not just a worry: on synthetic frames where the tongue is only
ever tracked near the spout, `detect_licks` returns **1 event for the whole session**. So
both the chosen parameters *and* the ILI argument in cells 12–13 have to be re-derived at
0.90, not carried over.

### 5. It is single-session, and the pooled machinery now exists

One session: `behavior_716325_2024-05-31_10-31-14`. That session **is** in
`all_tongue_movements_04022026.parquet` (it is 1 of 44, across 15 subjects), so it has
already been reprocessed by the current pipeline — a direct old-model/new-model comparison
on the same session is available.

---

## What to do

### A. Re-point the data loading at `intermediate_data/`, not raw CSVs

This is the single change that removes most of the rot. `run_batch_analysis` already writes,
per session, under `SCRATCH / "session_analysis_mlk" / <session_id> / "intermediate_data"/`
(the path `kin_05` §4 and `kin_06` §3 use):

- `tongue_kins.parquet` — per-frame, with `time_in_session` and NaN-masked `x`/`y`.
  Confirmed by reading `segment_movements_trimnans`: it only assigns/clears `movement_id`,
  it **does not drop rows**, so every frame survives. This is the substrate `detect_licks`
  needs.
- `kps_raw_spout_l.parquet` / `kps_raw_spout_r.parquet` — spout positions, replacing cell
  6/7's `np.mean(keypoint_dfs_trimmed['spout_r'], 0)`.
- `nwb_df_licks.parquet` — lickometer `timestamps`, replacing cell 6's
  `nwb.acquisition["left_lick_time"].timestamps - keypoint_timebase[0]`.

Consequences to handle explicitly:

- **Delete cells 3 and 4 entirely.** The NWB glob, `parseSessionID`,
  `trim_kinematics_timebase_to_match`, the raw-CSV load, the manual time-zeroing, and the
  masking step all collapse into two `pd.read_parquet` calls. This also removes the
  dependency on the `matt_test_DLC_LP_results_20240920` asset.
- **Clock.** Use `tongue_kins['time_in_session']` against `nwb_df_licks['timestamps']`
  directly. `annotate_licks_in_kinematics` compares exactly these two at a 10 ms tolerance
  and the pipeline reports >90 % coverage, so they share an origin. State this in a markdown
  cell; the old hand-rolled `- keypoint_timebase[0]` offset should not come back.
- **Flag the substrate change.** `tongue_kins` x/y are **post-`kinematics_filter`** (50 Hz
  Butterworth + cubic interpolation), whereas cell 4's `tongue_masked` was unfiltered.
  Filtering suppresses exactly the jitter that the refractory filter exists to remove, so
  the refractory argument in cells 12–14 may come out weaker. That is a finding, not a
  problem — but the notebook must say which series it ran on. If the unfiltered series is
  wanted, `kps_raw_tongue_tip_center.parquet` + `mask_keypoint_data` reconstructs it.
- **Keep the L/R spout swap note.** `spout_l` is the animal's **right** spout (bottom camera
  mirrors). Cells 6/7 already compensate; `kin_06` cell 9/10 documents the same swap. The
  swap is irrelevant to detection (min distance to either spout) but matters for the
  `nearest_spout` label in cell 18.

### B. Pick the import path now; batch the library fix

Import **only** from `tongue_lickometer_utils`, by full dotted path:

```python
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_lickometer_utils import (
    detect_licks, filter_timestamps_refractory, calculate_metrics,
    calculate_metrics_witheventkeys,
)
```

Rationale: it is the only module whose four signatures match the call sites, it is 225×
faster on `detect_licks`, and it carries no bogus `tn`. After change A the notebook needs
neither `load_keypoints_from_csv` nor `mask_keypoint_data`, so `tku`'s better
`load_keypoints_from_csv` is not a reason to split imports across modules.

Do **not** block this notebook on a library PR. The library fix is a deletion plus a pin
bump plus a Docker rebuild; `TODO.md` already says to batch library changes with the
outbound-metrics consolidation. Append the concrete resolution to the existing boundary
item (text in "Where this belongs" below) so it is not re-argued per function.

### C. Pin the fp/fn convention with an assertion, not a comment

Add a short section (new §2) that calls `calculate_metrics` on hand-built synthetic event
lists — one guaranteed hit, one detector-only event, one truth-only event — and asserts
which output slot each lands in. This runs locally with no data and is how the inversion in
§3 above was found. Then define, once, at the top:

```python
# ground_truth = lickometer, detected_events = video. Everything downstream is
# stated from the video detector's point of view.
tp, fp, fn = calculate_metrics(licko_times, video_times, overlap_s)
```

and swap the argument order at every call site (cells 6, 7, 14, 17). Re-derive
`recall`/`precision`/`FNR`/`FDR` from that. **F1 will not move**; every other rate will
transpose. Say so in the markdown, with the old numbers, so the change is auditable against
the stored outputs.

Correspondingly, in cell 19/20: `FP_times` must come from the **video** classified frame
and `FN_times` from the **lickometer** frame — the opposite of what is there now — and the
clip output directories follow.

### D. Pool the sweep across sessions

Benchmarked: the full cell-7 grid (9 spatial × 11 refractory × 50 overlap = 4 950
evaluations) takes **~38 s/session** on a synthetic 1.8 M-frame session →
**~28 min for 44 sessions**, plus parquet I/O. Affordable.

- Loop `process_session`-style over the sessions that have `intermediate_data/`, gated by
  `data_loading.load_session_quality_filter` so the sweep uses the same inclusion set as
  every `kin_*`/`eph_*` notebook.
- Emit one tidy frame: `session × spatial_threshold × t_refractory × time_threshold ×
  {tp, fp, fn, precision, recall, FDR, FNR, F1}`. **Cache it** — it is small (44 × 4 950
  rows) and it makes every figure below re-runnable without the 28-minute loop. Per the
  data-intermediates decision, cache it under `SCRATCH`, do not add it to the two
  canonical intermediates.
- **Session is the sampling unit**: the heatmaps (cells 9, 10) plot mean F1 across
  sessions, and the argmax is reported with a spread, not as a point. Add a per-session
  argmax scatter — whether the optimum is stable across sessions *is* the methods result,
  and a single session cannot answer it.
- Report by **subject** as well (15 subjects); sessions within a subject share a camera
  and spout geometry, so 44 sessions are not 44 independent draws of "what threshold works".
- Keep **one named `EXAMPLE_SESSION`** — `behavior_716325_2024-05-31_10-31-14`, the session
  the notebook already used — for the per-event inspection (cells 17–19), the ILI
  histograms (cell 13) and the video clips (cell 20). Those are illustrations, not
  estimates, and pooling them buys nothing.

On "is a single well-characterized session the honest scope?" — it would be, if the
intermediates were not already sitting there for 44. Given they are, and given the single
session leaves 30 px vs 35 px genuinely unresolved (cell 11 openly waffles about it),
pooling is both cheap and the thing that answers the open question. Gate it: if fewer than
~10 sessions have usable `intermediate_data/`, fall back to the single session and say so.

### E. Sections to carry over, with cell provenance

Proposed shape for `val_02_lick_detection.ipynb`, following `kin_00`/`kin_06`'s
header → setup → ENV → numbered sections:

| § | Content | From |
|---|---|---|
| 1 | Setup, `plotstyle.apply_style()`, ENV block | new (copy `kin_06` cell 3) |
| 2 | Scoring-convention self-test (synthetic events, assertions) | **new** — see C |
| 3 | Load `intermediate_data/`; detector definition and its known failure mode | replaces cells 3–4 |
| 4 | Single-session worked example at the chosen parameters | cell 6 |
| 5 | Parameter sweep, pooled across sessions | cells 7–8, extended per D |
| 6 | F1 heatmaps (3 pairwise) + facet grid over `t_refractory` | cells 9–10, restyled |
| 7 | ILI argument for the refractory filter (CDF by threshold; LP vs lickometer ILI histograms) | cells 12–13 |
| 8 | Overlap-threshold justification: rate curves vs `time_threshold`, and why 100 ms | cells 14–16 |
| 9 | Per-event FP/FN inspection + distance to nearest spout | cells 17–19 |
| 10 | Labeled video clips at example FP/FN timepoints (CO only) | cell 20 |
| 11 | **New**: the same scoring applied to the detector the pipeline actually uses | see F |

Cells 11 and the trailing comment blocks become markdown. Cell 19's inline
`plot_tongue_trajectory` stays in the notebook (one consumer — same call `kin_06` made for
`plot_standard_lick_landmarks`), restyled onto `plotstyle`'s `PALETTE`/`style_ax`.

For §10, keep `extract_clips_ffmpeg_encode`. Do **not** substitute
`video_clip_utils.extract_clips_ffmpeg_after_reencode`: that one uses `-c copy`, which
snaps to keyframes and will not land on an exact event timepoint unless the input was
pre-re-encoded. `extract_clips_ffmpeg_encode`'s `libx264` re-encode is the correct tool
here. Its eventual home is `video_clip_utils` (alongside the other clip helpers), not
`tongue_lickometer_utils` — note it in the boundary item.

### F. Add §11 — score the detector that is actually in use

The most important finding from this audit: **`detect_licks` feeds nothing.** The pipeline's
video↔lick correspondence is `annotate_licks_in_kinematics(tolerance=0.01)` +
`assign_movements_to_licks`, and it is `nearest_movement_id` / `has_lick` / `lick_time`
that flow into `all_tongue_movements.parquet` and every downstream `kin_*`/`eph_*`
notebook. So the sweep, as written, tunes a detector no current analysis uses.

That is not a reason to drop the sweep — it is the only precision measurement in the repo,
and it validates the underlying claim that video can detect licks at all. But §11 should
put the **live** detector on the same footing: take each session's movement start times
(or `lick`-annotated frames) as the detected-event series, score them against
`nwb_df_licks['timestamps']` with the same `calculate_metrics`, and sweep the one free
parameter the live path has — the 10 ms `tolerance`. That yields the precision number for
the detector the manuscript actually depends on, which currently does not exist anywhere.

### G. Supersession check — what NOT to re-port

- **`analyze_tongue_movement_quality` → `tongue_quality_stats.json`** (`tongue_analysis.py`
  L152–L311) computes `coverage_pct` = fraction of lickometer licks with a matched
  movement. That is a **recall** measure for the movement detector, already computed for
  every session and already pooled by `test_session_quality_analysis.ipynb`, and it is what
  `data_loading.load_session_quality_filter` thresholds at 90 %. **Do not re-derive
  recall-across-sessions from scratch** — cite `coverage_pct` and, in §11, show the two
  agree. It has no false-positive side, no sweep, and no precision/F1: those remain unique
  to this notebook.
- **`pixel_error.ipynb`** measures Lightning-Pose pixel error against human labels per
  keypoint. Different layer (keypoint accuracy, not event detection). **No overlap.**
- **`kin_05` §4** already inspects licks-without-movements with per-event trace plots —
  the same per-event FN inspection, for the movement detector. §9 here should
  cross-reference it rather than duplicate the plotting; what §9 adds is the FP side and
  the distance-to-nearest-spout panel.
- **`model_quality.ipynb`** — note that `REORG.md` describes this as "foraging behavioral-model
  quality". **That is wrong.** Reading it: it is a single-session Lightning-Pose/tongue-pipeline
  walkthrough (load → mask `tongue_tip_center` @0.90 → filter → segment → annotate → lick
  coverage → confidence analysis), already on modern library imports. Its lick-coverage
  cells (6, 12) are the single-session version of `coverage_pct`. Correct the REORG line;
  decide separately whether its unique content folds into §11 here.

---

## Notes

### What is verified vs unverified

**Verified** (locally, this session):
- All six function pairs diffed against the library's `main` (`b21eac0`); signature and
  arity drifts as listed.
- `detect_licks` timing 0.2 s vs 51.8 s; full-grid cost 38 s/session; the
  "tongue only tracked near the spout → 1 event/session" failure mode.
- The fp/fn inversion, against the notebook's own stored cell-8 and cell-16 outputs.
- Only `tongue_lickometer.ipynb` and `tongue_kinematics.ipynb` consume the six names;
  nothing in the library pipeline calls `detect_licks`.
- `segment_movements_trimnans` preserves all rows; `tongue_kins` derives from
  `kinematics_filter` output.
- `data/for_local/` holds the pooled parquet (246 359 × 49, 44 sessions, 15 subjects) and
  nothing else relevant — **no per-frame keypoints, no spout positions, no lick times**.
- `behavior_716325_2024-05-31_10-31-14` is present in the pooled parquet.

**Unverified — Code Ocean only, or could not be reached this session:**
- Whether the `matt_test_DLC_LP_results_20240920` and `video_preds_labeltest` data assets
  are still attached. Change A removes the need for the first; §10 still needs a labeled
  video for the second.
- How many of the 44 sessions actually have `intermediate_data/` under
  `session_analysis_mlk`. `kin_05`/`kin_06` assume the path; neither has run there.
- Real session frame counts. The 38 s/session figure assumes 1.8 M frames (1 h @ 500 Hz).
- That `nwb_df_licks['timestamps']` and `tongue_kins['time_in_session']` share an origin —
  inferred from `annotate_licks_in_kinematics` comparing them at 10 ms and coverage >90 %,
  not measured.
- `detect_licks_multiple` (used by `tongue_kinematics.ipynb` cells 103–105) exists **nowhere**:
  not in the library `main` or `fix/video-csv-header`, not in `code/`, not in local library
  history. The `LC_manuscript` branch could not be fetched this session, so this is
  "not found" rather than "does not exist". See the REORG correction below.

### Local vs Code Ocean, for the notebook header

- **Local**: §1, §2 (the synthetic self-test) and the notebook's structure. Nothing else —
  the sweep needs per-frame keypoints and NWB lick times, and neither is in
  `data/for_local/`. Expect the header to read closer to `eph_09`'s ("Code Ocean only") than
  `kin_06`'s.
- **Code Ocean**: §3–§11.

### Ordering

Do A → B → C first; that is the repair, and it is testable on the example session alone.
D (pooling) is a separate, larger step and should be a second commit. F is the part most
likely to change what anyone concludes, but it depends on A.

### Risk

C changes published-looking numbers. Keep the current stored outputs visible (quote them in
the markdown) so the transposition is auditable rather than silent. The parameter choice
(30 px / 0.1 s / 0.1 s) does not change from C alone — but it may well change from the
0.8 → 0.90 confidence shift in §4, and that is the point of re-running it.

---

## Where this belongs

**Both** — and one append.

1. **`TODO.md`** — everything from "Why" through "Notes" above, as one item titled
   *"Repair and modernize `tongue_lickometer.ipynb` → `val_02_lick_detection.ipynb`"*,
   placed after the `eph_09` item and before the library/repo boundary item (which it
   references).

2. **`REORG.md`** — a new prefix and four corrections, all small:

   - Rename the section **"KEEP — model-quality / methods evaluation"** to
     **"KEEP — `val_*` methods and model-quality validation"** and give it the layering the
     other series have:

     | Slot | Notebook | Layer |
     |---|---|---|
     | `val_00` | `pixel_error.ipynb` | keypoint accuracy vs human labels |
     | `val_01` | `test_session_quality_analysis.ipynb` | session-level quality + selection |
     | `val_02` | `tongue_lickometer.ipynb` → this item | lick-detection precision/recall vs lickometer |

     **They should not move together.** Only `val_02` is being rewritten; renaming
     `pixel_error` and `test_session_quality_analysis` is a `git mv` plus a reference sweep
     with no analysis change, and doing it now means three notebooks in flight at once.
     Record the intended slots in REORG, rename each when it is next touched. Accept the
     numbering gap in the meantime — `eph_*` already has gaps.

   - **`test_session_wrapper.ipynb` is misfiled.** It is a batch-runner wrapper (19 cells,
     most commented out, wrapping `run_batch_analysis`) — operations, not evaluation. Move
     its line to **"KEEP — pipeline / data generation"** now. One-line edit, no file change,
     no `val_` slot.

   - **Correct the `model_quality.ipynb` description** (see §G): it is a single-session
     Lightning-Pose pipeline walkthrough, not foraging behavioral-model quality. It gets no
     `val_` slot until someone decides whether its coverage cells fold into `val_02` §11 or
     it is archived as a pipeline demo.

   - **Correct the "Already covered elsewhere — do not port" line.** It currently says
     `tongue_kinematics` cells 78–79 duplicate `tongue_lickometer.ipynb`. True for 78–79.
     But **cells 103–105 are a different analysis** — a multi-keypoint detector
     (`detect_licks_multiple` over `tongue_center`/`tongue_left`/`tongue_right`) that the
     lickometer notebook does not have and that calls a function existing nowhere. It
     cannot run and cannot be ported as-is. Either re-derive it inside `val_02` as a
     "multi-keypoint variant" section or declare it dropped — but `tongue_kinematics.ipynb`
     should not be archived under the claim that it is already covered.

3. **Append to the existing "Define the boundary between the library and this repo's
   `code/` modules" item**, under *What to do* → "Resolve the library-internal duplication
   in (1)". The resolution is now concrete rather than open:

   > Resolved by inspection 2026-09-16 (see the `val_02` item). Keep
   > `tongue_lickometer_utils`'s `detect_licks` (vectorized; 225× faster than
   > `tongue_kinematics_utils`'s row-loop copy), `calculate_metrics` and
   > `calculate_metrics_witheventkeys` (the `tku` copies add a `tn` that subtracts counts
   > from a timestamp — delete, do not merge). Keep `tongue_kinematics_utils`'s
   > `load_keypoints_from_csv` (coerces to numeric; `tlu`'s does not).
   > `filter_timestamps_refractory` and `mask_keypoint_data` are byte-identical — keep one,
   > either side. Net: `tongue_lickometer_utils` owns detection + scoring,
   > `tongue_kinematics_utils` owns keypoint I/O, neither duplicates the other. Also move
   > `extract_clips_ffmpeg_encode` from `tongue_lickometer_utils` to `video_clip_utils`
   > (its `libx264` re-encode is the correct precise-seek variant; do not replace it with
   > `extract_clips_ffmpeg_after_reencode`, which is `-c copy` and keyframe-snapped).
   > Only two consumers exist ecosystem-wide — `tongue_lickometer.ipynb` (→ `val_02`) and
   > `tongue_kinematics.ipynb` (HOLD) — so this is a low-risk deletion, and `val_02` can
   > ship against `tongue_lickometer_utils` before the library PR lands.
