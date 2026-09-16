# DRAFT — Repair and refocus `tongue_lickometer.ipynb` → `val_02_spout_contact_detection.ipynb`

_Drafted 2026-09-16 on `wild`, revised after scope discussion. Staging file — see "Where this
belongs" at the bottom._

---

## The question this notebook asks

**Can Lightning-Pose tongue tracking detect a lick, where a lick is defined as the tongue
reaching the spout?**

This is a *spatial contact* definition, and it is the reason the notebook exists. Nothing
else in the repo or the library asks it:

| | Detector | Lick is… | Ground truth |
|---|---|---|---|
| **This notebook** | `detect_licks` — tongue within `threshold` px of either spout | **contact with the spout** | lickometer (for now) |
| Library pipeline / `kin_*` | `segment_movements_trimnans` → `annotate_licks_in_kinematics(tol=0.01)` | any tracked tongue excursion, matched to a lickometer time | lickometer (always) |

The library's path is **agnostic to whether the tongue reached the spout** — it segments
tongue movements and then asks whether a lickometer event happened nearby in time. Which is
why `kin_05` has a whole population of "movements without licks": under the library's
definition those are unexplained, and under this notebook's definition many of them are
simply protrusions that never reached the spout.

So `detect_licks` is not a stale duplicate of the pipeline's detector. It is the only
implementation of the contact definition, and it is the only thing in the repo that can
measure **precision** — how often video says "contact" when the lickometer says nothing.

### Where this is going

The endpoint is not "tune a detector to agree with the lickometer." It is to find a
parameter set at which pose tracking is trustworthy enough to serve as **QC on the
lickometer itself** — catching lickometer misses (bad contact, capacitive dropout) and
lickometer false triggers (spout jostling, paw contact).

That goal has a direct, concrete consequence for the analysis, and it is the main change in
this revision:

**F1 is the wrong objective.** F1 weights precision and recall equally. The two QC use cases
want opposite asymmetries:

- To flag a **lickometer miss** — pose says contact, lickometer says nothing — that event
  must be credible. Requires **high pose precision**, at whatever recall.
- To flag a **lickometer false trigger** — lickometer fires, pose sees no contact — the
  absence must be credible. Requires **high pose recall**, at whatever precision.

A single F1 argmax (the current cells 8–11 conclusion) is right for neither. The sweep
should produce a **precision–recall surface** over the parameter grid with two named
operating points, and report F1 only as a summary statistic. This also dissolves the
notebook's own unresolved 30-vs-35 px waffling in cell 11 — those are two different
operating points, not two candidates for one answer.

### Recasting the reference direction

Last pass I flagged the `calculate_metrics(LP_licks, all_licks, …)` argument order as an
inversion bug. Under the stated goal it is not a bug — treating pose as the reference *is*
the lickometer-QC framing, and cell 5's markdown ("sensitivity and specificity of lickometer
wrt LP licks") says the author was already thinking that way.

The real defect is that the notebook **never states which direction it is reporting**, and
contradicts itself. Cells 6/7/14 print bare `recall` / `precision` / `false_negative_rate`;
cells 19–20 then take `FP_times` from the *lickometer* frame and write clips into
`…/false_positive/`, where they are actually lickometer licks the video missed.

Fix: report **both directions, always, with names that carry the direction** — never bare
`precision`/`recall`. From the notebook's own stored cell-16 output (30 px / 0.1 s / 0.1 s,
`tp=5411, fp=435, fn=180`):

- `pose_recall_vs_lickometer` = 5411/5846 = **0.926** — of lickometer licks, the fraction
  pose also called contact
- `pose_precision_vs_lickometer` = 5411/5591 = **0.968** — of pose contacts, the fraction the
  lickometer confirmed
- 180 pose-only events and 435 lickometer-only events — **these two populations are the
  eventual QC product**, not error terms to be minimized away

`f1_score = 2tp/(2tp+fp+fn)` is symmetric, so every existing F1 heatmap and facet grid is
correct as drawn regardless of direction. Nothing already plotted has to be thrown out.

---

## What is broken

Agreed and settled; kept short since the fix is not in dispute.

**Every domain import is a dead bare name.** `tongue_kinematics_utils` and
`tongue_lickometer_utils` were promoted into
`aind-dynamic-foraging-behavior-video-analysis`; the package ships no `__init__.py` under
`kinematics/`, so the full dotted path is the only import form (what every modern notebook
in `code/` already uses).

**Both library modules define the same six names, and four have drifted** (diffed against
library `main`, `b21eac0`):

| Function | `tongue_lickometer_utils` | `tongue_kinematics_utils` | Verdict |
|---|---|---|---|
| `detect_licks` | `(tongue_df, spoutL, spoutR, threshold)`, vectorized | `(tongue_df, timestamps, spoutL, spoutR, threshold)`, row loop | **`tlu`** — signature matches all call sites, and 225× faster (0.2 s vs 51.8 s/session, benchmarked on 1.8 M frames) |
| `calculate_metrics` | `(tp, fp, fn)` | `(tp, fp, fn, tn)` | **`tlu`** — `tku`'s `tn` subtracts counts from a timestamp; it is meaningless |
| `calculate_metrics_witheventkeys` | 5-tuple | 6-tuple, same bogus `tn` | **`tlu`** |
| `load_keypoints_from_csv` | plain `read_csv` | `dtype=str` + `to_numeric(errors='coerce')` | **`tku`** — the one function where `tlu` is worse |
| `filter_timestamps_refractory` | — | — | byte-identical |
| `mask_keypoint_data` | — | — | byte-identical |

Against `tku`, cells 6/7/12/13/17 raise `TypeError` (4 positional args to a 5-param function)
and cells 6/7/14 raise `ValueError` (unpacking 3 from a 4-tuple). Loud failures, not silent
wrong numbers.

Blast radius is tiny: across the library **and** `code/`, the only consumers of these six
names are this notebook and `tongue_kinematics.ipynb` (HOLD). Nothing in the library's own
pipeline calls `detect_licks`.

**Convention rot.** `pip install` in cells 0–1 (both Dockerfile-pinned now); no ENV block
(three hardcoded `/root/capsule/data/...` paths); no `plotstyle`; cell 3 selects the session
by globbing NWB filenames with a date comparison (`datetime` vs the string `'05-31-2024'`)
that is always `False` and only works because the animal filter already leaves one row; cell
4 reimplements `trim_kinematics_timebase_to_match`, which
`integrate_keypoints_with_video_time` owns now.

---

## What to do

### A. Re-point data loading at `intermediate_data/` — fully verified path

`run_batch_analysis` writes, per session, under
`SCRATCH / "session_analysis_mlk" / <session_id> / "intermediate_data"/`. The reconstruction
is exact — I traced every column:

```python
inter = SESSION_DIR / EXAMPLE_SESSION / "intermediate_data"

# positions: raw masked keypoint, NOT tongue_kins — see below
kps       = {"tongue_tip_center": pd.read_parquet(inter / "kps_raw_tongue_tip_center.parquet")}
tongue    = mask_keypoint_data(kps, "tongue_tip_center", confidence_threshold=CONF)

# clock: same origin as nwb_df_licks['timestamps']
t0        = float(pd.read_parquet(inter / "nwb_df_trials.parquet")["goCue_start_time"].iloc[0])
tongue["time"] = tongue["time_raw"] - t0

spoutL    = pd.read_parquet(inter / "kps_raw_spout_r.parquet")[["x","y"]].mean()  # animal's left
spoutR    = pd.read_parquet(inter / "kps_raw_spout_l.parquet")[["x","y"]].mean()  # animal's right
licko     = pd.read_parquet(inter / "nwb_df_licks.parquet")["timestamps"].to_numpy()
```

- Every `kps_raw_*.parquet` carries `x`, `y`, `confidence`, `time` and `time_raw`
  (`integrate_keypoints_with_video_time` Step 5 inserts the last two; `time` is
  `Behav_Time − Behav_Time[0]`, *exactly* what cell 4 built by hand).
- `detect_licks` reads `tongue_df['time']`, so writing `time_in_session` into that column
  makes its output directly comparable to `nwb_df_licks['timestamps']` with **no offset
  arithmetic anywhere**. The hand-rolled `- keypoint_timebase[0]` does not come back.
- The `spout_l`/`spout_r` swap (bottom camera mirrors L/R) is preserved, matching cells 6/7
  and `kin_06` §3.

**Use `kps_raw_*` for positions, not `tongue_kins.parquet`.** `tongue_kins` is
post-`kinematics_filter`, which cubic-interpolates across NaN gaps, low-pass filters, then
reindexes back onto the full frame set. So the tracked frames at the *edges* of each
protrusion carry values contaminated by interpolation over the untracked gap — and the edges
of a protrusion are exactly where a contact threshold gets crossed. For a contact question
the honest input is the raw masked keypoint, which is also what the original notebook used.
(`tongue_kins` does keep every frame — I checked; the `kinematics_filter` docstring's
"retains only the originally available time points" refers to values, not rows.)

**Delete cells 3 and 4 entirely.** The NWB glob, `parseSessionID`, the broken date filter,
`trim_kinematics_timebase_to_match`, the raw-CSV load and the manual time-zeroing all
collapse into the block above. This also drops the dependency on the
`matt_test_DLC_LP_results_20240920` asset.

### B. Import only from `tongue_lickometer_utils`, full dotted path

```python
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_lickometer_utils import (
    detect_licks, filter_timestamps_refractory, calculate_metrics,
    calculate_metrics_witheventkeys,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (
    mask_keypoint_data,
)
```

(`mask_keypoint_data` is byte-identical in both — taken from `tku` only because that is where
the surviving copy should live, per the boundary-item text below.)

Do **not** block on a library PR. The library-side fix is a deletion plus a pin bump plus a
Docker rebuild; `TODO.md` already says to batch library changes with the outbound-metrics
consolidation. Concrete resolution text to append to the boundary item is at the bottom.

### C. Make confidence threshold an analysis axis, not a config constant

The original ran `'tongue_tip'` at `confidence_threshold=0.8`; the current pipeline uses
`'tongue_tip_center'` at `0.90`. Under the contact framing this is not drift to be silently
fixed — it is a parameter that acts directly on the detector's mechanism:

`detect_licks` emits a lick on the **rising edge** of "within threshold of a spout", and NaN
frames are skipped **without resetting the `is_licking` state** (true in both library
copies). The detector therefore needs the tongue tracked during *retraction* in order to
re-arm. Confidence gating is what decides whether retraction frames survive. Tightening
0.8 → 0.90 drops exactly those frames, merging adjacent contacts and inflating ILIs.

This is a demonstrated failure mode, not a worry: on synthetic frames where the tongue is
only ever tracked near the spout, `detect_licks` returns **one event for the entire
session**.

So: run the grid at **0.8 and 0.90** at minimum (ideally 0.7/0.8/0.9/0.95) and show the
effect on both the detection counts and the ILI distribution. The whole-grid cost is ~38 s
per confidence level on one session — adding this axis is far cheaper than adding sessions,
and unlike sessions it changes the answer.

### D. Express the spatial threshold in a transferable unit

The threshold is in **pixels**. Pixels are not a fixed physical quantity across sessions —
camera position, zoom and spout placement all vary, so "30 px" means a different physical
distance in each session. `kin_06` §3 already computes the jaw↔spout distance per session
from `kps_raw_jaw` / `kps_raw_spout_*` and uses it as the pixel scale.

Add a second x-axis (or a paired panel) expressing the threshold as a **fraction of the
jaw→spout distance**. Costs one extra parquet read and no compute. This matters more than it
sounds: see the sweep discussion below.

### E. Scoring convention, pinned by assertion

New §2: call `calculate_metrics` on hand-built synthetic event lists — one guaranteed
coincidence, one pose-only event, one lickometer-only event — and assert which output slot
each lands in. Runs locally, needs no data, and is how the labelling ambiguity above was
found in the first place.

Then define the direction once, at the top, and derive both directional rates from it. Never
print a bare `precision` or `recall`. Fix cells 19–20 to match: pose-only events come from
the pose-classified frame, lickometer-only events from the lickometer frame, and the clip
output directories follow.

Known limitation to state rather than fix: `calculate_metrics` uses a greedy two-pointer
match that advances the ground-truth index on mismatch. With a 100 ms window and real ILIs
of 100–150 ms, an interleaved run can be matched sub-optimally. It is not
mutual-nearest-neighbour, and the notebook should say so.

### F. Sections, with cell provenance

| § | Content | From |
|---|---|---|
| 1 | Header (the contact definition, and why it differs from the library's), setup, `plotstyle`, ENV | new — copy `kin_06` cell 3 |
| 2 | Scoring-convention self-test | **new** (E) |
| 3 | Load `intermediate_data/`; detector mechanism and its re-arming failure mode | replaces cells 3–4 |
| 4 | Worked example at one parameter set, both directional rates | cell 6 |
| 5 | Sweep: spatial × refractory × overlap × **confidence**, one session | cells 7–8, + C |
| 6 | **Precision–recall surface** + two named QC operating points; F1 heatmaps and facet grid demoted to summary | cells 9–10, restyled, reframed |
| 7 | ILI argument for the refractory filter, at each confidence level | cells 12–13 |
| 8 | Overlap-threshold justification (rate curves vs `time_threshold`; why 100 ms) | cells 14–16 |
| 9 | Per-event inspection of the two disagreement populations + distance to nearest spout | cells 17–19 |
| 10 | Labeled video clips at example disagreement timepoints (CO only) | cell 20 |
| 11 | Short markdown: how this relates to `coverage_pct` and `kin_05` — **no new analysis** | **new** |

Cell 11 and the trailing comment blocks become markdown. Cell 19's inline
`plot_tongue_trajectory` stays in the notebook (one consumer, same call `kin_06` made for
`plot_standard_lick_landmarks`), restyled onto `plotstyle`.

§9 is where the QC goal actually gets exercised: the pose-only and lickometer-only
populations are the *output*, not residual error. Inspect them for the signatures you would
expect — a lickometer-only event with the tongue nowhere near the spout is a candidate
lickometer false trigger; a pose-only event with a clean full protrusion to the spout is a
candidate lickometer miss.

For §10, keep `extract_clips_ffmpeg_encode`. Do **not** substitute
`video_clip_utils.extract_clips_ffmpeg_after_reencode` — that uses `-c copy`, which snaps to
keyframes and will not land on an exact event timepoint. The `libx264` re-encode is the
correct tool; its eventual home is `video_clip_utils`.

### G. §11 — state the boundary, don't re-measure it

Replaces the "score the live detector" section from the previous draft, which mis-framed
this as a competition between detectors. One markdown cell, no new analysis:

- `analyze_tongue_movement_quality` → `tongue_quality_stats.json` → `coverage_pct` is the
  fraction of lickometer licks with a nearby tongue **movement**. It is a recall measure for
  a *different, contact-agnostic* detector, already computed for every session, already
  pooled by `test_session_quality_analysis.ipynb`, and already thresholded at 90 % by
  `data_loading.load_session_quality_filter`. It has no precision side and no free
  parameter to sweep. Cite it; do not re-derive it.
- `kin_05` §4 inspects licks-without-movements with per-event traces — the same disagreement
  population, under the movement definition. Cross-reference rather than duplicate.
- The gap between the two definitions is itself informative: movements that are not contacts
  are `kin_05`'s "movements without licks" population, and one worthwhile sentence is what
  fraction of them this notebook's detector also rejects.
- `pixel_error.ipynb` measures keypoint pixel error against human labels. Different layer
  entirely. No overlap.

---

## Should the sweep run across all sessions?

**No — one session, and the reason is not statistical.**

The cost was never the obstacle: the full grid is ~38 s/session benchmarked, so all 44
sessions is ~28 min. Affordable, and I would have recommended it under the previous framing.
The contact framing changes the answer.

**The threshold is in pixels, and pixels are not transferable.** Pooling 44 sessions on a
raw-pixel threshold axis averages over sessions where 30 px is a different physical distance,
because camera and spout geometry vary. The resulting "mean F1 across sessions" would have a
tight-looking error bar around a quantity that does not mean the same thing in each
session — worse than the single-session number, because it looks more authoritative while
being less interpretable. Session-as-sampling-unit is the right convention for `kin_05/06/07`
because those pool *behavioral* measurements; it is the wrong convention for a parameter
whose units are session-specific.

So the ordering is:

1. **Now (this item):** one session, full grid, with the threshold *also* expressed in
   jaw→spout units (D) and confidence as a fourth axis (C). This characterizes the detector
   and produces the operating points.
2. **Later, as a separate question:** *does the normalized threshold transfer?* That is what
   multi-session is actually for, and it only becomes askable once step 1 has put the
   threshold in a transferable unit. Until then a pooled sweep answers nothing.

One cheap hedge worth taking, if you want it: run the grid on **2–3 additional sessions from
different subjects** — ~2 minutes, no new machinery, no pooled statistics — purely as a spot
check that the optimum is in the same neighbourhood elsewhere. If it is wildly session-specific,
that is worth knowing before anyone relies on the parameters; if it is stable, step 2 gets
cheaper to motivate. Frame it as a sanity check in a single table, not as a pooled analysis.

The honest scope caveat, stated plainly in the notebook header: these parameters are
characterized on `behavior_716325_2024-05-31_10-31-14` and are not yet demonstrated to
transfer.

---

## Notes

### Verified this session

- All six function pairs diffed against library `main` (`b21eac0`); signature/arity drifts
  as tabulated.
- `detect_licks` timing 0.2 s (`tlu`) vs 51.8 s (`tku`) per 1.8 M-frame session; full grid
  38 s/session.
- The re-arming failure mode (tongue tracked only near the spout → 1 event/session).
- Directional rates recomputed from the notebook's own stored cell-8 and cell-16 outputs.
- The full loading path: `kps_raw_*.parquet` carries `x`/`y`/`confidence`/`time`/`time_raw`;
  `nwb_df_trials.parquet` supplies the `goCue_start_time` origin; this reproduces
  `time_in_session` exactly, matching `nwb_df_licks['timestamps']`.
- `kinematics_filter` reindexes onto the full frame set and asserts the time base is
  unaltered — `tongue_kins` keeps every frame, but its values are interpolation-contaminated
  at protrusion edges.
- Only this notebook and `tongue_kinematics.ipynb` consume the six duplicated names; nothing
  in the library pipeline calls `detect_licks`.
- `data/for_local/` holds the pooled parquet (246 359 × 49, 44 sessions, 15 subjects) and
  nothing else relevant — no per-frame keypoints, no spout positions, no lick times.
- `behavior_716325_2024-05-31_10-31-14` is in the pooled parquet, so it has been reprocessed
  by the current pipeline.

### Unverified — Code Ocean only

- Whether the `video_preds_labeltest` asset (§10's labeled video) is still attached. §A
  removes the need for `matt_test_DLC_LP_results_20240920`.
- Whether `session_analysis_mlk/<session>/intermediate_data/` exists for the example session.
  `kin_05`/`kin_06` assume this path; neither has run there.
- Real session frame counts. The 38 s figure assumes 1.8 M frames (1 h @ 500 Hz).
- `detect_licks_multiple` (used by `tongue_kinematics.ipynb` cells 103–105) is not in library
  `main`, `fix/video-csv-header`, `code/`, or local library history. `LC_manuscript` could
  not be fetched this session — "not found", not "does not exist".

### Local vs Code Ocean

- **Local:** §1, §2 only. The sweep needs per-frame keypoints and NWB lick times; neither is
  in `data/for_local/`. Header should read like `eph_09`'s, not `kin_06`'s.
- **Code Ocean:** §3–§11.

### Ordering and risk

A → B → E is the repair and is testable on the example session alone; commit that first.
C, D and the §6 reframe are the analysis change and belong in a second commit.

Nothing already plotted becomes wrong: F1 is symmetric, so the existing heatmaps stand. What
changes is which figure carries the conclusion (precision–recall surface, not F1 argmax) and
whether the parameters survive the 0.8 → 0.90 confidence shift — which is the point of
re-running.

---

## Where this belongs

**Both**, plus one append.

1. **`TODO.md`** — everything above as one item, *"Repair and refocus
   `tongue_lickometer.ipynb` → `val_02_spout_contact_detection.ipynb`"*, placed after the
   `eph_09` item and before the library/repo boundary item it references.

   On the filename: `val_02_spout_contact_detection.ipynb` names the question rather than the
   instrument, which is the distinction this whole item turns on.
   `val_02_lick_detection.ipynb` is the alternative if you prefer continuity with the old
   name.

2. **`REORG.md`** — a new prefix plus four corrections:

   - Rename **"KEEP — model-quality / methods evaluation"** to **"KEEP — `val_*` methods and
     model-quality validation"**, with layering:

     | Slot | Notebook | Layer |
     |---|---|---|
     | `val_00` | `pixel_error.ipynb` | keypoint accuracy vs human labels |
     | `val_01` | `test_session_quality_analysis.ipynb` | session-level quality + selection |
     | `val_02` | `tongue_lickometer.ipynb` → this item | spout-contact detection; lickometer cross-validation |

     **They should not move together.** Only `val_02` is being rewritten; renaming the other
     two is a `git mv` plus a reference sweep with no analysis change. Record the slots now,
     rename each when next touched, accept the numbering gap (`eph_*` already has gaps).

   - **`test_session_wrapper.ipynb` is misfiled** — it is a batch-runner wrapper (19 cells,
     most commented out, wrapping `run_batch_analysis`). Operations, not evaluation. Move its
     line to "KEEP — pipeline / data generation". No `val_` slot.

   - **Correct the `model_quality.ipynb` description.** REORG calls it "foraging
     behavioral-model quality"; it is a single-session Lightning-Pose pipeline walkthrough
     (load → mask `tongue_tip_center` @0.90 → filter → segment → annotate → lick coverage),
     already on modern library imports. No `val_` slot until someone decides whether it is a
     pipeline demo or archive material.

   - **Correct the "Already covered elsewhere — do not port" line.** True for
     `tongue_kinematics` cells 78–79. But **cells 103–105 are a different analysis** — a
     multi-keypoint contact detector (`detect_licks_multiple` over
     `tongue_center`/`tongue_left`/`tongue_right`) that this notebook does not have and that
     calls a function existing nowhere. Under the contact framing it is a natural extension
     of `val_02` (three keypoints give a better contact estimate than one), so either
     re-derive it there or declare it dropped — but `tongue_kinematics.ipynb` should not be
     archived under the claim that it is already covered.

3. **Append to "Define the boundary between the library and this repo's `code/` modules"**,
   under *What to do* → "Resolve the library-internal duplication in (1)":

   > Resolved by inspection 2026-09-16 (see the `val_02` item). Keep
   > `tongue_lickometer_utils`'s `detect_licks` (vectorized; 225× faster than
   > `tongue_kinematics_utils`'s row-loop copy, and the only signature its call sites use),
   > `calculate_metrics` and `calculate_metrics_witheventkeys` (the `tku` copies add a `tn`
   > that subtracts counts from a timestamp — delete, do not merge). Keep
   > `tongue_kinematics_utils`'s `load_keypoints_from_csv` (coerces to numeric; `tlu`'s does
   > not) and its `mask_keypoint_data` / `filter_timestamps_refractory` (byte-identical —
   > keep one copy, `tku` side). Net: `tongue_lickometer_utils` owns **contact detection and
   > event scoring**, `tongue_kinematics_utils` owns **keypoint I/O, filtering and movement
   > segmentation**; the two implement different lick definitions and both are load-bearing,
   > so this is a de-duplication, not a consolidation. Also move `extract_clips_ffmpeg_encode`
   > from `tongue_lickometer_utils` to `video_clip_utils` (its `libx264` re-encode is the
   > correct precise-seek variant; do not replace it with `extract_clips_ffmpeg_after_reencode`,
   > which is `-c copy` and keyframe-snapped). Only two consumers exist ecosystem-wide —
   > `tongue_lickometer.ipynb` (→ `val_02`) and `tongue_kinematics.ipynb` (HOLD) — so this is
   > a low-risk deletion, and `val_02` can ship against `tongue_lickometer_utils` before the
   > library PR lands.
