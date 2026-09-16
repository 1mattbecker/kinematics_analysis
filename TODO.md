# TODO — kinematics_analysis

Deferred work items. Newest first. Dates are YYYY-MM-DD.

---

## Port plan for the five HOLD notebooks (overview)

_Logged 2026-09-11. Revised 2026-09-11 after a full cell-by-cell audit of all five HOLD
notebooks against the current `kin_*` / `eph_*` series._

Read this section before picking up any of the six port items below it. The five HOLD
notebooks in `REORG.md` are **not** five independent ports — their unreplicated content
clusters into five scientific questions that cut across notebooks, with real duplication
between them. Porting notebook-by-notebook would reproduce that duplication in the new
series.

### The question clusters → target notebooks

| Target | Question it answers | Source material |
|---|---|---|
| `kin_02_latency` *(extend)* — **done** | …and does the ordinal effect decompose as RT = RT₁ + (k−1)·Δt? | `tongue_latency` 15, 17, 18, 26; also 5, 8, 9, 14 (§11, single-session illustration — reassigned from `kin_05` below on 2026-09-14) |
| `kin_05_nonlick_movements` *(new)* — **done** | Do the lickometer and video streams describe the same events — and what are the movements that aren't licks? | `tongue_kinematics` 35–70; `cueresponse` 43; ~~`tongue_latency` 3, 5, 8, 9~~ (5, 8, 9 already ported into `kin_02` §11; 3 is a superseded draft of 5, not needed) |
| `kin_06_lick_geometry_choice` *(new)* — **done** | Does where the tongue goes carry choice information? | `cueresponse` 19, 49–56, 93–95; `tongue_kinematics` 60 |
| `kin_07_value_encoding` *(new)* — **written, unrun** | Do behavioral-model latents (Q, RPE) explain tongue kinematics? | `cueresponse` 17–28; `tongue_kinematics` 111–138 |
| `eph_07_bout_encoding` *(new)* — **done** | Do LC units respond differently to within-trial vs ITI movement bouts? | `intertrialmovs` 10–32 |
| `eph_09_structural_axes` *(new)* — **written, unrun** | Does the RT-encoding spatial gradient align with waveform / MERFISH / projection-target axes? | `spatial_axis_..._update` 13–38 |

Plus one new flat module, `spatial_axes.py` (see the items below). The bout-segmentation
helpers fold into the existing `ephys_utils.py` rather than getting a module of their own —
see the `eph_07` item for why.

### Two findings that change the previous plan

1. **`kin_07_value_encoding` is a topic this repo does not track at all.** Both
   `cueresponse` (cells 17–28) and `tongue_kinematics` (cells 111–138) carry a
   kinematics-vs-behavioral-model-latents analysis — `q_diff`, `q_sum`, `chosen_prob`,
   previous-trial `rpe`, screened with Spearman / mutual information / RidgeCV R² / random-forest
   importances. **No `kin_*` or `eph_*` notebook uses behavioral model latents** (`eph_06` uses
   Sue's model outputs, but against ephys, not kinematics). Because this content lives in *two*
   HOLD notebooks, `kin_07` gates **two** archivals — so on the previous plan neither
   `tongue_kinematics` nor `cueresponse` could actually be retired.

2. **`cueresponse` is not just "spatial geometry."** Cells 43–56 are a coherent
   choice-prediction analysis: donut of pre-lick movement counts, ridge-logistic regression
   (+ permutation importance, AUC) predicting left/right lick from pre-lick kinematics, violins
   of last-pre-lick excursion angle by lick direction, and binned P(right lick) vs `endpoint_y`
   and `excursion_angle_deg`. The landmark figures are the setup for that question, not the
   point of it.

### Why the boundaries are drawn where they are

- **The RT decomposition goes *into* `kin_02`, not into a separate notebook.** An earlier
  draft of this file floated `kin_05_latency_decomposition`. Don't. The decomposition
  *is the explanation* for the ordinal effect `kin_02` §4 already shows — it is the
  culmination of that notebook's existing line of questioning, not a new topic.
- **Lick↔movement correspondence and preparatory non-lick movements merge into one notebook.**
  Establishing that non-lick movements are real events (not lickometer misses) is the premise
  for asking where in the trial they occur. Split across two notebooks, neither stands alone.

### Known duplication between sources — port once

- `tongue_kinematics` cell 60 **==** `cueresponse` cell 95 (max-excursion-from-jaw scatter,
  lick vs non-lick). Port once, into `kin_06`.
- `tongue_latency` cells 10 and 37 **≈** `cueresponse` cell 42 (the 2×2 lick-latency-by-ordinal
  panel). Superseded by `kin_02` §4 — do not port.
- `intertrialmovs` cells 14, 15, 19, 22 are four near-identical population-PETH cells.
  Consolidate to one; keep the **cell 15** variant (session-wide z-scoring), which is the
  most defensible normalization.
- `tongue_kinematics` cells 78–79 (lick-detection FP/FN parameter sweep) are already covered
  by `val_02_lickometer.ipynb`, which is KEEP. **No port needed.**

### Orphan code that needs a home before its notebook is archived

Same category as `compute_outbound_metrics` (see the outbound item below) — analysis logic
living only in a notebook cell. Unlike the outbound case there is no duplicate to reconcile;
there is exactly one copy, and it is inside a notebook slated for archiving.

- `annotate_movement_bouts` — defined only at `intertrialmovs` cell 10; called 6× in that
  same notebook and referenced nowhere else in the repo. Verified **absent from the library
  on all three branches** (`main`, `LC_manuscript`, `video_alignment`), searching both the
  function name and the `mov_bout_*` columns it emits. → `ephys_utils.py`.
- ~~`plot_standard_lick_landmarks` — defined only at `cueresponse` cell 19.~~ **done
  2026-09-15** — landed in `kin_06` §3, restyled onto `plotstyle`'s palette. Left in the
  notebook, not promoted to `plotstyle.py`: still exactly one consumer, and `plotstyle.py`
  holds style, not domain plotting.

### Data-availability constraints (drives what is local-testable vs Code Ocean-only)

`all_tongue_movements_04022026.parquet` carries 49 columns. What matters for these ports:

- **Present, so pooled across sessions:** `has_lick`, `lick_count`, `lick_latency`, `event`,
  `movement_before_cue_response`, `cue_response`, `cue_response_movement_number`,
  `movement_number_in_trial`, `movement_latency_from_go`, `start_time`, `end_time`, `trial`,
  `session`, `goCue_start_time_in_session`, `endpoint_x/y`, `excursion_angle_deg`,
  `max_x_from_jaw`, `max_y_from_jaw`, `max_x_from_jaw_y`, `out_*`. **Caveat, found during the
  `kin_06` port:** `endpoint_x/y` and the `max_*_from_jaw` columns are *absolute pixel
  positions*, not jaw-relative — the jaw-relative distances are `max_x_distance` /
  `max_y_distance`. Being in the parquet does not make them pooled-safe: each session carries
  its own camera/jaw offset (per-session mean `endpoint_y` spans ~80 px across the 44
  sessions, against a ~51 px within-session SD). `kin_06` §2.1 recovers each session's jaw
  position from the absolute/distance column pairs and works in a jaw-centered frame — reuse
  `estimate_jaw_position()` from there rather than pooling these columns raw.
- **Absent:** `nearest_movement_id` (licks-without-movements needs per-session
  `nwb_df_licks.parquet`), **spout** landmark positions (per-session keypoint means — an
  earlier revision of this line called the `max_*_from_jaw` columns jaw-relative and therefore
  pooled-safe; they are not, see the caveat above. The *jaw* is recoverable from the pooled
  parquet, the spouts are not),
  and any behavioral-model latent (`q_*`, `chosen_prob`, `rpe`).

Consequence: `kin_02` ext is fully local-testable; `kin_05` and `kin_06` are mostly pooled with
one CO-only section each; `kin_07`, `eph_07`, `eph_09` are Code Ocean-only.

### Recommended order (value ÷ risk) and archiving gates

1. `kin_02` extension — pure pooled parquet, no new dependencies, fully local-testable. — **done**
2. `kin_05_nonlick_movements` — mostly pooled, one CO-only section (turned out to be two
   partial CO-only splits, within §3 and §5 — see the item below). — **done**
3. `eph_07_bout_encoding` (+ bout helpers into `ephys_utils.py`) — clear spec, reuses
   `eph_00`'s helpers. — **done**
4. `spatial_axes.py` + `eph_09_structural_axes`, then refactor `eph_08` onto the module. —
   **written 2026-09-15, not yet run on Code Ocean.** Assets confirmed mounted; `scanpy`
   added to the Dockerfile (image rebuild required). See the item below for what is
   still unverified.
5. `kin_06_lick_geometry_choice` — mostly pooled; two Code Ocean-only sections (§3–§4,
   which need per-session **spout** keypoints — the jaw turned out to be recoverable from
   the pooled parquet, see the item below). — **done**
6. `kin_07_value_encoding` — last; new dependency on `get_mle_model_fitting`. —
   **written 2026-09-15; Code Ocean-only sections unrun.** Dependency confirmed live
   (40 of 44 sessions have a fit; ~48 s for the full sweep), so the notebook pools
   rather than running single-session as both sources do.

Archive only when **all** gates for a notebook are met:

| HOLD notebook | Archive after |
|---|---|
| `tongue_latency.ipynb` | `kin_02` §8–§11 (done) — **no remaining gate, ready to archive** |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | `eph_07` (done) — **no remaining gate, ready to archive** |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | `eph_09` — written, **gate open until it runs on Code Ocean** |
| `tongue_kinematics.ipynb` | `kin_05` (done) **+** `kin_07` — written, **gate open until it runs on Code Ocean** |
| `tongue_kinematics_cueresponse.ipynb` | ~~`kin_06`~~ (done) **+** `kin_07` — written, **gate open until it runs on Code Ocean** |

---

## Extend `kin_02_latency` with the RT + IMI decomposition

_Logged 2026-09-11. Done 2026-09-14 — landed as `kin_02_latency.ipynb` §8–§10, executed
end-to-end locally against `data/for_local/all_tongue_movements_04022026.parquet`
(44 sessions, k=1..4). Nothing deferred; `kin_02` no longer gates on this item —
`tongue_latency.ipynb` archiving still waits on `kin_05_nonlick_movements`._

**Flag from the port:** §10's `std_aligned` and `std_raw` (source `tongue_latency`
cell 17 §6) are mathematically identical, not just empirically close — de-shifting a
k group by its own constant `(k−1)·Δt` cannot change that group's standard deviation.
The source notebook's "raw vs aligned" framing for this comparison doesn't test what
it appears to test. Ported both columns for continuity but added a note in the
notebook explaining the identity, and dropped the redundant duplicate line from the
plot itself.

**Correctness fix, same day.** §3's original filter (`dropna` on `cue_response_movement_number`
+ `movement_latency_from_go` only) never restricted to `movement_number_in_trial ==
cue_response_movement_number`. Since `cue_response_movement_number` is a trial-level constant,
grouping by k pooled in *every* movement from a k-labeled trial, not just the k-th one — only
~11% of the rows plotted in §4–§7 were actually the cue-response movement (checked directly:
6,958 / 57,925 at k=1, similarly ~11% at k=2–4). Fixed by splitting §3 into `movements_valid`
(all movements in trials with a valid k, unrestricted — needed by §8's Δt estimate, which
requires the whole up-to-cue-response movement sequence) and `df` (only the actual cue-response
movement, one row per trial — what §4–§7 use). §8's `lat_df` was re-pointed at `movements_valid`
accordingly. Re-executed end-to-end; §4–§7's numbers changed substantively (e.g. k=1 log-normality
n dropped from 57,925 to 6,958), §8–§10 were unaffected (Δt and per-trial counts already matched,
since §8 had its own independent restriction). Also added a second section, **§11 — single-session
illustration**, porting the example-trial and raster/colored-histogram figures from `tongue_latency`
cells 5, 8, 9, 14 (one example session, `behavior_716325_2024-05-31_10-31-14`, matching the
source's own choice) — checked `kin_00`/`kin_01`/`kin_03` first for duplicates (none: `kin_00`'s
rasters are unrelated QC/spatial-radius figures, `kin_01`/`kin_03`'s "raster" hits were the
`rasterized=True` matplotlib flag, not raster plots). The example-trial tongue-position trace
(cell 5) needs per-frame `tongue_kins.parquet`, not in the pooled parquet — Code Ocean only,
written but unexecuted; the two rasters and the colored-histogram panel run locally on the pooled
parquet filtered to the example session.

### Why

`kin_02_latency.ipynb` absorbed the clean parts of `tongue_latency.ipynb` — latency
distributions by ordinal (§4), k=1 log-normality (§5), session-level summaries (§6),
cross-ordinal correlation (§7). It shows *that* lick latency grows with the cue-response
movement ordinal k, but never explains *why*.

The explanation is the "nice story" left behind in `tongue_latency`: reaction time modeled as
a first-movement latency plus a sequence of inter-movement intervals,
**RT ≈ RT₁ + (k−1)·Δt**. This is the natural culmination of `kin_02`'s existing line of
questioning, which is why it belongs in `kin_02` rather than in a new notebook.

### What to do

Append to `kin_02` as §8–§10, continuing from the existing §7:

- **§8 — Δt estimate and de-shift.** Median within-trial inter-movement interval as Δt
  (`tongue_latency` cell 15, §1–3); overlay `lick_latency` distributions by k, then de-shift
  each by `(k−1)·Δt` and show they collapse. KDE variant is cell 17.
- **§9 — Does the collapse hold?** KS tests of each de-shifted k against k=1
  (`tongue_latency` cell 15 §4, cell 17 §4).
- **§10 — Noise propagation.** SD of latency vs k, raw and aligned, with bootstrap 95% CI
  (`tongue_latency` cell 17 §6 and cell 18 — `bootstrap_std_ci`). Then the cross-session
  grand mean ± SEM by cue-response movement number (cell 26), which is the population version
  of the same claim and should use session as the sampling unit.

Carry over the notebook's own caveat (`tongue_latency` cell 19, markdown): the residual
mismatch at k=1 and k=2 is attributed to *further* covert preparatory movements visible as a
secondary bump in those distributions. State it as the open question it is — `kin_05` is where
that claim gets tested.

### Notes

- Uses `lick_latency` (lickometer RT) conditioned on `cue_response_movement_number`, which is a
  **different quantity** from the `movement_latency_from_go` that `kin_02` §4–§7 already plot.
  Make the distinction explicit in the prose or the two halves of the notebook will read as
  contradictory.
- Reuse `kin_02`'s existing pooled-parquet load path. No spike data, no new dependency —
  this item is fully testable locally against `data/for_local/`.

---

## Create `kin_05_nonlick_movements.ipynb`

_Logged 2026-09-11. Done 2026-09-14 — landed as `kin_05_nonlick_movements.ipynb` §3–§8,
executed end-to-end locally (§4 excepted) against
`data/for_local/all_tongue_movements_04022026.parquet` (246,359 movements, 44 sessions).
`tongue_kinematics.ipynb` archiving now gates on `kin_07` alone._

**Correction from the port: §3 needed a partial CO-only split, not just §4.** The
"§3–§8 pooled, only §4 CO-only" reading below undercounted the CO-only surface by one.
Checked directly against the source (`tongue_kinematics` cell 35, not just its
data-availability summary): part (a) of §3, "licks without movement," computes
`nwb.df_licks['nearest_movement_id'].isna().sum()` — needs the per-session column, which
is confirmed absent from the pooled parquet's 49 columns. Parts (b) and (c) use
`tongue_movements['lick_count']`/`['has_lick']`, genuinely pooled. Landed §3 split
accordingly: (b)/(c) run and print locally (8,585/151,404 = 5.67% movements with >1
lick; 94,955/246,359 = 38.54% movements without licks); (a) sits under the same
Code-Ocean guard as §4, written but unexecuted, with the expected
`nwb_df_licks.parquet` schema documented in the notebook.

**Same finding recurs at §5.** Cell 43 (the `lick_count > 1` tally) is pooled, but cell
44 (the multi-lick example-trace figure) needs per-frame `tongue_segmented` plus
per-session `nwb.df_licks` — also CO-only, which the "§5 — cells 43, 44 — pooled" line
below doesn't distinguish. Landed §5 as the pooled `lick_count` distribution + per-session
rate (cell 43's tally is already covered by §3b, not re-run) rather than adding a third
CO-only guarded block for a qualitative single-session aside on an already-rare event.

**Bug fix in the port, not present in the source.** The per-trial structure (cells
64–68) grouped by `trial` alone in the source, which is only correct because that
notebook runs on a single session — `trial` numbers repeat across sessions in the pooled
parquet. §7 groups by `(session, trial)` instead; the naive group-by would have silently
pooled movement counts from unrelated trials that happen to share a number.

§8 uses the pooled `movement_before_cue_response` column directly rather than
reconstructing it from `nearest_movement_id` per-session, as the source did — the
reconstruction is unnecessary once the flag is already in the pooled parquet.

### Why

The lickometer and the video-derived movement stream do not describe the same set of events,
and the mismatch is the whole point: movements without licks are the candidate "covert
preparatory" movements that `kin_02`'s RT decomposition invokes but cannot test.

`kin_01` §5 currently compares lick vs no-lick movements on exactly two features
(`out_peak_velocity`, `out_duration`), pooled. Nothing in `kin_*` covers the correspondence
itself, the per-trial structure, or the pre-cue-response timing.

**Framing warning.** This is *not* a QC filter and must not be written as one. Per the
established noise definition, lickometer agreement, confidence, and duration are **not** valid
noise criteria — QC noise means confident misdetection of the wrong body part, which is
`kin_00`'s job. `kin_05` asks a definitional question (what are these events?), not a
filtering one. Keeping it out of `kin_00` is deliberate.

### What to do

One linear arc, roughly:

- **§3 — Three-way correspondence tally** (`tongue_kinematics` cell 35): licks without
  movements, movements with >1 lick, movements without licks, as counts and percentages.
- **§4 — Licks without movements** (`tongue_kinematics` cells 37, 39): tongue trace around
  unmatched licks, single and in sequences. **Code Ocean only** — needs per-session
  `nwb_df_licks.parquet` (`nearest_movement_id`), which is not in the pooled parquet. Gate it
  the way `kin_00` gates its trajectory/video sections.
- **§5 — Movements with multiple licks** (`tongue_kinematics` cells 43, 44): pooled via
  `lick_count > 1`.
- **§6 — Kinematic profile of non-lick movements** (`tongue_kinematics` cells 50, 59):
  duration and total-distance distributions, lick vs non-lick, including the fine-grained
  sub-50 ms histogram. Extends `kin_01` §5 to more features — cross-reference rather than
  duplicate.
- **§7 — Per-trial structure** (`tongue_kinematics` cells 64, 65, 67, 68): counts of lick vs
  non-lick movements per trial, percent non-lick, and their correlation.
- **§8 — Preparatory timing** (`tongue_kinematics` cells 69, 70; `cueresponse` cell 43):
  prevalence of ≥1 and ≥2 non-lick movements *before* the cue-response lick, via the pooled
  `movement_before_cue_response` column, plus the donut of trials by pre-lick movement count.
- ~~§9 — Illustrative single-trial and raster figures~~ **already ported, 2026-09-14** —
  landed as `kin_02_latency` §11 instead (one example trial's tongue position coloured by
  `has_lick`, CO-only; the trial raster coloured by movement type and by movement ordinal,
  both local). Don't re-port here.

### Notes

- `tongue_latency` cells 6/8 carry a `coerce_bool` helper written because `astype(bool)` on a
  string column turns any non-empty string True. `kin_02` §11 needed it (`cue_response` is
  object-dtype with `True`/`False`/`None`); check the pooled dtypes here too rather than
  porting reflexively.
- ~~Sections §3–§8 are pooled and therefore locally testable; only §4 is CO-only.~~
  **Wrong, see the Done note above** — §3's "licks without movement" tally and §5's
  example-trace figure are also CO-only (both need per-session `nwb.df_licks` /
  `nearest_movement_id`); everything else in §3, §5–§8 is pooled.

---

## Create `kin_06_lick_geometry_choice.ipynb`

_Logged 2026-09-11. Done 2026-09-15 — landed as `kin_06_lick_geometry_choice.ipynb` §3–§9
(35 cells), executed end-to-end locally against
`data/for_local/all_tongue_movements_04022026.parquet` (246,359 movements, 44 sessions).
§3–§4 are Code Ocean-only, written but unexecuted. `tongue_kinematics_cueresponse.ipynb`
now gates on `kin_07` alone; **not** archived this session._

**Correction from the port: `max_*_from_jaw` is not jaw-relative.** The "What to do"
list below (and the overview's data-availability note, now fixed) treats §5 as
pooled-safe because `max_x_from_jaw` / `max_y_from_jaw` are jaw-relative. They are not —
they are the *absolute* pixel coordinates of the point farthest from the jaw; the
distances are `max_x_distance` / `max_y_distance`. Neither those nor `endpoint_x/y` are
pooled-safe raw, since each session carries its own camera/jaw offset.

**Revised again: the notebook now prefers the real jaw keypoint over reconstructing it.**
`get_jaw_positions()` (§2.1) tries `kps_raw_jaw.parquet` for every session first — the same
asset §3's landmarks use — and only falls back to the algebraic reconstruction
(`estimate_jaw_position()`) where that file is missing, which locally means **all 44**
sessions (no `kps_raw_*.parquet` exists in `data/for_local/`). Checked first whether a
better local proxy existed instead: neither `startpoint_x/y` nor the movement bounding-box
columns (`min_x/y`, `max_x/y`) sit at a fixed point — all have 11–39 px within-session SD,
far noisier than the algebraic method's own ~0.1 px internal residual — so there is no
local column better than the reconstruction; the real keypoint is Code-Ocean-only.

The fallback logic is unchanged from the previous revision: `jaw_x = median(max_x_from_jaw
- max_x_distance)` is exact up to tracking noise (the tongue protrudes in one fixed
x-direction, so there's only one candidate per row); `jaw_y` has no fixed sign (excursions
go to either spout), so each row gives *two* candidates, `max_y_from_jaw ± max_y_distance`,
and a 1-D grid search finds the value minimizing the median distance to the nearer one.

**What changed: validation is now automatic, not deferred to one example session.** For
every session where both a keypoint and an algebraic estimate exist, `get_jaw_positions()`
reports the distance between them directly — this is the real check that §3's old
single-session cross-check only sampled once. On Code Ocean this should resolve most or
all of the "jaw_y is unverified" caveat immediately; locally it can't run at all (0 of 44
sessions have a keypoint file), so `jaw_y` for this run is still the unverified fallback.

**Why this got revisited:** investigating a user-reported anomaly in §9 (a few "wrong-sign"
cue-response endpoints, changed-mind trials pulling one marginal negative) surfaced that
**41 of 44 sessions show `|left-lick excursion| > right-lick excursion`** by a median of
13 px (up to 57 px) in the algebraic frame — plausibly real spout geometry (right spout
closer to the jaw's rest position for most animals), but only the keypoint-vs-algebraic
comparison above can rule out a reconstruction artifact instead. Worth checking this
asymmetry specifically once `get_jaw_positions()` runs on real data.

§5–§9 run in the resulting jaw-centered frame regardless. The **spouts** stay Code
Ocean-only, which is why §3–§4 do — the split is jaw-vs-spout, not
jaw-relative-vs-absolute as assumed below.

**Statistics: the source's evaluation could not be ported as-is.** `cueresponse` cell 47
calls `train_test_split(random_state=42)` on one session's trials; on pooled data that
leaks session identity across the split. §8 reports two evaluations instead —
`GroupKFold` with session as the group (held-out-session AUC **0.835 ± 0.100**) and
per-session fits (median AUC **0.943** over 39 qualifying sessions) — plus a
within-session shuffled-label null (mean 0.530, observed p = 0.005 at 200 permutations).
Class balance is 57.9% left / 42.1% right. Permutation importance puts `last_angle`
(ΔAUC 0.168), `mean_distance` (0.098) and `mean_angle` (0.070) on top, with both
peak-velocity terms and `session_time` at ~0.

**Block-structure caveat quantified, not just recorded** (§8.4–§8.5). P(stay) = 0.910 and
**previous choice alone reaches AUC 0.907** — better than kinematics. Kinematics still
reads AUC 0.815 / 0.772 *within* each previous-choice stratum, so it is not simply
re-encoding choice history, but the two are not separated here. Separating them needs
block-aware regressors → `kin_07`.

**`plot_standard_lick_landmarks` landed in `kin_06`, not `plotstyle.py`** — restyled onto
the project palette, still one consumer, so not pre-promoted per the note below.

### Why

`kin_*` carries `endpoint_*`, `excursion_angle_deg` and the `max_*_from_jaw` columns, but never
plots them in the anatomical frame that makes them interpretable (jaw and spout landmarks), and
never asks the question that frame sets up: **does the direction of a preparatory tongue
movement predict which spout the animal goes on to lick?**

This is the content `REORG.md` previously summarized as "cue-response spatial geometry", which
undersells it — the geometry is the setup, the choice prediction is the result.

### What to do

- **§3 — Landmark frame** (`cueresponse` cells 19, 94): `plot_standard_lick_landmarks` helper
  (promote it out of the notebook), plus the jaw↔spout distance printout that establishes the
  scale. **CO-only** — needs per-session keypoint means.
- **§4 — Endpoints by event type** (`cueresponse` cell 93): cue-response movement endpoints
  coloured by left/right lick event, over the landmarks.
- **§5 — Lick vs non-lick excursion geometry** (`tongue_kinematics` cell 60 == `cueresponse`
  cell 95 — port once): max excursion from jaw, with and without licks. ~~Jaw-relative, so~~
  **poolable across sessions** even though §3–§4 are not — but *not* because the columns are
  jaw-relative (they aren't; see the Done note above). Poolable because the notebook
  reconstructs the per-session jaw origin first.
- **§6 — Non-lick endpoints by movement ordinal** (`cueresponse` cells 54, 56).
- **§7 — Does pre-lick direction predict choice?** (`cueresponse` cells 49, 55): violins of
  last-pre-lick `excursion_angle_deg` by subsequent lick direction; binned P(right lick) vs
  pre-lick `endpoint_y` and vs `excursion_angle_deg`.
- **§8 — Ridge-logistic decode** (`cueresponse` cells 44, 47, 48): predict left/right from
  last-pre-lick kinematics, report AUC + classification report, with both absolute standardized
  coefficients and permutation importance.
- **§9 — Pre-lick → cue-response endpoint displacement** (`cueresponse` cells 50, 51, 52):
  paired per-trial `endpoint_y`, pre-lick vs cue-response, against the jaw midline.

### Notes

- `cueresponse` runs on a single session. Re-do §5–§9 pooled where the columns allow it
  (all of `endpoint_*`, `excursion_angle_deg`, `movement_before_cue_response`, `event` are in
  the pooled parquet) — pooling turns the logistic decode from anecdote into a result.
  Use session as the sampling unit for any population claim.
- The `event` column encodes lick side as `right_lick_time` / `left_lick_time`.

---

## Create `kin_07_value_encoding.ipynb`

_Logged 2026-09-11. **Written 2026-09-15** as `kin_07_value_encoding.ipynb` (40 cells,
§1–§8). All of §3–§7 below landed — nothing was deferred. `tongue_kinematics.ipynb` and
`tongue_kinematics_cueresponse.ipynb` are now both fully replicated; **neither archive
gate closes until this notebook actually runs on Code Ocean**, which is the one remaining
task on this item._

**What ran and what did not.** The notebook was executed end-to-end locally: every
Code-Ocean gate took its skip path, no errors, 23 code cells. Only **§6.1** (the kinematic
covariance structure) produced real output — everything from §3.4 on needs per-session
`nwb_df_trials.parquet`. The §6 screen machinery was verified separately against a
**synthetic** latent built as a known function of one kinematic column (real kinematics,
14,039 cue-response trials, 44 sessions; a hard target — the planted driver has
ρ = 0.92–0.97 twins): all four methods ranked it first at two noise levels, and the
pure-noise control returned q = 0.99 and held-out R² = −0.001. The screen is wired
correctly; the real data path is not yet exercised.

**Dependency status (checked before writing anything).** `get_mle_model_fitting` resolves,
but `han_pipeline.get_mle_model_fitting` — the path both sources import — is now a
**deprecated shim** forwarding to `df_mle_model_fitting.get_mle_model_fitting`; the
notebook prefers the new path with a fallback. Coverage measured over all 44 sessions in
~48 s: **40 have a `QLearning_L2F1_CKfull_softmax` fit**; the four sessions of subject
**751004** (2024-12-20 … 2024-12-23) have no MLE records at all.

**Flag — the "port once, defined identically" note below is wrong.** The two copies of
`attach_model_latents_to_trials` differ in the **sign of `q_diff`** (`cueresponse` cell 13:
`R − L`; `tongue_kinematics` cell 115: `L − R`), which flips every correlation involving
it, and in where `q_diff_c` is derived. Ported once as `R_value − L_value`, matching
`kin_06`'s "+y toward the animal's right" convention. See `CHANGELOG.md` 2026-09-15 (7)
for the full decision list (restored alignment assertions, the two left/right-convention
checks, `q_sum` added as the vigour control, why no jaw-reconstruction code was duplicated
from `kin_06`).

**Open follow-ups, in priority order:**

1. **Run it on Code Ocean.** Read §8's ordered checklist first — §3.5's L/R check must
   print SUPPORTED and §3.4's `latent_index_mode` must be "responded" for every session
   before anything downstream is worth reading.
2. **Then archive both HOLD notebooks** (a separate pass; see the gate table above).
3. **Block-aware regressors are still not done, and this is now the second notebook to
   defer them.** `kin_06` §8.5 hands the problem here; `kin_07` §6 conditions on nothing
   either. Value covaries with block structure, choice side and time in session, and
   `kin_06` §8.4 shows the previous-choice confound is strong in this dataset
   (P(stay) = 0.910). A partial-correlation or within-choice-stratum version of the §6
   screen is the obvious next step and deserves its own item once §6 has real output to
   condition.
4. `excursion_angle_deg` is circular and is treated linearly throughout, as both sources do.
5. One agent (`QLearning_L2F1_CKfull_softmax`), taken from the sources with no
   model-comparison step; that question lives in `model_quality.ipynb`.
6. Sessions are treated as exchangeable across-session, but come from 13 subjects with 1–4
   sessions each. A subject-level random effect would be more correct than the Wilcoxon.

### Why

**No notebook in the `kin_*` or `eph_*` series uses behavioral-model latents against
kinematics.** `eph_06` compares RT encoding to Sue's model outputs, but that is ephys.
The kinematics-side question — does action value or reward-prediction error show up in how the
tongue moves? — exists only in the two HOLD notebooks, in two partly-overlapping copies.

This item gates **two** archivals (`tongue_kinematics` and `cueresponse`), which is why it
cannot simply be dropped even though it is the largest new dependency.

### What to do

- **§3 — Attach model latents.** `attach_model_latents_to_trials` (defined identically at
  `tongue_kinematics` cell 115 and `cueresponse` cell 13 — port once) over
  `get_mle_model_fitting` from `aind_analysis_arch_result_access.han_pipeline`; merge onto
  movements by `trial`. Derive `q_diff`, `q_sum`, `q_diff_c`, `chosen_prob`.
- **§4 — Trajectories and endpoints coloured by value** (`cueresponse` cells 18, 20, 26):
  tongue trajectories coloured by `q_diff` over the landmark frame; 2-D endpoint scatter
  coloured by `q_diff_c`.
- **§5 — Binned kinematics vs value** (`cueresponse` cells 21, 26, 27): binned scatter with
  regression for the pairs the source pins — `q_diff`×`max_x_from_jaw_y`,
  `q_diff_c`×`max_x_from_jaw_y_distance`, `q_diff_c`×`max_x_from_jaw`,
  `chosen_prob`×`max_x_from_jaw_y_distance`.
- **§6 — Systematic screen** (`tongue_kinematics` cell 135): all kinematic × all model columns,
  four ways — Spearman ρ, mutual information, RidgeCV predictive R², random-forest importances —
  as the 2×2 summary figure. This is the cell that earns the notebook; the pinned pairs in §5
  should fall out of it rather than being asserted.
- **§7 — Previous-trial RPE** (`tongue_kinematics` cell 137): shift `rpe` by one trial, merge
  at movement level, repeat the Spearman + MI screen. Tests whether the *outcome* of the last
  trial changes this trial's movement, which is the LC-relevant version of the question.

### Notes

- **Code Ocean only** and the heaviest new dependency of the six items — `get_mle_model_fitting`
  hits the upstream pipeline. Confirm it still resolves before committing to the port.
- Both source copies are single-session. Decide early whether to pool; if pooling, model fits
  must be fetched per session, which is the main cost driver here.
- Scope risk is real. If the port stalls, land §3–§5 (which is all `cueresponse` needs) and
  leave §6–§7 for a follow-up — but note that `tongue_kinematics` cannot be archived until
  §6–§7 land.

---

## Create `eph_07_bout_encoding.ipynb`

_Logged 2026-09-11. Landed 2026-09-15 as `eph_07_bout_encoding.ipynb` — bout helpers
(`annotate_movement_bouts`, `classify_bout_times`, `get_session_bout_times`) in
`ephys_utils.py`, thresholds pinned and parameterized as specified below. Only the
bout-helper verification (§4) executed, against
`data/for_local/all_tongue_movements_04022026.parquet` (44 sessions, 246,359
movements → 31,081 bouts; pinned thresholds give 14,368 go-responsive / 9,297 ITI
bouts pooled). Everything from §5 on (single-unit raster, population PETH, per-unit
encoding comparison, qualitative figures, lick-bout robustness check) is Code Ocean
only — written and reasoned through, not executed. `eph_07` is now the sole remaining
gate on archiving `tongue_kinematics_ephys_intertrialmovs.ipynb`; archiving itself
was deliberately left for a separate step, after review of this port. See
`CHANGELOG.md` for the full list of design decisions (why the per-unit go-responsive
class was rebuilt on the primary movement-bout definition rather than porting source
cell 29's hybrid definition verbatim; why the lick-derived robustness check
reclassifies lick-bout starts the same way as movement-bout starts rather than
reproducing cell 26's mixed definition)._

### Why

`tongue_kinematics_ephys_intertrialmovs.ipynb` (43 cells) is a distinct ephys analysis with no
equivalent in the new `eph_*` series: it groups movements into **bouts** and compares LC unit
responses to **within-trial (go-response) bouts vs inter-trial-interval (ITI) bouts** —
bout-aligned rasters/PETHs, population overlays and waterfalls, and a per-unit go-responsive vs
ITI Δhz comparison. It is the ephys counterpart to the covert-preparatory-movement story that
`kin_02` and `kin_05` tell on the behavior side.

The notebook already imports the refactored `data_loading` and `ephys_utils`, so it is partway
to the new structure.

### The two bout definitions — pick deliberately

The source notebook runs the comparison **twice**, on two different notions of bout:

1. **Movement-derived** (cells 10–24): `annotate_movement_bouts` on `movs["start_time"]` with
   a 0.5 s gap threshold. This is the definition that depends on the orphan function.
2. **Lick-derived** (cells 26–32): `licks["bout_start"]`, produced upstream by
   `aind_dynamic_foraging_basic_analysis.licks.lick_analysis`, read from the per-session
   `nwb_df_licks.parquet`.

Choose one as the notebook's primary and keep the other as a robustness check — don't port both
at equal weight. The movement-derived definition is the better primary: it is the one that
matches the behavioral story, and it does not inherit the lickometer's blind spot for non-lick
movements (which is exactly what `kin_05` establishes).

### What to do

- **Bout helpers → `ephys_utils.py`** (do this first): `annotate_movement_bouts` (from
  `intertrialmovs` cell 10, the only copy in the repo — see the orphan-code note in the
  overview) plus the within-trial / ITI classifier currently copy-pasted into four
  `get_session_bout_times` variants. Parameterize the thresholds rather than hardcoding:
  `GAP_THRESHOLD_S=0.5`, `GO_RESPONSE_WINDOW_S`, `ITI_MIN_POST_CUE_S`, `ITI_MIN_PRE_NEXT_S` —
  the source uses **different values** in different cells (2.0/2.0/1.0 at cell 12 vs
  1.0/2.0/0.5 at cell 29), so pin one set in the notebook and state it.

  *Why `ephys_utils.py` and not a new `bout_utils.py`:* an earlier draft of this file
  proposed a separate module on the grounds that bout segmentation is behavioral (it touches
  `start_time` and go cues, never spikes) and so doesn't belong in an ephys module. That
  contradicts what `ephys_utils.py` already does — `build_trial_features(movs, licks,
  df_trials)` is also pure behavior, deriving per-trial RTs and `first_move_*` /
  `cue_response_*` columns with no spikes involved. The established convention here is that
  **behavior-derived features existing to serve ephys alignment live in `ephys_utils.py`**,
  and bout segmentation is the same category. `eph_07` already imports from `ephys_utils`,
  so this adds no new import surface. A module with one consumer is overhead: the smallest
  existing module, `ccf_utils.py` at 58 lines, earns its place with two consumers.
- **`eph_07`**: bout segmentation → within-trial vs ITI event times → reuse `eph_00`'s
  `make_rp_and_events` / `compute_psth` / `smooth_vector` / `plot_psth` helpers for the
  single-unit raster + PETH pair (cell 12), then the population heatmap, mean±SEM overlay
  (cell 16/20), and waterfall (cell 17). **Consolidate cells 14/15/19/22 into one** — keep the
  cell 15 variant, which z-scores against session-wide firing statistics rather than a
  pre-event baseline.
- Include the go-cue-aligned third condition (cell 24) as the reference the two bout conditions
  are read against, and the ITI-bouts-per-session count distribution (cell 18) plus the
  `MIN_BOUTS` session filter (cell 19) as the sampling-adequacy check.
- Per-unit encoding comparison (cells 29, 30, 31): paired go-responsive vs ITI Δhz with
  Wilcoxon, baseline-vs-response paired plots per class with per-unit significance, and the
  go-responsive vs ITI scatter. Register through `per_unit_stats_registry` /
  `encoding_methods` so it composes with `eph_01`–`eph_04`.
- Keep the event-raster and single-trial timeseries figures (cells 35, 36) as the qualitative
  setup. **Drop cell 37** (video-clip extraction) — that is a one-off, and the clip helpers are
  library-owned.

### Notes

- Bout event times can come from the pooled `all_tongue_movements` parquet — it carries
  `start_time`, `trial`, `session` and `goCue_start_time_in_session`. Align to spikes via the
  cached `filtered_ephys.pkl`.
- `annotate_movement_bouts` is a long-term **library** candidate — it is general movement
  segmentation, sibling to `annotate_movement_timing` and `add_lick_metadata_to_movements`
  which already live in `tongue_kinematics_utils.py`. Premature now: the definition is not
  settled (two bout definitions, thresholds varying across cells). Let `eph_07` exercise it
  first. Same staging as the outbound-metrics item; the placement criteria are the subject of
  the module-boundary item below.
- If a `kin_*` notebook later wants bouts, note that importing them from `ephys_utils` drags
  in `tongue_ephys` at module level to do pure behavior. That works, but it is the signal to
  revisit placement — fold it into the module-boundary item rather than splitting reflexively.

---

## Create `spatial_axes.py` and `eph_09_structural_axes.ipynb`

_Logged 2026-09-11. **Written 2026-09-15. `eph_08` replicated the reference figure on Code
Ocean 2026-09-16 (see "Replication" below); `eph_09` is still NOT executed there.**
Both external assets confirmed mounted by the user before starting; `scanpy==1.10.3` added to
`environment/Dockerfile` in its own layer, so **the capsule image must be rebuilt** before the
MERFISH block can run._

### Status: what landed, and what is still unverified

Landed: `code/spatial_axes.py`; `code/eph_09_structural_axes.ipynb` (34 cells); `eph_08`
refactored onto the module; `scanpy` in the Dockerfile.

**Verified locally** — `spatial_axes.py` is pure numpy/sklearn and was checked against
synthetic data with known planted axes (30 assertions, all passing): linear/CCA/LDA all
recover a planted direction, `compare_bootstrap_directions` returns ~0° for identical axes
and ~90° for orthogonal ones, `cone_half_angle` widens monotonically with noise and as n
falls, and the module's CCA pair is **bit-identical** to the inline copies deleted from
`eph_08` (same axis, same bootstrap cloud, same cone, same projections, same seed).
Both notebooks execute clean locally through their skip paths.

**NOT verified — needs a Code Ocean run:**

1. ~~**`eph_08` output equivalence on real data.**~~ **Done 2026-09-16** — `eph_08`
   reproduces the reference figure exactly (r=0.184 displayed as 0.18, **p=0.0681**, n=99).
   It took two fixes; see "Replication of the reference figure" below.
2. **Everything in `eph_09` past §1.** No cell touching real data has run. Expect to debug
   column names on first contact, in particular:
   - ~~`all_counts_df` must carry `baseline_spike_count`~~ — superseded: see the
     `all_counts_df` fix below; it is now built locally with `baseline_window_s=(-1.0, 0.0)`,
     so the column is guaranteed present.
   - `all_counts_df` must carry `baseline_spike_count` for the `T_rt_bl` axis.
   - The `(session_prefix, unit_str)` merge in §2 — confirm the join is not silently empty
     (§2 prints the surviving unit counts; if they are 0, the unit-key canonicalization is
     the suspect).
   - `retro_ccf` must have `injection_region`, `x`, `y`, `z`.
3. **The MERFISH block specifically** needs the rebuilt image. Until then it will print
   `Could not load MERFISH data: No module named 'scanpy'` and set `HAS_MERFISH = False` —
   which is the guard working, not a bug. Re-run after the rebuild.
4. **`eph_09` §8 projects *signed* `T_rt`**, while the reference figure is `|T_rt|`
   (abs). Its wf panel therefore will **not** match `rt_response_projection_abs.svg` until
   `feat_proj` is wrapped in `np.abs`. Decide which is the headline before running — the
   reference's signed-`T_rt` panels are a separate figure (its cells 38/40, all ns).
5. **No anatomical filter outside `eph_08`/`eph_09`.** The `z_ccf` bounds filter added here
   is the only thing excluding mislocalised units; `data_loading`'s QC checks spike quality
   only. `eph_04` and `spatial_encoding.py` project the same unit table and still include
   unit 85. Consider lifting the filter into `data_loading` or `spatial_encoding`.
6. **`scanpy==1.10.3` against the pinned block.** Isolated `RUN` layers separate pip's
   *resolution*, not the environment — scanpy can still move shared packages. `scipy==1.13.0`
   is re-asserted in that layer as a tripwire. If the build fails there, that is the tripwire
   firing: resolve it rather than dropping the pin.

### Decisions taken during the port (revisit if you disagree)

- **RT stats go through `AnalysisSpec` / `fit_encoding` / `PerUnitStatsRegistry`**, not a
  third inline copy of `build_rt_encoding_stats`. This answers source cell 40's
  commented-out registry sketch properly, and makes `eph_09`'s `T_rt` the same quantity
  `eph_01`–`eph_04` use. Session/unit QC likewise goes through `data_loading`.
- **No RT trial window** (`trial_query=""`, `min_trials=50`), matching the source and
  `eph_08` rather than `eph_01`'s `RT_QUERY` (0.05–1.0 s). The axes therefore describe the
  same units `eph_08`'s poster figure projects. `RT_QUERY` is defined in §2 for a
  sensitivity check — **worth running once**, since the two conventions could give
  different axes and only one can be the headline number.
- **Okabe-Ito colors** (via `plotstyle`) instead of the source's palette. The source used
  the upstream red/orange/green/purple/peach to match his figures; swap `COLORS` back in §1 if you
  need a side-by-side with those.
- **Axis labels** on the three-plane figure say `ML/AP/DV (mm)` rather than the source's
  `dim 0` / `dim 1`.

### Fixed 2026-09-15 (second pass): `all_counts_df` was read from a path that does not exist

First Code Ocean run of `eph_08` failed at its §2 with
`FileNotFoundError: /root/capsule/scratch/all_counts_df.parquet`.

**Root cause, and it predates this port.** `eph_08` and (by inheritance) `eph_09` were the
only two notebooks that *read* `all_counts_df.parquet` from a path; `eph_01`–`eph_06` all
**build** it via `ephys_utils.build_all_counts_df`. The read came from source notebook
`spatial_axis_..._update`, where that very line is **commented out** (cell 7) next to a NOTE
saying the pipeline functions had to be pasted in — i.e. it was a placeholder that was
probably never executed as written. `/root/capsule/scratch` is also not persistent on Code
Ocean, so even a file that once existed there is gone.

**Fix:** both notebooks now follow the `eph_02` pattern —
`load_units_with_spike_times` → `build_all_counts_df(units_with_spikes, cfg, base_dirs)`.

> **Superseded 2026-09-16 on the window values only.** That fix used the project-wide
> `count_window_s=(0.0, 0.2)`, `baseline_window_s=(-1.0, 0.0)`. Those are `eph_01`–`eph_06`'s
> windows but **not** the reference figure's, and they are why the first Code Ocean run gave
> r=0.013 instead of 0.184. Both notebooks now use `(0.0, 0.5)` / `(-2.0, 0.0)`. Building
> `all_counts_df` in-notebook (rather than reading a path) was and remains correct.

### Replication of the reference figure (resolved 2026-09-16)

**Target.** `rt_response_projection_abs.svg` — three panels, `|T_rt|` (response, abs)
projected onto the waveform / MERFISH / retrograde axes; wf panel r=0.184, p=0.0681, n=99.

**Where it came from.** `code/archive/spatial_axis_comparison_rt_encoding.ipynb` cell 42
(`execution_count` 37), saved to `/root/capsule/scratch/figures/poster/`, SVG timestamp
2026-05-04 00:34:41, committed 16 minutes later in `2f20188`. **Not** `_update` — that file
was *created* in the same commit, after the figure existed. Confirmed independently: the
SVG's per-panel x-tick colours are that notebook's `COLORS` entries for the three axes, and
cell 42's stored output carries the figure's r/p/n verbatim.

**Two root causes, both now fixed:**

1. **Spike-count windows.** Cell 9 of that run (`execution_count` 7) builds `all_counts_df`
   with `count_window_s=(0.0, 0.5)`, `baseline_window_s=(-2, 0.0)` — also the published
   analysis's windows (500 ms post-cue, 2 s pre-cue baseline). `eph_08`/`eph_09` had
   `(0.0, 0.2)` / `(-1.0, 0.0)`, inherited from a **commented-out** cfg in `_update` whose
   values had already been changed. Fixing this moved r from 0.013 to 0.19.
2. **ML fold sign.** The reference folds unit coordinates to `+ML`
   (`ccfs[:, ml] = np.abs(...)`, commented "POSITIVE, matching upstream"); `eph_08`/`eph_09`
   folded to `-ML` while the structural axes are *fitted* in `+ML` space
   (`ccf_wf[:, ml] = np.abs(...)`), flipping the ML component of every projection relative to
   the axis it is projected onto. Fixing this moved r from 0.19 to 0.184.

Also added: the reference's `z_ccf` ∈ [-5.2, -3.5] anatomical filter, which drops exactly one
mislocalised unit (`behavior_758017_2025-02-06_11-26-14` unit 85, `z_ccf = -2.174`, 2.25 mm
from the mesh centroid vs 0.79 mm for every other unit), taking n from 100 to 99. In `eph_09`
it sits **after** the axis fits and **before** §8, matching the reference's ordering (axis fit
n=100, projections n=99).

**Deliberately not matched: the mesh.** The reference uses `new_core_mesh.obj` with a bespoke
transform; both notebooks keep `20250418_transformed_remesh_10_ccf25.obj` + `pir_to_lps`.
Centering is a pure translation along the projection axis, so r, p and n are unaffected and
only the plotted x-axis shifts (~0.19 mm; ours spans ≈ -0.65…+0.58, the reference
-0.78…+0.44). Verified locally: with the `+ML` fold, both meshes give identical r.

**Method note worth keeping.** The whole case rested on the archived notebook's **stored
outputs and `execution_count` values** — they pinned the windows, the axis, the unit counts
and the run ordering. `_update` had its outputs cleared and could prove none of it. Clearing
notebook outputs is lossy for provenance.

**Assumption to check on the next run:** that this cfg matches whatever produced the
now-missing cached parquet. If `eph_08`'s poster figure comes out numerically different from
the published version, **the count/baseline windows are the first thing to suspect** — not the
`spatial_axes.py` refactor, which is separately verified bit-identical.

Build cost: `build_all_counts_df` loops every session × unit, so `eph_08` and `eph_09` each
pay it. If that becomes annoying, cache it once to `SCRATCH` and read-if-present — but do it
in `ephys_utils`, for all the `eph_*` notebooks at once, not ad hoc in these two.

### Fixed 2026-09-15 (third pass): waveform CSV path pointed at the upstream layout, not this capsule

Second Code Ocean failure, in the waveform block:
`FileNotFoundError: /root/capsule/data/LC-NE_scratch_data_1/combined/waveforms_np/combined_features.csv`

**Root cause.** The CSV is a **separately attached Code Ocean data asset**, not part of
`LC-NE_scratch_data_1`. The archived predecessor
(`code/archive/spatial_axis_comparison_rt_encoding.ipynb` cell 23) hardcodes the real path:
`/root/capsule/data/results-59472bbb-4c3a-40f9-a1f5-b0c5113e4ab9-waveforms_np/combined_features.csv`.
The `_update` source rewrote it to `FIG_PREP_DIR/waveforms_np/` — the upstream capsule's own layout —
and `eph_08` / `eph_09` inherited that. `combined_unit_tbl.pkl` is unaffected; it really does
live under `FIG_PREP_DIR/combine_unit_tbl/`.

**Fix:** both notebooks now call a `find_wf_features_csv(DATA)` resolver — tries the upstream layout,
then the known asset id, then `*waveforms_np*/combined_features.csv`, then any
`**/combined_features.csv`, and raises listing everything it tried. Globbing means a
re-attached asset with a new id still resolves. Tested locally against five synthetic layouts.

**Note this was masked in `_update` and would have been masked in `eph_09`.** The source's
waveform block is wrapped in `try/except`, so it printed "Could not load waveform data" and set
`HAS_WAVEFORM = False` — meaning **the `_update` notebook very likely never fitted the waveform
axis either**, and its comparison table would have silently omitted it. `eph_09` has the same
guard by design, so watch the `HAS_*` flags and the "OMITTED (asset unavailable)" line in §6
rather than assuming all five axes are present. `eph_08` has no such guard, which is why it
raised and surfaced the problem at all.

**Still unconfirmed:** whether the asset id above is current in this capsule. A diagnostic that
lists the data mounts, globs for the CSV, and checks whether `combined_unit_tbl.pkl` already
carries the seven `wf_feature_cols` (in which case the CSV is unnecessary) was handed to the
user; the answer has not come back yet.

### Follow-ups this port surfaced (not blocking)

- **`eph_08` still duplicates two things it need not.** Its §2 reimplements the session-QC
  and unit-QC filter inline although `data_loading.load_session_quality_filter` /
  `filter_ephys_units` already do exactly that, and it carries its own
  `build_rt_encoding_stats` and `get_regression_CI`. Left alone deliberately this session —
  `eph_08` is a poster-figure path and the brief was a pure move. Fold it in once `eph_08`
  has been re-run and confirmed unchanged.
- **`build_rt_encoding_stats` is orphan code** in the same sense as `annotate_movement_bouts`
  was: defined only in `eph_08` §2 and source cell 8. `eph_09` avoids adding a third copy by
  using `fit_encoding`; `eph_08`'s copy should follow once it is safe to touch.
- **`get_regression_CI` now exists twice** — in `spatial_axes.py` and inline in `eph_08` §5.
  Same fix, same gating.
- The source's `get_regression_CI` computed `se` twice, the first immediately overwritten by
  the second. The dead line is dropped in the module (`eph_08`'s copy had already dropped
  it); no numerical change.
- **`compare_bootstrap_directions` no longer mutates its inputs.** The source did
  `np.asarray(b_x_boot, float)` and then flipped signs in place, so passing a float64 array
  (or a *slice* of one, as source cell 24 does) rewrote the caller's bootstrap cloud. In
  this pipeline the flip mask is always empty — the clouds arrive already hemisphere-aligned
  — so this changes no result, but the module copies defensively now.


### Why

`spatial_axis_comparison_rt_encoding_update.ipynb` (41 cells) compares the RT-encoding spatial
axis against **four** structural axes — waveform (CCA), MERFISH transcriptomics (CCA), and
retrograde tracing (LDA) — with bootstrap direction comparison, 95% confidence cones in
azimuth-elevation, and projection scatters.

`eph_08_waveform_axis.ipynb` is thinner than previously recorded. It ports the waveform CCA axis
and the `|T_rt|` projection scatter, and **it never fits the RT-encoding spatial axis itself**
(§6 of the source). So MERFISH, retrograde/LDA, `compare_bootstrap_directions`,
`cone_half_angle`, and the multi-axis cone visualization have no home anywhere in `eph_*`.

`spatial_encoding.py` is not that home either: it provides `SpatialEncoder` (CCF maps, subgroup
maps, permutation tests) and contains **no axis-fitting machinery at all**. `eph_08` currently
carries private inline copies of `fit_spatial_axis_cca` and `bootstrap_spatial_axis_cca`.

(`spatial_axis_comparison_rt_encoding.ipynb` without `_update` was the older duplicate and is
already archived. `_update` is the source for this port.)

### What to do

- **`spatial_axes.py`** (new flat module), from `spatial_axis_..._update` cells 14 and 15:
  `fit_spatial_axis_linear` / `_cca` / `_LDA` and their bootstrap wrappers, `cone_half_angle`,
  `compare_bootstrap_directions`, `vectors_to_az_el`, `plot_projected_arrow_with_cone`, and
  `plot_projection_scatter` (cell 31). Keep it separate from `spatial_encoding.py` — different
  question (direction of a gradient vs clustering in space), different inputs.
- **Refactor `eph_08`** to import from `spatial_axes.py`, deleting its two inline copies.
  Behavior must not change; the projection figure is a poster figure.
- **`eph_09_structural_axes.ipynb`**:
  - Fit the RT-encoding spatial axis (linear, scalar `T_rt` → OLS) and the baseline `T_rt_bl`
    axis, with 2000-resample bootstrap and cone half-angles (cells 17, 18). *This is the piece
    `eph_08` skipped.*
  - Load and fit the three structural axes (cells 20, 21, 22), each in its own guarded block —
    the source already uses `HAS_WAVEFORM` / `HAS_MERFISH` / `HAS_RETRO` flags so a missing
    asset degrades instead of failing. Keep that pattern.
  - Pairwise `compare_bootstrap_directions` over all axes, with angle, Wald W, χ² p and
    bootstrap p (cell 24), and the summary table (cell 29).
  - The three-plane projected-arrow figure with 95% confidence cones (cell 26) and the
    azimuth-elevation bootstrap scatter (cell 27).
  - Projection scatters of `T_rt` and `T_rt_bl` onto each structural axis, plus the combined
    table (cells 34, 36, 38).
  - Carry over the interpretation guide (cell 28 markdown) — it states how to read
    angle × p-value, and without it the summary table is hard to act on.
- Drop cell 40 (a commented-out registry sketch); register through `per_unit_stats_registry`
  properly or not at all.

### Notes

- ~~Confirm the external assets are reachable on Code Ocean before starting this item~~ —
  **done 2026-09-15: the user confirmed both are still mounted**
  (`merfish_data/adata/adata_mer_subset_2_2k.h5ad` and
  `LC_retro/manual_proofread_ccf_18brains.csv`, under `/root/capsule/data`, from the upstream
  capsule), so the degraded fallback was not needed and all three structural axes are
  written. `scanpy` was indeed absent from the pip block and is now added — see the status
  section above.
- Retrograde LDA sign is arbitrary; the source aligns it to the waveform axis (cell 22). Keep
  that convention or the cone figure flips between runs.
- Code Ocean only, like `eph_08`. Use the same `IS_CO` skip-guard structure so it runs clean
  locally.

---

## Repair and refocus `tongue_lickometer.ipynb` (→ `val_02_lickometer`)

_Logged 2026-09-16. The notebook does not run: every domain import is a bare name for a module
that moved into the library. Scope settled as: one session, two detector implementations
compared._

> **Status 2026-09-16: implemented on `wild` in two commits** — the repair (A1/A2/A3 +
> Implementation A) and then Implementation B with the six comparison figures. The notebook
> now runs. **Nothing has been executed against the real session**: local runs cover §1, §2,
> §4.1's synthetic demonstrations and §5's five unit tests; every Code Ocean-only cell was
> smoke-tested against synthetic stand-in `intermediate_data/` parquets only.
>
> **Still open, and what to do next:**
> - **Run it on Code Ocean.** That is the whole remaining point — every claim about B is still
>   a prediction from synthetic traces.
> - **Two assumptions this item flagged are still unverified**, and are now checked at runtime
>   rather than assumed: `session_analysis_mlk/<session>/intermediate_data/` exists (the loader
>   raises with the missing-file list, no fallback), and confidence is lowest during retraction
>   (§3.2 measures it and prints a verdict plus the consequence for B's change 4).
> - **Two departures from this plan**, both found while building and both recorded in
>   `CHANGELOG.md`: the overlap window must be held **fixed** during operating-point selection,
>   since it is a scoring parameter and letting it float buys agreement by widening the window;
>   and the flat-in-refractory test as posed was confounded by the filter also deleting genuine
>   fast licks, so flatness is now checked only below the shortest genuine ILI and paired with
>   a direct count of what the filter still deletes.
> - **One correction to a claim above:** A's re-arming failure is *not* unconditional. It
>   collapses to one event per session precisely when the confidence mask is at least as tight
>   as the spatial threshold — measured boundary at a 30 px threshold: masking at 35 px gives
>   the correct 4 events, masking at 30 px gives 1.
> - Still deferred as written: multi-session (§12), promoting B into the library, the `val_`
>   rename, and the library-side de-duplication (the notebook already imports the paths that
>   survive it, so it does not block).

### Why — the question this notebook asks

**Can Lightning-Pose tongue tracking detect a lick, where a lick is defined as the tongue
reaching the spout?**

This is a *spatial contact* definition, and it is why the notebook exists. Nothing else in the
repo or the library asks it:

| | Detector | A lick is… | Reference |
|---|---|---|---|
| **This notebook** | tongue within *N* px of a spout | **contact with the spout** | lickometer (for now) |
| Library / `kin_*` | `segment_movements_trimnans` → `annotate_licks_in_kinematics(tol=0.01)` | any tracked tongue excursion matched to a lickometer time | lickometer (always) |

The library's path is **agnostic to whether the tongue reached the spout**. Which is why
`kin_05` has a population of "movements without licks" — under the library's definition those
are unexplained; under this one, many are simply protrusions that never made contact.

So `detect_licks` is not a stale duplicate of the pipeline's detector. It is the only
implementation of the contact definition, and the only thing in the repo that can measure
**precision** — how often video claims contact when the lickometer says nothing.

**Where this is going.** The endpoint is not "tune a detector to agree with the lickometer" but
to reach a parameter set where pose tracking can serve as **QC on the lickometer** — catching
lickometer misses (bad contact, capacitive dropout) and false triggers (spout jostling, paw
contact). Two consequences shape the notebook:

- **F1 is the wrong objective.** F1 weights precision and recall equally; the two QC uses want
  opposite asymmetries. To flag a *lickometer miss* — pose says contact, lickometer says
  nothing — that event must be credible: **high pose precision**. To flag a *false trigger*,
  the absence must be credible: **high pose recall**. A single F1 argmax (cells 8–11) is right
  for neither. Produce a **precision–recall surface** with two named operating points; keep F1
  as a summary only. This also dissolves the notebook's own 30-vs-35 px waffling in cell 11 —
  those are two operating points, not two candidates for one answer.
- **No supervised fitting to lickometer labels.** A detector trained to reproduce the
  lickometer cannot audit it. Every parameter must be justifiable from geometry and tongue
  kinematics alone. Say so in the notebook; it is the first thing a reader will suggest.

**Reference direction.** Report **both directions, always, with names that carry the
direction** — never a bare `precision`/`recall`. Cells 6/7/14 currently print undirected rates
and cells 19–20 contradict them (taking `FP_times` from the *lickometer* frame and writing
clips into `…/false_positive/`, where they are lickometer licks the video missed). From the
notebook's own stored cell-16 output (30 px / 0.1 s / 0.1 s; `tp=5411, fp=435, fn=180`):
`pose_recall_vs_lickometer` = **0.926**, `pose_precision_vs_lickometer` = **0.968**, with 180
pose-only and 435 lickometer-only events. **Those two populations are the eventual QC product**,
not error terms to be minimized away. `f1 = 2tp/(2tp+fp+fn)` is symmetric, so every existing F1
heatmap is correct as drawn regardless of direction — nothing already plotted is discarded.

### Why — what is broken

**Every domain import is a dead bare name.** `tongue_kinematics_utils` and
`tongue_lickometer_utils` were promoted into `aind-dynamic-foraging-behavior-video-analysis`.
The package ships no `__init__.py` under `kinematics/`, so the full dotted path is the only
import form — what every modern notebook in `code/` already uses.

**Both library modules define the same six names; four have drifted** (diffed against library
`main`, `b21eac0`):

| Function | `tongue_lickometer_utils` | `tongue_kinematics_utils` | Keep |
|---|---|---|---|
| `detect_licks` | `(tongue_df, spoutL, spoutR, threshold)`, vectorized | extra `timestamps` arg, row loop | **`tlu`** — matches all call sites, 225× faster (0.2 s vs 51.8 s/session) |
| `calculate_metrics` | `(tp, fp, fn)` | `(tp, fp, fn, tn)` | **`tlu`** — `tku`'s `tn` subtracts counts from a timestamp; meaningless |
| `calculate_metrics_witheventkeys` | 5-tuple | 6-tuple, same bogus `tn` | **`tlu`** |
| `load_keypoints_from_csv` | plain `read_csv` | `dtype=str` + `to_numeric(errors='coerce')` | **`tku`** |
| `filter_timestamps_refractory` | — | — | byte-identical |
| `mask_keypoint_data` | — | — | byte-identical |

Against `tku`, cells 6/7/12/13/17 raise `TypeError` and cells 6/7/14 raise `ValueError`. Loud
failures, not silent wrong numbers. Only this notebook and `tongue_kinematics.ipynb` (HOLD)
consume these six names ecosystem-wide; nothing in the library pipeline calls `detect_licks`.

**Convention rot.** `pip install` in cells 0–1 (both Dockerfile-pinned now); no ENV block
(three hardcoded `/root/capsule/data/...` paths); no `plotstyle`; cell 3 selects the session by
globbing NWB filenames with a date comparison (`datetime` vs the string `'05-31-2024'`) that is
always `False`, and only works because the animal filter already leaves one row; cell 4
reimplements `trim_kinematics_timebase_to_match`, which `integrate_keypoints_with_video_time`
owns now.

### Scope

- **One session**: `behavior_716325_2024-05-31_10-31-14`, the one already in the notebook. It
  is in `all_tongue_movements_04022026.parquet` (1 of 44), so it has been reprocessed by the
  current pipeline and its `intermediate_data/` exists.
- **Two detector implementations, compared**: **A** = the notebook's current detector (spatial
  threshold + refractory filter), kept as the baseline; **B** = a gap-aware, hysteretic
  detector timestamped at closest approach.
- Everything modernized to current library definitions, reading per-session
  `intermediate_data/` parquets.

Multi-session is **deferred, not skipped** — see Notes.

### What to do — A1. Re-point data loading at `intermediate_data/`

Verified end to end; every column traced.

```python
inter = SESSION_DIR / EXAMPLE_SESSION / "intermediate_data"

kps    = {"tongue_tip_center": pd.read_parquet(inter / "kps_raw_tongue_tip_center.parquet")}
tongue = mask_keypoint_data(kps, "tongue_tip_center", confidence_threshold=CONF)

t0 = float(pd.read_parquet(inter / "nwb_df_trials.parquet")["goCue_start_time"].iloc[0])
tongue["time"] = tongue["time_raw"] - t0          # == time_in_session

spoutL = pd.read_parquet(inter / "kps_raw_spout_r.parquet")[["x","y"]].mean()  # animal's LEFT
spoutR = pd.read_parquet(inter / "kps_raw_spout_l.parquet")[["x","y"]].mean()  # animal's RIGHT
licko  = pd.read_parquet(inter / "nwb_df_licks.parquet")["timestamps"].to_numpy()
```

- Every `kps_raw_*.parquet` carries `x`, `y`, `confidence`, `time`, `time_raw`
  (`integrate_keypoints_with_video_time` Step 5 inserts the last two; `time` is
  `Behav_Time − Behav_Time[0]`, *exactly* what cell 4 built by hand).
- Writing `time_in_session` into `tongue["time"]` makes detector output directly comparable to
  `nwb_df_licks['timestamps']` with **no offset arithmetic anywhere**. The hand-rolled
  `- keypoint_timebase[0]` does not come back.
- The `spout_l`/`spout_r` swap (bottom camera mirrors L/R) is preserved, matching cells 6/7 and
  `kin_06` §3.

**Use `kps_raw_*` for positions, not `tongue_kins.parquet`.** `tongue_kins` is
post-`kinematics_filter`, which cubic-interpolates across NaN gaps, low-pass filters, then
reindexes onto the full frame set. Tracked frames at the *edges* of each protrusion therefore
carry values contaminated by interpolation over the untracked gap — and protrusion edges are
exactly where a contact threshold gets crossed. Raw masked keypoints are the honest input, and
are what the original notebook used. (`tongue_kins` does keep every row — checked; the
docstring's "retains only the originally available time points" refers to values, not rows.)

**Delete cells 3 and 4 entirely.** The NWB glob, `parseSessionID`, the broken date filter,
`trim_kinematics_timebase_to_match`, the raw-CSV load and the manual time-zeroing all collapse
into the block above. Also drops the dependency on the `matt_test_DLC_LP_results_20240920`
asset.

### What to do — A2. Imports

```python
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_lickometer_utils import (
    detect_licks, filter_timestamps_refractory, calculate_metrics,
    calculate_metrics_witheventkeys,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (
    mask_keypoint_data,
)
```

Do **not** block on a library PR. The library-side de-duplication is a deletion plus a pin bump
plus a Docker rebuild; batch it with the outbound-metrics consolidation below. Resolution
recorded in the boundary item.

### What to do — A3. Scoring layer, shared by both implementations

Both detectors emit a list of event times; everything downstream is common.

- **New §2 — convention self-test.** Call `calculate_metrics` on hand-built synthetic event
  lists (one guaranteed coincidence, one pose-only, one lickometer-only) and assert which
  output slot each lands in. Runs locally, needs no data, and is how the labelling ambiguity
  above was found.
- Derive `pose_recall_vs_lickometer` / `pose_precision_vs_lickometer` from a single stated
  direction. Never print a bare `precision`/`recall`.
- Fix cells 19–20 to match: pose-only events from the pose-classified frame, lickometer-only
  from the lickometer frame; clip output directories follow.
- State, don't fix: `calculate_metrics` uses a greedy two-pointer match that advances the
  ground-truth index on mismatch. With a 100 ms window and real ILIs of 100–150 ms an
  interleaved run can be matched sub-optimally. It is not mutual-nearest-neighbour, and the
  notebook should say so.

### What to do — Implementation A (baseline)

Unchanged in behavior. `detect_licks(tongue, spoutL, spoutR, threshold)` →
`filter_timestamps_refractory(licks, t_refractory)`. Ported as-is so the comparison is honest;
**do not quietly improve it.**

Grid (the notebook's own, cell 7): spatial ∈ `arange(10,51,5)` (9) × refractory ∈
`arange(0,0.11,0.01)` (11) × overlap ∈ `arange(0.005,0.251,0.005)` (50) = 4 950 scorings, 9
detector runs. **Benchmarked: ~38 s.**

Three properties to document, because they are what B responds to:

1. **`.dropna()` discards the time base.** After it the loop indexes a compacted array where
   consecutive entries may be 2 ms or 2 s apart, and nothing downstream can tell.
2. **State re-arms only on a surviving frame outside the threshold.** Since the confidence gate
   preferentially removes low-confidence retraction frames — the same frames that do the
   re-arming — raising confidence merges adjacent contacts. Demonstrated: on synthetic frames
   where the tongue is tracked only near the spout, `detect_licks` returns **one event for the
   entire session**.
3. **Events fire on the rising edge**, so event time depends on the spatial threshold.

**The confidence axis exists to expose (2).** Run A at `confidence_threshold` ∈ {0.80, 0.90} —
the original's value and the current pipeline's. Not a tuning exercise: it is the measurement
showing A's detection count and ILI distribution are hostage to a masking parameter the
notebook treats as a constant. One extra pass (~38 s).

### What to do — Implementation B (gap-aware hysteretic contact detector)

Four changes, each fixing one of the above.

1. **Keep the full time base; gaps terminate a contact.** Work on `d(t)` = distance to nearest
   spout over *all* frames; close an open contact when an untracked gap exceeds `max_gap_s`. A
   tracking dropout becomes explicit evidence instead of being invisible.
2. **Timestamp at closest approach**, not first entry — threshold-invariant, and the right
   physical analogue of the lickometer's electrical contact (deepest protrusion).
3. **Hysteresis**: enter at `d ≤ enter`, exit at `d > exit`, with a dead band between. Acts on
   the *signal* at the moment of decision rather than deleting events from the output.
4. **Asymmetric confidence — high bar to assert, low bar to deny.** `conf ≥ conf_assert` to
   *open* a contact, `conf ≥ conf_deny` to *close* one. A retraction frame at 0.84 is not
   trusted to claim contact but is plenty to establish the tongue left. Directly fixes the
   fusion failure, and the asymmetry is right for a QC instrument where a false contact is
   expensive and a false gap is cheap.

Vectorizable — no Python loop over frames. Prototyped and benchmarked:

```python
on  = tracked & (d <= enter) & (conf >= conf_assert)
off = (tracked & (d > exit_) & (conf >= conf_deny)) | (~tracked & (gap > max_gap_s))
last_on  = np.maximum.accumulate(np.where(on,  idx, -1))
last_off = np.maximum.accumulate(np.where(off, idx, -1))
inside   = last_on > last_off          # one event per contiguous run, at argmin(d)
```

Grid: `enter` ∈ `arange(10,51,5)` (9) × dead band ∈ {5,10,15,20,25} px (5) × overlap (50).
**Benchmarked: 133 ms per detector run, 6 s for the 45-run grid** — cheaper than A.

Fixed rather than swept, with stated justification:

- `max_gap_s = 0.020` — matches the library's own segmentation tolerance
  (`segment_movements_trimnans(max_dropped_frames=10)` at 500 Hz). Use the library's number
  rather than inventing one.
- `conf_assert = 0.90` — the current pipeline's value.
- `conf_deny = 0.60` — sanity-check, do not tune. Report sensitivity in a line, not an axis.

**Choosing the dead band.** There is a wide valid window, which is why this is not trading one
arbitrary number for two. *Lower bound*: the band must exceed the tracker's position noise, or
a stationary tongue wanders across it — that noise is measurable, and `pixel_error.ipynb`
already reports it for `tongue_tip_center`. *Upper bound*: the band must sit well inside a full
retraction excursion (jaw→spout distance, tens of px), or genuine retractions stop crossing it
— `kin_06` §3 computes that distance per session. Flatness between those bounds is itself a
checkable result.

**The falsifiable claim.** Put the dead band **and** the refractory period on B's grid
together. Prediction: **once the dead band exceeds the noise scale, F1 goes flat along the
refractory axis** — the refractory filter buys nothing, because hysteresis has already removed
what it was there to remove. If it does not go flat, hysteresis is not capturing everything and
there is a second source of double-triggering to find. Either outcome is a result.

### What to do — the comparison

Six figures, all on the one session. Total sweep cost well under two minutes.

| # | Figure | What it shows |
|---|---|---|
| 1 | Detection counts + ILI distributions, A vs B, at conf 0.80 and 0.90 | A's count and ILI tail move with the masking parameter; B's should not |
| 2 | **Precision–recall surface** per implementation, with the two QC operating points marked | the deliverable; replaces the F1 argmax |
| 3 | Event-time offset vs matched lickometer events (median + spread) vs spatial threshold | A drifts with threshold; B should be flat |
| 4 | F1 vs refractory period, for B at several dead bands | the falsifiable claim above |
| 5 | Per-event inspection of the two disagreement populations at B's operating point, + distance to nearest spout | the QC product |
| 6 | Labeled video clips at example disagreement timepoints (CO only) | ground-truth eyeball |

On figure 3 — from the prototype, on a synthetic ramped protrusion, A fires **14–22 ms early
and drifts 8 ms across the 25–40 px range**, while B lands exactly at closest approach at every
threshold. If that reproduces on real data it explains a structural feature of the cell-9
heatmap: the spatial and overlap axes are coupled by construction, so the apparent improvement
along the diagonal is partly the overlap window absorbing a threshold-induced latency shift,
not a real optimum. Cell 11 flagged the symptom (*"maximizing both … produces best performance;
however, unsure if this is truly warranted"*) without the explanation. With B the two axes
decouple, and the required overlap window should shrink well below 100 ms — sharpening the §8
argument.

On the ILI argument (cells 12–13): the sub-100 ms population is a **mixture** of jitter
artifacts and genuinely fast licks. The refractory filter cannot separate them — it deletes
both, which is why the argument has to lean on "faster than a real lick" as a blanket claim.
Hysteresis separates them by construction, because only the artifacts lack a retraction. So
whatever short ILIs survive B are real, and the distribution becomes interpretable instead of
an assumption to defend.

### What to do — section plan

| § | Content | From |
|---|---|---|
| 1 | Header (contact definition; why it differs from the library's), setup, `plotstyle`, ENV | new — copy `kin_06` cell 3 |
| 2 | Scoring-convention self-test | **new** (A3) |
| 3 | Load `intermediate_data/`; the distance signal `d(t)` | replaces cells 3–4 |
| 4 | **Implementation A** — detector, its three properties, worked example at 30 px / 0.1 s | cell 6 |
| 5 | **Implementation B** — detector, the four changes, worked example at matched settings | **new** |
| 6 | Sweeps for both; precision–recall surfaces; two QC operating points | cells 7–10, reframed |
| 7 | Confidence sensitivity: A vs B at 0.80 / 0.90 (figure 1) | **new** |
| 8 | Event-time accuracy vs threshold (figure 3); overlap-threshold justification revisited | cells 14–16 + new |
| 9 | ILI structure and the refractory-vs-hysteresis test (figure 4) | cells 12–13, reframed |
| 10 | Per-event inspection of disagreement populations (figure 5) | cells 17–19 |
| 11 | Labeled video clips (CO only) | cell 20 |
| 12 | Short markdown: relation to `coverage_pct` and `kin_05` — **no new analysis** | **new** |

Cell 11 and the trailing comment blocks become markdown. Cell 19's inline
`plot_tongue_trajectory` stays in the notebook (one consumer — same call `kin_06` made for
`plot_standard_lick_landmarks`), restyled onto `plotstyle`.

For §11, keep `extract_clips_ffmpeg_encode`. Do **not** substitute
`video_clip_utils.extract_clips_ffmpeg_after_reencode`, which uses `-c copy` and snaps to
keyframes — it will not land on an exact event timepoint. The `libx264` re-encode is correct
here; its eventual home is `video_clip_utils`.

§12 states the boundary, does not re-measure it: `analyze_tongue_movement_quality` →
`tongue_quality_stats.json` → `coverage_pct` is the fraction of lickometer licks with a nearby
tongue **movement** — a recall measure for a contact-agnostic detector, already computed per
session, already pooled by `test_session_quality_analysis.ipynb`, already thresholded at 90 % by
`data_loading.load_session_quality_filter`. No precision side, no free parameter. Cite it.
`kin_05` §4 inspects licks-without-movements with per-event traces — cross-reference rather
than duplicate. `pixel_error.ipynb` measures keypoint error against human labels: different
layer, no overlap.

**Where B lives.** In the notebook, not the library, until validated on real data.
`detect_licks` stays as the baseline. Once B has earned it, promote it into
`tongue_lickometer_utils` *alongside* `detect_licks`, not replacing it — they implement
different decision rules and the comparison should stay reproducible.

### Notes

**Verified 2026-09-16:**
- All six function pairs diffed against library `main` (`b21eac0`).
- `detect_licks` 0.2 s (`tlu`) vs 51.8 s (`tku`) per 1.8 M-frame session; A's full grid 38 s.
- B vectorized: 133 ms/run, 6 s for the 45-run grid; agrees with a loop prototype on a toy trace.
- A's re-arming failure (tongue tracked only near the spout → 1 event/session); A emits 4 events
  on one noisy protrusion where B emits 1; A fires 14–22 ms early with 8 ms of
  threshold-dependent drift, B lands at closest approach at every threshold.
- Directional rates recomputed from the notebook's own stored cell-8 / cell-16 outputs.
- Full loading path: `kps_raw_*.parquet` carries `x`/`y`/`confidence`/`time`/`time_raw`;
  `nwb_df_trials.parquet` supplies the `goCue_start_time` origin; together these reproduce
  `time_in_session`, matching `nwb_df_licks['timestamps']`.
- `kinematics_filter` reindexes onto the full frame set and asserts the time base is unaltered
  — `tongue_kins` keeps every row, but values are interpolation-contaminated at protrusion edges.
- `behavior_716325_2024-05-31_10-31-14` is in the pooled parquet.

**Unverified:**
- **Everything about B's behavior is synthetic so far.** The latency offset's real magnitude
  depends on actual protrusion kinematics; the flat-in-refractory prediction is a prediction.
- **"Confidence is lowest during retraction"** — the premise behind B's asymmetric gate — is a
  physical assumption about the tracker, not measured. `plot_keypoint_confidence_analysis`
  (saved per session by `analyze_tongue_movement_quality`) is where to check it before leaning
  on `conf_deny`.
- Whether `session_analysis_mlk/<session>/intermediate_data/` exists for the example session.
  `kin_05`/`kin_06` assume this path; neither has run there.
- Whether the `video_preds_labeltest` asset (§11's labeled video) is still attached.
- Real session frame count. The 38 s / 6 s figures assume 1.8 M frames (1 h @ 500 Hz).

**Local vs Code Ocean.** Local: §1, §2 only — plus B's unit tests, which run on synthetic
traces with no data. The sweeps need per-frame keypoints and NWB lick times; neither is in
`data/for_local/`. Header should read like `eph_09`'s, not `kin_06`'s. Code Ocean: §3–§12.

**Deferred: multi-session.** Not skipped — **not yet well-posed.** The threshold is in pixels,
and pixels are not transferable: camera position, zoom and spout placement vary, so 30 px is a
different physical distance in each session. Pooling would put a tight-looking error bar around
a quantity that does not mean the same thing session to session. The prerequisite is expressing
the threshold in jaw→spout units (`kin_06` §3 computes that scale, one extra parquet read).
Report the conversion factor for this session alongside the pixel value so the parameter is
quotable later; then "does the normalized threshold transfer?" becomes a real second question,
and 44 sessions at ~44 s each is ~30 min whenever it is asked.

**Ordering and risk.** A1 → A2 → A3 is the repair and is testable on the example session alone
— commit that first. Implementation B and the comparison figures are the second commit. Nothing
already plotted becomes wrong: F1 is symmetric, so the existing heatmaps stand. What changes is
which figure carries the conclusion (precision–recall surface, not F1 argmax), and whether A's
parameters survive the 0.80 → 0.90 confidence shift — which is the point of re-running.

**On the `val_` prefix.** The category is real — these notebooks validate the *instrument*, not
the behavior, and fit neither `kin_` nor `eph_`. **Done 2026-09-16: renamed to
`val_02_lickometer.ipynb`.** `pixel_error` / `test_session_quality_analysis` were deliberately
left alone — moving them is a `git mv` plus a reference sweep with no analysis change, and
belongs to a separate repo-wide decision.

### Adjacent findings — not part of this item

Turned up while auditing `code/` for this plan. Independent of the lickometer notebook; each
should be decided on its own.

1. **`REORG.md` describes `model_quality.ipynb` wrongly** — listed as "foraging
   behavioral-model quality"; it is a single-session Lightning-Pose pipeline walkthrough (load
   → mask `tongue_tip_center` @0.90 → filter → segment → annotate → lick coverage), already on
   modern library imports. One-line factual correction.
2. **`test_session_wrapper.ipynb` is filed under methods evaluation but is a batch runner** —
   19 cells, most commented out, wrapping `run_batch_analysis`. Operations, not evaluation;
   arguably belongs under "KEEP — pipeline / data generation". One-line reclassification.
3. **`tongue_kinematics.ipynb` cells 103–105 call `detect_licks_multiple`, which exists
   nowhere** — not in library `main`, `fix/video-csv-header`, `code/`, or local library history
   (`LC_manuscript` could not be fetched; "not found", not "does not exist"). This matters to
   *that* notebook's archive gate: REORG's "already covered elsewhere — do not port" line is
   true for cells 78–79, but 103–105 are a different, unrunnable analysis (a multi-keypoint
   contact detector). Either it gets re-derived somewhere or it is explicitly dropped, but
   `tongue_kinematics.ipynb` should not be archived under the claim that it is already covered.
   If re-derived, this notebook is the natural host — three tongue keypoints give a better
   contact estimate than one — but that is a future Implementation C, not scope here.

---

## Define the boundary between the library and this repo's `code/` modules

_Logged 2026-09-15. Do this before the next round of "should this go in the library?" decisions —
it is currently answered ad hoc, and at least three open items depend on the answer._

### Why

Both `aind-dynamic-foraging-behavior-video-analysis` and this repo's `code/` carry "kinematics
utils" and "ephys utils", with no stated rule for what belongs where. The result is duplication,
inconsistent figure styling, and a recurring per-function argument.

Current state, for reference:

**Library** (`aind_dynamic_foraging_behavior_video_analysis`)
- `kinematics/tongue_kinematics_utils.py` (1437 lines) — segmentation, aggregation, annotation,
  lick detection, keypoint masking/filtering, video/CSV discovery, **and 5 plot functions**
- `kinematics/tongue_analysis.py` (563) — batch runner, QC stats, intermediate generation
- `kinematics/tongue_lickometer_utils.py` (275), `video_clip_utils.py` (218),
  `kinematics_nwb_utils.py` (88)
- `ephys/tongue_ephys.py` (516) — session dir/intermediate loading, event construction,
  raster/PSTH primitives, `RasterPlotter`, **and 5 plot functions**

**This repo** (`code/`, flat — notebooks import by bare name)
- `data_loading.py` (187), `ephys_utils.py` (323), `encoding_methods.py` (558),
  `encoding_plots.py` (895), `per_unit_stats_registry.py` (442), `spatial_encoding.py` (629),
  `plotstyle.py` (190), `ccf_utils.py` (58), + planned `spatial_axes.py`

### Concrete problems this causes

1. **Duplication inside the library.** `tongue_lickometer_utils.py` and
   `tongue_kinematics_utils.py` define the same six functions by the same names:
   `filter_timestamps_refractory`, `calculate_metrics`, `calculate_metrics_witheventkeys`,
   `detect_licks`, `mask_keypoint_data`, `load_keypoints_from_csv`. Which one a notebook gets
   depends on which module it imported.
2. **Plotting has no home rule.** The library owns ~10 plot functions (movement tiles, processing
   steps, keypoint confidence, raster/PSTH, unit panels); this repo owns `plotstyle.py` +
   `encoding_plots.py`. The library's plotters predate and ignore `plotstyle.py`, which is why
   every port has to restyle figures by hand.
3. **"Ephys utils" in both.** `tongue_ephys.py` (primitives) vs `ephys_utils.py` (analysis config,
   spike counting, session bundles, `all_counts_df`). The layering is actually reasonable —
   repo-on-top-of-library — but nothing states it, so new code lands in whichever was opened last.
4. **Silent cross-boundary coupling.** `data_loading.load_session_quality_filter` reads
   `tongue_quality_stats.json`, which the library's `analyze_tongue_movement_quality` writes.
   Neither side declares the contract; a library-side field rename breaks this repo quietly.

### Proposed criteria (starting point — decide, then write it into CLAUDE.md)

One test: **would another AIND project doing tongue kinematics want this?**

- **Library** — produces or annotates the per-session intermediates; runs in the batch pipeline;
  generic to tongue-kinematics work; must stay stable because other capsules depend on it.
  *Segmentation, aggregation, annotation, QC stats, lick detection, video/NWB I/O.*
- **This repo** — analysis built *on top of* those intermediates, specific to the LC-NE RT-encoding
  question; fast-moving; free to churn. *Encoding models, per-unit registries, spatial topography
  and axes, manuscript figure style.*

Sorting the open cases under that test: `compute_outbound_metrics` → library (already the plan);
`annotate_movement_bouts` → library, once settled; `encoding_methods`, `per_unit_stats_registry`,
`spatial_encoding`, `spatial_axes`, `encoding_plots`, `plotstyle` → repo.
`ephys_utils.build_trial_features` is the genuinely ambiguous one — it derives generic per-trial
kinematic features, but its column set (`kcols`) is chosen for this project's encoding models.

### What to do

- Agree the criteria, then record them in `CLAUDE.md` so the rule is enforced at the point new
  code gets written, not rediscovered per function.
- Resolve the library-internal duplication in (1) — one definition, one import path.

  > **Resolved by inspection 2026-09-16** (see the `tongue_lickometer` item above). Keep
  > `tongue_lickometer_utils`'s `detect_licks` (vectorized; 225× faster than
  > `tongue_kinematics_utils`'s row-loop copy, and the only signature its call sites use),
  > `calculate_metrics` and `calculate_metrics_witheventkeys` (the `tku` copies add a `tn`
  > that subtracts counts from a timestamp — delete, do not merge). Keep
  > `tongue_kinematics_utils`'s `load_keypoints_from_csv` (coerces to numeric; `tlu`'s does
  > not) and its `mask_keypoint_data` / `filter_timestamps_refractory` (byte-identical — keep
  > one copy). Net: `tongue_lickometer_utils` owns **contact detection and event scoring**,
  > `tongue_kinematics_utils` owns **keypoint I/O, filtering and movement segmentation**; the
  > two implement different lick definitions and both are load-bearing, so this is a
  > de-duplication, not a consolidation. Also move `extract_clips_ffmpeg_encode` from
  > `tongue_lickometer_utils` to `video_clip_utils` (its `libx264` re-encode is the correct
  > precise-seek variant; do **not** replace it with `extract_clips_ffmpeg_after_reencode`,
  > which is `-c copy` and keyframe-snapped). Only two consumers exist ecosystem-wide —
  > `val_02_lickometer.ipynb` and `tongue_kinematics.ipynb` (HOLD) — so this is a low-risk
  > deletion, and the lickometer notebook can ship against `tongue_lickometer_utils` before
  > the library PR lands. A validated Implementation B would land here too, **alongside**
  > `detect_licks` rather than replacing it.
- Decide the plotting rule in (2): either the library's plotters adopt `plotstyle.py`, or the
  library stops shipping plot functions and this repo owns presentation entirely.
- State the layering in (3) explicitly in both modules' docstrings.
- Give (4) a declared contract — a documented schema for `tongue_quality_stats.json`, or a
  library-side accessor this repo calls instead of reading the JSON directly.

### Notes

- Library changes need their own branch/PR + a pin bump here, so batch them: do this alongside
  the outbound-metrics consolidation below rather than as a separate round trip.
- Blocks nothing immediately, but every deferred "promote to the library?" note in this file
  (`compute_outbound_metrics`, `annotate_movement_bouts`, the `kin_06` landmark helper) resolves
  faster once the criteria exist.

---

## Fold outbound metrics into the library's movement aggregation step

_Logged 2026-09-11._

### Why

The outbound-phase metrics (`out_duration`, `out_peak_velocity`, `out_mean_velocity`,
`out_total_distance` — computed over the outbound segment only, start → max-jaw-excursion frame)
are **not** part of the standard segmentation/aggregation in
`aind-dynamic-foraging-behavior-video-analysis`. The library's
`kinematics/tongue_kinematics_utils.py::aggregate_tongue_movements()` only emits *whole-movement*
metrics (`peak_velocity`, `duration`, `total_distance`, excursion endpoints). No branch
(`main`, `LC_manuscript`, `video_alignment`) computes an outbound split.

As a result the logic lives as orphan code in this repo, duplicated in two places:
- `code/add_outbound.ipynb` — the canonical generator (`compute_outbound_metrics` +
  `augment_movs_with_outbound_in_place`), which backfills `out_*` into each per-session
  `tongue_movs.parquet` after the fact.
- `code/tongue_movements_all.ipynb` cell 30 — a duplicate copy of `compute_outbound_metrics`.

Every downstream consumer already depends on these columns: `ephys_utils.py` (`kcols` →
`first_move_out_*` / `cue_response_out_*` in `all_counts_df`), `eph_02/03/04`, and `kin_00/01/03/04`.

### What to do

- Move `compute_outbound_metrics()` into the library, computing `out_*` **inside**
  `aggregate_tongue_movements()` (it already has `movement_id`, `x`, `y`, `v`, `time_in_session`
  and the jaw keypoint it needs) so every session's movement table carries `out_*` by default —
  no separate augmentation pass.
- Once the library emits `out_*` natively: delete the two duplicate defs
  (`add_outbound.ipynb`, `tongue_movements_all.ipynb` cell 30), retire the
  `augment_movs_with_outbound_in_place` backfill, and re-run the per-session pipeline so the
  intermediates are regenerated with `out_*` baked in from segmentation.
- Verify column parity against the current `all_tongue_movements_04022026.parquet` before/after
  (same values, same dtypes) so `ephys_utils.py` and the `kin_*`/`eph_*` notebooks are unaffected.

### Notes

- Do this in the library repo (its own branch/PR), then bump the pin here. Batch it with the
  library-vs-repo module-boundary item above — both touch the library and both need a pin bump,
  so one round trip is cheaper than two.
- Unblocks archiving `add_outbound.ipynb` and simplifies `tongue_movements_all.ipynb`
  (see REORG.md).

---

## Upgrade the Code Ocean environment off Python 3.9

_Logged 2026-09-07. Not scheduled — do this deliberately, on a branch, not mid-analysis._

### Why

`environment/Dockerfile` builds from
`codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04`. Python 3.9 reached
end-of-life in October 2025. That single pin is the root cause of several unrelated-looking
workarounds scattered through this repo:

1. **`rachel-analysis-utils` needs `--ignore-requires-python`.** The package declares
   `requires-python>=3.10`. The Dockerfile installs it in an isolated `RUN` layer with
   `--ignore-requires-python` specifically so that flag does not leak into the main pip block
   (where it would let pip resolve a too-new numpy and break the build). See the comment above
   that layer.
2. **`rachel_analysis_utils.analysis_utils` cannot be imported at all** on 3.9 — it has a
   nested-quote f-string at `analysis_utils.py:294` (`f'{x.split('_dff')...}'`), which is 3.12+
   syntax. Because of this, `fip_00_explore.ipynb` reimplements `enrich_df_trials`'s
   `num_reward_past` locally as `enrich_streaks` (unit-tested to match the package). That
   reimplementation exists *only* to dodge the import and can be deleted after the upgrade.
3. **The repo-wide "Python 3.9 compatible syntax only" rule** in `CLAUDE.md` — no `|` type
   unions, no structural pattern matching, no walrus in complex contexts — exists solely to
   match this image.

Upgrading the base image removes all three.

### Risk (why this is not a quick change)

The main pip block pins a mutually-consistent stack that was resolved against 3.9:

```
spikeinterface[full]==0.100.0   scipy==1.13.0        pynwb==3.0.0
hdmf-zarr==0.11.0               zarr==2.18.2         statsmodels==0.14.2
pyarrow==21.0.0                 seaborn==0.13.2      opencv_python==4.11.0.86
moviepy==1.0.3                  scikit-image==0.24.0 aind-ephys-utils==0.0.15
```

Several of these will need version bumps on a newer interpreter, and `spikeinterface[full]`
in particular pulls a large transitive tree. ~41 notebooks depend on this environment, with
the `eph_*` and `kin_*` families being the most exposed (they use spikeinterface, pynwb and
the ephys stack directly). A bad upgrade silently breaks working analyses.

### Suggested approach

- Do it on a dedicated branch, and use Code Ocean's environment versioning so the current
  working environment stays recoverable.
- Bump the base image to the Code Ocean `python3.11` JupyterLab image (matches
  `aind-motion-energy`, which already targets 3.11).
- Relax the pins in the main block to `>=` where possible, rebuild, then re-pin to whatever
  actually resolves — do not hand-guess versions.
- Drop `--ignore-requires-python` from the `rachel-analysis-utils` layer and fold it back
  into the main pip block.
- Verification pass, in rough order of blast radius: `eph_00`, `kin_00`, then `fip_00`.
  Confirm `import rachel_analysis_utils.analysis_utils` succeeds, then delete the local
  `enrich_streaks` reimplementation in `fip_00_explore.ipynb` and switch back to
  `enrich_df_trials`.
- Remove the Python 3.9 syntax rule from `CLAUDE.md` once the image is live.

### Related cleanups unblocked by this

- `fip_00_explore.ipynb` cell 3 pip-installs `fastparquet` at runtime because the image does
  not ship it (`rachel_analysis_utils.load_nwb_list` hardcodes `engine="fastparquet"`).
  That belongs in the Dockerfile pip block regardless of the Python version — it does not
  need to wait for this upgrade.
