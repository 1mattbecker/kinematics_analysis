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
| `kin_06_lick_geometry_choice` *(new)* | Does where the tongue goes carry choice information? | `cueresponse` 19, 49–56, 93–95; `tongue_kinematics` 60 |
| `kin_07_value_encoding` *(new)* | Do behavioral-model latents (Q, RPE) explain tongue kinematics? | `cueresponse` 17–28; `tongue_kinematics` 111–138 |
| `eph_07_bout_encoding` *(new)* — **done** | Do LC units respond differently to within-trial vs ITI movement bouts? | `intertrialmovs` 10–32 |
| `eph_09_structural_axes` *(new)* | Does the RT-encoding spatial gradient align with waveform / MERFISH / projection-target axes? | `spatial_axis_..._update` 13–38 |

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
  by `tongue_lickometer.ipynb`, which is KEEP. **No port needed.**

### Orphan code that needs a home before its notebook is archived

Same category as `compute_outbound_metrics` (see the outbound item below) — analysis logic
living only in a notebook cell. Unlike the outbound case there is no duplicate to reconcile;
there is exactly one copy, and it is inside a notebook slated for archiving.

- `annotate_movement_bouts` — defined only at `intertrialmovs` cell 10; called 6× in that
  same notebook and referenced nowhere else in the repo. Verified **absent from the library
  on all three branches** (`main`, `LC_manuscript`, `video_alignment`), searching both the
  function name and the `mov_bout_*` columns it emits. → `ephys_utils.py`.
- `plot_standard_lick_landmarks` — defined only at `cueresponse` cell 19. → `kin_06`, or
  `plotstyle.py` if a second consumer appears.

### Data-availability constraints (drives what is local-testable vs Code Ocean-only)

`all_tongue_movements_04022026.parquet` carries 49 columns. What matters for these ports:

- **Present, so pooled across sessions:** `has_lick`, `lick_count`, `lick_latency`, `event`,
  `movement_before_cue_response`, `cue_response`, `cue_response_movement_number`,
  `movement_number_in_trial`, `movement_latency_from_go`, `start_time`, `end_time`, `trial`,
  `session`, `goCue_start_time_in_session`, `endpoint_x/y`, `excursion_angle_deg`,
  `max_x_from_jaw`, `max_y_from_jaw`, `max_x_from_jaw_y`, `out_*`.
- **Absent:** `nearest_movement_id` (licks-without-movements needs per-session
  `nwb_df_licks.parquet`), spout/jaw **absolute** landmark positions (per-session keypoint
  means — note the `max_*_from_jaw` columns are jaw-*relative* and therefore pooled-safe),
  and any behavioral-model latent (`q_*`, `chosen_prob`, `rpe`).

Consequence: `kin_02` ext is fully local-testable; `kin_05` and `kin_06` are mostly pooled with
one CO-only section each; `kin_07`, `eph_07`, `eph_09` are Code Ocean-only.

### Recommended order (value ÷ risk) and archiving gates

1. `kin_02` extension — pure pooled parquet, no new dependencies, fully local-testable. — **done**
2. `kin_05_nonlick_movements` — mostly pooled, one CO-only section (turned out to be two
   partial CO-only splits, within §3 and §5 — see the item below). — **done**
3. `eph_07_bout_encoding` (+ bout helpers into `ephys_utils.py`) — clear spec, reuses
   `eph_00`'s helpers. — **done**
4. `spatial_axes.py` + `eph_09_structural_axes`, then refactor `eph_08` onto the module.
   **Confirm the MERFISH / retrograde assets are reachable on Code Ocean before starting.**
5. `kin_06_lick_geometry_choice`.
6. `kin_07_value_encoding` — last; new dependency on `get_mle_model_fitting`.

Archive only when **all** gates for a notebook are met:

| HOLD notebook | Archive after |
|---|---|
| `tongue_latency.ipynb` | `kin_02` §8–§11 (done) — **no remaining gate, ready to archive** |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | `eph_07` (done) — **no remaining gate, ready to archive** |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | `eph_09` |
| `tongue_kinematics.ipynb` | `kin_05` (done) **+** `kin_07` |
| `tongue_kinematics_cueresponse.ipynb` | `kin_06` **+** `kin_07` |

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

_Logged 2026-09-11._

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
  cell 95 — port once): max excursion from jaw, with and without licks. Jaw-relative, so
  **poolable across sessions** even though §3–§4 are not.
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

_Logged 2026-09-11._

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
  hits Han's pipeline. Confirm it still resolves before committing to the port.
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

_Logged 2026-09-11._

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

- **Confirm the external assets are reachable on Code Ocean before starting this item** —
  `merfish_data/adata/adata_mer_subset_2_2k.h5ad` and
  `LC_retro/manual_proofread_ccf_18brains.csv`, both under `/root/capsule/data`, both from
  Han's capsule. The MERFISH path also needs `scanpy`, which is not in the current pip block.
  If either is gone, land the RT-axis fit + waveform comparison and record the gap.
- Retrograde LDA sign is arbitrary; the source aligns it to the waveform axis (cell 22). Keep
  that convention or the cone figure flips between runs.
- Code Ocean only, like `eph_08`. Use the same `IS_CO` skip-guard structure so it runs clean
  locally.

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
