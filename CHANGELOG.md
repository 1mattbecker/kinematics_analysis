# Changelog

Notable changes to this project. Newest first. Dates are YYYY-MM-DD.

## 2026-09-16

### eph_08 / eph_09: use the reference run's spike-count windows and z_ccf filter
- Traced the poster figure `rt_response_projection_abs.svg` (2026-05-04 00:34:41) to
  `code/archive/spatial_axis_comparison_rt_encoding.ipynb` cell 42 (`execution_count` 37),
  committed 16 min later in `2f20188`. Confirmed by the SVG's per-panel tick colours, which
  are that notebook's `COLORS` entries for the waveform / MERFISH / retrograde axes, and by
  cell 42's stored output matching the figure's r/p/n exactly.
- **Root cause of the failed replication: the spike-count windows.** Cell 9 of that run
  (`execution_count` 7) builds `all_counts_df` with `count_window_s=(0.0, 0.5)` and
  `baseline_window_s=(-2, 0.0)`. `eph_08`/`eph_09` had `(0.0, 0.2)` / `(-1.0, 0.0)`,
  inherited from a commented-out cfg in `spatial_axis_comparison_rt_encoding_update.ipynb`
  — a file created *after* the figure, in the same commit, whose commented block had already
  been changed. The 500 ms / 2 s windows also match the published analysis.
- Changed both notebooks to `(0.0, 0.5)` / `(-2.0, 0.0)`; both continue to build
  `all_counts_df` in-notebook. Added a markdown note before each counts cell recording the
  provenance and a numeric check (response window should give 50 nominally / 43 FDR
  significant units of 103; 42 / 36 means the old windows are still in effect), plus prints
  of those counts.
- Added the reference's `z_ccf` ∈ [-5.2, -3.5] anatomical filter with its own note. It drops
  exactly one mislocalised unit (`behavior_758017_2025-02-06_11-26-14` unit 85,
  `z_ccf = -2.174`, 2.25 mm from the mesh centroid vs 0.79 mm for every other unit), taking
  the projection from n=100 to the reference's n=99. In `eph_09` it is placed after the axis
  fits and before §8, matching the reference's ordering (axis fit n=100, projections n=99).
- Verified locally against `data/for_local/`: with the old windows the pipeline reproduces
  the current `eph_08` output exactly (r=0.011, p=0.9109, n=100, proj max 2.196,
  |T_rt| max 12.096), and reproduces the reference's mesh centroid to 8 decimals, its
  unit counts, and its waveform axis — isolating the windows as the only remaining
  difference. Rebuilding counts needs spike times, so the final check must run on Code Ocean.

## 2026-09-15 (8)

### `kin_07_value_encoding.ipynb` — two fixes from the first Code Ocean run

- **Fixed a `KeyError: 'max_x_from_jaw_y'` in §5.** `build_trial_tables` selected only
  `KIN_COLS`, so the column §5 derives `max_x_from_jaw_y_rel` / `_distance` from never
  reached `trials_cue`. Added `AUX_KIN_COLS` — carried into both trial tables but
  deliberately excluded from §6's screen grid, since the raw absolute version is a
  near-duplicate of `max_y_from_jaw` and would add a redundant heatmap row.
- **Replaced the RidgeCV "beats null by 2 SD" criterion with a permutation p-value.**
  Found by the new integration test below: on fabricated noise latents, `summarize_screen`
  reported `rpe` as beating its null on an R² of +0.0000 against a null of −0.0024. With
  3 shuffles the null SD is itself noise, so the criterion was meaningless. `ridge_r2_screen`
  now defaults to 20 shuffles and returns `p_perm` = `(1 + #{null >= r2}) / (1 + n_shuffles)`;
  a latent counts as predicted only if R² > 0 **and** `p_perm` < 0.05, starred on the panel.
  §8's checklist item 3 updated to say so — this feeds the notebook's headline conclusion.
- **`run_value_screen` gained a `methods` argument.** §6.3 reads only the Spearman panel but
  was paying for the random forest and both nulls; it now requests `("spearman",)`. Skipped
  panels come back NaN-shaped so the plot/summary helpers still accept the result.
- Added a loud warning when `add_jaw_relative_columns` resolves zero jaw keypoints, instead
  of silently producing all-NaN columns, and guarded §5's Wilcoxon the way `spearman_screen`
  already was.

**New verification.** `verify_pipeline.py` fabricates a `trial_latents` table with the real
schema from the local parquet and runs every stage §3.6 onward — `build_trial_tables`,
`add_jaw_relative_columns`, `per_session_slopes`, `run_value_screen` (all four methods and
spearman-only), `add_previous_trial_rpe`, `plot_value_screen`, `summarize_screen`. It
validates *column availability at each stage*, not any scientific result, and is what would
have caught the `max_x_from_jaw_y` bug before the Code Ocean run. All stages pass; the
synthetic-latent screen test still passes unchanged.

## 2026-09-15 (7)

### `kin_07_value_encoding.ipynb` — new: do behavioural-model latents explain tongue kinematics?

Sixth and last of the HOLD ports. Merges the kinematics-vs-model-latents analysis that
existed in two partly-overlapping single-session copies — `tongue_kinematics.ipynb` cells
111-138 and `tongue_kinematics_cueresponse.ipynb` cells 17-28 — into one pooled notebook.
No `kin_*` or `eph_*` notebook previously used behavioural-model latents against
kinematics (`eph_06` uses them against ephys), so this is a topic the refactored series
did not track at all.

**Status: written, executed locally only to the extent the data allows.** §6.1 (the
kinematic covariance structure) runs on the pooled parquet and has real output; everything
from §3.4 on needs per-session `nwb_df_trials.parquet` and is Code Ocean-only, so it is
written and reasoned through but unexecuted. The notebook was run end-to-end locally to
confirm every gate takes its skip path cleanly (no errors, 23 code cells).

**Dependency check, done before writing anything.** `get_mle_model_fitting` still resolves,
and the fetch is fast enough that pooling was never in question:

- The import path both sources use —
  `aind_analysis_arch_result_access.han_pipeline.get_mle_model_fitting` — is now a
  **deprecated shim** forwarding to `...df_mle_model_fitting.get_mle_model_fitting`. The
  notebook prefers the new path and falls back to the shim, so it survives the shim's
  removal. `environment/Dockerfile` installs the package unpinned, so Code Ocean tracks
  whatever is current — which is the argument for not importing through the shim. No
  Dockerfile change needed (and none made).
- **Coverage, measured over all 44 sessions in ~48 s:** 40 sessions have a
  `QLearning_L2F1_CKfull_softmax` fit; the four sessions of subject **751004**
  (2024-12-20 … 2024-12-23) have no MLE records at all and drop out.
- Latent shapes confirmed: `q_value` and `choice_kernel` are `(2, N+1)`, `choice_prob`
  `(2, N)`, `rpe` `(N,)`.

**Pooled, not single-session.** Both sources run on one session. At ~1 s/session the fetch
is not the cost driver, and pooling is what makes **session** available as a sampling unit
— which is the whole point, given the statistics below. 40 sessions.

### Three statistical problems in the sources, not inherited

- **Pseudo-replication.** Both sources merge trial-level latents onto **movement**-level
  rows (`tongue_with_q = tongue_movements.merge(...)`), so *n* becomes the movement count
  while the predictor only varies per trial. Measured inflation in this dataset: 246,359
  movements against ~14,000 trials. Fixed two ways together — aggregate to one row per
  trial (`trials_cue`, the cue-response movement; `trials_mean`, the per-trial mean over
  all movements), **and** make session the unit of inference (statistic computed within
  session, tested across sessions). The one-cue-response-movement-per-trial assumption is
  checked rather than assumed: 14,138 trials carry one, 9 carry two (0.06%), and the first
  is kept.
- **Multiple comparisons.** §6's grid is 24 kinematics × 11 latents = 264 tests;
  uncorrected, ~13 would clear p < 0.05 with nothing present. Every p is BH-FDR corrected
  across the whole grid (`multipletests(method="fdr_bh")`), matching `encoding_methods.py`
  and `per_unit_stats_registry.py`. Noted in the notebook: a Wilcoxon on ~40 sessions has
  a p-value **floor** (~8e-13 at n=44), so strong and very strong effects report the same
  q — rank by |ρ|, not by q.
- **Correlated features.** Quantified rather than asserted, in §6.1 (which runs): median
  |ρ| among the 276 kinematic pairs is a mild 0.27, but 21 pairs exceed 0.7 and **five
  exceed 0.99**. Within-session z-scored, `max_x` / `max_x_from_jaw` / `max_x_distance` are
  literally the same variable (ρ = 1.000), as are `time_to_endpoint` / `out_duration`;
  `endpoint_y` / `max_y_from_jaw` sit at 0.996. The prose next to the 2×2 figure says
  plainly that RF importance splits credit arbitrarily across such blocks and MI
  double-counts them, and that **only the RidgeCV R² panel is a genuinely predictive
  measure** — it uses all kinematics jointly and is scored on held-out sessions
  (`GroupKFold`, the same discipline `kin_06` §8 uses).

### Other decisions worth recording

- **`TODO.md` was wrong that `attach_model_latents_to_trials` is defined identically in the
  two sources.** They differ in the sign of `q_diff` — `cueresponse` cell 13 computes
  `R_value − L_value`, `tongue_kinematics` cell 115 computes `L_value − R_value` — which
  flips the sign of every correlation involving it. Ported once, as **`R_value − L_value`**,
  so that positive `q_diff` and positive `endpoint_y` in `kin_06`'s jaw-centred frame both
  mean "toward the animal's right". The second difference: `tongue_kinematics` derives
  `q_diff_c` from `animal_response` inside the function while `cueresponse` has that block
  commented out and derives it later from the movement's `event`; the `animal_response`
  version is used, being defined for every responded trial.
- **Restored the alignment assertions both sources commented out.** A length mismatch
  between the latent arrays and the trial table would otherwise shift every latent by an
  unknown offset without raising. Because the upstream API does not document which trials
  it fits, the function checks the fit length against *both* candidates (responded trials
  vs all trials) and records which matched in a `latent_index_mode` column, so a mixture
  across sessions is visible rather than silent.
- **Added two checks of the left/right row convention (§3.5), which neither source makes.**
  Both assume row 0 of each `(2, N)` array is left; a swap would negate `q_diff` and turn
  `chosen_prob` into 1 − `chosen_prob` without erroring. Check 1 compares mean
  `chosen_prob` against the fit's own reported `prediction_accuracy` (under a swap it would
  match 1 − that instead); check 2 asserts mean `q_diff_c` > 0. Both fall out of quantities
  the fit already reports.
- **`q_sum` is new, not ported.** `TODO.md` §3 asks for it and neither source has it. It is
  the natural control for `q_diff`: `q_diff` is about which option is better, `q_sum` about
  how good the environment is overall — the vigour-like quantity a motor effect would most
  plausibly track.
- **§5's four pinned pairs are framed as descriptive, and §6.3 tests them.** The source
  picked those pairs after looking at the data and attached p-values that ignore the
  search. §6.3 re-runs the screen with the two jaw-derived columns appended and reports
  each pinned pair's q and its rank within the full FDR-corrected grid, so "do the pinned
  pairs fall out of the screen" is answered by the notebook rather than by the reader.
- **`rpe_prev` is computed on the complete trial table before the merge**, not after.
  Shifting after a merge that drops trials would make `rpe_prev` the RPE of the previous
  trial *that had a cue-response movement*, which can be several trials back. The sources
  happen to get this right; it is made explicit because the ordering is easy to invert.
  §7 screens both `trials_cue` and `trials_mean`, since "how the animal moves on the next
  trial" includes the preparatory non-lick movements `kin_05` establishes are real.
- **No jaw-reconstruction code duplicated from `kin_06`.** `kin_06` §2.1 carries an
  algebraic fallback because that notebook must run locally. `kin_07` cannot run locally at
  all, so it reads the real `kps_raw_jaw.parquet` keypoint directly (`get_jaw_y`, 6 lines)
  and the ~100-line fallback is deliberately not copied.
- Dropped from the port as redundant or superseded: `tongue_kinematics` cells 119-134
  (one-off `plot_kinematics_vs_q` calls, subsumed by the §6 screen), cells 131-134
  (hexbin/scatter of kinematics against each other, no latents involved — that is `kin_01`
  territory), and `cueresponse` cells 17, 22-24 (a commented-out plot helper and the
  keypoint loading `kin_06` already owns).

### Verification

The model-fit path cannot run here, so the §6 screen machinery was tested **independently
of the real latents**: real kinematics from
`data/for_local/all_tongue_movements_04022026.parquet` (14,039 cue-response trials, 44
sessions) against a *synthetic* latent built as a known linear function of one kinematic
column plus noise, at two noise levels, with a pure-noise negative control. The driver was
chosen to be a hard target — `excursion_angle_deg` sits at ρ = 0.92-0.97 with `endpoint_y`,
`min_y`, `max_y` and `max_y_from_jaw` — so this tests discrimination against near-twins,
not just against unrelated columns. Result: **all four methods ranked the planted driver
first at both noise levels**; the noise control gave driver q = 0.99 and held-out
R² = −0.001; RidgeCV ordered strong (0.80) > weak (0.31) > noise (−0.001). The test
execs the notebook's own cell sources, so it exercises the shipped code rather than a copy.

That run also produced the number quoted in §6's prose: a latent planted on **one**
kinematic column made **17 of 24** columns clear FDR in the Spearman panel. Sixteen were
not findings — which is the correlated-feature caveat, measured.

### Files

- `code/kin_07_value_encoding.ipynb` — new, 40 cells (8 sections).
- `TODO.md`, `REORG.md` — `kin_07` moved out of planned; both HOLD rows updated.

## 2026-09-15 (6)

### `kin_06_lick_geometry_choice.ipynb` — jaw position now read from real keypoints, preferentially

Three follow-ups to the `kin_06` port (previous entry), landed interactively after publishing it.

- **Stopped overclaiming validation of the jaw reconstruction.** The notebook's own §2.1
  prose said its fit residual "confirms the reconstruction" — it doesn't. That residual is
  an internal self-consistency check (how tightly each row's two `max_y_from_jaw ±
  max_y_distance` candidates cluster), not a comparison to the true jaw keypoint, and can't
  catch a systematic offset. Corrected in the notebook (§2.1/§3 prose, docstrings, print
  statements) and in `TODO.md`/`REORG.md`, which repeated the same overclaim.
- **§9's example-session panel now reuses `EXAMPLE_SESSION`** (the same session §3/§4 use,
  and the source `cueresponse` notebook's own session) instead of auto-picking whichever
  session has the most paired trials, so one example recurs through the notebook.
- **Investigated a user-flagged anomaly** in §9's pooled scatter (a few cue-response
  endpoints on the "wrong" side of the midline; the changed-mind marginal shifted
  negative). Traced both to real, non-bug causes: (1) the marginal shift is exactly the
  sign-flip definition applied to an already-known asymmetry — right-lick trials flip side
  25.1% of the time vs left-lick's 10.4%, so changed-mind trials skew toward the
  "flipped-to-right" case (753 of 1,185); (2) a handful of individual right-lick trials
  (10 of 3,003, 0.33%) do cross the fitted midline, concentrated in sessions with unusually
  small right-lick excursion magnitude. That surfaced a bigger, previously unflagged
  finding: **41 of 44 sessions show `|left-lick excursion| > right-lick excursion`** from
  the fitted jaw, by a median of 13 px (up to 57 px) — plausibly real spout geometry (right
  spout closer to the jaw's rest position for most animals), but only checkable against the
  true keypoint, not from the pooled parquet alone.
- **Checked whether a better local jaw estimate existed before reaching for Code Ocean.**
  Neither `startpoint_x/y` nor the movement bounding-box columns (`min_x/y`, `max_x/y`) sit
  at a fixed point — within-session SD 11-39 px, all far noisier than the algebraic
  reconstruction's own ~0.1 px internal residual. Confirms the reconstruction was already
  the best available *local* estimate; the open question was never precision, it was
  whether that recovered constant is the *same* point Code Ocean's `kps_raw['jaw']` calls
  the jaw.
- **New `load_jaw_from_keypoints()` + `get_jaw_positions()`** (§2.1). `get_jaw_positions`
  now tries every session's real `kps_raw_jaw.parquet` mean first and only falls back to
  `estimate_jaw_position()`'s algebraic reconstruction where that file is missing — locally,
  all 44 sessions, since no keypoint data exists in `data/for_local/`. Where both exist, it
  reports the gap between them directly, so the fallback's accuracy is validated
  automatically across every available session the first time this runs on Code Ocean,
  rather than spot-checked on one example the way §3's old cross-check did. `estimate_jaw_
  position()` itself is unchanged — it's now explicitly documented as the fallback, not the
  primary path. Re-executed end-to-end locally: prints "0 of 44 sessions from the real
  keypoint, 44 from the algebraic fallback," all downstream numbers unchanged.

## 2026-09-15 (5)

### New `code/kin_06_lick_geometry_choice.ipynb` — does pre-lick tongue direction predict choice?

Phase 2 of the repo reorg: the fifth of the six HOLD-notebook ports. Source is
`tongue_kinematics_cueresponse.ipynb` cells 19, 49-56, 93-95 plus `tongue_kinematics`
cell 60. `tongue_kinematics_cueresponse.ipynb` now gates on `kin_07` alone.

**35 cells, §1-§9.** Structured so the landmark figures read as the setup (§3-§4) and the
choice decode as the result (§7-§8), per the `TODO.md` outline. Executed end-to-end
locally against `data/for_local/all_tongue_movements_04022026.parquet` (246,359 movements,
44 sessions); §3-§4 print skip messages and are unexecuted.

**Correction to `TODO.md`'s data-availability note: `max_*_from_jaw` is NOT jaw-relative.**
`max_x_from_jaw` / `max_y_from_jaw` are the *absolute* pixel coordinates of the point
farthest from the jaw; the jaw-relative distances are `max_x_distance` / `max_y_distance`
(pooled means ~368 px vs ~25 px). So those columns are **not** pooled-safe as written, and
neither are `endpoint_x/y` — per-session mean `endpoint_y` spans ~80 px across the 44
sessions against a ~51 px within-session SD. Pooling them raw would have smeared the
anatomical frame the whole notebook rests on.

**The jaw origin turns out to be recoverable from the pooled parquet** — no Code Ocean
needed. New `estimate_jaw_position()` (§2.1) solves for each session's jaw keypoint from
the absolute/distance column pairs: `jaw_x = median(max_x_from_jaw - max_x_distance)`
(the tongue protrudes in +x, so the minus branch is correct — within-session SD ~4 px vs
~20 px for the plus branch), and `jaw_y` by a 1-D search over the two per-row candidates
`max_y_from_jaw ± max_y_distance`. Worst-session residual 0.25 px across all 44 sessions —
**but that residual is an internal self-consistency check, not external validation.** It
measures how tightly each row's two candidates cluster around the fitted point; it cannot
detect a systematic offset (e.g. `jaw_y` skewed toward whichever spout draws more excursions).
The real check — comparing against the actual `kps_raw['jaw']` keypoint mean — is written into
§3 but has **not executed**, since no `kps_raw_*.parquet` exists in `data/for_local/`. Treat
the reconstructed `jaw_y` as unverified until that cell runs on Code Ocean.
Everything from §5 on works in the resulting jaw-centered `*_rel` frame. This is what lets
§5-§9 pool at all while §3-§4 stay Code Ocean-only: the **spouts** still need per-session
keypoint means (`kps_raw_*.parquet`); the jaw is the only landmark recoverable without them,
and even that recovery is pending confirmation.

- **§3-§4 (Code Ocean only, unexecuted)** — `plot_standard_lick_landmarks` (source cell 19,
  the only copy in the repo) brought in here and restyled onto `plotstyle`; **not**
  promoted to `plotstyle.py`, which is style-only and has no second consumer. Plus the
  jaw↔spout scale printout (cell 94), a cross-check of the §2.1 reconstruction against the
  true keypoint mean, and cue-response endpoints coloured by lick side over the landmarks
  (cell 93). The source's mirrored spout naming (`spout_l` is the animal's *right* spout —
  the bottom camera flips left/right) is preserved and documented.
- **§5 — lick vs non-lick excursion geometry.** `tongue_kinematics` cell 60 **==**
  `cueresponse` cell 95; **ported once**, here, as `TODO.md` specifies. Pooled and
  jaw-centered, as 2-D density rather than the source's alpha scatter (246k points
  saturate where the source's ~7k did not). With lick 65.3 ± 10.9 px from the jaw vs
  40.8 ± 15.8 px without; per-session paired medians 66.2 vs 39.3 px, Wilcoxon p = 1.1e-13
  (n = 44 sessions).
- **§6 — non-lick endpoints by movement ordinal** (cells 54, 56). The source referenced two
  names (`nonlick`, `licks`) it never defined — leftovers from another notebook;
  reconstructed here as the obvious `has_lick` split.
- **§7 — does pre-lick direction predict choice?** (cells 49, 55). 7,138 trials, 44
  sessions; class balance **57.9% left / 42.1% right** (majority-class accuracy 0.579).
  Last pre-lick excursion angle: median -60.0° before a left lick vs +38.5° before a right
  lick (session-paired Wilcoxon p = 1e-11, n = 44). Binned P(right lick) rises monotonically
  from 0.08 to 0.94 across `endpoint_y_rel`, crossing the base rate within one bin of the
  jaw midline, and 0.09 → 0.88 across `excursion_angle_deg`.
- **§8 — ridge-logistic decode** (cells 44, 47, 48). **The source's evaluation could not be
  reused.** It calls `train_test_split(random_state=42)` on one session's trials; applied
  to pooled data that leaks session identity across the split. Replaced with two
  evaluations, reported side by side because they answer different questions:
  - **`GroupKFold`, session as the group (primary)** — held-out-session AUC
    **0.835 ± 0.100** (SD over 5 folds), pooled out-of-fold AUC 0.836, accuracy 0.808.
    Asks whether *one* geometry→choice mapping generalizes across animals and rigs.
  - **Per-session fits (secondary)** — median AUC **0.943** over the 39 sessions with
    n ≥ 40 and ≥ 10 per class; 97% above 0.5, Wilcoxon p = 7.3e-12. Higher than the shared
    decoder, which is what session-specific camera geometry predicts — the reason both are
    reported.
  - **Shuffled-label null**, labels permuted *within* session over 200 permutations: mean
    0.530, 95th pct 0.545, observed 0.835, p = 0.005 (the permutation resolution floor).
    The null sits above 0.5 because within-session shuffling preserves session base rates,
    which a pooled evaluation can exploit; that offset, not 0.5, is the right reference.
  - **Permutation importance** (on held-out sessions, not held-out trials): `last_angle`
    0.168, `mean_distance` 0.098, `mean_angle` 0.070, `mean_duration` 0.035 ΔAUC; both
    peak-velocity terms and `session_time` ≈ 0. Direction carries the decode, not vigor.
- **§8.4-§8.5 — the block-structure caveat, quantified rather than asserted.** Left/right
  choice in this task is block-driven, so a lick-side decoder may be reading recent choice.
  Measured: **P(stay) = 0.910** and **previous choice alone reaches AUC 0.907** — *better*
  than kinematics. But kinematics still reads AUC 0.815 / 0.772 *within* each
  previous-choice stratum, so it is not simply re-encoding choice history. Not resolved
  here; separating them needs block-aware regressors, which is `kin_07`'s dependency.
- **§9 — pre-lick → cue-response displacement** (cells 50-52). Pre-lick and cue-response
  endpoints land on the same side of the jaw midline in **84.0% ± 2.2%** of trials (SEM,
  session as the sampling unit), with the cue-response lick carrying the tongue 15-23 px
  *further* from the midline (Wilcoxon p < 1e-300 both sides). Per-session
  r(pre-lick y, cue-response y) median 0.776.

Other port decisions:

- **Every grouping is on `(session, trial)`**, not `trial` — trial numbers repeat across
  sessions in the pooled parquet. Same class of bug as the one fixed in `kin_05` §7.
- **Session is the sampling unit for every population claim** — §5's paired excursion
  medians, §7's violin panel and binned curves (binned *within* session on pooled quantile
  edges, then mean ± SEM *across* sessions), §8's grouping, §9's same-side fraction.
- Dropped from the source throughout: hardcoded hex colors, seaborn violin/`palette`
  defaults, `plt.grid(True)`, and the absolute `/root/capsule/scratch/figures` save paths.
  All 8 figures end with `save_fig(..., fig_dir=FIG_DIR, save=SAVE_FIG)`.
- Label joins use merges rather than `pd.MultiIndex.from_frame(...).map(...)`, and §7's
  binning uses integer bin labels rather than Interval categoricals — both to keep pandas
  off its object-hashing path, which emitted a `RuntimeWarning` on the NaN-bearing `trial`
  column. Notebook now executes warning-free.
- `fmt_p()` reports SciPy p-value underflow as `< 1e-300` rather than printing `0`.

Deliberately **not** done: no `git mv` of `tongue_kinematics_cueresponse.ipynb`. Unlike
`eph_07` / `eph_09`, `kin_06` does not clear its source on its own — that notebook is gated
on `kin_06` **and** `kin_07` (the value-encoding port), so it stays in `code/`.

## 2026-09-15 (4)

### New `code/spatial_axes.py` + `eph_09_structural_axes.ipynb`; `eph_08` refactored onto the module

Phase 2 of the repo reorg: the fourth of the six HOLD-notebook ports. Source is
`spatial_axis_comparison_rt_encoding_update.ipynb` cells 13-38, which is now fully
replicated and (pending a Code Ocean run) becomes archivable.

- **New `code/spatial_axes.py` (~700 lines).** Fitting and comparing 3-D spatial *axes* —
  the direction along which a feature varies fastest. `fit_spatial_axis_linear` / `_cca` /
  `_LDA` (scalar / multivariate / categorical features) with bootstrap wrappers,
  `compare_bootstrap_directions` (tangent-plane Wald test), `cone_half_angle`,
  `vectors_to_az_el`, `plot_projected_arrow_with_cone`, `get_regression_CI` and
  `plot_projection_scatter`. Ported from source cells 14, 15, 31 with function bodies
  unchanged. Kept separate from `spatial_encoding.py` deliberately: that module asks
  *where* a statistic is large (CCF maps, permutation tests) and has no axis-fitting
  machinery; this one asks *in which direction* it changes.
- **`eph_08_waveform_axis.ipynb` refactored onto it** — its two private inline copies of
  `fit_spatial_axis_cca` / `bootstrap_spatial_axis_cca` (51 lines) deleted in favor of
  `from spatial_axes import bootstrap_spatial_axis_cca`. Structured as a pure move: no
  other change, because its projection figure is the poster figure
  `rt_response_projection_abs`. **Not verified against real data** — `eph_08` is Code
  Ocean only and skips locally. What *was* verified: the module and the deleted inline
  copies produce bit-identical axes, bootstrap clouds, cone half-angles and downstream
  projections on synthetic waveform-shaped input (same seed, same RNG draw sequence).
- **New `code/eph_09_structural_axes.ipynb` (34 cells).** Asks whether the RT-encoding
  spatial gradient is the *same* gradient as LC's structural ones.
  - Fits the RT-encoding axis (`T_rt`) and its baseline control (`T_rt_bl`) —
    **the step `eph_08` skipped**; `eph_08` projects onto the waveform axis but never
    fits the RT axis, so there was previously nothing to compare against.
  - Three structural axes, each behind its own `HAS_WAVEFORM` / `HAS_MERFISH` /
    `HAS_RETRO` guard so a missing asset drops that axis from every downstream
    comparison rather than failing or being substituted.
  - Pairwise direction comparison (angle, Wald W, chi2 p, bootstrap p) + summary table,
    three-plane arrow-and-cone figure, azimuth-elevation bootstrap scatter, and
    projection scatters of `T_rt` / `T_rt_bl` onto each structural axis.
  - Carries over the source's interpretation guide, extended with the two failure modes
    the summary table hides (small angle + significant p; large angle + wide cone).
  - Source cell 40 (a commented-out registry sketch) dropped. Instead the RT statistics
    are computed through the real machinery — `AnalysisSpec` / `fit_encoding` /
    `PerUnitStatsRegistry`, as `eph_01` does — so `T_rt` is the same quantity
    `eph_01`-`eph_04` report and no third inline copy of `build_rt_encoding_stats` was
    created. Session/unit QC likewise goes through `data_loading` rather than being
    reimplemented inline as `eph_08` does.
  - **One deliberate departure from `eph_01`:** no RT trial window (`trial_query=""`,
    `min_trials=50`), matching the source notebook and `eph_08` so the fitted axes
    describe the same units the poster figure projects. `RT_QUERY` is left in place for
    a sensitivity check.
- **Retrograde LDA sign convention preserved** — an LDA axis sign is arbitrary, so it is
  pinned to the waveform axis as the source does, and the notebook says so explicitly
  (and warns when no waveform axis is available to pin against).
- **`environment/Dockerfile`: added `scanpy==1.10.3`** in its own layer, following the
  `rachel-analysis-utils` precedent. The MERFISH axis cannot run without it, and it was
  absent from the pip block. 1.10.3 is the last scanpy supporting python 3.9 (1.10.4+
  require >=3.10); `scipy==1.13.0` is re-asserted alongside it as a tripwire so a future
  scanpy bump fails the build loudly instead of silently moving numpy/scipy under the
  other pins. **This triggers a Code Ocean image rebuild.**

**Verification status.** `spatial_axes.py` is pure numpy/sklearn math and was tested
locally against synthetic data (30 checks, all passing): `fit_spatial_axis_linear`
recovers a planted gradient to 0.00 deg noise-free and 3.4 deg at heavy noise;
`_cca` and `_LDA` recover planted axes to ~1-4 deg; `compare_bootstrap_directions`
gives 0.00 deg / p=1.0 for identical axes, 0.25 deg / p=0.98 for two samples of the same
axis, and 89.7 deg / p<0.002 for orthogonal ones; `cone_half_angle` widens monotonically
with noise (0.46 -> 2.49 -> 6.43 -> 20.26 deg) and as n falls (5.3 -> 13.2 deg).
`eph_08` and `eph_09` both execute clean locally through their skip paths, but **neither
has been run on Code Ocean**, so no real-data output — including `eph_08`'s poster-figure
equivalence — has been confirmed. See `TODO.md`.

## 2026-09-15 (3)

### Extract shared `fip_*` setup into `code/fip_utils.py` (Pass 1 of 2)

- **New `code/fip_utils.py`.** `fip_00_explore.ipynb`, `fip_01_movement_value_coding.ipynb` and
  `fip_02_ne_only_events.ipynb` each carried their own copy of the same load -> curate ->
  session-setup preamble (283/248/241 non-blank lines, ~490 of them pure duplication). Every bug
  fix this cycle had to be applied 2-3 times by hand; the `regex=False` substring fix alone
  touched 7 call sites. The module follows the existing `plotstyle.py`/`ephys_utils.py`
  convention (flat module in `code/`, bare import, no `sys.path` juggling) and carries the
  loading/curation, session select, `parse_event`/`get_trace`/`build_meta`/`pick_example`,
  trial enrichment, motion-energy-on-the-FIP-clock, signal helpers, and the multi-session
  `process_session` pipeline. Notebook code drops 1657 -> 893 lines (-764).
- **Memory fix is now structural.** `load_curated_sessions` holds the pre-curation `nwb_list_raw`
  and the unused second curated list as function locals, so both are released on return. The
  `del nwb_list_raw, _nwb_list_curated_unused; gc.collect()` incantation is no longer something
  each notebook has to remember.
- **Two switches onto upstream functions** (everything else is a move, not a change):
  - `fu.enrich_trials` calls `rachel_analysis_utils.analysis_utils.enrich_df_trials`, with a
    local two-column reimplementation only as a fallback. This replaces *two* separate
    reimplementations — `fip_00`'s `enrich_streaks` (which never tried upstream) and `fip_01`'s
    `_enrich_streaks_and_rpe_bins_fallback`. The reason `fip_00` avoided upstream ("won't parse
    on Python 3.9, nested-quote f-string at `analysis_utils.py:294`") is stale: that file is 177
    lines and `python3.9 -m py_compile` clean. The fallback keeps `fip_00`'s `ses_idx` grouping,
    which `fip_01`'s bare `.shift(1)` lacked and which multi-session needs so streaks don't run
    across session boundaries.
  - Onset detection consumes the upstream `data_z` column
    (`fu.threshold_onsets(..., already_z=True)`) instead of z-scoring internally, so one
    normalization convention runs throughout. `fu.zscore` switches to `ddof=1` to match
    `enrich_dfs.zscore_fip` exactly — verified equal to
    `scipy.stats.zscore(x, ddof=1, nan_policy="omit")` to 4e-16, and the `ddof` change moves zero
    samples across threshold at n=100k, so onset times are unaffected.
- **`attach_me_to_df_fip` convention reconciled to raw ME.** `fip_00`'s multi-session copy stored
  `me_z` while `fip_01`/`fip_02` stored raw. Raw is correct: `fip_psth_inner_compute` takes a
  `data_column` argument (default `"data"`) and never z-scores internally, so raw-in-`data` plus
  `data_column="data_z"` is the composable form, and the `enrich_dfs` functions z-score `data`
  themselves. Only affects `fip_00`'s `BUILD_NWB_LIST_ME=False` cell, which prints guidance.
- **`fip_00`'s internal duplication resolved** — `build_meta`, `locate_me_assets` and
  `attach_me_to_df_fip` were each defined twice, once in the single-session cells and again in
  the multi-session helpers cell. The multi-session cell keeps only the four functions Pass 2
  will replace (`session_etr_mean`, `aggregate_series`, `collect_region_etr`, `plot_by_subject`)
  and imports the rest by name, leaving the four analysis cells and results table untouched.
- **`fip_01` sheds a dead helper cell** — `zscore`/`threshold_onsets`/`peri_event` became unused
  there when Figure 3 moved to `fip_02`.
- Each notebook's imports cell gains `%load_ext autoreload` / `%autoreload 2`: a session load is
  tens of GB and minutes, so without it every `fip_utils.py` edit would cost a kernel restart
  and a full reload.
- **Upstream curation API has been rewritten — Dockerfile pin needed.** At
  `rachel-analysis-utils` main (`b7b7487`, "CSV-based data curation inputs"),
  `apply_curation_nwb_list` no longer exists; `data_curation_helpers.py` is now
  `load_curation` + `apply_curation_df_fip`, reading CSVs via
  `aind_bwnm_fiber_data_curation_utils` — imported at module top level but declared in no
  packaging metadata (`dependencies = []` at main as well as at the pin), so pip installs
  nothing for it and the import raises `ModuleNotFoundError` — while overwriting
  `df_fip['event']` with target names and dropping `intended_measurement` entirely. `environment/Dockerfile:60` installs `@main`
  unpinned; the cached image still has the old API, so the next rebuild would break the curation
  cell and then `parse_event`/`build_meta`/`pick_example`. Recommended pin
  (`864550d55356ecb4906d05801cdae694bd000fde`) is written up in `code/fip_todo.md` — not applied
  here, since CLAUDE.md puts `/environment` off limits.
- Verification is static only (CO-only data assets): `python3.9 -m py_compile` on the module,
  `nbformat` validate + `nbconvert --to script` + `compile()` + `pyflakes` (no undefined names,
  no shadowed module functions) on all three notebooks. Pass 1 has **not** been run on Code
  Ocean yet.

## 2026-09-15 (2)

### Create `eph_07_bout_encoding.ipynb` — bout helpers consolidated into `ephys_utils.py`
- **Bout helpers ported into `ephys_utils.py`.** `annotate_movement_bouts` (previously the sole
  copy, living only in `tongue_kinematics_ephys_intertrialmovs.ipynb` cell 10) is relocated
  unchanged. The within-trial/ITI classifier — copy-pasted across four near-identical
  `get_session_bout_times` cells in the source, with drifting thresholds (`2.0/2.0/1.0` at cell
  12 vs `1.0/2.0/0.5` at cell 29) — is consolidated into two functions: `classify_bout_times`
  (the shared dt-to-nearest-go-cue logic, generalized to take arbitrary bout onset times) and
  `get_session_bout_times` (wraps it with `annotate_movement_bouts` for the movement-bout
  definition), both with the thresholds as explicit parameters rather than hardcoded. Verified
  locally against `data/for_local/all_tongue_movements_04022026.parquet` (44 sessions, 246,359
  movements → 31,081 bouts; median bout size 5, row-weighted mean 17.9; pinned thresholds
  `GAP_THRESHOLD_S=0.5, GO_RESPONSE_WINDOW_S=2.0, ITI_MIN_POST_CUE_S=2.0,
  ITI_MIN_PRE_NEXT_S=1.0` give 14,368 go-responsive and 9,297 ITI bouts pooled across sessions).
- **`eph_07_bout_encoding.ipynb`** ports the movement-bout-derived within-trial vs ITI ephys
  comparison from `tongue_kinematics_ephys_intertrialmovs.ipynb`, following `TODO.md`'s port
  plan: movement-derived bouts as primary (doesn't inherit the lickometer's blind spot for
  non-lick movements, per `kin_05`'s finding), lick-bout-derived (`licks["bout_start"]`) as a
  lighter-weight robustness check in §9. Consolidated the four near-identical population-PETH
  cells (source 14/15/19/22) into one pass, keeping the session-wide z-scoring normalization
  (source cell 15). Per-unit go-responsive-vs-ITI Δ firing rate comparison (source cells 29–31)
  registered through `encoding_methods.fit_encoding` + `per_unit_stats_registry` (binary `is_iti`
  predictor, OLS) so it composes with `eph_01`–`eph_04` rather than standing alone as a one-off
  Wilcoxon test. Dropped source cell 37 (video-clip extraction — a one-off, and the clip helpers
  are library-owned).
- Reused `eph_00`'s `make_rp_and_events`/`compute_psth`/`smooth_vector`/`plot_psth` raster/PSTH
  path throughout (single-unit example in §5 and the population loop in §6), rather than the
  source's hand-rolled `peth_for_unit` — the population loop still builds a `RasterPlotter` per
  unit × condition, at a coarser 20 ms bin size to keep the loop over hundreds of units cheap
  (matching the source's own reason for that bin-size choice, not a reason to avoid the shared
  path).
- **Only §4 (the bout-helper verification) executed this session** — everything from §5 onward
  needs per-session spike times and intermediate data that exist only on Code Ocean. Written and
  reasoned through, syntax-verified (the notebook was actually executed end-to-end locally; every
  Code-Ocean-only cell took its `ENV == "local"` skip branch without error, which at minimum
  confirms every such cell parses), but not run — none of it should be treated as a finding until
  it runs there.
- **Flag, not acted on:** `CHANGELOG.md` already had a `## 2026-09-15` section (now below,
  unnumbered) placed *after* the `## 2026-09-14 (9)`…`(2)` block instead of above it — out of the
  file's stated newest-first order. Left as found; worth a cleanup pass if noticed independently.
- Not archived this session: `eph_07` is `tongue_kinematics_ephys_intertrialmovs.ipynb`'s sole
  remaining port gate (see `TODO.md`/`REORG.md`), but archiving is a separate step after review.

## 2026-09-14 (9)

### `fip_02_ne_only_events`: add a z-threshold sensitivity sweep
- New closing section: reruns onset detection and all four quantitative NE-DA
  analyses (shuffle control, lag distribution, peak-amplitude correlation, ME
  alignment) across `Z_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]`, one four-panel summary
  figure per threshold, to check how much of the main analysis depends on the
  choice of detection threshold. Ends with a small per-threshold summary table
  (onset counts, coincidence rate + p-value, median lag, Pearson r).
- Refactored as `analyze_at_zthresh(z_thresh)` returning a dict, rather than five
  copies of the module-level analysis cells — reuses the existing `threshold_onsets`,
  `peak_in_window`, `peri_event`, and `coincidence_rate` helpers unchanged; only
  `z_thresh` varies across the sweep, `REFRACTORY`/`MIN_RUN`/`WINDOW` stay fixed at
  the notebook's main values. Shuffle count reduced to 500 (from 1000) per sweep
  point since the whole pipeline reruns five times.
- Not executed — same Code-Ocean-only data assets as the rest of the `fip_*` series.
  Validated via `nbformat.validate` read-back and `ast.parse`/`py_compile` on the
  converted script.

## 2026-09-14 (8)

### `fip_02_ne_only_events`: add raw-trace and cross-correlation views ahead of onset analysis
- Two new sections inserted right after the z-scored NE/DA/ME traces are built, **before**
  onset detection — a qualitative continuous-signal look precedes the discrete-event analysis
  that follows it:
  - **NE/DA raw traces over a window** (`TRACE_WINDOW_DUR`, default 10s, plus a
    `plot_ne_da_window(t_start, duration)` helper to scan other parts of the session) — overlay
    the two z-scored traces to get a visual sense of how correlated they look before
    quantifying anything.
  - **NE↔DA cross-correlation**: resample both onto a common 20 Hz grid over their time overlap
    (same recipe `fip_00_explore.ipynb` uses for motion energy × FIP) and run `norm_xcorr`
    (added to the Helpers cell, same function `fip_00` defines). Sign convention deliberately
    set to match the onset-lag distribution further down the notebook (positive = NE leads DA).
- No changes to the onset-detection/classification/quantitative sections below — this is a
  qualitative + continuous-signal complement to the existing analyses, placed to build the
  notebook's narrative forward (raw look → continuous relationship → discrete events →
  event classification → event-level quantification → behavioral comparison).
- Not executed — needs the same Code Ocean-only data assets the rest of the `fip_*` series does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on the converted script.

## 2026-09-14 (7)

### Remove Figure 3 from `fip_01`; new `fip_02_ne_only_events` — NE/DA transient dissociation
- **`fip_01_movement_value_coding.ipynb`**: removed Figure 3 ("NE"-only onsets aligned to
  movement) — it treated every PL-GCaMP ("NE") onset alike, which is now a distinct notebook's
  question. Also removed the now-unused `NE_EVENT` selection and trimmed the title/example-
  signals markdown accordingly. `fip_01` is left with its original two figures (movement per
  RPE bin; z-scored AUC per consecutive R-/R+), both still looped over all example FIP channels.
- **New `code/fip_02_ne_only_events.ipynb`**: which NE (PL-GCaMP) transients happen *without* a
  concurrent DA (NAc dLight) transient ("NE-only") versus with one ("NE+DA"), and does movement
  (motion energy) look different around the two kinds of NE event. Classification: independent
  `threshold_onsets()` on NE and DA (same parameters used throughout this notebook family), an
  NE onset counts as NE+DA if a DA onset falls within ±0.5 s of it, otherwise NE-only — reuses
  the existing onset-detection machinery symmetrically rather than a separate DA-magnitude rule.
  Four analyses quantify the relationship beyond that binary split: DA z-score peak distribution
  around every NE onset (with `z_thresh` marked, to show whether the split is a real dichotomy
  or a continuum); a circular-shift shuffle control for the coincidence rate (empirical p-value
  against each signal's own baseline event rate); lag distribution for coincident pairs (does DA
  lead/lag NE); and NE-vs-DA peak-amplitude correlation (Pearson + Spearman) for coincident
  pairs. A QC figure (example NE-only vs. NE+DA traces) sits before those. Main figure: ME
  aligned to each onset group, mean ± SEM overlaid.
  - Only needs continuous-time `data_z` (via `aind_dynamic_foraging_data_utils.enrich_dfs.
    zscore_fip`, same as `fip_01`) — deliberately skips `fip_01`'s per-trial
    `enrich_fip_in_df_trials`/`remove_tonic_df_fip` pipeline, since this analysis is onset-based,
    not per-trial. Setup cells (data load, curation + memory fix, session select, `pick_example`,
    motion-energy alignment) copied from `fip_01`, same as `fip_01` copied from `fip_00` — now a
    3rd notebook duplicating this setup; `fip_todo.md` updated to flag the shared-module
    extraction as overdue.
- Not executed — needs the same Code Ocean-only data assets the rest of the `fip_*` series does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on both notebooks; confirmed
  no leftover `NE_EVENT`/Figure-3 references remain in `fip_01`.

## 2026-09-14 (6)

### Create `fip_01_movement_value_coding.ipynb` — movement × RPE/value coding
- New notebook, built on the example session `fip_00_explore.ipynb` loads (`SESSION_IDX=0`,
  subject 808054). Extends the analysis used for the FIP-only phasic/tonic-DA poster figure
  (baseline AUC by consecutive-reward streak, outcome traces by RPE bin) to motion energy (ME)
  and to the "NE" channel (this dataset's curation has no nLight sensor; PL-GCaMP stands in for
  NE per project convention). Three figures: (1) session-averaged movement per RPE bin, via
  `aind_dynamic_foraging_basic_analysis.plot.plot_fip.plot_fip_psth_compare_alignments` on an
  ME pseudo-channel, aligned to choice and split by `RPE-binned3` — same function/window/color
  convention as the reference pipeline's `PAC_2026.ipynb` recipe; (2) z-scored AUC per
  consecutive R-/R+, via `aind_dynamic_foraging_data_utils.enrich_dfs.enrich_fip_in_df_trials`
  + `remove_tonic_df_fip` (Rachel's actual baseline/tonic-normalization functions — not a
  reimplementation), paired with `num_reward_past` shifted by one trial (the exact pairing
  convention traced from `power_analysis.ipynb`/`foraging_summary_plots.py::plot_baseline_corr`,
  easy to get backwards); (3) "NE"-only onsets aligned to movement — no analog in Rachel's
  pipeline (she never analyzed motion energy), reuses `fip_00_explore.ipynb`'s own
  `threshold_onsets`/`peri_event` machinery restricted to the PL-GCaMP channel.
- Traced the recipe through `rachel-analysis-utils`, `aind-dynamic-foraging-basic-analysis`,
  `aind-dynamic-foraging-data-utils`, and (for plotting conventions only, not a dependency)
  `AllenNeuralDynamics/DA_phasic_tonic`. Full trace, function-by-function, and a dedicated
  review of deviations from Rachel's actual pipeline (what was fixed vs. inherent since ME was
  never part of her analysis) are in the implementation plan for this notebook.
- Incidental finding while tracing this: `rachel_analysis_utils.analysis_utils`'s
  Python-3.9-incompatible nested-quote f-string (cited in `TODO.md`'s Python-3.11-upgrade
  section) appears fixed upstream — a fresh clone of `main` compiles cleanly under
  `python3.9 -m py_compile`. Not yet verified on Code Ocean; logged in the new `fip_todo.md`
  (see below) rather than acted on.
- **New file `fip_todo.md`**: a short to-do list scoped to the `fip_*` series, separate from
  `TODO.md`/`REORG.md`'s `kin_*`/`eph_*` port-plan tracking. Seeded with the `fip_utils.py`
  extraction (this notebook currently duplicates `fip_00_explore.ipynb`'s setup cells rather
  than sharing a module), the `analysis_utils` Python-3.9 re-verification above, and the
  "anticipatory movement during CS+Delay" figure that was requested alongside these three but
  deferred (no literal CS+/delay epoch exists in this dynamic-foraging task).
- Not executed — needs the same Code Ocean-only data assets `fip_00_explore.ipynb` does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on the converted script only.

## 2026-09-14 (5)

### Create `kin_05_nonlick_movements.ipynb` (Phase 2 port, item 2/6)
- New notebook: do the lickometer and video-derived movement streams describe the same
  events, and what are the movements that aren't licks? Framed as a definitional
  question, not a QC filter — lickometer agreement, confidence, and duration are not
  valid noise criteria (that's `kin_00`'s job). §3 three-way correspondence tally
  (licks w/o movements, movements w/ >1 lick, movements w/o licks), §4 licks without
  movements, §5 movements with multiple licks, §6 kinematic profile of non-lick
  movements (extends `kin_01` §5 to `duration`/`total_distance`, not a duplicate of its
  `out_peak_velocity`/`out_duration`), §7 per-trial lick vs. non-lick structure, §8
  preparatory timing (prevalence of non-lick movements before the cue-response lick +
  donut of trials by pre-lick movement count). §9 (illustrative single-trial/raster
  figures) already landed in `kin_02` §11 last session — not re-ported.
- **Correction to the port plan: §3 needed a partial Code-Ocean-only split, not just
  §4.** `TODO.md` labeled §3 "pooled and locally testable" with only §4 flagged
  Code-Ocean-only. Checked directly against the source (`tongue_kinematics` cell 35):
  part (a), "licks without movement," computes
  `nwb.df_licks['nearest_movement_id'].isna().sum()` — a per-session column, confirmed
  absent from all 49 columns of `all_tongue_movements_04022026.parquet`. Parts (b) and
  (c) use `tongue_movements['lick_count']`/`['has_lick']`, both pooled. Split §3
  accordingly: (b)/(c) compute and print locally (verified: 8,585/151,404 = 5.67%
  movements with >1 lick; 94,955/246,359 = 38.54% movements without licks); (a) moved
  under the existing §4 Code-Ocean guard (written, unexecuted, with the expected
  `nwb_df_licks.parquet` schema documented).
- **Second plan gap, same category:** `TODO.md`'s §5 spec ("cells 43, 44 ... pooled")
  is only true of cell 43 (the `lick_count > 1` tally, already covered by §3b — not
  re-run). Cell 44, the multi-lick example-trace figure, needs per-frame `tongue_segmented`
  and per-session `nwb.df_licks` — Code-Ocean-only, same as §4. Given the tally is
  already the pooled result and the per-session example is a qualitative aside on an
  already-rare event (~0.03–32% by session, median ~2.9%), didn't port it as a second
  CO-only guarded block; ported a pooled `lick_count` distribution + per-session rate
  instead, which is the actual population-level finding cell 44 can't provide from a
  single session.
- **Bug fix in the port, not present in the source:** the source's per-trial structure
  (cells 64-68) grouped by `trial` alone, correct only because that notebook runs on one
  session. `trial` numbers repeat across sessions in the pooled parquet, so §7 groups by
  `(session, trial)` instead — silent cross-session pooling would have inflated the
  per-trial movement counts.
- §8 uses the pooled `movement_before_cue_response` column directly rather than
  reconstructing it from `nearest_movement_id` (what the source did) — the reconstruction
  needs the per-session lick table and is unnecessary since the flag is already pooled.
- Verified end-to-end locally against `data/for_local/all_tongue_movements_04022026.parquet`
  (246,359 movements, 44 sessions) via `jupyter nbconvert --execute`; 0 errors, 6 figures
  rendered. Only §4 (licks without movements) is unexecuted, Code-Ocean-only.
- `tongue_kinematics.ipynb` archiving now gates on `kin_07` alone (`kin_05` gate met).

## 2026-09-14 (4)

### `fip_00_explore`: drop the `video_alignment`-branch workaround, use `read_video_csv`
- **Cleanup, no behavior change on the current dataset.** `aind-dynamic-foraging-behavior-
  video-analysis`'s `video_alignment` module (previously only on a `video_alignment` branch)
  is now merged to `main`, and the Dockerfile already installs `@main` (`environment/
  Dockerfile:48`). Section 8b's import cell no longer needs to git-fetch/checkout the
  `video_alignment` branch inside the editable install at `/src` — replaced with a plain
  `import ... as va`, falling back to a pip install from `@main` only for a from-scratch env.
- **Use the package's own CSV reader.** `video_alignment` now ships `read_video_csv` (auto-
  detects the Old/flat headerless camera CSV vs. the New/AIND headered layout, which names the
  behavior-time column `ReferenceTime` instead of `Behav_Time`) plus the `DEFAULT_COLUMNS` /
  `TIME_COLUMN_ALIASES` constants. `motion_energy_to_session` now calls `va.read_video_csv`
  and resolves the behavior-time column from `va.TIME_COLUMN_ALIASES` instead of hand-rolling
  `pd.read_csv(..., header=None, names=CAM_COLUMNS)` with a hardcoded `cam["Behav_Time"]`
  lookup — so it no longer silently mis-reads a New/AIND-layout CSV as headerless data.
- Verified `video_alignment.py`'s public function signatures (`compute_video_session_offset`,
  `get_first_frame_behavior_time`, `behavior_time_to_video_time`, `video_time_to_session_time`)
  are unchanged from what the notebook already called. Not executed locally (needs Code Ocean
  data assets); validated via `nbformat` read-back and `ast.parse` on the converted script.

## 2026-09-14 (3)

### `kin_02_latency` §11: fix raster rendering artifact and illegible legends
- **Rendering bug.** Both §11 raster figures (movement type, movement ordinal) showed
  spurious horizontal gaps — bands of trials with no visible ticks. Root cause: at the
  notebook's default `figure.dpi` (110, from `plotstyle.apply_style()`), a 6×6 in raster
  with ~570 trial rows renders at ~649×649 px, under 1.2 px/trial; matplotlib's
  anti-aliasing drops some trial rows unevenly at that density. Confirmed directly —
  extracted the actual embedded PNG bytes from the executed notebook, reproduced the
  identical banding at 649×649 px, and confirmed it disappears entirely by ~1180×1180 px
  (dpi 200). Fixed by passing `dpi=220` explicitly to `plt.subplots()` in both raster
  cells (scoped to just those two cells, not a `plotstyle.py`-wide change — the other
  figures in this notebook aren't dense enough to need it).
- **Illegible legend.** Both raster legends sat directly over dense scatter data with no
  background (`plotstyle`'s global `legend.frameon=False` convention), making them
  unreadable. Added `frameon=True, facecolor="white", edgecolor="none", framealpha=0.9`
  to both legend calls — a scoped exception to the frameless convention for these two
  data-dense figures specifically.
- Re-executed end-to-end; both figures confirmed banding-free with readable legends.

## 2026-09-14 (2)

### `kin_02_latency`: fix §3–§7 ordinal filter; add §11 single-session illustration
- **Correctness fix.** §3's filter never restricted to `movement_number_in_trial ==
  cue_response_movement_number` — since `cue_response_movement_number` is a trial-level
  constant, grouping by k pooled in every movement from a k-labeled trial, not just the k-th
  one. Only ~11% of the rows plotted in §4 (and consumed by §5–§7) were actually the
  cue-response movement (verified: 6,958 / 57,925 at k=1). Split §3 into `movements_valid`
  (all movements in a trial with a valid k — what §8's Δt estimate needs) and `df` (only the
  cue-response movement itself, one row per trial — what §4–§7 use); re-pointed §8's `lat_df`
  at `movements_valid`. Re-executed end-to-end: §4–§7's numbers changed substantively (e.g.
  k=1 log-normality n: 57,925 → 6,958; medians now cleanly spaced ~0.13–0.66 s across k=1–4),
  §8–§10 were unaffected since they already carried their own independent restriction.
- **New: §11, single-session illustration.** Ported the example-trial and raster/histogram
  figures from `tongue_latency.ipynb` cells 5, 8, 9, 14 (one example session,
  `behavior_716325_2024-05-31_10-31-14` — the session the source notebook itself used):
  a trial raster colored by movement type (cue-response lick / other lick / non-lick), a
  raster colored by movement ordinal (viridis, cue-response movement outlined), and a
  colored-histogram panel (lick vs. 1st/2nd-move latency, and lick latency by k reusing §8's
  `k_colors` so the same k means the same color throughout the notebook). The single-trial
  tongue y-position trace (source cell 5) needs per-frame `tongue_kins.parquet`, not in the
  pooled parquet — written with the `ENV`/`kin_00`-style guard but Code Ocean only, unexecuted.
  Checked `kin_00`/`kin_01`/`kin_03` first for duplicates: none (`kin_00`'s rasters are
  unrelated QC/spatial-radius figures; `kin_01`/`kin_03`'s "raster" hits were the
  `rasterized=True` matplotlib flag, not raster plots).
  - This content was previously slated for `kin_05_nonlick_movements` (not yet built);
    reassigned here since it fit naturally as `kin_02`'s closing illustration and the user
    asked for it directly. Added a `coerce_bool` helper (`tongue_latency` cells 6/8) since
    `cue_response` is object-dtype with `True`/`False`/`None` in the pooled parquet.
  - **Consequence:** `tongue_latency.ipynb`'s entire audited unreplicated-content list (RT +
    IMI decomposition, single-trial example, both rasters) is now ported. Its archive gate no
    longer includes `kin_05` — it has no remaining port gate. Not archived this session (out
    of scope); `TODO.md`/`REORG.md` updated to reflect this so a future session (or the user)
    can `git mv` it directly.
- Updated `TODO.md` (item body, overview table, archive-gate table) and `REORG.md` (HOLD table,
  `kin_02` KEEP note, planned-additions table) to match.

## 2026-09-14

### `kin_02_latency`: RT + IMI decomposition (Phase 2 port, item 1 of 6)
- Appended §8–§10 to `kin_02_latency.ipynb`, porting the "nice story" left behind in
  `tongue_latency.ipynb` (cells 15, 17, 18, 26): reaction time modeled as a first-movement
  latency plus a sequence of inter-movement intervals, RT ≈ RT₁ + (k−1)·Δt.
  - §8 estimates Δt as the median within-trial inter-movement interval, then de-shifts
    `lick_latency` by `(k−1)·Δt` per cue-response movement ordinal k and shows the
    distributions partially collapse onto k=1 (hist + KDE overlay).
  - §9 runs KS tests of each de-shifted k against de-shifted k=1.
  - §10 checks whether the spread of `lick_latency` grows with k (bootstrap 95% CI on SD),
    then the cross-session population version (grand mean ± SEM by k, session as the
    sampling unit).
  - Carried over the source notebook's caveat as closing markdown: the residual mismatch at
    k=1/k=2 is attributed to further covert preparatory movements, stated as an open question
    that `kin_05_nonlick_movements` (not yet built) is meant to test.
- Made the `lick_latency` (conditioned on `cue_response_movement_number`) vs
  `movement_latency_from_go` (what §4–§7 already plot) distinction explicit in prose, per
  `TODO.md`'s note — these are different quantities on different event streams.
- Flagged and fixed a latent issue in the source: `tongue_latency` cell 17's "raw vs aligned SD"
  comparison compares a quantity to itself (shifting a group by its own constant offset cannot
  change that group's SD) — kept both columns in the results table for continuity but dropped
  the redundant duplicate line from the plot and added an explanatory note.
- Fully local-testable; executed end-to-end against
  `data/for_local/all_tongue_movements_04022026.parquet` (44 sessions) with no new dependencies.
  Δt ≈ 0.172 s; KS tests reject full collapse at k=2/3/4 (p ≪ 0.001), consistent with the
  carried-over caveat.
- Updated `TODO.md` (item marked done, archive-gate table) and `REORG.md` (moved the entry out
  of "Planned additions" into the `kin_02_latency` KEEP note, updated the HOLD table and
  execution-plan checklist). `tongue_latency.ipynb` archiving still waits on
  `kin_05_nonlick_movements`.

## 2026-09-15

### Reorg planning refinements
- Dropped the planned `bout_utils.py`. `annotate_movement_bouts` and the within-trial/ITI
  classifier now fold into the existing `ephys_utils.py` — which already holds behavior-derived
  features that serve ephys alignment (`build_trial_features` takes movs/licks/trials and touches
  no spikes), so bout segmentation is the same category. `eph_07` already imports from
  `ephys_utils`, and a module with one consumer isn't worth the file. `spatial_axes.py` remains
  the only new module planned. Rationale recorded in `TODO.md`'s `eph_07` item.
- Added a `TODO.md` item: **define the boundary between the library and this repo's `code/`
  modules.** Both sides carry "kinematics utils" and "ephys utils" with no stated rule, which
  causes: duplicate definitions inside the library (`tongue_lickometer_utils` and
  `tongue_kinematics_utils` share six identically-named functions); no home rule for plotting
  (the library ships ~10 plot functions that predate and ignore `plotstyle.py`, so every port
  restyles by hand); undocumented layering between `tongue_ephys.py` and `ephys_utils.py`; and
  an undeclared cross-boundary contract on `tongue_quality_stats.json`. Item proposes criteria
  to agree and record in `CLAUDE.md`, and should be batched with the outbound-metrics
  consolidation since both need a library PR + pin bump.

## 2026-09-11

### Repo reorganization — Phase 2 planning (HOLD notebook port plan)
- Audited all five HOLD notebooks cell-by-cell (`tongue_latency`, `tongue_kinematics`,
  `tongue_kinematics_cueresponse`, `tongue_kinematics_ephys_intertrialmovs`,
  `spatial_axis_comparison_rt_encoding_update`) against the current `kin_*`/`eph_*` series to
  determine what is genuinely unreplicated.
- Found the leftover content clusters into five scientific questions that cut *across* notebooks
  rather than mapping one-notebook-to-one-port; rewrote the plan around question-oriented target
  notebooks: `kin_02` §8–§10 (RT + IMI decomposition), `kin_05_nonlick_movements`,
  `kin_06_lick_geometry_choice`, `kin_07_value_encoding`, `eph_07_bout_encoding`,
  `eph_09_structural_axes`, plus modules `bout_utils.py` and `spatial_axes.py`.
- Two findings that changed the previous plan: (1) kinematics × behavioral-model latents
  (Q values, RPE) is a topic no `kin_*`/`eph_*` notebook covers, present in *two* HOLD notebooks,
  so it gates two archivals; (2) `tongue_kinematics_cueresponse` is a choice-prediction analysis,
  not only "spatial geometry". Also confirmed `eph_08` never fits the RT spatial axis itself, and
  that `spatial_encoding.py` has no axis-fitting machinery.
- Recorded known cross-notebook duplication (port-once cases), content already covered by
  `tongue_lickometer.ipynb` (do not port), orphan code needing a home
  (`annotate_movement_bouts` — verified absent from the library on all three branches;
  `plot_standard_lick_landmarks`), pooled-parquet column availability (drives local vs Code
  Ocean-only sections), a recommended work order, and per-notebook archiving gates.
- Rewrote `TODO.md` (overview section + six work items) and updated `REORG.md` (planned
  additions, audited HOLD table, execution plan). No code changed.

### Repo reorganization — Phase 1 (archive superseded notebooks)
- Reviewed every notebook/script in `code/` against the refactored `eph_*`/`kin_*`/`fip_*` series
  and shared modules; recorded the full classification in `REORG.md`.
- Created `code/archive/` (+ `code/archive/reference/`) and `git mv`'d 15 files there (renames,
  history preserved): superseded pipelines/notebooks (`rt_ols_registry_pipeline`,
  `registry_usage_example.py`, `umap`, `tongue_kinematics_ephys`, `tongue_kinematics_ephys_figures`,
  `spatial_axis_comparison_rt_encoding` old dup), library-promoted/dev scratch
  (`old_functions_tongue_kinematics`, `cue_response_lick_processing_example`,
  `tongue_segmentation_test`, `batch_clips`, `create_labeled_clip`, `example`,
  `extract_tongue_kinematics`, `event_timeline`), and a collaborator reference notebook
  (`F_ephys_behavior_action&outcome` → `archive/reference/`).
- Tagged `wild-prereorg` as a restore point before moving anything.
- Kept in place: active analysis + shared modules, data-generation/pipeline infra, model-quality
  evaluation notebooks, and 5 notebooks with not-yet-replicated content held for porting.
- `TODO.md`: logged the outstanding ports (RT+IMI latency story, ITI bout ephys, MERFISH/retrograde
  spatial axes, lick↔movement correspondence, cue-response geometry) and the outbound-metrics
  library consolidation.

## 2026-09-07

### Repo housekeeping
- Added `TODO.md` for deferred work. First entry: upgrading the Code Ocean environment off
  Python 3.9, which is the shared root cause of the `--ignore-requires-python` layer in the
  Dockerfile, the un-importable `rachel_analysis_utils.analysis_utils` (and the local
  `enrich_streaks` reimplementation that works around it), and the repo-wide 3.9 syntax rule.
- `.gitignore`: ignore `*.code-workspace`. VS Code multi-root workspace files point at
  machine-specific paths (e.g. `../.venv/src/...`) and are not portable to Code Ocean.

### Upstream: `video_alignment` merged to main
- `aind-dynamic-foraging-behavior-video-analysis` merged the `video_alignment` branch to `main`
  (PR #2). The module is now on `main`, which `environment/Dockerfile` already tracks.
- **Follow-up, not yet done:** rebuild the Code Ocean environment, then delete the runtime
  `git fetch`/`git checkout` bootstrap in `fip_00_explore.ipynb` section 8b — it exists only
  because `main` previously lacked `video_alignment.py`, and becomes a no-op after the rebuild.
  Worth adding `fastparquet` to the Dockerfile pip block at the same time, to retire the other
  runtime `pip install` bootstrap in the imports section.

## 2026-06-30

### fip_00_explore.ipynb — multi-session comparison
- Added a "Multi-session comparison" section that reruns the single-session pipeline over all
  curated sessions and pools results with session as the sampling unit (mean ± SEM), grouped
  by region × subject. The single-session cells are unchanged and still serve as a detailed view.
- New helpers: `process_session` (per-session enrich + ME load + onsets), `build_meta`,
  `locate_me_assets`, `attach_me_to_df_fip` (injects motion energy as a `df_fip` pseudo-channel
  `event="ME"` so the upstream `plot_fip` PSTH machinery can treat it like a FIP channel), plus
  cross-session aggregation helpers (`session_etr_mean`, `aggregate_series`, `iter_region_signals`,
  `collect_region_etr`, `plot_by_subject`, `window_mean`, `streak_go_cues`).
- Four cross-session analyses: (1) ETR of FIP from ME onsets, (2) ETR of ME from FIP transients,
  (3) within-trial (0–2 s) vs ITI (2–4 s) relative to go cue, (4) peri-go-cue responses binned by
  consecutive rewards/failures via `rachel_analysis_utils.analysis_utils.enrich_df_trials`
  (`num_reward_past`). Scalar summaries collected into an in-memory `df_results`.
- `peri_event` gained an optional `censor_times` passthrough (additive) so streak/ITI go-cue
  subsets are censored against the full go-cue set.
- Sessions missing ME/video assets are skipped and logged, not fatal.
- 3.9 fix: dropped the `rachel_analysis_utils.analysis_utils` import (that module has a
  nested-quote f-string at line 294 that doesn't parse on Python 3.9, the CO env). Reimplemented
  the only piece we use — `num_reward_past` — as a local 3.9-safe `enrich_streaks` helper
  (verified to match the package's definition exactly).

## 2026-06-29

### fip_00_explore.ipynb
- Added example-signal analyses (`134bdcc`): pick one NAc DA (dLight), PL (GCaMP), and
  NAc ACh (rAch) series via curated `intended_measurement`; full-session + 60 s traces,
  peri-go-cue averages, z-scored motion energy, FIP↔ME onset alignment, ME×FIP xcorr.
- Refactored into 4 phases (`e195cc0`): imports → data loading → data processing → data viz
  (42→37 cells). Removed old single-FIBER plots + FIBER/VARIANT scaffolding; consolidated
  helpers; excluded `pearsonR` series (signal-signal correlations, not photometry); curation
  set to `..._firstpass` (has `correct_mapping`; `secondpass` does not).
- Onset detection made causal: dropped the centered `uniform_filter1d` smoothing (acausal,
  biased onsets early). `threshold_onsets` now uses a sustained-crossing rule (`min_run`
  consecutive samples above threshold; onset time = true first crossing). `me_z` is raw
  z-scored ME (no pre-smoothing); the xcorr runs on raw z-scored traces with NaN-safe
  interpolation only.
- Motion energy: pad a leading 0 so ME is 1-to-1 with video frames. `aind-motion-energy`
  emits a consecutive-frame difference (N frames → N−1 values, no value for frame 0); the
  pad is gated on the ME metadata (`n_me_frames` vs `n_frames_decoded`) so it auto-disables
  once the library pads upstream. Length-mismatch warning now fires only on genuine anomalies.
- Simplified helpers: assert (don't sort) that `df_fip` timestamps are time-ordered per
  event after session-pick, so `get_trace` no longer re-sorts; merged
  `peri_event`/`peri_event_series` into one array-based
  `peri_event(t, y, event_times, censor=...)` (FIP traces pass `*get_trace(df_fip, ev)`).

### eph_00_single_unit_inspection.ipynb
- Import fix (`9f4b4ec`): `load_intermediate_data` / `find_session_dir` now come from
  `aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys`, not `data_loading`
  (where they don't exist — the old import raised ImportError).

### Other
- Started this CHANGELOG; CLAUDE.md Git-workflow section now points at it.
- Updated CLAUDE.md fip_00 description.
- Diagnosed bad curation JSON: `DA_NE_4channel_datacuration_secondpass.json` is malformed
  (line 4) and lacks `correct_mapping`; use `firstpass`.
