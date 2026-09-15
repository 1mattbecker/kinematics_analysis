# Changelog

Notable changes to this project. Newest first. Dates are YYYY-MM-DD.

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
  `load_curation` + `apply_curation_df_fip`, reading CSVs from a new dependency
  (`aind_bwnm_fiber_data_curation_utils`), overwriting `df_fip['event']` with target names, and
  dropping `intended_measurement` entirely. `environment/Dockerfile:60` installs `@main`
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
