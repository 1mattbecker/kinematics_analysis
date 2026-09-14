# Changelog

Notable changes to this project. Newest first. Dates are YYYY-MM-DD.

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
