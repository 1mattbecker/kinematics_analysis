# Changelog

Notable changes to this project. Newest first. Dates are YYYY-MM-DD.

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
